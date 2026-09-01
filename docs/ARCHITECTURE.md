# Architecture

How BookTalk turns a PDF into a book you can talk to, and why each piece is shaped the
way it is. The binding contract is [SPEC.md](../SPEC.md) — §1–§13 are the v1 design,
§14 is the research-driven v2 revision. Section numbers below refer to it.

Everything on the server is hand-rolled Python: no LangChain, no LlamaIndex, no vector
database.
The whole retrieval stack is a few hundred readable lines, which is the point — this is
the part of the system being judged, so it should be legible and ablatable rather than
configured.

§1–§7 are the server. §8 is the client, which is where the latency work is actually
heard and the citations are actually seen.

## Module map

```
backend/app/
  main.py            FastAPI factory · CORS · static mount · reranker pre-warm
  config.py          frozen Settings dataclass from os.environ + optional .env
  db.py              SQLite DDL, connection factory, migrate-on-start
  services/
    openai_client.py AIClient Protocol · OpenAIClient · FakeAIClient   ← the only seam
  ingest/
    pdf.py           PyMuPDF geometry (lines, font sizes, positions), scanned preflight
    markdown.py      pymupdf4llm text layer, driven by our own headings → Block stream
    structure.py     the five-rung structure ladder → BookStructure
    toc.py           printed-contents-page parser · LLM TOC-level assignment
    headings.py      heading-family voting (RAGFlow bullets_category)
    chunker.py       three-pass HybridChunker port · naive baseline for the eval
    enrich.py        contextual enrichment · chapter/book summary nodes
    vocab.py         proper-noun harvesting for ASR biasing and repair
    prompts.py       every ingest-side prompt as a module constant
    ingestor.py      background-thread orchestration + progress
  index/
    store.py         persistence + the per-book runtime bundle (LRU 3)
    dense.py         cosine top-k over a normalized float32 matrix
    bm25.py          BM25Okapi + CJK/latin tokenizer
    fusion.py        weighted reciprocal rank fusion
    rerank.py        FlashRank cross-encoder, loaded once, never fatal
  rag/
    retriever.py     hybrid retrieve → rerank → parent expansion → context pack
    rewrite.py       conversational query rewrite
    citations.py     post-hoc sentence → chunk attribution
    pipeline.py      answer_stream(): the SSE state machine
    prompts.py       every answer-side prompt as a module constant
  speech/
    sentences.py     streaming two-phase splitter (vendored LiveKit port)
    pcm.py           the 24 kHz s16le constants and duration math
    turns.py         turn identity · cancellation · AudioPacer back-pressure
    phonetics.py     Double Metaphone entity repair
  api/               books.py · asr.py · chat.py
  models.py          Pydantic request/response bodies — the API boundary only
  sessions.py        in-memory history, LRU 100 sessions × 20 turns
```

---

## 1. Ingestion

Runs on a background thread the moment an upload lands, writing progress into the `books`
row that `GET /api/books/{id}/status` polls. The bar is one 0…1 float split across stages
so it always moves forward:

```
parsing 0.00–0.25 · structuring 0.25 · chunking 0.30
contextualizing 0.35–0.60 · summarizing 0.60–0.70 · embedding 0.70–1.00
```

Stages that are switched off simply do not happen and the bar jumps — a stage that did
not run should not fake a duration.

### 1.1 Two extraction passes, on purpose

`ingest/pdf.py` walks pages lazily with PyMuPDF, reducing each `get_text("dict")` payload
to small `Line` records (text, font size, bold flag, x/y) and discarding the full payload
before reading the next page. Peak memory therefore stays flat on a 1000-page book.

Those geometric records are not what gets indexed. They are what *structure detection*
reads — font sizes, the top-of-page test, the cross-page repeated-line detector all need
coordinates that markdown has thrown away. The **indexed text** comes from
`ingest/markdown.py`, which calls `pymupdf4llm.to_markdown`.

Why two passes instead of one: pymupdf4llm undoes ligatures and end-of-line hyphenation
that raw `get_text()` leaves in place. Measured on a real 244-page book, raw extraction
produced 129 ligature characters and 68 hyphen line-breaks; pymupdf4llm produced **0 and
0**. De-hyphenation is a direct BM25 win, because `light-\nhouse` does not tokenize to
`lighthouse`. What we do *not* take from it is its `IdentifyHeaders` font-ranking — we
already ran a five-rung ladder over the same document, so the detected outline is fed
back in through the documented `hdr_info=` hook. The markdown that comes out carries
*our* `#` levels, which makes the `Block` stream a cheap, faithful serialization of the
structure we detected.

Two more extraction rules:

- **Scanned-PDF preflight** (§14.1): characters-per-page over the first 10 pages. A
  born-digital book measures ~1400+; an image-only scan measures ~0. Below 20 the upload
  fails with a clear message. OCR is out of scope and says so rather than silently
  producing an empty index.
- **Running heads and feet** are removed by a cross-page repeated-line detector, not by
  cropping margins. A margin crop also eats the first line of body text on a page whose
  header sits low, and misses a header that sits high on a short page.

### 1.2 The structure ladder (§4.2, §14.2)

Five rungs, tried in order, each broader and weaker than the one above it:

| Rung | Signal | Notes |
|---|---|---|
| 1 | **Embedded TOC** — `doc.get_toc()` | The publisher already told us the answer. |
| 2 | **Printed contents page** | State machine over the pages a reader would call "Contents": find the marker (English or CJK), read `title … page` entries, **validate by monotonic page numbers**, then map printed numbers onto physical pages (they differ by the front-matter offset). |
| 3 | **Heading-family voting** | Ported from RAGFlow's `bullets_category`. Instead of asking "does this line look like a heading?", score whole *families* of numbering conventions (`Chapter <roman\|word\|number>`, `PART ONE`, `^\d{1,2}[.、 ]`, `第..[章节部]`, markdown `#`) across every line in the book and take the argmax. A book numbers its chapters exactly one way, so the winning family *is* the book's convention — and every line matching it is a heading, however plain its typography. Matching walks patterns deepest-first, so `1.2.3 Ferries` is a third-level heading, not a first-level `1.`. |
| 4 | **Font histogram** | The v1 heuristic: lines ≥1.3× the modal body size, ≤80 characters, in the top 25% of a page, clustered by size into chapter and section levels. |
| 5 | **Fallback** | One synthetic "Full text" chapter. The app still works; the UI shows no TOC. |

**Plus an LLM rung that is not a rung.** Rungs 1 and 2 can produce a *flat* list with no
hierarchy — a contents page with no numeric prefixes, for instance. When that happens and
no offline signal (the heading families) can supply levels, one strict-JSON ingest-time
call assigns an outline level per title, order preserved (RAGFlow's `assign_toc_levels`).
It is a repair for a specific defect, not a tier in the ladder, and it never runs when an
offline signal exists.

Every unit carries an `anchor` — the `(page_no, y)` of the line it starts on — and
chapter `idx` is 1-based, matching the "Chapter N" a reader sees. **A bare heading never
becomes its own chunk**; it opens the body of the first chunk beneath it.

### 1.3 Chunking: Docling's HybridChunker, ported (§4.3, §14.3)

Three passes over the `Block` stream, with `semchunk` as the leaf splitter:

1. **Assemble.** Walk the blocks keeping a `heading_by_level` dict. An incoming heading
   pops every key **at or below** its own level — an h2 ends the previous h2 and
   everything under it, but not its h1 ancestor — so every base chunk carries the full
   ancestor breadcrumb it was written under.
2. **Split.** Any base chunk whose *contextualized* token count exceeds the cap is split.
   Contextualized is the load-bearing word: the budget is `available = max_tokens −
   header_tokens`, so a chunk under a deep heading path gets less body, and the thing
   that gets embedded never overflows. `semchunk` recurses through paragraph → sentence →
   word separators and cuts at the most semantic boundary that fits.
3. **Merge.** Adjacent undersized chunks merge **only when their heading paths are
   identical**, so a chunk can never straddle two sections. A final split pass guarantees
   the cap survives merging.

Children are **256–320 tokens with zero overlap**. Chroma's chunking study (5 corpora,
472 queries) measured recursive 200/0 at 88.1% recall and found every overlap arm worse:
overlap buys duplication, not recall.

**Breadcrumb budget rules** — the header is `«Book» — Chapter — Section`, and three
guardrails keep it from eating the chunk:

- the header may not exceed **25%** of the chunk's token budget;
- the body may not fall below a **96-token content floor**;
- when both cannot be satisfied, the path is trimmed **from the head** (book title first,
  outermost level next) because the most specific level is the one that actually
  distinguishes this chunk — and if even a bare path fails, the header is dropped rather
  than the body starved;
- a chunk that **opens a section** already contains its heading in the body, so it is
  indexed with no header at all. Repeating it would index the same words twice.

**Parents.** The `parent_text` of a child is the surrounding section text, grown
symmetrically outward from the child until a 1600-token cap. Children are what retrieval
*scores*; parents are what the model *reads*.

The eval's arm A baseline lives in the same module (`naive_chunks`): fixed 800-character
chunks off the raw PDF lines, no header, no parent, running heads kept — so the ablation
measures the extraction win too, not just the chunking win.

### 1.4 Enrichment (§14.4, §14.5)

Two optional LLM passes, both behind the `AIClient` seam, both run through an 8-worker
thread pool because they are IO-bound and a 600-chunk book would otherwise serialize ten
minutes of round trips.

- **Contextual enrichment** (`BOOKTALK_CONTEXTUAL=1`, default on) — Anthropic's contextual
  retrieval: a short LLM-written sentence saying where each chunk sits, prepended to what
  gets embedded and BM25-indexed (published effect: 5.7% → 2.9% retrieval failures). The
  prompt is scoped to the chunk's **chapter**, not the whole book, which is what makes it
  affordable. With the flag off, the static breadcrumb header stays — that is eval arm B.
- **Summary nodes** (RAPTOR-lite) — one synopsis per chapter plus one for the book,
  indexed as ordinary retrievable chunks but tagged with a `kind` and a `level`. They
  answer the thematic questions ("what is chapter 7 about") that flat chunks provably
  miss, and double as TOC synopses in the UI. Full RAPTOR clustering was rejected: the
  chapter tree already exists, so the clustering step is redundant.

**Vocabulary harvesting** runs in the same phase. `ingest/vocab.py` extracts the book's
proper nouns with a deliberately dumb distributional test — a term must appear
capitalized **away from a sentence start** at least twice, and must almost never appear
lowercase. That keeps "Harbour" the place and drops "However" the adverb and "harbour"
the common noun. No POS tagger, no NER model. CJK is skipped: it has no capitalization to
read and Double Metaphone is an English algorithm, so a CJK book gets an empty vocabulary
and the untouched v1 behaviour. Capped at 200 terms — beyond that the ASR prompt is full
anyway and the long tail is where false repairs come from.

### 1.5 Embedding and persistence

Child chunks are embedded in batches of 128 (`header` is the embedded document, not
`text`) and stored as float32 BLOBs. Ingestion is idempotent by file sha256: re-uploading
the same content returns the book that already exists — **200 rather than 202** —
whatever stage it has reached, so uploading a book that is still being indexed joins the
run in progress instead of starting a second one. A book whose previous attempt *failed*
is the exception: its row and its id are reused, and ingestion starts again.

---

## 2. Retrieval (§5.2 step 2, §14.6)

```
dense top-40 ┐                        ┌ FlashRank over 20 CHILD chunks ┐
             ├─ weighted RRF (k=20) ──┤                                ├─ top-8
BM25  top-40 ┘                        └ (~8 ms, cross-encoder)         ┘
                                                  │
          ≤5 blocks, ≤4500 tokens ── reading order ── parent expansion (score-gated)
```

**Wide legs.** 40 per leg, not 24. Fusion can only reorder what the legs surfaced, and
the reranker can only sharpen what fusion kept. Widening costs a slice of numpy and a
BM25 argsort.

**Weighted RRF at k=20.** `score(d) = Σ_lists w / (k + rank(d))`, ranks 1-based. RRF
merges by *position*, so a BM25 score of 14.2 and a cosine of 0.83 never have to be made
commensurable. `k` controls how fast weight decays with rank; the familiar k=60 comes
from Cormack et al.'s TREC corpora of millions of documents, where a flat curve rewarding
cross-list agreement is the right bet. One book is four orders of magnitude smaller and
its ranks are far more informative, so small-corpus guidance (k=10–20) applies. Swept in
eval over {10, 20, 60} × three weight splits — the result was flat, so the default stands
on reasoning, not measurement. Ties break on first-seen order, which keeps output
deterministic.

**Rerank over children only.** FlashRank's `ms-marco-TinyBERT-L-2-v2` truncates at 512
tokens. A 1600-token parent would be judged on its opening third, so only the ~320-token
children are scored. Three properties the module exists to guarantee: it is **never on
the import path** (both imports live inside `load_reranker`, so the app and the test suite
boot without them installed); it is **never a surprise download** (`allow_download=False`
refuses to construct a Ranker unless the model is already unpacked, which is what the
offline suite relies on); and it is **never fatal** (every failure path returns `None` and
the caller keeps its fusion order). It is loaded once at startup on a daemon thread —
first inference on a cold session is ~10× steady state, and that is not latency to spend
on a reader's first question.

**Gated parent expansion.** A weak match that drags in 1600 tokens of section text buys
noise. A chunk earns its parent window only if its final score is within **0.3 ×** the
best hit's score. Expressed as a *ratio*, deliberately: it then means the same thing
whether `final` is a cross-encoder probability or a normalized fused rank, and the top
chunk always expands, so no question can end up with child-only context by accident.

**Reading-order assembly.** Chunks are grouped into blocks by parent section (chapter
synopses are kept in their own block — merging a summary with the prose it summarizes
would present it as if it were that page). Blocks compete for the 4500-token budget
**best-first**, because the budget must protect the strongest evidence, then the survivors
are re-sorted by `(chapter_idx, ord)` and handed to the model in the book's own sequence
rather than in a relevance-ordered shuffle. Within a block, paragraphs are merged and
deduped — which collapses both the small overlap between neighbouring children and the
much larger overlap between two parent windows cut from the same section — and any jump
between non-adjacent chunks is marked `[…]` so the model cannot mistake a gap for
continuous prose. At most 5 blocks.

**Scores are kept, not discarded.** Every kept chunk carries normalized dense / BM25 /
RRF / rerank values in 0..1, and the result carries `top_score`, which leaves on the
`sources` event as `confidence`. It is only an *absolute* quantity when a cross-encoder
ran: without one, fused ranks are normalized against the best hit and it reads 1.0 for
anything that matched at all.

**Nothing is gated on it.** There is no server-side abstain gate — refusal is a rule in
the answer prompt, not a threshold — and the client's warning is computed from the
answer, not from this score (§8.5). That is a measured decision, not a stylistic one. In
the eval's arm F — the shipped configuration: structure-aware chunks, hybrid retrieval,
the cross-encoder, summary nodes and contextual enrichment — the `confidence` column
averages 0.985 on single-fact questions and 0.994 on ASR-corrupted ones against 0.038 on
questions the book cannot answer (0.003 in arm D) — but 0.469 on cross-chapter questions
and 0.950 on chapter-summary ones, so the separation exists in some categories and not in
others.
Without a reranker every category reads 1.000. On a real book the ordering inverted
outright: *"what happens in chapter 2"* — answered correctly from four cited passages —
scored **0.0017**, while a subject the book never mentions scored **0.9953**. Wrong in
both directions. The number is still computed, still sent, and still reported per question
by the eval; it just decides nothing.

**Query rewrite** (`rag/rewrite.py`) runs only when the session has history: one call
turning the question plus the last 3 exchanges into a standalone query. With no history
it is the identity function and costs nothing.

An **optional single-call LLM rerank** (`BOOKTALK_RERANK=1`, SPEC §12) still exists but is
off by default: it adds a full round trip before the first token, and the cross-encoder
now covers the same ground for ~8 ms. A malformed reply leaves the order untouched.

---

## 3. Post-hoc citation attribution (§14.7)

v1 asked the model to write "(Chapter 3)" inline. That is the wrong instrument for a
product that is *spoken*: the marker gets read aloud, it costs tokens on the critical
path, and the model is guessing at attribution it was never shown.

So attribution happens after the fact, on `aux_pool`, while the TTS tail is still
synthesizing — the same trick as RAGFlow's `insert_citations`:

1. Split the finished answer into **display sentences** — deliberately *not* the streaming
   splitter, which will happily emit a fragment at a comma. Citations want units a reader
   can see a source attached to, so this splitter only cuts at a real sentence end
   (abbreviation list, decimal guard, CJK enders self-delimiting). It returns
   `(sentence, start, end)` triples rather than strings; the offsets index into the answer
   exactly as it was generated, which is what the client needs (below).
2. Embed all sentences in **one batched call**.
3. Score each sentence against every candidate chunk with a hybrid similarity:
   **0.9 × cosine + 0.1 × token overlap**. Vector similarity catches paraphrase, which is
   most of what a generated sentence is; token overlap catches the names, numbers and
   dates embeddings smooth away — and, capped at 0.1, can never manufacture a citation out
   of function words alone.
4. Select with a **descending threshold**: start at **0.63**, multiply by **0.8** per step,
   never go below **0.3**. A confident match is required first; a weaker one is accepted
   only when nothing better exists. At most **4 sources per sentence**.
5. A sentence with no match above the floor gets **no citation at all**. "We do not know
   where this came from" is the honest answer, and the client renders that sentence
   unmarked.

Candidates are only the chunks that **actually reached the model and were sent to the
client** — a chunk dropped by the context budget fails on both counts, since the model
never read it and the client has no id to render against. Chunk embeddings are read
straight out of the book's in-memory matrix (already L2-normalized), so attribution never
re-embeds the book. The module is pure: one embedding call, no database, no HTTP, no
globals.

The event goes out as `citations` immediately before `done`:

```json
{"map": [{"sentence_idx": 0, "span": [0, 74], "chunk_ids": ["…", "…"]}],
 "turn_id": "…"}
```

**`span` is the load-bearing field.** It holds `[start, end)` character offsets into the
answer string that `done` carries — the same string, unstripped, that the splitter
measured — so the client highlights a cited sentence by *slicing* the answer at those
offsets. It never re-splits.

A splitter on the client would be a second opinion about where sentences are, and the map
is only meaningful under the server's. The two would disagree in exactly the places that
are hard: decimals (`3.5`), abbreviations (`e.g.`), domains (`example.com`), CJK enders.
A single disagreement would not fail loudly — it shifts every subsequent `sentence_idx`,
so citations quietly attach to the wrong sentences from there on, which is worse than no
citations at all. Offsets remove the question.

---

## 4. The speech pipeline (§5.2 step 4, §14.8)

The latency trick is here. Time-to-first-audio is dominated by how many LLM tokens we
wait for before the first synthesis call, so we wait for as few as we honestly can.

### 4.1 Two-phase sentence segmentation

**Where does a sentence end?** A vendored port of LiveKit Agents' rule-based tokenizer
(Apache-2.0, attributed at the vendored block in `speech/sentences.py`): a cascade of
regexes that *mark* every full stop which is not a boundary — `Dr.`, `e.g.`, `U.S.`,
`3.5`, `example.com` — leaving the unmarked ones as sentence ends. Stdlib `re` only: no
NLTK, no `nltk.download()` at runtime, no corpora to ship. Marks use same-length
private-use sentinels so every offset in the analysed copy still addresses the original
string. Three guards are bolted on top:

- **pipecat's non-whitespace lookahead** — on seeing `.?!;…`, wait for the next
  non-whitespace character before committing. Their example is exactly the case that
  matters: `"$29."` is not a sentence, `"$29. Next"` is.
- **stream2sentence's 12-character trailing guard** — never look for a boundary inside the
  last 12 characters of the buffer, because the token that would disambiguate them has not
  arrived yet.
- **Never split numbers** — a full stop with digits on both sides is a decimal point or a
  thousands separator, never a sentence end.

CJK enders (`。！？…`) are self-delimiting and always boundaries.

**When is it worth speaking?** Two phases:

| | Trigger | Minimum |
|---|---|---|
| **First fragment** | any soft delimiter — `,;:.!?…` and CJK equivalents | ≥ **10** characters, force-emitted at the next space after ~**30 words** |
| **Every later chunk** | a true sentence boundary only | ≥ **20** characters (LiveKit `min_sentence_len`), shorter ones merge forward |

"In chapter seven," is a perfectly good thing to start speaking. Fragmenting costs
nothing in prosody past the opening, because phase two never does it.

### 4.2 Raw PCM on the wire

TTS is requested as `response_format="pcm"`: headerless signed 16-bit little-endian mono
at 24 kHz. The audio event is
`{"turn_id", "seq", "format": "pcm", "sample_rate": 24000, "b64"}`.

No container, no codec, no `decodeAudioData` step, no ffmpeg on either side. And because
the format is fixed and headerless, **byte length is duration** — exactly, not
approximately — which is what the back-pressure estimate runs on. The browser pushes the
samples straight onto one AudioContext timeline, scheduling each buffer at
`max(currentTime, lastEnd)`, strictly in `seq` order; out-of-order arrivals wait for their
predecessors and are never reordered.

### 4.3 Back-pressure

Synthesis runs ahead of playback by design — that is the win — but running too far ahead
is pure waste. Three bounds:

- **TTS concurrency 2**, with a queue bound of **1** sentence behind what is in flight.
- **Synthesis pauses entirely** while `AudioPacer` estimates more than **5 seconds** of
  emitted audio is still unplayed (PCM byte count vs. wall clock).
- **A 30-second absolute cap** on time spent waiting, because an estimate must never be
  able to wedge a stream. On expiry it logs, acknowledges, and resumes.

The result: a turn cancelled after two sentences has paid for roughly two sentences, not
six. The cancellation check sits immediately before the TTS call, so anything still queued
when a cancel lands is never paid for at all. And `finish()` is a generator rather than a
list, so each segment reaches the socket the moment it is synthesized — releasing the
whole chain at once would delay the tail, start the pacer's clock on bytes the client has
not been sent, and leave a mid-tail cancel with nothing left to stop.

A TTS failure on one sentence degrades that sentence to text-only. It never kills an
answer the reader can already see.

### 4.4 Turn lifecycle, cancel and history truncation

A spoken answer is not finished when the text is finished — it is finished when the audio
has been *heard*. That gap is what `speech/turns.py` owns.

**Every SSE event of a turn carries the same `turn_id`.** Two invariants are enforced in
the `answer_stream` wrapper rather than at each `yield` inside the body, so a later edit
cannot forget one: the id is stamped on every event, and **no event escapes once the turn
is cancelled** — including the segment already in flight when the cancel arrived. The
client keeps a set of dead turn ids and drops anything belonging to them.

**Cancel** is `POST /api/chat/turns/{turn_id}/cancel` with `{played_seq}` — the last
segment the browser actually finished playing. Closing the socket alone is not enough: the
server would keep synthesizing sentences nobody will hear, and stored history would claim
the assistant said things that were never spoken. So the client aborts *and* posts.

**History truncation** (the vocode pattern) is the payoff. On cancel, the stored exchange
is rewritten to the sentences actually played, so a follow-up ("what about its harbour?")
is resolved against a transcript the user recognises. Recording an empty answer drops the
exchange entirely, which is the right outcome when the user cut in before a word was
spoken. The endpoint is idempotent and deliberately forgiving: cancelling a turn that has
already finished still corrects the history, because "the user stopped listening after
sentence two" is true whether or not the server had already written sentence six. 64
turns are retained per process so a cancel racing the end of a stream still finds its
turn.

### 4.5 ASR hardening (§14.9)

Named-entity corruption causes 87–96% of ASR-induced RAG failures (arXiv 2608.22872), and
a corrupted proper noun is invisible: a BM25 miss plus an embedding pointing somewhere
else. A book hands us the fix for free, because it is a closed corpus whose names we have
already read. Two uses of the harvested vocabulary:

1. **Bias the decoder** — the top **60** terms go into the transcription request's
   `prompt`, behind a fixed opener (`"Proper nouns from the book being discussed: …"`).
   Whisper truncates the prompt at 224 tokens, so the list is capped at **896 characters**
   and terms are added in frequency order: a truncated hint still carries the names most
   likely to be spoken.
2. **Repair what still came back wrong** — Double Metaphone over the same vocabulary. It
   encodes how a word *sounds* and returns two codes for genuinely ambiguous cases
   ("Thorne" → `0RN` and `TRN`), so a transcript that heard the second pronunciation still
   matches.

A wrong repair is worse than the miss it replaces, because the reader sees a plausible
sentence that quietly retrieves the wrong chapter. So a token is rewritten only when *all*
of: ≥4 characters, latin, not a common English word, not already in the vocabulary,
phonetic code ≥3 characters (two-character codes like `FL` collide across half the
language), textual similarity ≥0.6 on top of the phonetic match, **and exactly one**
vocabulary term survives all of it. Two candidates mean we do not know, and the original
wins.

**A hint has two failure modes, and the shipped code guards both.** The `prompt`
parameter is advisory, and advisory parameters fail in ways that are not errors:

- **The request is rejected for carrying it.** `OpenAIClient.transcribe` catches that one
  class of error and retries the same audio with no prompt. A transcript with a mangled
  name beats no transcript.
- **The response *is* the prompt.** A hinted decoder that hears no speech sometimes emits
  its own hint back as the transcript — the reader records a question, and what returns is
  the server's list of names. `app/api/asr.py` tests every transcript with
  `speech/phonetics.py:echoes_prompt`; when it fires, the same audio is transcribed again
  with no hint at all, because the vocabulary is worth having and the question is worth
  more. If the retry echoes too, the transcript is dropped and the endpoint reports what
  actually happened — silence — as the 422 the composer renders as "nothing intelligible
  was heard".

**Why the echo test matches the prompt's opening words.** `echoes_prompt` lowercases the
transcript and asks whether it *starts with* the first 24 characters of the template's
fixed opener — `"proper nouns from the bo"`. The first instrument tried was a
token-overlap ratio between transcript and prompt, which looked more thorough and was
wrong in the one direction that matters. A book's vocabulary contains most of the content
words a short question is made of — *"Who settled Kestrel Cove?"* names exactly the place
the hint lists — so real questions scored high overlap and were being thrown away. The
opener, by contrast, is server-written text no reader would ever say, which makes matching
it evidence rather than correlation. Twenty-four characters is the trade: long enough to be
unambiguous, short enough to survive the decoder dropping a word or changing the
punctuation after it.

`book_id` is optional on `/api/asr` and an unknown book is ignored rather than rejected.

---

## 5. Storage

One SQLite file, `data/booktalk.db`, WAL mode, `synchronous=NORMAL`, 30-second busy
timeout. Connections are created per call and never shared across threads, because
ingestion runs on a background thread while the API serves requests.

```sql
books    (id PK, title, filename, sha256 UNIQUE, pages, status, progress, error, created_at)
chapters (book_id, idx, title, page_start, page_end, PRIMARY KEY(book_id, idx))
chunks   (id PK, book_id, chapter_idx, section_title, page_start, page_end,
          text, header, parent_text, embedding BLOB, ord,
          kind TEXT NOT NULL DEFAULT 'text', level INTEGER NOT NULL DEFAULT 0)
vocab    (book_id, term, count, ord, PRIMARY KEY(book_id, term))

idx_chunks_book(book_id, ord) · idx_chunks_kind(book_id, kind)
idx_books_sha(sha256)         · idx_vocab_rank(book_id, ord)
```

`text` is the source preview, `header` is what gets embedded and BM25-indexed (breadcrumb
or LLM-written context + the chunk), `parent_text` is what the model reads. `kind`/`level`
tag summary nodes; `vocab.ord` is the rank, so the top N reads without re-sorting.

**Migrate-on-start ordering matters.** `init_db` executes tables first, then column
migrations (`ALTER TABLE … ADD COLUMN` for anything a pre-existing database lacks), then
indexes — because `idx_chunks_kind` is an index *over a migrated column* and would fail
against a database created before that column existed. This ordering is a bug fix with a
regression test; see [AI_WORKFLOW.md](AI_WORKFLOW.md).

**The runtime bundle.** On the first query for a book, its rows are loaded once into a
`BookIndex`: a normalized float32 numpy matrix, a BM25 index and the chunk metadata list.
Bundles are cached for the 3 most recently used books. At book scale (tens of thousands of
chunks) cosine top-k is a single matrix-vector product well under a millisecond, which is
why there is no vector database here.

---

## 6. The AIClient boundary

```python
class AIClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def transcribe(self, audio: bytes, filename: str, prompt: str | None = None) -> str: ...
    def chat_stream(self, messages: list[dict], **kw) -> Iterator[str]: ...
    def chat(self, messages: list[dict], **kw) -> str: ...
    def tts(self, text: str) -> bytes: ...   # raw 24 kHz s16le PCM
```

Every network call the app makes goes through this Protocol. That single seam is what
makes the whole pipeline — including the HTTP surface, the SSE stream and the eval
harness — runnable offline and deterministically.

- **`OpenAIClient`** wraps the official SDK, 2 retries, 60 s timeout. `base_url` is
  **always passed explicitly**: with `None` the SDK falls back to the `OPENAI_BASE_URL`
  environment variable, and a blank `OPENAI_BASE_URL=` line in `.env` (which dotenv loads
  as `""`) then silently breaks every request. `config.py` normalizes empty to `None` and
  the client substitutes the real default. ASR and TTS fall back to `whisper-1` / `tts-1`
  when the configured model is unavailable on an account.
- **`FakeAIClient`** returns deterministic hash-seeded embeddings (`sha1(text)` →
  `RandomState` → unit vector), per-sentence synthetic PCM, a canned transcript and canned
  chat replies. Callers tag completions with `purpose="rewrite" | "rerank" | "answer" |
  "toc_levels" | "context" | "summary"`; `OpenAIClient` strips the tag, and `FakeAIClient`
  uses it to return a reply of the right *shape* for that call site — a JSON level list
  for `toc_levels`, a situating sentence for `context`, and so on. Identical text always
  yields an identical vector, but *similar* texts get unrelated vectors: tests may rely on
  exact-text determinism only, never on semantic similarity.

`create_app(settings=..., ai=..., store=...)` takes all three, so the entire HTTP surface
runs against the fake with no monkeypatching. With no key at all, `build_client` returns
nothing and the model-dependent endpoints answer 503 with instructions instead of failing
obscurely.

---

## 7. The wire

```
GET    /api/health                     → {ok, openai_configured}
POST   /api/books  (multipart "file")  → 202 {book_id, status} for new content
                                       | 200 {book_id, status} for a sha256 already
                                         ingested or ingesting (§1.5)
GET    /api/books                      → {books:[{book_id, title, pages, status,
                                                  progress, created_at}]}
GET    /api/books/{id}/status          → {status, progress, error?}
GET    /api/books/{id}/toc             → {title, chapters:[{idx, title, page_start,
                                                  page_end, synopsis?, opening?}]}
DELETE /api/books/{id}                 → 204
POST   /api/asr    (multipart "audio", optional "book_id")
                                       → {text, duration_ms}
POST   /api/chat   {book_id, session_id, question}
                                       → SSE
POST   /api/chat/turns/{turn_id}/cancel  {played_seq}
                                       → {turn_id, cancelled, played_seq, answer}
```

`synopsis` and `opening` are **absent, not null**, for a chapter that has neither: the
response is served with `response_model_exclude_unset`, so `chapter.synopsis` is the only
check the client needs. `duration_ms` on `/api/asr` is how long the transcription took —
the client already knows the length of its own recording, and the server cannot measure
audio duration without decoding the container.

SSE is written by hand over `StreamingResponse` — the wire format is four lines of code
and one fewer dependency than `sse-starlette`. Headers include `X-Accel-Buffering: no` so
no proxy sits on the stream. Event order for one turn:

```
status(rewriting|retrieving) → question → [status(retrieving)] → sources
→ status(generating) → delta* → status(speaking) → audio(seq 0..n)* → citations → done
```

| Event | Payload |
|---|---|
| `status` | `{"stage": "rewriting" \| "retrieving" \| "generating" \| "speaking"}` |
| `question` | `{"text"}` — the query retrieval actually ran, i.e. the rewrite when there was history to rewrite against, else the question unchanged |
| `sources` | `{"sources": [{"chunk_id", "chapter_idx", "chapter_title", "section_title", "pages": [start, end], "preview", "text", "scores": {"dense", "bm25", "rrf", "rerank", "final"}}], "confidence"}` |
| `delta` | `{"text"}` — one token-ish slice of the answer, as it streams |
| `audio` | `{"seq", "format": "pcm", "sample_rate": 24000, "b64"}` |
| `citations` | `{"map": [{"sentence_idx", "span": [start, end], "chunk_ids"}]}` (§3) |
| `done` | `{"answer", "timings": {"rewrite_ms", "retrieve_ms", "first_token_ms", "first_audio_ms", "total_ms"}}` |
| `error` | `{"message"}` — replaces the tail on failure |

`preview` is the first 200 characters of `text`; `scores` are the normalized signals of
§2 and `confidence` is the best `final` among them. Every event carries `turn_id`, stamped
in one place (§4.4) rather than at each `yield`.

---

## 8. The client

`frontend/src` is React 18 on Vite, in TypeScript: ~4,100 lines of TSX and ~3,500 lines of
CSS. No state library, no router, no component framework. The dependencies that are here
earn their place at a boundary: `eventsource-parser` for SSE framing, three Radix
primitives (tooltip, popover, collapsible) for focus and keyboard behaviour nobody should
re-implement, `use-stick-to-bottom` for the thread's scroll anchoring, and three
`@fontsource-variable` faces so the type is not a runtime network dependency.

```
frontend/src/
  main.tsx           React root · the three font faces · the three stylesheets
  App.tsx            the two views, the shared AudioQueue, the flight rects, toasts
  hooks/
    useChat.ts       one live SSE turn → per-book threads · cancellation by turn id
    useLibrary.ts    book list · upload · 1 s status polling while ingesting · TOC cache
  lib/
    api.ts           typed fetch wrappers + ApiError · fire-and-forget cancelTurn
    sse.ts           POST-based SSE reader (eventsource-parser + AbortController)
    audioQueue.ts    base64 PCM → one AudioContext timeline, gapless, strictly by seq
    recorder.ts      MediaRecorder push-to-talk; exposes its input node for the meter
    types.ts         the wire contract of §7, in TypeScript
  components/
    Shelf.tsx        the landing screen: tick rail, rows, scroll restore, help
    ShelfBook.tsx    one volume in CSS 3D · the opening move · the flight home
    ShelfRail.tsx    the fixed tick index down the left edge
    Reader.tsx       one book open: the bar, the book panel, the conversation column
    BookPanel.tsx    the flying cover · the chapter page lifted out of it · the pager
    Conversation.tsx the thread · one exchange · span-sliced answer · the sources panel
    SourceCard.tsx   one margin note; its reference turns the book to that passage
    Composer.tsx     the mic and the text field, and the ASR round trip
    MicButton.tsx    the record control and its live level ring
    Visualizer.tsx   one canvas waveform, pointed at either direction of audio
    Hero.tsx         the empty-library panel, shared by both screens
    IngestPanel.tsx  the six ingest stages, while a book is being indexed
    StatTile.tsx     the one stat surface: the percentage, counting up
    Toast.tsx · Tip.tsx · Icons.tsx   notifications, tooltips, the icon set
    cloth.ts         the per-book palette: shelf position → five custom properties
  styles/
    tokens.css       the design tokens — one light world, one theme colour
    base.css         reset, typography, themed browser surfaces, reduced motion
    components.css   every component's rules
```

### 8.1 Two screens, no router

`App.tsx` holds `view: "shelf" | "chat"` in a `useState`, and that is the entire
navigation model. Three things the code does would each have to be worked around under a
router:

- **Everything that must survive the crossing lives above both screens.** One `AudioQueue`
  (`useMemo`, constructed once), the chat threads, the library and its polling all hang
  off `App`. A route swap that unmounted the reader would silence a spoken answer and drop
  a live SSE turn on the floor. Here, leaving the reader stops the turn because
  `backToShelf` calls `stopAll()` — a decision, not a teardown.
- **Leaving is not instantaneous.** On the way back, the shelf branch *also* renders the
  departing `Reader` above itself for 300 ms (`leavingReader`), so both screens can move at
  once (§8.2). Two screens mounted at the same time is awkward to express as one route.
- **The transition is passed as measured geometry.** The flight is driven by a `DOMRect`
  handed from one screen to the other as a prop. A URL cannot carry that, and a deep link
  would arrive with nothing to fly from.

The cost is real and worth naming: there is no shareable link to a book, and the browser's
Back button leaves the app rather than returning to the shelf. For two screens that is a
fair trade; a third screen would change the answer.

**The shelf** is the landing screen and the only one that can stand with an empty library.
**The reader** is one book — the book itself on the left, the conversation on the right,
the composer docked under it. Its main column is chosen from the book's state: `Hero` when
the library is empty, `IngestPanel` while the book is being indexed or has failed,
`Conversation` once it is ready, and an empty `.studio__body` as the default so the
composer stays anchored to the bottom in transitional frames instead of jumping. The
departing copy of the reader gets `flightFrom={null}`, no-op handlers and an empty body:
at that moment it is scenery, and the object it used to hold is on the shelf.

Two guards close the loop. Removing the last book while the reader is open sends the view
back to the shelf — there is nothing left to be in. And changing `activeBookId` runs
`stopAll()` *and* clears the open citation, because chapter numbers mean nothing across
books: kept, it would have turned the next book to an unrelated chapter.

**State ownership** is split by lifetime, not by screen:

| Hook | Owns | Notes |
|---|---|---|
| `useLibrary` | `books`, `loaded`, `activeBookId`, `tocs` | Polls `/status` every 1 s while any book is ingesting and re-lists when one turns ready. Its `INGESTING` set mirrors the backend's non-terminal statuses — omitting one silently stops the poll mid-ingest. A failed TOC fetch is swallowed: the TOC is an enhancement, the book still answers. |
| `useChat` | `threads` (per book), `streamingBookId`, `speaking` | One live turn at a time (`activeRef`), a set of dead turn ids, and one session id minted per book on its first question and kept for the life of the page — so server-side history is scoped to the book, and switching books does not resolve a follow-up against another book's transcript. |
| `App` | `view`, `flightFrom`, `returnFlight`, `leavingReader`, `cited`, toasts | Everything that is about *the crossing* rather than about a book. |

Toasts dedupe by message, so a failing poll cannot stack the same sentence six deep, and
one health probe at startup raises the "running without an OpenAI key" notice above the
composer rather than letting the first question discover it.

### 8.2 One object crosses

Opening a book is one continuous movement of one object. Everything else on both screens
gets out of its way and comes back.

**Out, shelf → reader.**

1. The click sets `opening` on the row. CSS takes the volume out of its shelf pose — the
   46° tilt goes flat, it rises 16px and scales 1.08 over 150 ms — and three leaves spread
   out from under the cover, staggered 40 ms apart.
2. After **280 ms**, `ShelfBook` measures `.shelf-book__volume` and hands the rect up. The
   measurement happens *after* the spread deliberately: the flight has to start from where
   the book is now, not from where it stood before it opened.
3. `App.openBook` records the shelf's `scrollTop`, selects the book, stores the rect as
   `flightFrom`, and switches the view.
4. `BookPanel` runs a FLIP against that rect in a `useLayoutEffect`: the panel renders
   where it belongs, and the cover is animated in from where the book was.

What actually moves is one node — `.tome__block`, the cover, carrying the book's title.
The cover carries the type across because without it the book arrived blank and the title
vanished at the hand-off, which read as a cut rather than a move. The wordmark does not
animate at all: both screens draw the same `.brand` block at the same inset, so it is
continuous rather than transitioned. The background is one element too — `App` renders
`.app-ground` (three slow colour fields) outside both branches and only changes
`--ground-strength`, 1 on the shelf and 0.28 in the reader over 900 ms, so crossing is a
change of attention rather than a change of place.

Three decisions inside the animation, each of them a fix:

- **Web Animations API, not a CSS transition.** A transition released inside
  `requestAnimationFrame` lost its release whenever React re-rendered mid-flight. An
  `Animation` object survives that, and can be cancelled.
- **The box is animated, not a transform.** The book is a wide bar on the shelf and an
  upright cover in the reader, so the flight changes its *shape* as well as its place. A
  scale would stretch the title with it; animating `width`/`height` lets the title reflow
  as the cover turns.
- **Running animations are cancelled before measuring.** React mounts effects twice in
  development, and a surviving flight parks the element on its first keyframe — where its
  rect is the shelf's, not the destination's.

**Where the landing is guaranteed.** `land()` is scheduled *before* anything that can
fail, and it is idempotent behind a `done` flag. It is reached from four directions: the
animation's `finished` promise, the reduced-motion branch, an early return when there is
no node or the destination measures 0×0, and a `FLIGHT_MS + 200` = **820 ms** fallback
timer. Landing is what opens the book — it sets `landed`, which reveals the chapter page
and adds `.tome--open`, and calls `onArrived` so `App` can clear `flightFrom`. A flight
that cannot run is a bad transition; a page that never opens is a broken screen, so the
timer is armed first and the animation is the optimisation.

The chapter page is held back until landing for a related reason: it is a child of the
flying element, so drawing it early sent a half-transparent chapter drifting across the
conversation.

**Back, reader → shelf.** `backToShelf` stops the turn, measures `.tome__block` where it
now stands, stores `{bookId, rect}`, and switches the view *immediately* — then keeps the
reader mounted as a fixed, pointer-transparent layer above the shelf for 300 ms. That
overlap is the point: the book starts morphing back into its row at the same moment the
reader's panels leave, rather than after them.

`ShelfBook` runs the reverse FLIP with one step the outbound flight does not need: it
**measures the row flat**. A volume's resting rect is the bounding box of an object tilted
46°, which is 103px tall for a book that is 148px tall; ending the animation on that number
left the book visibly squashed. So the transform comes off for exactly one measurement and
goes straight back, and the flight ends on the real box with the tilt and lean restored.
Before any of that it calls `onBeforeMeasure()`, because the shelf has to be scrolled back
to where the reader left it before a row can say where it is — which is also why `Shelf`
keeps its scroll restore as a *function* that runs once rather than in the scroll
container's ref callback: the returning row's layout effect runs first.

**Reduced motion** is honoured at each seam rather than suppressed globally: the spread
delay collapses to 0, both flights short-circuit straight to their landing, the leaves,
the screen entrances and exits, the cited page's ring and the drifting ground all stop, and
the rail's `scrollIntoView` switches from `smooth` to `auto`.

### 8.3 The audio path

```
POST /api/chat ─▶ eventsource-parser ─▶ useChat.onEvent ─▶ audioQueue.enqueue
                                                                 │
                                        base64 → Int16LE → Float32│
                                                                 ▼
                                AudioBuffer started at max(currentTime, lastEnd)
```

**SSE over POST.** The browser's `EventSource` is GET-only and cannot carry a JSON body or
be aborted, so `lib/sse.ts` posts and pipes `response.body` through `TextDecoderStream`
into `eventsource-parser`'s `EventSourceParserStream`. The library owns the wire grammar —
events split across chunks, multi-line `data:`, comments — and this module owns the two
parts that are ours: the POST contract, and an `AbortSignal` that cancels the reader so no
event is delivered after an abort. A non-2xx never becomes a stream: it throws an
`ApiError` carrying the server's own `detail`, which is what the toast shows.

**Turn identity comes first.** Before dispatching on the event name, `useChat` reads
`turn_id`: if it is in the dead set the event is dropped, which is exactly the late audio
segment of a turn the reader has already interrupted. The first id seen is stored on the
turn and patched onto the exchange, so the cancel endpoint has something to name.

**PCM, decoded by hand.** `decode()` base64-decodes, walks the bytes as signed 16-bit
little-endian samples into a `Float32Array` (`/ 32768`; a trailing odd byte is ignored
rather than counted as half a sample) and copies them into `ctx.createBuffer(1, n, 24000)`.
No `decodeAudioData`, no container, no codec. A format that is not `"pcm"` falls back to
`decodeAudioData`, and a decode that throws returns `null` and is *skipped* rather than
allowed to stall the queue.

**Gapless scheduling.** `drain()` releases only the segment whose `seq` is next, starts it
at `max(ctx.currentTime, nextStartTime)`, and advances `nextStartTime` by that buffer's own
duration — so consecutive segments abut exactly on one timeline, and an out-of-order
arrival waits in `pending` for its predecessors instead of being reordered. `currentTime`
is the floor for the first segment and after any underrun, which keeps the queue from
scheduling into the past. A decode that finishes after a `stop()` is discarded by an
`epoch` counter: it belongs to a dead turn.

**`playedSeq` means heard, not sent.** The only writer of `maxPlayedSeq` is a source node's
`onended`, and `stop()` nulls every `onended` before stopping its node — so reaching that
handler can only mean the segment played to its natural end. That number is what the server
truncates stored history against (§4.4), so it has to mean exactly that.

**Holes are released at the end.** On `done` the client calls `finish()`: no missing `seq`
can arrive now, so anything buffered behind a gap is played. The backend skips a `seq` when
one sentence's TTS fails (§4.3), and waiting on that hole would silence the whole rest of
the answer over one lost sentence.

**Cancellation, in this order.** `stopAll()`:

1. adds the live `turn_id` to the dead set, so anything already in flight is ignored;
2. reads `audioQueue.playedSeq` and posts `/api/chat/turns/{id}/cancel` with it —
   fire-and-forget, `keepalive: true`, errors swallowed, because cancelling must never
   block the next question;
3. aborts the fetch;
4. finalizes the exchange as `stopped`, keeping the partial text already on screen;
5. stops every scheduled node, clears `pending`, and resets the ordering to seq 0.

`playedSeq` is the one number the server rewrites history against, so it is protected from
both ends of that teardown: `useChat` reads it before step 5, and `stop()` itself *freezes*
`maxPlayedSeq` into `frozenPlayedSeq` (and the elapsed seconds into `frozenPlayed`) before
resetting them, so the getter still answers correctly for anything that reads it after the
queue has been torn down.

`ask()` calls `stopAll()` first, so asking a new question is also how you interrupt the old
one. `audioQueue.start()` is called synchronously inside the click or submit handler —
Safari and iOS only allow an `AudioContext` to be created or resumed inside a user gesture
— and `Composer` does the same through `primeAudio` on the mic press, before any await.

**One waveform, both directions.** `Visualizer` takes a context and a node and draws real
analyser data; the composer points it at the recorder's `MediaStreamAudioSourceNode` while
recording, and the answer card points it at `audioQueue.outputNode` — the gain node the
played signal passes through — while speaking. The stroke is the element's own `color`, so
a class picks the direction: coral for live input, blue for playback.

### 8.4 The citation path

The client is a renderer of §3, and its design follows from one rule: **the server's
sentence boundaries are the only ones**.

**Highlighting by slicing.** `FinalAnswer` takes the `answer` string from `done` and the
`map` from `citations`, keeps the entries whose `span` is well-formed and in range
(`0 ≤ start < end ≤ answer.length`, at least one chunk id), sorts them by start offset and
walks the string with a cursor: text before a span is emitted plain, the span itself
becomes a `.sentence` element, and the tail after the last span is emitted plain. A span
overlapping the previous one is skipped rather than nested, so prose is never duplicated
or lost. Sentences with no entry stay unmarked — the honest rendering of "we do not know
where this came from".

There is no sentence splitter anywhere in the client, deliberately. It would be a second
opinion about where sentences end, disagreeing with the server on decimals and
abbreviations, and one disagreement silently shifts every later `sentence_idx` onto the
wrong text.

**While streaming**, the answer is the accumulated `delta` segments, unhighlighted: spans
cannot exist yet, because attribution runs on the finished answer. On `done` the segments
are dropped and the server's `answer` becomes the single string the offsets index into.

**The link runs both ways** off one piece of state per exchange (`hoveredChunk`): hovering
a cited sentence lights the source cards it names, and hovering a card lights every
sentence citing it — a `--blue-soft` wash on the sentence, an inset blue ring and a 1px
lift on the card.

**The sources panel** is a Radix `Collapsible` labelled "Used N sources", holding two
renderings of the same list:

- **margin notes** (`SourceCard`) — white cards in the right margin beside the answer, each
  with its number, its `Ch. N · pp. a–b` reference, chapter and section titles, and the
  passage itself, clamped to four lines. A "Full passage" toggle appears only when the card
  is actually holding something back: the passage runs past its 200-character `preview`, or
  that preview is long enough for the clamp to clip it;
- **a compact trace** — one row per source: index, reference, chapter and section, chunk id.

Which is visible is decided by a **container query on the conversation column**, not by the
viewport: `.reader__talk` is an inline-size container, and below 980px *of column* the
margin is dropped and the same `SourceCard`s move inline into the disclosure (with a
viewport backstop at 1280px for the same switch). The reader gives the conversation 53% of
the screen, so a viewport-keyed breakpoint would have printed margin notes into a column
with no margin to print them in.

**Turning the book to a cited passage.** A card's reference is a button. Pressing it lifts
`{chapterIdx, pages, text, chunkId}` to `App` as `cited`; `BookPanel` finds the chapter
with that `idx` and turns to it — but prints **the passage's own pages and the passage's
own words**, not the chapter's opening. "The chapter it came from" is not where the answer
came from, and the top of a thirty-page chapter would leave the reader hunting for the
sentence they just heard. The page is keyed on the chunk id, so a second citation replays
the lift out of the book instead of cross-fading text in place; it arrives ringed in blue
and settles over 1.4 s — long enough to find, short enough not to sit there glowing — and
the passage carries the same wash the cited sentence in the answer carries, so both ends of
the link are visibly the same colour. Moving the pager clears the citation back to the
chapter's opening; so does switching books.

### 8.5 The grounding warning is measured on the answer

The warning above an answer — *"None of this answer could be traced to a passage — check
it against the book"* — is **not** driven by the retrieval confidence score. It appears
when three things hold: the turn is finished, `citations` arrived and is **empty**, and the
answer is longer than **80 characters**.

The condition is a count: how many of the answer's own sentences post-hoc attribution
could trace back to a retrieved passage. Zero of them, on an answer long enough to have
said something, is the case worth flagging — and it is a judgement about the answer rather
than about the query. The reason it is measured there is itself measured, on a real book:
*"what happens in chapter 2"*, answered correctly from four cited passages, scored a
retrieval confidence of **0.0017**, while a subject the book never mentions scored
**0.9953**. Wrong in both directions, so the score cannot carry this.
Citation coverage separates the same two cases cleanly, because it is computed from the
thing the reader is being asked to trust.

The 80-character floor is the other half of the rule: a refusal ("the book does not cover
that") is short and cites nothing *by design*, and warning about it would be nonsense.

The score is still computed, still sent on the `sources` event, and still reported per
question by the eval as `confidence`; `useChat` parses it onto the exchange. Nothing in the
answer card is conditioned on it.

### 8.6 The styling system

Three stylesheets, loaded in order by `main.tsx`: `tokens.css` (the vocabulary),
`base.css` (reset, typography, browser surfaces), `components.css` (every rule). No
CSS-in-JS and no utility framework. The only styles JavaScript writes are custom properties
whose *values* are data: a book's cloth, the mic's live level, the ground's strength.

**Tokens.** One light world, deliberately — no dark surface exists anywhere in the system,
so there is no second palette to keep honest. One canvas (`#e9ebf2`) carrying two static
radial washes on `<body>`, white cards, one decisive theme colour (`--blue`), and coral
(`--live`) reserved for recording and nothing else. The scales are fixed rather than fluid:
eight type sizes from 12 to 44, a 4px space scale, three card shadows built as offset +
blur from a single light source above, plus two more for the shelf's volumes, which sit
*on* the ground rather than floating over it and so take a contact shadow under the cast
one. Every pairing carries its measured contrast ratio in a comment beside it, which is
what makes the palette checkable rather than merely asserted.

**Glass, on three surfaces.** The toasts, the shelf's help button and the chapter pager
frost — the surfaces that genuinely float over scrolling content, which is the rule. Each
is declared twice, and the order matters: the *default* declaration is the opaque
`--glass-solid` (0.92 white), and the translucent 0.72 variant with `backdrop-filter` is
layered on only inside `@supports (backdrop-filter: blur(1px))`. A browser without the
filter therefore gets an opaque panel rather than text over moving content, and both states
were measured — which is why `tokens.css` records a ratio for each. The two ambient washes
exist for this: glass needs colour underneath it, or it frosts grey over grey.

**The per-book cloth.** `cloth.ts` is the single source of a book's colour, and it is a
pure function of the book's *position in the library*. `clothForIndex(index)` returns five
custom properties — two gradient stops, the page-edge tint, an on-cloth ink and the
gradient's angle — which the shelf row and the reader's cover both read, so a book is the
same colour in both places. That is what lets one object cross between the screens without
a costume change.

The palette is 16 composed hue pairs × 2 tones = **32 cloths**. Stepping a hue wheel in 32
equal steps was tried and was wrong twice over: at one saturation it read as a highlighter
set, and with both stops a few degrees apart no cover showed a gradient at all. These pairs
each travel 24–70° of hue and are chosen as combinations — peach into rose, mint into sky,
periwinkle into orchid — with tone as the second axis: an airy and a deeper version of the
same pair. Walking the palette *by position* rather than hashing the title is what makes a
shelf look composed: the first sixteen books take the sixteen pairs, the next sixteen
repeat them deeper, and only the thirty-third book wears a colour already on the shelf.
Hashing guaranteed that to nobody — two books could collide on the second row. The gradient
angle walks too (`112 + (slot % 5) × 14`), so neighbours differ by more than hue. The
module records the contrast that constrains all of it: `--ink` measures ≥7.7:1 on every
stop of all thirty-two, and the tinted on-cloth secondary ink 4.89:1 on the worst.

**Motion is state, not decoration.** 120–240 ms with one shared easing for interface
feedback; the long numbers (620 ms flights, 900 ms ground) belong to the one movement that
*is* the subject rather than feedback about it. `base.css` collapses animation and
transition durations under `prefers-reduced-motion`, and every JS-driven movement checks
the same query itself — `element.animate()` is not a CSS animation, so that blanket rule
never reaches it.

---

## 9. Borrowed algorithms

Re-implemented against our own data structures, never copy-pasted, except the one
explicitly vendored block. Full survey and the rejected set: [RESEARCH.md](RESEARCH.md).

| Borrowed | From | Where it lives |
|---|---|---|
| `bullets_category` heading-family voting | RAGFlow | `ingest/headings.py` |
| `assign_toc_levels` strict-JSON level assignment | RAGFlow | `ingest/toc.py` |
| `insert_citations` hybrid sentence attribution | RAGFlow | `rag/citations.py` |
| `HybridChunker` three-pass assembly | Docling (IBM, MIT) | `ingest/chunker.py` |
| Chunk size / zero-overlap evidence | Chroma chunking technical report | `ingest/chunker.py` |
| Contextual retrieval | Anthropic | `ingest/enrich.py` |
| RAPTOR, degenerate chapter-level case | RAPTOR (ICLR 2024) | `ingest/enrich.py` |
| `split_sentences` rule tokenizer (**vendored**, Apache-2.0) | LiveKit Agents | `speech/sentences.py` |
| Non-whitespace lookahead | pipecat | `speech/sentences.py` |
| `minimum_first_fragment_length`, `force_first_fragment_after_words`, `context_size` | stream2sentence | `speech/sentences.py` |
| `SentenceStreamPacer` back-pressure shape | LiveKit Agents / RealtimeTTS | `speech/turns.py` |
| Generation-id'd cancellation, PCM stream player | openai-realtime-console | `speech/turns.py`, `frontend/src/lib/audioQueue.ts` |
| History truncation to what was played | vocode | `rag/pipeline.py` |
| Honest-UI warning when an answer may be ungrounded | kotaemon | `Conversation.tsx` — the *pattern* is borrowed, the signal is not: kotaemon warns from a retrieval score, which measured wrong in both directions here (§2), so the banner counts traced sentences instead (§8.5) |
| Eval harness shape (categories, per-query CSV) | Onyx `run_search_eval.py` | `eval/run_eval.py` |
