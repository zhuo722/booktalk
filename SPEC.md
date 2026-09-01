# BookTalk — Technical Specification

Voice-based question answering over an uploaded PDF book.
Pipeline: browser mic → ASR → RAG over the book → LLM answer → TTS → browser playback.

This document is the single source of truth for implementation. Implementation agents
must follow the file tree, API contract, and module interfaces exactly. Where this spec
is silent, prefer the simplest solution consistent with the design rationale.

---

## 1. Goals and quality bars

- Upload a PDF book of any size (target: handles a 1000-page book; must handle ≥10 chapters).
- Ask questions by voice; hear the answer spoken back, with visible text + sources.
- Answers grounded strictly in the book; refuse when the book doesn't cover it.
- Retrieval must beat naive fixed-size chunking measurably (eval harness proves it).
- Latency target: first audible audio ≤ ~3.5 s after the question is transcribed.
- Works in Chinese and English (both book language and question language).
- All OpenAI calls behind one injectable client wrapper → fully testable offline.

## 2. Stack

- Backend: Python (≥3.9, system python3 on dev machine is 3.9.6 — all deps must support it), FastAPI + uvicorn. No LangChain/LlamaIndex — RAG is hand-rolled (small, auditable, testable; a deliberate design decision to document).
- Frontend: React 18 + TypeScript + Vite. No UI framework (no MUI/AntD/Tailwind) — hand-written CSS with design tokens. No component library "AI template" look.
- Storage: SQLite (stdlib `sqlite3`), one DB file `data/booktalk.db`. Embeddings stored as float32 BLOBs. Uploaded PDFs kept under `data/books/{book_id}.pdf`.
- OpenAI models (all configurable via env):
  - ASR: `gpt-4o-mini-transcribe` (fallback `whisper-1`)
  - LLM: `gpt-4o-mini` (env override e.g. `gpt-4o`)
  - TTS: `gpt-4o-mini-tts` (fallback `tts-1`), voice `alloy`, format mp3
  - Embeddings: `text-embedding-3-small` (1536 dims)
- `OPENAI_BASE_URL` env override supported (proxy-friendly).

## 3. Repository layout

> **Divergence note — added at the end of the project; the tree below is left as written.**
> This section records what was planned, not what shipped, and three things moved it:
> §14's research-driven revisions, which state what they override; the tests, which grew
> from the six-file sample sketched here to 326 unit and 43 integration tests (see
> [docs/TESTING.md](docs/TESTING.md)); and the interactive UI phase, which replaced the
> frontend outright — `Shell`, `Library`, `Toc`, `BookCard`, `SourceChip` and `Waveform`
> are gone, and `Shelf`, `ShelfBook`, `ShelfRail`, `Reader`, `BookPanel`, `SourceCard`,
> `Visualizer` and a `hooks/` directory took their place. The backend gained the modules
> §14 called for (`ingest/toc.py`, `headings.py`, `markdown.py`, `vocab.py`, `enrich.py`,
> `prompts.py`; `index/rerank.py`; `rag/citations.py`, `prompts.py`; `speech/pcm.py`,
> `turns.py`, `phonetics.py`). **The tree as built, module by module, is the "Module map"
> at the top of [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — that file, not this one,
> is where the current shape is documented.

```
booktalk/
  SPEC.md
  README.md                      # written in docs phase
  .gitignore                     # python, node, data/, .env
  docs/
    ARCHITECTURE.md
    TESTING.md
    AI_WORKFLOW.md
  backend/
    pyproject.toml               # PEP 621 metadata + deps; installable with `pip install -e .[dev]`
    .env.example                 # OPENAI_API_KEY=, OPENAI_BASE_URL=, BOOKTALK_LLM_MODEL=, ...
    app/
      __init__.py
      main.py                    # FastAPI app factory, CORS, static mount, router registration
      config.py                  # Settings dataclass read from env (no pydantic-settings dep)
      db.py                      # sqlite connection helpers, schema DDL, migrations-on-start
      models.py                  # pydantic response/request models (API layer only)
      services/
        openai_client.py         # AIClient protocol + OpenAIClient + FakeAIClient
      ingest/
        pdf.py                   # PyMuPDF extraction → PageText[] (text + font spans + page no)
        structure.py             # chapter/section detection → BookStructure tree
        chunker.py               # structure-aware chunking → Chunk[] (child + parent linkage)
        ingestor.py              # orchestrates parse→structure→chunk→embed→persist, progress callback
      index/
        store.py                 # persistence of books/chunks/embeddings; load per-book index bundle
        dense.py                 # cosine top-k over numpy matrix
        bm25.py                  # BM25 index (rank-bm25) + tokenizer (jieba for CJK, regex for latin)
        fusion.py                # reciprocal rank fusion
      rag/
        retriever.py             # hybrid retrieve → children → parent expansion → context pack
        rewrite.py               # conversational query rewrite (skips when no history)
        prompts.py               # all prompt templates as module constants
        pipeline.py              # answer_stream(): retrieve → LLM stream → sentence split → TTS
      speech/
        sentences.py             # incremental sentence splitter for streaming text (CJK + latin punctuation)
      api/
        books.py                 # upload/list/status/toc/delete
        asr.py                   # POST /api/asr
        chat.py                  # POST /api/chat (SSE)
      sessions.py                # in-memory chat history per (session_id, book_id), LRU capped
    tests/
      conftest.py                # FakeAIClient fixture, tmp DB, app fixture, synthetic PDF builders
      unit/
        test_structure.py
        test_chunker.py
        test_fusion.py
        test_sentences.py
        test_bm25_tokenizer.py
        test_dense.py
      integration/
        test_upload_and_ask.py   # full flow with FakeAIClient
        test_asr_endpoint.py
        test_error_paths.py      # bad file, unknown book, empty question, no key
    eval/
      make_fixture_book.py       # generates fixture PDF (≥12 chapters) + golden.json from a fact table
      facts.py                   # the deterministic fact table (see §10)
      run_eval.py                # retrieval + answer eval, --config full|naive, --fake for offline smoke
      README.md                  # how to run, and (later) results table
  frontend/
    package.json  vite.config.ts  tsconfig.json  index.html
    src/
      main.tsx  App.tsx
      lib/
        api.ts                   # typed fetch wrappers for the contract in §6
        sse.ts                   # POST-based SSE reader (fetch + ReadableStream parser)
        audioQueue.ts            # gapless-ish sequential playback of base64 mp3 segments (WebAudio)
        recorder.ts              # MediaRecorder (audio/webm;codecs=opus) + analyser levels
        types.ts
      components/
        Shell.tsx                # two-pane layout
        Library.tsx              # upload dropzone + book cards + ingest progress
        Toc.tsx                  # chapter list from /toc
        Conversation.tsx         # message list, sources chips, autoscroll
        MicButton.tsx            # record toggle + live level ring + states
        Composer.tsx             # text input fallback + mic
        SourceChip.tsx           # "Ch. 5 · p.123" → expandable quote
        Waveform.tsx             # live input level visualization
      styles/
        tokens.css  base.css  components.css
    mock/
      server.mjs                 # node http server implementing §6 with canned data (frontend dev without backend)
  scripts/
    dev.sh                       # starts backend (uvicorn) + frontend (vite) together
```

## 4. Ingestion design (the retrieval-quality core)

### 4.1 PDF parsing (`ingest/pdf.py`)
- PyMuPDF (`fitz`). For each page: `page.get_text("dict")` → lines with font size/flags.
- Output `PageText { page_no, lines: [{text, size, bold, y}] }`. Keep it memory-streamed:
  parse lazily per page; never hold list of full-res dicts for the whole book at once
  (accumulate reduced line records only).

### 4.2 Structure detection (`ingest/structure.py`)
Priority order:
1. Embedded TOC: `doc.get_toc()`. If ≥3 entries → use levels 1–2 as chapters/sections
   (map TOC page numbers to boundaries).
2. Heuristics: a line is a heading candidate if (a) matches
   `^(chapter|part|book)\s+([0-9ivxlc]+|one|two|...)\b.*` case-insensitive, or `^第[一二三四五六七八九十百0-9]+[章回部节].*`,
   or (b) font size ≥ 1.3 × body size (body size = mode of line sizes) AND ≤ 80 chars AND in top 25% of the page.
   Cluster candidates by size into chapter (larger) vs section (smaller) levels.
3. Fallback: no headings → one synthetic chapter "Full text". App still works; UI shows no TOC.

Output `BookStructure { title, chapters: [{title, page_start, page_end, sections: [...] }] }`.
Book title: PDF metadata title, else filename stem.

### 4.3 Chunking (`ingest/chunker.py`)
- Within each section (or chapter when no sections): rebuild paragraphs (join lines; blank-line
  or indentation breaks; drop running headers/footers — lines repeating on >30% of pages).
- Pack paragraphs into **child chunks**: target 450 tokens, hard max 600, overlap = 1 trailing paragraph.
  Token counting: `tiktoken` `cl100k_base`; fallback `len(text)//4` if tiktoken unavailable.
- **Parent context**: the section text, capped at 1600 tokens centered on the child.
- **Contextual header** (embedded and BM25-indexed with the child, not shown to LLM twice):
  `«{book_title}» — {chapter_title}[ — {section_title}]\n\n{chunk_text}`.
  Rationale: cheap version of contextual retrieval — situates the chunk without per-chunk LLM calls.
- Chunk metadata: `chapter_idx, chapter_title, section_title, page_start, page_end, char_span`.

### 4.4 Embedding + persistence (`ingest/ingestor.py`, `index/store.py`)
- Batch embed child chunks (batch 128 inputs per API call), store float32 BLOBs.
- Ingestion runs in a background thread; progress states: `parsing → structuring → chunking → embedding (with pct) → ready | failed(reason)`. Progress stored in DB, polled via `/status`.
- Idempotent: re-uploading same content (sha256 of file) returns existing ready book.

### 4.5 SQLite schema
```sql
books(id TEXT PK, title TEXT, filename TEXT, sha256 TEXT UNIQUE, pages INT,
      status TEXT, progress REAL, error TEXT, created_at TEXT);
chapters(book_id TEXT, idx INT, title TEXT, page_start INT, page_end INT,
         PRIMARY KEY(book_id, idx));
chunks(id TEXT PK, book_id TEXT, chapter_idx INT, section_title TEXT,
       page_start INT, page_end INT, text TEXT, header TEXT,
       parent_text TEXT, embedding BLOB, ord INT);
```
Per-book runtime bundle (LRU-cached, max 3 books): numpy matrix (normalized), BM25 index,
chunk metadata list.

## 5. Query pipeline

### 5.1 ASR (`api/asr.py`)
- `POST /api/asr` multipart `audio` (webm/opus from MediaRecorder; also accept mp4/m4a for Safari).
- Calls transcription model, returns `{ "text": str, "duration_ms": int }`.
- Reject empty/whisper-hallucinated results (len < 2 after strip) with 422 `{detail}`.

### 5.2 Chat (`api/chat.py` + `rag/pipeline.py`)
`POST /api/chat` JSON `{ book_id, session_id, question }` → SSE stream (`text/event-stream`
via StreamingResponse; no sse-starlette dep). Event sequence:

```
event: status   data: {"stage":"rewriting"|"retrieving"|"generating"|"speaking"}
event: question data: {"text": "<final question used for retrieval (post-rewrite)>"}
event: sources  data: {"sources":[{"chunk_id","chapter_idx","chapter_title","section_title","pages":[s,e],"preview":"<=200 chars"}]}
event: delta    data: {"text":"<token(s)>"}
event: audio    data: {"seq":0,"format":"mp3","b64":"..."}
event: done     data: {"answer":"...","timings":{"rewrite_ms","retrieve_ms","first_token_ms","first_audio_ms","total_ms"}}
event: error    data: {"message":"..."}
```

Steps inside `answer_stream()`:
1. **Rewrite** (`rag/rewrite.py`): only if session history non-empty — one `gpt-4o-mini` call
   turning the question + last 3 exchanges into a standalone query. Skip (identity) otherwise.
2. **Retrieve** (`rag/retriever.py`): dense top-24 (cosine over normalized matrix) + BM25 top-24
   → RRF (k=60) → top 8 children → group by parent, keep ≤5 parents, total context ≤ 4500 tokens.
3. **Generate**: streaming chat completion. System prompt (in `prompts.py`): answer only from
   excerpts; cite chapters inline as (Chapter N); if the excerpts don't contain the answer, say the
   book doesn't cover it — never improvise; reply in the language of the question; spoken register,
   2–6 short sentences unless asked for more; no markdown, no lists (it will be read aloud).
4. **Sentence-pipelined TTS** (`speech/sentences.py` + pipeline): as deltas stream, an incremental
   splitter emits complete sentences (`.!?。!?…` boundaries, min 25 chars — merge shorter with next;
   flush remainder at end). Each sentence → TTS call → `audio` event. Concurrency 2, order preserved
   by seq. First sentence goes to TTS as soon as it completes — this is the latency win.
5. Persist exchange into session history (in-memory, `sessions.py`, LRU 100 sessions × 20 turns).

### 5.3 Error handling
- Missing/invalid `OPENAI_API_KEY` → 503 on ASR/chat with actionable message; `GET /api/health`
  reports `{ok, openai_configured}`.
- OpenAI call failures mid-stream → `error` SSE event then close; frontend shows retry.

## 6. API contract (frontend ⇄ backend)

```
GET    /api/health                → {ok:true, openai_configured:bool}
POST   /api/books   (multipart "file")
                                  → 202 {book_id, status:"parsing"}  (or 200 existing ready book)
GET    /api/books                 → {books:[{book_id,title,pages,status,progress,created_at}]}
GET    /api/books/{id}/status     → {status, progress, error?}
GET    /api/books/{id}/toc        → {title, chapters:[{idx,title,page_start,page_end}]}
DELETE /api/books/{id}            → 204
POST   /api/asr     (multipart "audio")
                                  → {text, duration_ms}
POST   /api/chat    {book_id, session_id, question}
                                  → SSE per §5.2
```
Upload constraints: `.pdf` only, verify `%PDF` magic bytes, max size env-configurable
(default 200 MB). Errors: FastAPI default `{detail}` shape, proper status codes.
CORS: allow `http://localhost:5173` in dev.

## 7. Frontend UX spec

> **Divergence note — added at the end of the project; this section is left as written.**
> The interactions below — upload and progress, the mic's states, streaming with audio,
> stop, empty states, error handling — largely shipped as described. The surfaces they
> ship on did not, and the responsive rules here (a rail collapsing to a sheet) describe a
> layout that no longer exists. The warm-paper theme and its tokens were replaced twice —
> by the "Studio & Shelf" direction and then by
> **Studio Light**, whose palette, type and glass rules are
> [frontend/DESIGN.md](frontend/DESIGN.md). The 300px-rail two-pane layout is gone: the app
> is now two screens — a **Shelf** of books and a **Reader** that is the open book itself,
> with one cover flying between them ([frontend/DESIGN-SHELF.md](frontend/DESIGN-SHELF.md)
> for the shelf, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §8 for how the crossing and
> the two screens actually work). Sources are margin-note cards that turn the book to the
> cited passage rather than chips that expand a quote. Audio is raw PCM, not mp3 (§14.8),
> and the mock server sends generated PCM tones rather than a silent mp3. One correction
> that belongs here rather than in §14.6: **the low-confidence banner is not driven by the
> retrieval score.** It counts how many of the answer's sentences the post-hoc attribution
> could trace to a passage, and appears only when a substantial answer (>80 characters)
> traced none — the score was measured wrong in both directions on a real book (0.0017 for
> a question answered correctly from four cited passages, 0.9953 for a subject the book
> never mentions). The score is still computed and still sent on the `sources` event.

Layout: left rail (300px) = Library + TOC; main pane = conversation + composer.
Single committed visual theme — warm paper light theme:

```css
/* tokens.css */
--bg: #f6f1e7;          /* warm paper */
--surface: #fffdf8;
--ink: #1f1b16;         /* near-black warm */
--ink-soft: #6b6459;
--accent: #b4542a;      /* terracotta */
--accent-ink: #fff;
--line: #e3d9c8;
--radius: 10px;
font-display: "Fraunces", Georgia, serif;   /* book titles, headings (Google Fonts) */
font-body: "Inter", -apple-system, sans-serif;
```

> **Design authority note (added after skill adoption):** the design skills under
> `.claude/skills/` (`impeccable-craft`, `minimalist-ui`, `redesign-existing-projects`)
> are the binding authority for visual/interaction quality in ALL frontend work (build and
> polish) and override token details above where they conflict — notably: body font must NOT be Inter
> (pick a characterful sans per `minimalist-ui`), shadows near-invisible, one accent only,
> all component states (hover/focus/active/disabled/loading/error/empty) required.

Key interactions:
- Upload: drag-drop or click; during ingestion the book card shows stage + progress bar
  (poll /status every 1s); ready books show title, page count, chapter count.
- Mic: click to start (button morphs to a pulsing ring driven by real input level from
  the analyser), click to stop → "Transcribing…" → transcript appears as the user message
  → SSE answer streams in with a subtle text cursor; audio segments play as they arrive.
- Speaking state: small equalizer animation on the assistant message; a stop button
  cancels playback and aborts the stream (AbortController).
- Sources: chips under the answer ("Ch. 3 · pp. 41–44"); click expands the excerpt preview.
- Composer also has a text input (accessibility + testing without mic). Enter sends.
- Empty states: no book → centered upload hero with one-line explanation of the pipeline;
  book ready but no messages → 3 suggested questions (generic, e.g. "What is Chapter 1 about?").
- Errors: toast + inline retry on the failed message; mic-permission-denied explains browser fix.
- No dark mode (single committed look). Responsive down to 900px (rail collapses to a sheet).

Audio playback (`audioQueue.ts`): WebAudio — decode each base64 mp3 segment via
`decodeAudioData`, schedule buffers back-to-back on one `AudioContext` timeline
(next.start(max(ctx.currentTime, lastEnd))); expose stop() and an `onLevel` callback for
the equalizer. Handle iOS/Safari autoplay: create AudioContext on the mic click gesture.

Mock server (`frontend/mock/server.mjs`): implements §6 with canned data + a tiny silent
mp3 base64 for audio events; `npm run mock` runs it on :8000 so the UI is fully drivable
without the backend or an API key.

## 8. Configuration

`backend/.env` (loaded in `config.py` via os.environ + optional python-dotenv):
```
OPENAI_API_KEY=
OPENAI_BASE_URL=            # optional
BOOKTALK_LLM_MODEL=gpt-4o-mini
BOOKTALK_ASR_MODEL=gpt-4o-mini-transcribe
BOOKTALK_TTS_MODEL=gpt-4o-mini-tts
BOOKTALK_TTS_VOICE=alloy
BOOKTALK_EMBED_MODEL=text-embedding-3-small
BOOKTALK_MAX_UPLOAD_MB=200
BOOKTALK_DATA_DIR=./data
```

## 9. AIClient wrapper (`services/openai_client.py`)

```python
class AIClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def transcribe(self, audio: bytes, filename: str) -> str: ...
    def chat_stream(self, messages: list[dict], **kw) -> Iterator[str]: ...  # yields text deltas
    def chat(self, messages: list[dict], **kw) -> str: ...
    def tts(self, text: str) -> bytes: ...  # mp3
```
`OpenAIClient` implements with the `openai` SDK (retries: 2, timeout 60s).
`FakeAIClient` (used by tests and `--fake` eval): deterministic hash-based embeddings
(seeded RandomState from sha1(text), unit-normalized — similar texts do NOT get similar
vectors, tests must not assume semantic similarity, only exact-text determinism),
canned chat stream (echoes a template naming the retrieved chapter), 1-second silent mp3,
transcribe returns a fixed string. Injected via FastAPI dependency (`app.state.ai`).

## 10. Fixture book + golden set (`backend/eval/`)

- `facts.py`: deterministic fact table for a fictional non-fiction book
  “A Field Guide to the Meridian Archipelago” — 12 chapters (islands), each with ~8 structured
  facts (founder names, years, populations, landmarks, local dishes, distances...). Facts are
  invented (contamination-free: the LLM cannot know them from pretraining — this is the point;
  document it).
- `make_fixture_book.py`: renders the PDF with PyMuPDF's insert APIs or `reportlab`
  (pick reportlab; add to dev deps): title page, per-chapter heading “Chapter N — <Island>”
  in 20pt bold, body 11pt, several paragraphs per chapter weaving the facts into prose,
  ≥2 pages per chapter. Also emits `golden.json`:
  `[{q, expected_answer_substrings:[...], chapter_idx}]` — ~4 questions/chapter (48 total),
  derived from the same fact table, including some cross-chapter and some
  “not in the book” questions (expected refusal).
- `run_eval.py`:
  - Ingests the fixture with config `full` (this spec) or `naive`
    (800-char fixed chunks, no header, dense-only) — the ablation baseline.
  - Retrieval metrics: recall@5 / recall@8 (a retrieved chunk’s chapter_idx matches),
    MRR. Answer metrics: substring match rate on expected_answer_substrings;
    refusal correctness on out-of-book questions.
  - `--fake` runs the whole harness with FakeAIClient (validates plumbing, not quality).
  - Outputs a markdown table to stdout and `eval/report.md`.

## 11. Testing strategy

- Unit (offline, no network): structure detection on synthetic in-test PDFs (built with
  PyMuPDF: pages with sized fonts), chunker invariants (max tokens respected, no empty chunks,
  metadata correct, headers/footers stripped), RRF ordering math, incremental sentence splitter
  (latin + CJK + trailing flush), tokenizer (jieba path for CJK).
- Integration (offline): TestClient + FakeAIClient: upload small synthetic PDF → poll to ready →
  toc correct → chat SSE yields the exact event sequence (status→sources→delta+→audio+→done) →
  error paths (bad magic bytes 400, oversized 413, unknown book 404, no key 503).
- Frontend: `npm run build` must pass with zero TS errors; vitest units for `sse.ts` parser and
  `audioQueue` ordering logic (mock AudioContext).
- Answer-quality eval: §10 harness with real key (run later; numbers go into README).
- Manual E2E checklist in docs/TESTING.md (mic permission, Safari, large book, stop button).

## 12. Non-goals

- No auth/multi-user, no deployment infra (local run only; README documents `scripts/dev.sh`).
- No dark mode. No i18n of the UI chrome (English UI; answers follow question language).
- No cross-encoder reranker by default (latency); LLM rerank exists behind
  `BOOKTALK_RERANK=1` env flag: single gpt-4o-mini call scoring top-8 → keep top-5 parents.

## 13. Dependencies (pin compatible with Python 3.9)

Backend runtime: fastapi, uvicorn[standard], python-multipart, openai (≥1.40),
pymupdf, numpy (<2 if needed for 3.9 wheels), rank-bm25, jieba, tiktoken, python-dotenv.
Dev: pytest, httpx, reportlab.
Frontend: react, react-dom, typescript, vite, @vitejs/plugin-react, vitest (dev).

## 14. v2 revisions (research-driven; override earlier sections where stated)

Source: five-agent GitHub research pass (see docs/RESEARCH.md). Applied as an upgrade
phase on top of the v1 backend. Each item names the section it overrides.

### 14.1 Extraction (overrides §4.1)
- Use `pymupdf4llm` (verified installable on py3.9, ~50 MB) as the text-extraction layer,
  injecting our own structure detection via its `hdr_info=` hook. Measured payoff on a real
  244-page book: 129 ligature chars and 68 line-end hyphen breaks with raw `get_text()` vs
  0 and 0 with pymupdf4llm — de-hyphenation is a direct BM25 win. Note AGPL-3.0 in README.
- Scanned-PDF preflight: measure chars/page over the first 10 pages; an image-only PDF
  yields ~0 (born-digital ~1400+). Reject with a clear UI message (OCR out of scope).
- Keep repeated-line header/footer stripping (cross-page detector, not margin crops).

### 14.2 Structure detection ladder (overrides §4.2)
Tier 1: embedded TOC (`get_toc()`). Tier 2: printed-Contents-page parser (state machine
over the book's own contents pages; validate entries by monotonic page numbers). Tier 2.5:
heading-family voting, ported from RAGFlow `bullets_category`: score regex families
(`Chapter <roman|word|number>`, `PART ONE..`, `^\d{1,2}[.、 ]`, `第..[章节部]`, markdown `#`)
across all lines, take the argmax family. Tier 3: font-size histogram (v1 heuristic).
Plus: when a TOC exists but has no usable levels, one ingest-time LLM call assigns levels
(RAGFlow `assign_toc_levels`: strict JSON `[{level,title}]`, order preserved).
Never let a bare heading become its own chunk.

### 14.3 Chunking (overrides §4.3)
Port Docling `HybridChunker`'s three-pass algorithm (~150 lines, MIT) with `semchunk`
(verified py3.9) as the leaf splitter: (a) maintain `heading_by_level`, popping keys ≥ an
incoming heading's level, so each chunk carries its full ancestor breadcrumb; (b) budget by
CONTEXTUALIZED token count — `available = max_tokens - header_tokens`; (c) merge adjacent
undersized chunks only when heading paths are identical. Child target 256–320 tokens,
**zero overlap** (Chroma evidence: overlap hurt; 200/0 recursive hit 88.1% recall).
Guardrails: header ≤25% of chunk tokens and content ≥ floor (drop header rather than starve
body); omit header on a section's first chunk; truncate long header paths from the head
(keep most-specific levels).

### 14.4 Contextual enrichment upgrade (extends §4.3)
Config `BOOKTALK_CONTEXTUAL=1` (default on): per-chunk LLM-generated situating context
(Anthropic contextual retrieval; published 5.7%→2.9% retrieval-failure numbers), with the
context prompt scoped to the chunk's CHAPTER (not whole book) for cost. Concurrent ingest
calls; progress reflected in ingestion status. The static breadcrumb header remains the
fallback when the flag is off — and is an eval ablation arm.

### 14.5 Chapter summary nodes (extends §4.4; RAPTOR-lite)
One LLM summary per chapter (+one book-level), indexed as retrievable nodes tagged
`level`. Answers thematic questions flat chunks provably miss ("what is chapter 7 about").
Full RAPTOR clustering rejected (UMAP/numba, 2–3 pt gain, redundant with a real TOC tree).
Summaries double as TOC synopses in the UI.

### 14.6 Retrieval (overrides §5.2 step 2)
dense top-40 + BM25 top-40 → weighted RRF (sweep k∈{10,20,60} × weight splits in eval;
TREC's k=60 assumes million-doc corpora) → **FlashRank rerank** (`ms-marco-TinyBERT-L-2-v2`,
`flashrank` + `onnxruntime==1.19.2` — verified on this machine: 5 MB model, 7.6 ms per 20
passages, cp39 wheel resolves; pre-warm at startup) over top-20 CHILD chunks (max_length
512 would truncate parents) → top 8 → parent expansion only for chunks above a score
threshold → group by chapter, sort by reading order, merge contiguous runs, dedupe →
context ≤4500 tokens. Keep normalized per-signal scores alongside ranks: they drive an
abstain gate and a low-confidence UI banner ("Context relevance is low — double-check
this answer", kotaemon pattern).

### 14.7 Citations become post-hoc (overrides §5.2 step 3's inline "(Chapter N)")
The generation prompt no longer forces inline citation markers (spoken text stays clean).
After the answer completes: split into sentences, batch-embed, match each sentence against
retrieved chunk embeddings with hybrid similarity (tkweight 0.1 / vtweight 0.9), descending
threshold 0.63 → ×0.8 → floor 0.3, ≤4 sources/sentence (RAGFlow `insert_citations`).
Runs concurrently with the TTS tail; new SSE event `citations`
(`{turn_id, map:[{sentence_idx, chunk_ids}]}`) emitted before `done`. Zero added
time-to-first-audio.

### 14.8 Speech pipeline (overrides §5.2 step 4 and the audio event format)
- TTS output `response_format="pcm"` (24 kHz, s16le, mono). Audio SSE event:
  `{"turn_id","seq","format":"pcm","sample_rate":24000,"b64"}`. No mp3, no decode step.
- Sentence segmentation, two-phase (latency-optimal): first fragment emits on any soft
  delimiter (`,;:.!?…`) once ≥10 chars, force-emit at next space after ~30 words
  (stream2sentence params); every later chunk requires a true sentence boundary and ≥20
  chars (LiveKit `min_sentence_len`). Vendor LiveKit's `split_sentences` (~60 lines,
  Apache-2.0, attribution in file) + pipecat's non-whitespace lookahead ("$29." vs "$29. Next")
  + 12-char trailing guard + never-split-numbers. Table-driven unit tests.
- Back-pressure: sentence→TTS queue bound 1; pause synthesis while >5 s of audio remains
  unplayed. A cancelled turn wastes one sentence of spend, not the whole answer.
- Every SSE event carries `turn_id`. Cancellation: client aborts the stream and POSTs
  `/api/chat/turns/{turn_id}/cancel` with `{played_seq}`; server cancels LLM/TTS tasks,
  drains queues, truncates stored history to sentences actually played (vocode pattern) so
  follow-up questions see accurate context.

### 14.9 ASR hardening (extends §5.1; voice-native differentiator)
Named-entity corruption causes 87–96% of ASR-induced RAG failures (arXiv 2608.22872), and a
book gives us its entity vocabulary for free: at ingest, extract frequent proper
nouns/capitalized tokens; pass the top entities in the ASR request's prompt parameter
(verify current OpenAI semantics); post-ASR, run a Double Metaphone fuzzy-match layer
(pure Python) over the book vocabulary to repair entity mistranscriptions before retrieval.
The fixture book's invented names make this directly evaluable (ASR-corruption eval arm).

### 14.10 Eval upgrades (overrides §10/§11 metrics and arms)
- Metrics: recall@{1,3,5,10}, MRR, token-level IoU (Chroma-style precision signal),
  answer substring match, refusal correctness; per-category CSV + rank distribution.
- Categories: single-fact / cross-chapter synthesis / chapter-summary / not-in-book
  (abstain) / ASR-corrupted phonetic variants.
- Ablation arms: A naive fixed-800 dense-only · B +structure-aware chunks+breadcrumbs ·
  C +hybrid RRF · D +rerank · E +chapter summaries · F +LLM contextual enrichment.
  Plus the (k, weight) RRF sweep.
- ragas/deepeval REJECTED with reasons (both resolve on py3.9 but pull langchain/datasets/
  scipy resp. grpcio/otel/posthog; LLM-judge scores are poor instruments for small ablation
  deltas). Hand-rolled faithfulness judge (~30-line prompt on our client) instead.

### 14.11 Explicit rejections (document in README)
Late chunking (needs token-level embeddings; OpenAI returns pooled vectors only). HyDE
(full LLM round-trip on the first-audio critical path + entity hallucination risk).
Full RAPTOR (see 14.5). Semantic chunking as primary path (NAACL 2025 Qu/Tu/Bao + Chroma:
cost unjustified vs fixed-size on real documents) — optional ablation arm only.

### 14.12 Dependency additions (extends §13)
Backend: pymupdf4llm, semchunk, flashrank, onnxruntime==1.19.2, a Double Metaphone
implementation (small pure-python dep or vendored). Frontend (already instructed to the
frontend builder): radix-ui@1.6.7, eventsource-parser@4.1.0, use-stick-to-bottom@1.1.6,
@fontsource-variable/literata; PCM stream player; Tufte-sidenote citation layout;
vintage-paper palette (#f5f1e6 / #fffcf5 / #4a3f35 / #7d6b56 / #dbd0ba / accent #a67c52,
highlight #FBDA83); radius 0.25rem; offset shadows.

## 15. Definition of done per phase

- Backend agent: all endpoints per §6 implemented; `pytest tests/unit` green locally;
  app boots with no key (health reports openai_configured:false) — write the unit tests
  alongside the code.
- Frontend agent: `npm run build` green; UI fully drivable against `mock/server.mjs`.
- Tests agent: integration suite green offline; coverage of every §6 endpoint + SSE sequence.
- Eval agent: fixture book generates (≥12 chapters, ≥25 pages); `run_eval.py --fake` completes.
- Docs agent: README (quickstart ≤10 commands, architecture diagram in mermaid, design
  decisions incl. rejected alternatives, eval table placeholder), ARCHITECTURE.md,
  TESTING.md, AI_WORKFLOW.md skeleton.
