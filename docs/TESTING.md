# Testing

Three layers, all of them cheap enough to run on every change, plus one measurement pass
that costs money and one checklist that needs a human with a microphone.

| Layer | What it is | Cost |
|---|---|---|
| **369 offline tests** | `pytest` against `FakeAIClient`; no network, no key, no real data directory | ~seconds |
| **11 frontend units + `tsc`** | vitest over the SSE parser, the audio queue, and the request deadlines; a zero-error TypeScript build | ~seconds |
| **Eval harness** | six ablation arms over a fixture book, `--fake` offline or real-key | ~6 s fake, ~minutes + API spend real |
| **Manual E2E** | the things a test runner cannot hold: a microphone, Safari, a lock screen | ~10 minutes |

The first three are the fast loop. The fourth is where the bugs that a test runner cannot
see get found: the two backend bugs that shipped, and most of the defects in the
interactive UI phase, which were found by running the thing rather than by testing it —
see [AI_WORKFLOW.md](AI_WORKFLOW.md).

---

## Running everything

```bash
cd backend
.venv/bin/python -m pytest                      # 369 tests, offline
.venv/bin/python -m pytest tests/unit -q        # 326
.venv/bin/python -m pytest tests/integration -q # 43

cd ../frontend
npm run test                                    # 11 vitest units
npm run build                                   # tsc -b && vite build, must be zero-error
```

No `OPENAI_API_KEY` is needed, and the suite refuses to depend on one: `conftest.py`
clears every `OPENAI_*` and `BOOKTALK_*` variable, points the data directory at a
`tmp_path`, and injects `FakeAIClient`. A developer with a key in their shell and a
developer with none run exactly the same tests.

---

## What makes it possible: one seam and one fake

Every model call in the app goes through the `AIClient` Protocol
(`app/services/openai_client.py`). `FakeAIClient` implements it deterministically:

- **Embeddings** are seeded from `sha1(text)` → `numpy.RandomState` → unit-normalized
  vector. Identical text always yields an identical vector, so retrieval is reproducible.
  **Similar texts get unrelated vectors** — dense similarity is noise by construction, and
  tests may rely on exact-text determinism only, never on semantic closeness. This is a
  deliberate constraint: a fake that faked semantics would let a broken pipeline pass.
- **Chat** is canned per call site. Callers tag completions with `purpose="rewrite" |
  "rerank" | "answer" | "toc_levels" | "context" | "summary"`, and the fake returns a reply
  of the right *shape* for each — a strict-JSON level list for `toc_levels`, a situating
  sentence for `context`, an ordered index list for `rerank`. That is what lets ingestion's
  LLM tiers be exercised offline instead of skipped.
- **TTS** returns synthetic PCM whose length is a function of the sentence, so
  `seq` ordering, the byte-count-is-duration invariant and the `AudioPacer` back-pressure
  arithmetic are all testable without audio.
- **Transcription** returns a fixed string and records the `prompt` it was given, which is
  how the ASR vocabulary-biasing tests assert the hint was actually sent.

The fake also records every call (`embed_calls`, `chat_calls`, `tts_calls`,
`transcribe_prompts`), so tests assert on *what was asked of the model*, not just on the
answer.

**Synthetic PDFs** are built in-test with PyMuPDF at real coordinates — sized fonts,
margins, running heads and folios — so structure detection, the repeated-line detector and
the page-number mapping are exercised against actual geometry rather than a mock.

---

## Unit layer — 326 tests

Densest where the logic is subtlest:

| Area | Tests | What is pinned |
|---|---|---|
| `test_retriever.py` | 32 | Leg widths, weighted RRF wiring, rerank ordering and its failure fallbacks, the 0.3 parent-expansion gate, reading-order assembly, gap markers, dedupe across overlapping parents, the 4500-token budget, normalized score fields. |
| `test_citations.py` | 29 | The 0.9/0.1 hybrid similarity, descending threshold 0.63 → ×0.8 → 0.3, the ≤4-source cap, candidate restriction to chunks the client was actually sent, the display-sentence splitter (abbreviations, decimals, CJK, closers), and the character spans that let the client highlight by slicing the answer instead of re-splitting it. |
| `test_sentences.py` | 23 | Table-driven streaming splitter: `Dr.`, `e.g.`, `U.S.`, `3.5`, URLs, `"$29."` vs `"$29. Next"`, the 12-char trailing guard, first-fragment soft-delimiter emission, the ~30-word force, the 20-char floor after the first, CJK, trailing flush. |
| `test_pipeline.py` | 22 | The SSE event sequence and its ordering invariants, `turn_id` stamping, cancellation silence, back-pressure pausing and the 30 s wait cap, TTS failure degrading one sentence only, citation hand-off to the aux thread. |
| `test_chunker.py` | 21 | Three-pass invariants: token cap respected *after* contextualization, no empty chunks, headings never standalone, merge only within identical heading paths, header ≤25% and the 96-token content floor, head-first path trimming, parent windows, the naive baseline. |
| `test_phonetics.py` | 19 | Every repair guard: length, code length, similarity floor, single-candidate rule, common-word rejection, and the ASR prompt cap. |
| `test_vocab.py` | 15 | The distributional proper-noun test, CJK skip, the 200-term cap, and re-ingest/delete lifecycle in the `vocab` table. |
| `test_asr_api.py` · `test_chat_api.py` | 14 · 9 | Endpoint contracts, validation, 503 without a key. |
| `test_turns.py` | 13 | Turn registry, cancel idempotence, `played_seq` truncation, the 64-turn retention window, `AudioPacer` arithmetic. |
| `test_structure.py` · `test_toc_parser.py` · `test_headings.py` | 13 · 11 · 10 | Each rung of the ladder in isolation and the order they are tried in: embedded TOC, printed contents (monotonic validation, front-matter offset), family voting argmax, font histogram, fallback. |
| `test_fusion.py` · `test_dense.py` · `test_bm25_tokenizer.py` | 12 · 9 · 11 | RRF math and tie determinism, cosine top-k and stable ties, the CJK/latin tokenizer split. |
| `test_rerank.py` | 11 | The three guarantees: no import at module load, no download without the cached model, never fatal. |
| `test_enrich.py` · `test_ingestor.py` · `test_markdown_blocks.py` | 10 · 7 · 10 | Enrichment call shapes and skip paths, stage/progress sequencing, the block stream and running-head removal. |
| `test_speech_client.py` · `test_client_params.py` | 9 · 6 | PCM constants and duration math; kwarg normalization at the SDK boundary (notably `model=None` meaning "unset", never a literal null), plus the two base-URL regressions — a blank `OPENAI_BASE_URL=` in the environment never reaching the SDK, and a configured one being used. |
| `test_sessions.py` | 8 | LRU 100 sessions × 20 turns, thread safety. |
| `test_db_migration.py` | 2 | **Regression:** `init_db` against a v1-era `chunks` table with no `kind`/`level`. Asserts the columns are added *and* that `idx_chunks_kind` exists afterwards — the index-before-column ordering bug. |

---

## Integration layer — 43 tests

`TestClient` + `FakeAIClient`, exercising the real HTTP surface end to end.

**`test_upload_and_ask.py` (16)** — the happy path and its edges: `/api/health` reports
that the app runs without a key at all; ingestion walks the documented stages to `ready`;
re-uploading the same content returns the existing book; the library and TOC match the
book's own structure; each chapter gets a retrievable summary node (and a chapter with no
summary has no synopsis field); the TOC carries each chapter's own opening prose, which is
what the reader prints on a chapter page; **the answer stream follows the documented event
order**; the `sources` event names real chunk ids and carries a confidence; audio segments
arrive as ordered PCM the client can schedule; an answer that quotes nothing ships uncited
while a quoted answer is attributed to the chunk it came from; a follow-up question is
rewritten against the history; deleting a book removes both the row and the file; and
**cancelling mid-stream stops the events and truncates the history**.

**`test_error_paths.py` (18)** — every way a user can be wrong, and every way the app can
be unconfigured: not a PDF, missing `%PDF` magic bytes, empty upload, over the size limit,
a corrupt PDF that is accepted then fails *with a reason*, a scanned PDF rejected with an
actionable message, a failed book re-uploaded and retried, unknown books 404 everywhere, a
still-ingesting book 409, a failed book 409 carrying the ingestion error, a blank question
refused before the stream opens, a malformed request, 503 from the model endpoints with no
key, ingestion failing with configuration instructions with no key, a mid-stream model
failure becoming an `error` event, and three ways a recording is refused before it costs
anything: an unsupported container, one over the size limit, and one with no speech in it.

**`test_asr_endpoint.py` (9)** — transcription, the vocabulary hint actually being sent,
repair of a mistranscribed name against an ingested book, ordinary words *never* being
rewritten into names, unknown/absent books degrading to plain transcription, empty
recordings rejected, a decoder that hands back its own vocabulary prompt not being believed
(it is retried without the hint) — and the one that matters most: **the repaired question
is the one that reaches retrieval**, not just the one shown on screen.

---

## Frontend

```bash
npm run test     # vitest, node environment
npm run build    # tsc -b && vite build — zero TypeScript errors is the bar
```

Eleven units over the three pieces of client logic that are pure functions of their input
and break silently when wrong:

- the **POST-based SSE reader** — chunk boundaries mid-event, multi-line `data:`, abort
  behaviour;
- the **audio queue** — base64 → int16 → float32 conversion, and `seq`-ordered scheduling
  where out-of-order arrivals wait rather than reorder;
- the **request deadlines** — a fetch that never settles has to become an error the UI can
  show. This one exists because the failure was real: with the backend stopped, the dev
  server's proxy accepted the connection and never answered, so the mic button sat in its
  busy state and the startup health probe never settled, taking the toast that names the
  cause down with it. The tests hold a hanging fetch against fake timers and assert that an
  ordinary call gives up at 15 s, that transcription is still waiting there and gives up at
  60 s, that an unreachable server reads differently from a slow one, and that a started
  answer stream keeps running well past the 20 s header deadline.

Everything else on the client is DOM and audio hardware, which is what the manual checklist
is for.

The `mock/server.mjs` backend is also a test surface: it implements the full API contract
with canned data, so the entire UI — shelf, ingest progress, the flight into the reader,
streaming answer, PCM playback, cancel — is drivable with `npm run mock` and no key. It
has to be kept honest deliberately, because nothing fails when it drifts: its chapters
carry `opening` prose and its sources carry their passage in full (`text`, of which
`preview` is the first 200 characters, sliced exactly as the server slices it), which is
what makes the reader's chapter page and the "Full passage" control real offline rather
than quietly inert.

---

## Evaluation harness

Not a test — a measurement. It answers "is this retrieval actually better than naive
chunking, and which part is doing the work?" Full description of the fixture, question set,
arms and metrics: [`backend/eval/README.md`](../backend/eval/README.md). Results:
[`backend/eval/report.md`](../backend/eval/report.md).

```bash
cd backend
.venv/bin/python eval/make_fixture_book.py     # ~2 s — writes fixture PDF + golden.json
.venv/bin/python eval/run_eval.py --fake       # ~6 s, offline, no key
.venv/bin/python eval/run_eval.py --fake --sweep

.venv/bin/python eval/run_eval.py              # all six arms, real models
.venv/bin/python eval/run_eval.py --arms A,F   # just the two endpoints
.venv/bin/python eval/run_eval.py --sweep-only --no-answers
```

Useful flags: `--limit N` (N questions per category), `--no-answers` (retrieval metrics
only — removes about three quarters of the spend), `--reuse` (keep the previous run's
indexes in `eval/.work`), `--data-dir`, `--report`, `--results`.

A real run builds four indexes over a 25-page book and asks 88 questions of each of six
arms: roughly 500 chat completions and 800 embedding calls, plus one contextual-enrichment
call per chunk for arm F's index. The key is read from the environment or `backend/.env`
at run time and never printed.

**What `--fake` proves and does not prove.** `FakeAIClient`'s embeddings are noise by
construction, so the dense-only arms score near zero and everything above them is carried
by BM25; the canned chat reply means every answer metric measures the fake; arm D's
cross-encoder is swapped for a deterministic lexical stand-in so no model is downloaded. A
green `--fake` run proves the harness, the four index configurations, the six retrieval
configurations, the ASR repair path and the report writer all execute end to end.
**Numbers worth quoting come from a real-key run**, and `report.md` states which kind of run
produced it in its own header.

`make_fixture_book.py` will not write a fixture it cannot vouch for: before emitting
anything it checks that every expected answer substring is printed in the book, that every
evidence sentence survives PDF extraction intact, that the finished PDF re-parses into
twelve correctly named chapters through the app's *own* structure detector, and that all
twelve ASR corruptions are repairable against the vocabulary `app.ingest.vocab` extracts
from the same prose. `run_eval.py` re-validates `golden.json` on load, so a hand-edited
golden set cannot quietly produce meaningless numbers.

---

## Manual E2E checklist

The offline suite cannot hold a microphone, and the two production bugs that actually
shipped were both found by running the real thing with a real key. So were the interactive
faults of the UI phase — a transition that measured its own destination wrong, a warning
that fired on good answers, a mic that transcribed correctly and then silently threw the
transcript away — none of which any assertion in this repository would have caught. Run
this before calling a change done.

That last one is worth stating in full, because it is the shape of bug this checklist
exists for. The composer keeps an `alive` ref so a transcription that resolves after the
reader has left does not ask its question on a screen nobody is looking at. The cleanup set
it to `false`; nothing set it back to `true`. React's StrictMode mounts, unmounts and
remounts every component in development, so the flag was false from the first render on:
`/api/asr` returned 200, the early return dropped the transcript, and the `finally` clause
— gated on the same flag — never took the mic out of its busy state. Every layer reported
success. The server log is what named it: `POST /api/asr 200 OK` with no `POST /api/chat`
behind it, over and over. **When a step's own logs all say "fine", check what the next step
never received.**

**Setup:** real `OPENAI_API_KEY` in `backend/.env`, backend on `:8000`, `npm run dev` on
`:5173`, Chrome and Safari both open.

1. **Cold start with no key.** Move `.env` aside and boot. The app serves, `/api/health`
   reports `openai_configured: false`, the UI loads, and an upload fails with
   configuration instructions rather than a stack trace. No network call is made at all —
   the reranker pre-warm must not fire without a key.
2. **Mic permission — grant.** Click the mic on a fresh profile. The browser prompt
   appears, the button becomes a live ring driven by the real input level, and stopping
   moves to "Transcribing…" then to the streamed answer. Watch the button return to idle
   and the question appear in the thread — a transcript that arrives and is dropped looks
   identical to one that never arrived, and `npm run dev` runs under StrictMode where
   production does not.
3. **Mic permission — deny.** Deny it, then click again. The UI must explain how to
   re-enable it in the browser, not fail silently or spin forever.
4. **First audio.** Ask a factual question and *listen*. Audio should begin while text is
   still streaming — check the `done` event's `first_audio_ms` in the network panel
   against what you heard. Sentences must be gapless and in order; a click or a repeat is
   a `seq` scheduling bug.
5. **Stop button.** Interrupt mid-answer. Audio stops immediately, the stream aborts, and
   the visible answer is truncated to what was actually spoken. Then ask a follow-up
   ("what about its harbour?") — it must resolve against the *truncated* transcript, which
   is the whole point of `played_seq`.
6. **Safari.** MediaRecorder produces `audio/mp4` there, not webm, and the AudioContext
   must be created inside the click gesture or nothing plays. Record, transcribe, listen.
   Then repeat on iOS Safari if a device is available.
7. **A large real book.** Upload something 300+ pages. Watch the stage sequence and the
   progress bar move monotonically; confirm memory stays flat during parsing; confirm the
   TOC matches the book's printed contents; ask a question about a late chapter and check
   the cited pages are actually right by opening the PDF.
8. **A scanned book.** Upload an image-only PDF. It must be rejected with the
   characters-per-page message, not accepted and silently indexed empty.
9. **Lock-screen / background behaviour.** Start an answer, then lock the screen or switch
   tabs. WebAudio is suspended by the browser; on return, playback must not desynchronize
   from the transcript or resume as a burst of stale audio. If it does, the fix is client
   side — the server has already emitted those segments.
10. **A book in Chinese.** Upload one and ask in Chinese. Structure detection should find
    `第N章` headings, BM25 should tokenize through jieba, and the answer should come back
    in the question's language. Note that the ASR vocabulary/repair path is English-only by
    design and simply does nothing here.
11. **Two browsers, one book.** Ask from both at once. Sessions are keyed by
    `(session_id, book_id)` and must not bleed into each other's history.
12. **The grounding warning.** Ask something the book answers well: sentences pick up
    their sidenotes and **no banner appears**. Ask about a subject the book has never
    heard of: the answer is a short refusal, which cites nothing by design and is kept
    out of the warning by the >80-character test. To see the banner itself, ask something
    long-winded that the model can answer from its own knowledge without the passages
    supporting it. What must never happen is a banner over a well-cited answer. The
    warning counts how many of the answer's own sentences the post-hoc attribution traced
    back to a passage — **not** the retrieval confidence, which was measured wrong in both
    directions on a real book: 0.0017 for "what happens in chapter 2", answered correctly
    from four cited passages, and 0.9953 for a subject the book never mentions. That score
    is still computed and still travels on the `sources` event; it just no longer decides
    anything the reader sees.
13. **A source, opened.** Expand the sources under an answer and press a card's
    "Ch. N · pp. x–y" reference. The book beside the conversation must turn to **that
    passage** — its chapter, its pages, its own words, washed the colour the cited
    sentences carry — not merely to the chapter it came from. Then press "Full passage":
    it must show more than the card was already printing (the card holds a 200-character
    preview; the source carries the whole chunk). Page away with the chapter strip and the
    page returns to the chapter's opening prose.
14. **Shelf → reader → shelf.** Click a book on the shelf. It turns square and rises where
    it stands, and then the same cover flies to the reader and the page opens once it has
    landed. Only the cover should cross: the wordmark stays exactly where it is and the
    rest of each screen hands over around it. Leaving mirrors the arrival, rightward.
    Watch for the failures this transition has actually had — a cover that snaps to
    another shape or size at the hand-off, the page drawn over the conversation while the
    book is still travelling, and a reader left half-arrived when the flight could not be
    measured (an unmeasurable destination must still open the page). Then repeat with
    reduced motion enabled.
15. **Coming back to where you left.** Scroll well down the shelf, open a book from the
    bottom of the stack, then go back. The shelf must return at the scroll position it was
    left at, with the book landing in its own row — not at the top of the stack with the
    flight aimed at a row nobody can see.
