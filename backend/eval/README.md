# BookTalk evaluation harness

Does the pipeline in `app/` actually retrieve better than naive chunking, and
which of its parts is doing the work? This directory answers that with a fixture
book nobody has ever read, 88 questions derived from the same table the book was
written from, and six ablation arms that differ by exactly one thing each
(SPEC §10, §14.10).

```
facts.py               the invented fact table — 12 islands × 8 facts
make_fixture_book.py   renders the PDF and emits golden.json
run_eval.py            runs the arms, writes report.md and the CSVs
fixture/               the generated PDF (regenerate, do not edit)
golden.json            the generated question set (regenerate, do not edit)
report.md              the last run's tables
results/               per_question.csv · per_category.csv · sweep.csv
```

## Running it

```bash
cd backend
.venv/bin/python eval/make_fixture_book.py     # ~2 s, writes the PDF + golden.json
.venv/bin/python eval/run_eval.py --fake       # ~6 s, offline, no API key
.venv/bin/python eval/run_eval.py --fake --sweep
```

A real-model run reads `OPENAI_API_KEY` from the environment (or `backend/.env`)
at run time and never prints it:

```bash
.venv/bin/python eval/run_eval.py              # all six arms, real models
.venv/bin/python eval/run_eval.py --arms A,F   # just the two endpoints
.venv/bin/python eval/run_eval.py --sweep-only --no-answers
```

Useful flags: `--limit N` (N questions per category, for a smoke test),
`--no-answers` (retrieval metrics only — no generation spend), `--reuse` (keep
the indexes built by the previous run in `eval/.work` instead of rebuilding
them), `--data-dir`, `--report`, `--results`.

A real run builds four indexes over a 25-page book and asks 88 questions of each
of six arms: roughly 500 chat completions and 800 embedding calls on
`gpt-4o-mini` / `text-embedding-3-small`, plus one contextual-enrichment call
per chunk for arm F's index. `--no-answers` removes about three quarters of it.

## The fixture book

*A Field Guide to the Meridian Archipelago* — 25 pages, 12 chapters, two
sections per chapter, an embedded outline, a running head and folios. Everything
in it is invented, and that is the point: no model has read about Kestrel Cove,
so a correct answer is evidence that retrieval found the passage rather than
evidence that the model already knew. Each chapter carries eight structured
facts (founder, charter year, population, landmark and its height, local dish,
distance from the capital, festival and month, export) woven into ordinary
prose, surrounded by near-miss prose about the same island.

`make_fixture_book.py` will not write a fixture it cannot vouch for. Before
emitting anything it checks that every expected answer substring is printed in
the book, that every evidence sentence survives PDF extraction intact, that the
finished PDF re-parses into twelve correctly named chapters through the app's
own structure detector, and that all twelve ASR corruptions are repairable
against the vocabulary `app.ingest.vocab` extracts from the same prose.

## The question set

| Category | n | What it tests |
|---|---|---|
| `single-fact` | 48 | One fact, one chapter, one sentence of evidence. |
| `cross-chapter` | 8 | A comparison that needs both chapters retrieved. |
| `chapter-summary` | 12 | "What is chapter 7 about" — thematic, which flat chunks answer badly and the §14.5 summary nodes answer well. |
| `not-in-book` | 8 | Plausible questions the book does not answer. The correct behaviour is to refuse. |
| `asr-corrupted` | 12 | A fact question with the island's name mis-heard ("Castrel Cove"), repaired against the book's vocabulary before retrieval (§14.9). |

Each entry in `golden.json` carries `question`, `expected_answer_substrings`,
`chapter_idx`, `gold_chapters` (two, for the cross-chapter category),
`expect_refusal`, and `evidence` — the exact sentences a perfect retriever would
return, which is what token-IoU is scored against. `run_eval.py` re-validates
the whole file on load, so a hand-edited golden set cannot quietly produce
meaningless numbers.

## The arms

| Arm | Index | Retrieval |
|---|---|---|
| A | fixed 800-character chunks, no header, no parent | dense only |
| B | structure-aware chunks with breadcrumb headers | dense only |
| C | as B | dense + BM25, weighted RRF |
| D | as B | + cross-encoder rerank |
| E | + chapter and book summary nodes | as D |
| F | + LLM contextual enrichment per chunk | as D |

Four indexes are built per run (A, B/C/D, E, F share where they can) and every
arm runs through `ingest_book`, `retrieve` and `build_messages` exactly as the
server calls them. Retrieval runs to depth 10 rather than the production 8, so
that recall@10 is defined; nothing else about the configuration is changed.

## The metrics

- **recall@{1,3,5,10}** — the share of a question's *gold chapters* found in the
  top k. Averaging over gold chapters rather than asking "did any hit" is what
  makes the cross-chapter category honest: finding one of two chapters is half a
  success, not a whole one.
- **MRR** — reciprocal rank of the first chunk from any gold chapter.
- **token-IoU** — Chroma's precision signal: token-multiset intersection over
  union between the gold evidence sentences and the packed context the model is
  actually shown. Absolute values are small by construction (a sentence against
  a 4500-token budget); only the comparison between arms means anything.
- **answer match / answer full** — the share of expected substrings present in
  the generated answer, and the share of questions where all of them were.
- **refusal ok** — scored in both directions. Refusing the out-of-book questions
  *and* not refusing the rest; a pipeline that abstains from everything would
  otherwise look perfect.
- **confidence** — `Retrieval.top_score`. Only an absolute quantity when a
  reranker ran. It is measured here and nothing gates on it: this eval is what
  showed it inverts on a real book (0.0017 for a correctly answered question
  about chapter 2, 0.9953 for a subject the book never mentions), so the
  shipped grounding warning counts traced sentences instead, and no abstain
  gate was built.
- **repair rate** — how often the ASR corruption was restored exactly.

Plus a first-hit rank distribution, because two arms with the same MRR can have
very different shapes, and the `--sweep` grid over RRF `k ∈ {10, 20, 60}` and
dense/BM25 weight splits (TREC's k=60 assumes a million-document corpus; a
single book is four orders of magnitude smaller, which is why this is measured
rather than assumed).

## What `--fake` does and does not prove

`FakeAIClient` seeds its embeddings from `sha1(text)`, so similar texts get
unrelated vectors: **dense retrieval is noise by construction**, the dense-only
arms A and B score near zero, and everything above them is carried by BM25. Its
chat completion is a canned sentence, so every answer metric measures the fake.
Arm D's cross-encoder is replaced by a deterministic lexical stand-in, because
FlashRank would have to download a model.

A green `--fake` run proves that the harness, the four index configurations, the
six retrieval configurations, the ASR repair path and the report writer all
execute end to end. Numbers worth quoting come from a real-key run, and
`report.md` states which kind of run produced it in its own header.

## Results

`report.md` in this directory holds the most recent run — currently a real-model
run (six arms plus the RRF sweep, 88 questions; its header states the models).
The same table and the per-category analysis appear in the project README;
`results/` holds the per-question, per-category and sweep CSVs behind it.
