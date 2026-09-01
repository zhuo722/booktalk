# BookTalk retrieval evaluation

Generated 2026-08-31T19:42:28+00:00 · real models · fixture «A Field Guide to the Meridian Archipelago» (25 pages, 12 chapters) · 88 questions · retrieval depth 10

Models: chat `gpt-4o-mini`, embeddings `text-embedding-3-small`. Reranker: `ms-marco-TinyBERT-L-2-v2`. RRF k=20, weights dense 0.5 / BM25 0.5.

## Ablation arms

| Arm | Configuration | chunks | recall@1 | recall@3 | recall@5 | recall@10 | MRR | token-IoU | answer match | answer full | refusal ok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** naive | fixed 800-char chunks, dense only | 85 | 0.856 | 0.938 | 0.950 | 0.988 | 0.929 | 0.016 | 0.931 | 0.925 | 0.943 |
| **B** +structure | structure-aware chunks + breadcrumbs | 65 | 0.938 | 0.994 | 0.994 | 1.000 | 0.994 | 0.010 | 0.988 | 0.988 | 0.989 |
| **C** +hybrid | dense + BM25, weighted RRF | 65 | 0.950 | 0.988 | 0.994 | 1.000 | 1.000 | 0.010 | 0.975 | 0.975 | 0.977 |
| **D** +rerank | cross-encoder over the fused head | 65 | 0.950 | 0.975 | 0.994 | 1.000 | 1.000 | 0.013 | 0.975 | 0.975 | 0.977 |
| **E** +summaries | chapter synopsis nodes indexed | 78 | 0.950 | 0.975 | 0.988 | 1.000 | 1.000 | 0.014 | 0.963 | 0.950 | 0.977 |
| **F** +contextual | LLM situating context per chunk | 78 | 0.950 | 0.975 | 0.994 | 1.000 | 1.000 | 0.014 | 0.969 | 0.963 | 0.977 |

Each arm adds one thing to the arm above it. `chunks` is the size of the index that arm retrieved from; `answer match` is the share of expected substrings found in the generated answer and `answer full` the share of questions where every one of them was; `refusal ok` scores both directions — refusing the out-of-book questions and *not* refusing the rest.

Read `token-IoU` against `recall`, never on its own. It is a precision signal — evidence tokens over total context tokens — so arm A scores well on it partly by retrieving *less*: naive chunks carry no parent window, so a naive context is a third the size of a structured one even when it contains nothing the question asked for.

### Index build

| index | mode | contextual | summaries | chunks | summary nodes | enriched | build s |
|---|---|---|---|---|---|---|---|
| naive | naive | no | no | 85 | 0 | 0 | 1.0 |
| structured | full | no | no | 65 | 0 | 0 | 1.3 |
| summarized | full | no | yes | 78 | 13 | 0 | 8.6 |
| contextual | full | yes | yes | 78 | 13 | 65 | 21.8 |

### Category: single-fact

| Arm | n | recall@1 | recall@3 | recall@5 | recall@10 | MRR | token-IoU | answer match | answer full | refusal ok | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** naive | 48 | 0.979 | 1.000 | 1.000 | 1.000 | 0.990 | 0.014 | 0.958 | 0.958 | 0.958 | 1.000 |
| **B** +structure | 48 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.008 | 0.979 | 0.979 | 0.979 | 1.000 |
| **C** +hybrid | 48 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.009 | 0.979 | 0.979 | 0.979 | 1.000 |
| **D** +rerank | 48 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.012 | 0.958 | 0.958 | 0.958 | 0.982 |
| **E** +summaries | 48 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.013 | 0.958 | 0.958 | 0.958 | 0.983 |
| **F** +contextual | 48 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.013 | 0.958 | 0.958 | 0.958 | 0.985 |

### Category: cross-chapter

| Arm | n | recall@1 | recall@3 | recall@5 | recall@10 | MRR | token-IoU | answer match | answer full | refusal ok | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** naive | 8 | 0.438 | 0.875 | 1.000 | 1.000 | 0.938 | 0.027 | 0.812 | 0.750 | 0.875 | 1.000 |
| **B** +structure | 8 | 0.500 | 0.938 | 0.938 | 1.000 | 1.000 | 0.019 | 1.000 | 1.000 | 1.000 | 1.000 |
| **C** +hybrid | 8 | 0.500 | 0.875 | 0.938 | 1.000 | 1.000 | 0.020 | 1.000 | 1.000 | 1.000 | 1.000 |
| **D** +rerank | 8 | 0.500 | 0.750 | 0.938 | 1.000 | 1.000 | 0.020 | 1.000 | 1.000 | 1.000 | 0.482 |
| **E** +summaries | 8 | 0.500 | 0.750 | 0.875 | 1.000 | 1.000 | 0.020 | 0.875 | 0.750 | 1.000 | 0.469 |
| **F** +contextual | 8 | 0.500 | 0.750 | 0.938 | 1.000 | 1.000 | 0.020 | 0.938 | 0.875 | 1.000 | 0.469 |

### Category: chapter-summary

| Arm | n | recall@1 | recall@3 | recall@5 | recall@10 | MRR | token-IoU | answer match | answer full | refusal ok | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** naive | 12 | 0.500 | 0.667 | 0.667 | 0.917 | 0.610 | — | 0.833 | 0.833 | 0.833 | 1.000 |
| **B** +structure | 12 | 0.917 | 1.000 | 1.000 | 1.000 | 0.958 | — | 1.000 | 1.000 | 1.000 | 1.000 |
| **C** +hybrid | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | — | 1.000 | 1.000 | 1.000 | 1.000 |
| **D** +rerank | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | — | 1.000 | 1.000 | 1.000 | 0.501 |
| **E** +summaries | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | — | 1.000 | 1.000 | 1.000 | 0.949 |
| **F** +contextual | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | — | 1.000 | 1.000 | 1.000 | 0.950 |

### Category: not-in-book

| Arm | n | recall@1 | recall@3 | recall@5 | recall@10 | MRR | token-IoU | answer match | answer full | refusal ok | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** naive | 8 | — | — | — | — | — | — | — | — | 1.000 | 1.000 |
| **B** +structure | 8 | — | — | — | — | — | — | — | — | 1.000 | 1.000 |
| **C** +hybrid | 8 | — | — | — | — | — | — | — | — | 1.000 | 1.000 |
| **D** +rerank | 8 | — | — | — | — | — | — | — | — | 1.000 | 0.003 |
| **E** +summaries | 8 | — | — | — | — | — | — | — | — | 1.000 | 0.017 |
| **F** +contextual | 8 | — | — | — | — | — | — | — | — | 1.000 | 0.038 |

No gold chapter exists, so the retrieval columns are blank by design — this category only asks whether the pipeline declines to answer.

### Category: asr-corrupted

| Arm | n | recall@1 | recall@3 | recall@5 | recall@10 | MRR | token-IoU | answer match | answer full | refusal ok | confidence | repair rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** naive | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.014 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **B** +structure | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.009 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **C** +hybrid | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.009 | 0.917 | 0.917 | 0.917 | 1.000 | 1.000 |
| **D** +rerank | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.012 | 1.000 | 1.000 | 1.000 | 0.995 | 1.000 |
| **E** +summaries | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.013 | 1.000 | 1.000 | 1.000 | 0.994 | 1.000 |
| **F** +contextual | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.013 | 1.000 | 1.000 | 1.000 | 0.994 | 1.000 |

The question text is what a transcription model heard, with the island's name mangled into a phonetic near-miss. It is repaired against the book's own vocabulary before retrieval (SPEC §14.9); `repair rate` is how often the original name was restored exactly, and it varies between arms because the vocabulary is harvested from whatever chunks that index was built from.

### First-hit rank distribution

| Arm | rank 1 | rank 2 | rank 3 | rank 4–5 | rank 6–10 | rank miss |
|---|---|---|---|---|---|---|
| **A** naive | 72 | 4 | 0 | 0 | 3 | 1 |
| **B** +structure | 79 | 1 | 0 | 0 | 0 | 0 |
| **C** +hybrid | 80 | 0 | 0 | 0 | 0 | 0 |
| **D** +rerank | 80 | 0 | 0 | 0 | 0 | 0 |
| **E** +summaries | 80 | 0 | 0 | 0 | 0 | 0 |
| **F** +contextual | 80 | 0 | 0 | 0 | 0 | 0 |

## RRF sweep

Fusion in isolation: arm C's index, no reranker, retrieval only, over every question with a gold chapter.

| k | dense | bm25 | recall@1 | recall@3 | recall@5 | recall@10 | MRR | token-IoU |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.3 | 0.7 | 0.950 | 0.975 | 0.994 | 1.000 | 1.000 | 0.010 |
| 10 | 0.5 | 0.5 | 0.950 | 0.994 | 0.994 | 1.000 | 1.000 | 0.010 |
| 10 | 0.7 | 0.3 | 0.950 | 0.994 | 0.994 | 1.000 | 1.000 | 0.010 |
| 20 | 0.3 | 0.7 | 0.950 | 0.975 | 0.994 | 1.000 | 1.000 | 0.010 |
| 20 | 0.5 | 0.5 | 0.950 | 0.988 | 0.994 | 1.000 | 1.000 | 0.010 |
| 20 | 0.7 | 0.3 | 0.950 | 0.994 | 0.994 | 1.000 | 1.000 | 0.010 |
| 60 | 0.3 | 0.7 | 0.950 | 0.975 | 0.988 | 1.000 | 1.000 | 0.010 |
| 60 | 0.5 | 0.5 | 0.950 | 0.988 | 0.994 | 1.000 | 1.000 | 0.010 |
| 60 | 0.7 | 0.3 | 0.950 | 0.994 | 0.994 | 1.000 | 1.000 | 0.010 |

recall@3 is the only column with movement in it, and it separates the weight splits the
same way at every k (dense-heavy 0.994 · balanced 0.988–0.994 · BM25-heavy 0.975); best
to worst is 0.019 of mean recall, about one and a half of the 80 questions that have a
gold chapter. It was added to this table by hand from
`results/sweep.csv` — `sweep_table()` in `run_eval.py` does not print it, so a re-run
will drop it again.

