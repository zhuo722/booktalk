# Open-source research pass

Before finalizing the design, five parallel research agents surveyed GitHub and the
literature — one per domain — with instructions to read actual source and verify claims
(python version support, dependency weight, latency) rather than answer from memory.
Several recommendations were benchmarked on the target machine during research (FlashRank,
pymupdf4llm, chonkie+model2vec). Full ranked recommendations:
[research/raw-recommendations.txt](research/raw-recommendations.txt). The adopted set is
specified in SPEC.md §14; this file records where each idea came from and what was
deliberately rejected.

## What was adopted, and from where

| Adoption | Source | Evidence that carried the decision |
|---|---|---|
| Post-hoc sentence→chunk citation attribution | RAGFlow `insert_citations()` | Keeps spoken text free of markers; ~60 lines + one embedding batch running concurrently with TTS → zero added first-audio latency |
| Heading-family voting fallback for structure | RAGFlow `bullets_category` | Pure-regex voting across 5 heading conventions; makes structure detection survive PDFs with no embedded TOC |
| LLM TOC-level assignment (ingest-time) | RAGFlow `assign_toc_levels` prompt | Strict-JSON one-shot pass; fixes books whose TOC has no numeric prefixes |
| Chapter summary index nodes (RAPTOR-lite) | RAPTOR (ICLR 2024) via RAGFlow | Chapter tree already exists; ~15 LLM calls capture most of RAPTOR's thematic-question gain without UMAP/GMM clustering |
| pymupdf4llm extraction layer | pymupdf4llm | Measured on a real 244-page book: 129 ligature chars / 68 hyphen line-breaks raw vs 0 / 0 — de-hyphenation is a direct BM25 recall win; installs on py3.9 |
| Three-pass chunk assembly | Docling `HybridChunker` (IBM, MIT) | Named, citable algorithm: breadcrumb via `heading_by_level`, contextualized token budget, merge only within identical heading paths; `semchunk` leaf splitter verified on py3.9 |
| ~256–320-token children, zero overlap | Chroma chunking technical report | 5 corpora / 472 queries: recursive 200/0 = 88.1% recall; overlap measurably hurt |
| FlashRank reranking after RRF | FlashRank (TinyBERT-L-2-v2) | Benchmarked on this machine: 5 MB model, 7.6 ms for one isolated `rank()` over 20 passages, torch-free, cp39 wheel — for the technique Anthropic measured at 2.9%→1.9% failure. In the eval the *per-query* cost is larger: mean retrieve time 187 ms (arm C) → 267 ms (arm D), ~80 ms around the whole retrieve call rather than the cross-encoder alone |
| LLM contextual enrichment per chunk | Anthropic contextual retrieval | Published 5.7%→2.9% retrieval-failure reduction; scoped per chapter to keep ingest cost sane; validated by our own ablation rather than assumed |
| RRF constant + weight tuning | Elasticsearch/OpenSearch docs, Anthropic cookbook | k=60 comes from million-doc TREC corpora; small-corpus guidance is k=10–20; Anthropic's cookbook itself uses weighted fusion (0.8/0.2) |
| Two-phase TTS sentence segmentation | stream2sentence + LiveKit Agents | First fragment on soft delimiter ≥10 chars (cuts ~0.4–0.8 s off first audio); later chunks need true sentence boundary ≥20 chars |
| Vendored sentence splitter | LiveKit `_basic_sent` (Apache-2.0) + pipecat lookahead | Handles Dr./Ph.D/3.5/URLs with stdlib `re` only; no NLTK download at runtime |
| PCM streaming playback | openai-realtime-console `WavStreamPlayer` | Raw 24 kHz s16le over SSE: gapless, no mp3 decode, no ffmpeg anywhere, sample-accurate "how much was heard" tracking |
| Generation-id'd cancellation + history truncation | openai-realtime-console + vocode | Late chunks dropped by dead turn_id; stored history truncated to sentences actually played |
| TTS back-pressure (queue bound, >5 s unplayed pause) | LiveKit `SentenceStreamPacer`, RealtimeTTS | A cancelled turn wastes one sentence of TTS spend, not the whole answer |
| ASR entity biasing + Double Metaphone repair | arXiv 2608.22872 | Named-entity corruption causes 87–96% of ASR-induced RAG failures; the book's closed vocabulary makes the fix nearly free |
| Retrieval honesty UI (low-confidence banner) | kotaemon | The honest-UI pattern was adopted; kotaemon's *mechanism* was not kept. Its banner is driven by normalized retrieval scores, and on a real book ours inverted — 0.0017 for a correctly answered "what happens in chapter 2", 0.9953 for a subject the book never mentions. The shipped banner counts how many of the answer's sentences post-hoc attribution could trace to a passage, and shows when a substantial answer (>80 chars) traced none |
| Cited / also-retrieved source split | Onyx | Dedupe by chunk id, cited-first ordering |
| Eval harness shape (categories, per-query CSV) | Onyx `run_search_eval.py` | Categories mapped to real failure modes incl. abstain and ASR-corruption arms |
| Radix primitives, no component framework | frontend survey verdict | Unstyled, zero-dep behavior (focus trap, portal, aria) without the recognizable shadcn look |
| eventsource-parser / use-stick-to-bottom / Literata | frontend survey | Each replaces hand-rolled code where chat UIs visibly break; all zero/near-zero-dep, React 18 verified |
| Tufte-sidenote citation layout + paper palette | Tufte CSS, Readwise Reader, tweakcn vintage-paper, Midday | Margin sidenotes at ~65ch measure are both the most distinctive layout available and a direct display of retrieval quality |

## What was rejected, and why

| Rejection | Reason |
|---|---|
| LangChain / LlamaIndex | Retrieval is the graded core; abstractions hide the decisions we need to show and test |
| ragas / deepeval | Both resolve on py3.9 but pull heavy trees (langchain/datasets/scipy; grpcio/otel/posthog telemetry); LLM-judge scores are poor instruments for small ablation deltas. What was actually built is deterministic and nothing else: recall@{1,3,5,10}, MRR, token-level IoU, expected-substring match, refusal correctness. The hand-rolled ~30-line faithfulness judge SPEC §14.10 planned alongside them was never written — `backend/eval/` has no judge in it |
| Late chunking (Jina) | Architecturally impossible here: needs token-level embeddings; OpenAI returns pooled vectors only; published gain just +1.5–1.9 nDCG |
| HyDE | Full LLM round-trip on the first-audio critical path; hallucinates entities — the exact failure a voice pipeline can least afford |
| Full RAPTOR | UMAP/numba ingest cost for 2–3 pt gain, largely redundant with a real TOC tree; chapter-summary degenerate case adopted instead |
| Semantic chunking as primary | NAACL 2025 (Qu, Tu & Bao) and Chroma both find cost unjustified vs fixed-size on real docs; kept only as an optional ablation arm |
| Docling / marker / MinerU as parser | Torch-scale model stacks; wrong weight class for a py3.9 CPU app when pymupdf4llm + own structure ladder measures clean |
| Tailwind / shadcn / assistant-ui / AI Elements | The default shadcn look is the most recognizable AI-template aesthetic — the rubric explicitly penalizes it; Radix primitives give the behavior without the look |
| wavesurfer.js for playback visualization | Requires a decodable blob with precomputed peaks; a sentence-pipelined PCM queue never has one — one ~40-line analyser+canvas component covers mic and playback |
| Browser VAD (@ricky0123/vad-web) now | ~4.8 MB first load, repo issues on iOS and barge-in reliability; push-to-talk default + RMS endpointer later if time allows |

## Method note

Research ran as five parallel agents (RAG products / PDF parsing / voice pipelines /
retrieval+eval / frontend), ~500k tokens, 309 tool calls, ~16 minutes wall-clock. Each
returned structured findings with per-repo `concerns` fields; recommendations that
survived were folded into SPEC.md §14 and the frontend build instructions. Star counts and
maintenance status were checked against live GitHub data (notably: Verba is archived,
chatbot-ui and vocode are ~2 years stale — popularity was not treated as fitness).
