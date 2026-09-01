"""The ablation harness: does the pipeline actually beat naive chunking?

SPEC §10 asked for one number against one baseline. §14.10 replaced that with a
ladder, because "it is better" is not a finding — *which change* is better, and
by how much, is. Six arms, each adding exactly one thing to the one before it:

    A  naive         fixed 800-character chunks, dense retrieval only
    B  +structure    structure-aware chunks with breadcrumb headers
    C  +hybrid       BM25 alongside dense, fused with weighted RRF
    D  +rerank       a cross-encoder over the fused head
    E  +summaries    chapter synopsis nodes in the index (SPEC §14.5)
    F  +contextual   LLM-written situating context per chunk (SPEC §14.4)

Every arm runs through the application's own code — ``ingest_book`` with its
real mode flags, ``retrieve`` with its real parameters, ``build_messages`` with
the shipped prompt. Nothing about the pipeline is re-implemented here, which is
the only way an ablation can be evidence about the product rather than about the
harness. What this file adds is the scoring.

Metrics, per SPEC §14.10:

* **recall@{1,3,5,10}** — the share of a question's gold chapters that appear
  among the top *k* retrieved chunks. Averaging over gold chapters rather than
  asking "did any hit" is what makes the cross-chapter category meaningful: a
  synthesis question is only half answered if only one of its two chapters was
  found.
* **MRR** — reciprocal rank of the first chunk from any gold chapter.
* **token-IoU** — Chroma's precision signal. The gold evidence is the exact
  sentences that carry the fact; the retrieved side is the packed context the
  model is actually shown. Intersection over union on token multisets, so
  dragging in four irrelevant sections costs you, and a tighter context that
  still contains the evidence wins. Absolute values are small by construction
  (one sentence against a 4500-token budget); only the comparison between arms
  means anything.
* **answer match** — share of the expected substrings present in the generated
  answer, with a stricter "all of them" rate beside it.
* **refusal correctness** — the out-of-book questions must be refused, and the
  answerable ones must not be. Both directions are scored; a pipeline that
  refuses everything would otherwise look excellent.

``--fake`` runs the whole thing offline against :class:`FakeAIClient`. That
validates the plumbing and nothing else: the fake's embeddings are seeded from
``sha1(text)``, so *similar* texts get unrelated vectors and dense retrieval is
noise by design. Expect the dense-only arms (A, B) to score near zero in fake
mode and the BM25 arms to carry the run. Answer metrics are meaningless there
too — the fake returns a canned sentence. The report says so in its own header
so a fake-mode table can never be mistaken for a result.

Real mode reads ``OPENAI_API_KEY`` from the environment at run time and never
prints it, or anything derived from it.

    python eval/run_eval.py --fake
    python eval/run_eval.py --arms A,C,F
    python eval/run_eval.py --sweep-only --fake
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent
for _path in (str(BACKEND_DIR), str(EVAL_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from make_fixture_book import (  # noqa: E402
    CATEGORIES,
    CATEGORY_ABSENT,
    CATEGORY_ASR,
    DEFAULT_GOLDEN,
    DEFAULT_PDF,
    validate_golden,
)

from app.config import Settings, load_settings  # noqa: E402
from app.db import init_db  # noqa: E402
from app.index.rerank import get_reranker  # noqa: E402
from app.index.store import STATUS_READY, Store  # noqa: E402
from app.ingest.ingestor import MODE_FULL, MODE_NAIVE, ingest_book  # noqa: E402
from app.rag.pipeline import ANSWER_TEMPERATURE, build_messages  # noqa: E402
from app.rag.retriever import retrieve  # noqa: E402
from app.services.openai_client import FakeAIClient  # noqa: E402
from app.speech.phonetics import repair_transcript  # noqa: E402

#: How deep retrieval runs for scoring. Production takes the top 8 children;
#: recall@10 needs ten, and the extra two change nothing else — the context pack
#: is bounded by parents and tokens, not by this number.
RANK_DEPTH = 10
RECALL_AT = (1, 3, 5, 10)
DEFAULT_REPORT = EVAL_DIR / "report.md"
DEFAULT_RESULTS = EVAL_DIR / "results"
DEFAULT_WORK = EVAL_DIR / ".work"

#: The (k, weights) grid of SPEC §14.6. TREC's k=60 assumes a million-document
#: corpus; a single book is four orders of magnitude smaller, which is the whole
#: reason this is swept rather than assumed.
SWEEP_K = (10, 20, 60)
SWEEP_WEIGHTS = ((0.3, 0.7), (0.5, 0.5), (0.7, 0.3))

NO_KEY_MESSAGE = (
    "OPENAI_API_KEY is not set. Export it (or put it in backend/.env) for a real "
    "run, or pass --fake to run the harness offline."
)


# --------------------------------------------------------------------- arms


@dataclass(frozen=True)
class IngestConfig:
    """One way of building the index. Shared by every arm that needs it."""

    key: str
    mode: str
    contextual: bool
    summaries: bool

    @property
    def book_id(self) -> str:
        return "eval-{0}".format(self.key)


@dataclass(frozen=True)
class Arm:
    """One rung of the ablation."""

    key: str
    title: str
    detail: str
    ingest: IngestConfig
    dense_only: bool
    rerank: bool


NAIVE = IngestConfig("naive", MODE_NAIVE, contextual=False, summaries=False)
STRUCTURED = IngestConfig("structured", MODE_FULL, contextual=False, summaries=False)
SUMMARIZED = IngestConfig("summarized", MODE_FULL, contextual=False, summaries=True)
CONTEXTUAL = IngestConfig("contextual", MODE_FULL, contextual=True, summaries=True)

ARMS: Tuple[Arm, ...] = (
    Arm("A", "naive", "fixed 800-char chunks, dense only", NAIVE, True, False),
    Arm("B", "+structure", "structure-aware chunks + breadcrumbs", STRUCTURED, True, False),
    Arm("C", "+hybrid", "dense + BM25, weighted RRF", STRUCTURED, False, False),
    Arm("D", "+rerank", "cross-encoder over the fused head", STRUCTURED, False, True),
    Arm("E", "+summaries", "chapter synopsis nodes indexed", SUMMARIZED, False, True),
    Arm("F", "+contextual", "LLM situating context per chunk", CONTEXTUAL, False, True),
)
ARMS_BY_KEY = OrderedDict((arm.key, arm) for arm in ARMS)


# ---------------------------------------------------------------- questions


@dataclass
class Question:
    """One golden question, as the harness needs it."""

    id: str
    category: str
    text: str
    substrings: List[str]
    gold: List[int]
    expect_refusal: bool
    evidence: List[str]
    asr: Optional[Dict[str, str]] = None

    @property
    def scored_for_retrieval(self) -> bool:
        return bool(self.gold)


def load_questions(path: Path, limit: Optional[int] = None) -> Tuple[List[Question], Dict[str, Any]]:
    """Read and validate ``golden.json``.

    Validation runs on every load, not only at generation: a hand-edited golden
    file that no longer matches its book would produce numbers that look fine
    and mean nothing.
    """
    golden = json.loads(path.read_text(encoding="utf-8"))
    problems = validate_golden(golden)
    if problems:
        raise SystemExit(
            "golden set at {0} is invalid:\n  {1}".format(
                path, "\n  ".join(problems[:20])
            )
        )
    questions = [
        Question(
            id=item["id"],
            category=item["category"],
            text=item["question"],
            substrings=list(item["expected_answer_substrings"]),
            gold=[int(idx) for idx in item["gold_chapters"]],
            expect_refusal=bool(item["expect_refusal"]),
            evidence=list(item.get("evidence", [])),
            asr=item.get("asr"),
        )
        for item in golden["questions"]
    ]
    if limit:
        kept: List[Question] = []
        seen: Counter = Counter()
        for question in questions:
            if seen[question.category] < limit:
                seen[question.category] += 1
                kept.append(question)
        questions = kept
    return questions, golden


# ------------------------------------------------------------------ scoring

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

#: How a refusal reads once the answer prompt has had its way with it. Matched
#: on a normalized answer, so casing and line breaks do not matter.
REFUSAL_MARKERS = (
    "does not cover",
    "doesn't cover",
    "do not cover",
    "not covered",
    "does not mention",
    "doesn't mention",
    "no mention",
    "does not say",
    "doesn't say",
    "does not discuss",
    "doesn't discuss",
    "does not contain",
    "doesn't contain",
    "does not include",
    "not in the book",
    "not in this book",
    "nothing in the book",
    "no information",
    "cannot find",
    "can't find",
    "could not find",
    "couldn't find",
    "does not appear",
    "doesn't appear",
    "no excerpts matched",
    "not addressed",
)


def tokens(text: str) -> List[str]:
    return _TOKEN.findall((text or "").lower())


def token_iou(gold: Sequence[str], retrieved: str) -> Optional[float]:
    """Chroma-style token IoU between the gold evidence and the packed context.

    Multisets, not sets: a context that repeats a word ten times should not get
    ten times the credit, and the union in the denominator is what makes a
    bloated context cost something. ``None`` when the question has no evidence
    span (summaries and refusals), so it is excluded rather than scored zero.
    """
    gold_text = " ".join(gold).strip()
    if not gold_text:
        return None
    left, right = Counter(tokens(gold_text)), Counter(tokens(retrieved))
    if not left and not right:
        return None
    intersection = sum((left & right).values())
    union = sum((left | right).values())
    return intersection / union if union else 0.0


def recall_at(chapters: Sequence[int], gold: Sequence[int], k: int) -> float:
    """Share of the gold chapters represented in the top ``k`` results."""
    if not gold:
        return 0.0
    found = {chapter for chapter in chapters[:k] if chapter in set(gold)}
    return len(found) / len(set(gold))


def reciprocal_rank(chapters: Sequence[int], gold: Sequence[int]) -> float:
    wanted = set(gold)
    for position, chapter in enumerate(chapters, start=1):
        if chapter in wanted:
            return 1.0 / position
    return 0.0


def substring_score(answer: str, expected: Sequence[str]) -> Tuple[float, bool]:
    """``(share found, all found)`` — whitespace- and case-insensitive."""
    if not expected:
        return (0.0, False)
    haystack = _flat(answer)
    found = sum(1 for item in expected if _flat(item) in haystack)
    return (found / len(expected), found == len(expected))


def looks_like_refusal(answer: str) -> bool:
    flat = _flat(answer)
    return any(marker in flat for marker in REFUSAL_MARKERS)


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


# ------------------------------------------------------------------ results


@dataclass
class QuestionResult:
    """Everything one (arm, question) pair produced."""

    arm: str
    question: Question
    chapters: List[int]
    recall: Dict[int, float]
    mrr: float
    first_rank: Optional[int]
    iou: Optional[float]
    confidence: float
    retrieve_ms: int
    answer: str = ""
    answer_ms: int = 0
    substring_rate: Optional[float] = None
    substring_all: Optional[bool] = None
    refused: Optional[bool] = None
    refusal_correct: Optional[bool] = None
    repaired: Optional[bool] = None
    query: str = ""
    error: str = ""


@dataclass
class ArmRun:
    """One arm's results plus the facts about the index it ran against."""

    arm: Arm
    results: List[QuestionResult] = field(default_factory=list)
    chunks: int = 0
    summaries: int = 0
    contextualized: int = 0
    ingest_s: float = 0.0
    reranker: str = "none"


# ------------------------------------------------------------------ harness


class LexicalReranker:
    """Offline stand-in for FlashRank, used by ``--fake`` (SPEC §14.6).

    The real cross-encoder needs a 5 MB ONNX model fetched from HuggingFace,
    which an offline harness must not do. Scoring query/passage Jaccard overlap
    is not a cross-encoder, but it is deterministic, it genuinely reorders the
    fused head, and it exercises exactly the injection path the real reranker
    uses — which is what ``--fake`` is for. Reports label it, so a fake-mode
    arm D can never be read as a measurement of the real model.
    """

    model_name = "lexical-jaccard (offline stand-in)"

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        wanted = set(tokens(query))
        scores: List[float] = []
        for passage in passages:
            other = set(tokens(passage))
            union = wanted | other
            scores.append(len(wanted & other) / len(union) if union else 0.0)
        return scores


class Harness:
    """Ingests each index configuration once, then scores every arm on it."""

    def __init__(
        self,
        ai: Any,
        store: Store,
        settings: Settings,
        pdf: Path,
        reranker: Optional[Any],
        answers: bool = True,
        reuse: bool = False,
        verbose: bool = True,
    ) -> None:
        self.ai = ai
        self.store = store
        self.settings = settings
        self.pdf = pdf
        self.reranker = reranker
        self.answers = answers
        self.reuse = reuse
        self.verbose = verbose
        self._ingested: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------- indexing

    def index_for(self, config: IngestConfig) -> Dict[str, Any]:
        """Ingest one configuration (once per run) and return its facts."""
        if config.key in self._ingested:
            return self._ingested[config.key]

        book_id = config.book_id
        existing = self.store.get_book(book_id)
        if self.reuse and existing is not None and existing.status == STATUS_READY:
            self._log("reusing index {0}".format(config.key))
            info = {
                "book_id": book_id,
                "title": existing.title,
                "chunks": self.store.count_chunks(book_id),
                "summaries": 0,
                "contextualized": 0,
                "seconds": 0.0,
            }
        else:
            if existing is None:
                self.store.create_book(
                    book_id, "pending", self.pdf.name, "sha-{0}".format(config.key), "parsing"
                )
            self._log("ingesting {0} ({1})".format(config.key, config.mode))
            started = time.monotonic()
            result = ingest_book(
                self.store,
                self.ai,
                book_id,
                self.pdf,
                self.pdf.name,
                mode=config.mode,
                contextual=config.contextual,
                summaries=config.summaries,
                model=self.settings.llm_model,
            )
            info = {
                "book_id": book_id,
                "title": result.title,
                "chunks": result.chunks,
                "summaries": result.summaries,
                "contextualized": result.contextualized,
                "seconds": time.monotonic() - started,
            }
            self._log(
                "  {0} chunks, {1} summaries, {2} contextualized in {3:.1f}s".format(
                    info["chunks"], info["summaries"], info["contextualized"],
                    info["seconds"],
                )
            )
        info["chapters"] = self._chapter_map(book_id)
        info["vocabulary"] = self.store.get_vocabulary(book_id)
        self._ingested[config.key] = info
        return info

    def _chapter_map(self, book_id: str) -> Dict[int, int]:
        """Golden chapter number → the index this book actually assigned it.

        Resolved by island name rather than trusted to line up: the golden set
        numbers the fact table, the store numbers whatever the structure ladder
        detected, and an eval that assumes those agree would report a silent
        zero if they ever stopped agreeing.
        """
        import facts

        titles = {chapter.idx: chapter.title for chapter in self.store.get_chapters(book_id)}
        mapping: Dict[int, int] = {}
        for island in facts.ISLANDS:
            match = [idx for idx, title in titles.items() if island.name in (title or "")]
            mapping[island.idx] = match[0] if match else island.idx
        return mapping

    # -------------------------------------------------------------- scoring

    def run_arm(
        self,
        arm: Arm,
        questions: Sequence[Question],
        rrf_k: Optional[int] = None,
        weights: Optional[Sequence[float]] = None,
    ) -> ArmRun:
        info = self.index_for(arm.ingest)
        index = self.store.load_index(info["book_id"])
        reranker = self.reranker if arm.rerank else None
        run = ArmRun(
            arm=arm,
            chunks=int(info["chunks"]),
            summaries=int(info["summaries"]),
            contextualized=int(info["contextualized"]),
            ingest_s=float(info["seconds"]),
            reranker=_reranker_name(reranker),
        )
        self._log("arm {0} {1}".format(arm.key, arm.title))
        for question in questions:
            run.results.append(
                self._score(
                    arm.key,
                    question,
                    index,
                    info,
                    dense_only=arm.dense_only,
                    reranker=reranker,
                    rrf_k=rrf_k if rrf_k is not None else self.settings.rrf_k,
                    weights=weights if weights is not None else self.settings.rrf_weights,
                    answers=self.answers,
                )
            )
        return run

    def retrieval_only(
        self,
        label: str,
        questions: Sequence[Question],
        config: IngestConfig,
        rrf_k: int,
        weights: Sequence[float],
    ) -> List[QuestionResult]:
        """One sweep cell: retrieval scored, no generation, no reranker."""
        info = self.index_for(config)
        index = self.store.load_index(info["book_id"])
        return [
            self._score(
                label,
                question,
                index,
                info,
                dense_only=False,
                reranker=None,
                rrf_k=rrf_k,
                weights=weights,
                answers=False,
            )
            for question in questions
        ]

    def _score(
        self,
        label: str,
        question: Question,
        index: Any,
        info: Dict[str, Any],
        dense_only: bool,
        reranker: Optional[Any],
        rrf_k: int,
        weights: Sequence[float],
        answers: bool,
    ) -> QuestionResult:
        query = question.text
        repaired: Optional[bool] = None
        if question.asr:
            # The real query path: /api/asr repairs the transcript against the
            # book's vocabulary before anything retrieves on it (SPEC §14.9).
            fixed = repair_transcript(query, info["vocabulary"])
            query = fixed.text
            repaired = query == question.asr.get("spoken", query)

        mark = time.monotonic()
        result = retrieve(
            self.ai,
            index,
            query,
            dense_only=dense_only,
            model=self.settings.llm_model,
            child_top_k=RANK_DEPTH,
            rrf_k=rrf_k,
            weights=weights,
            reranker=reranker,
        )
        retrieve_ms = int(round((time.monotonic() - mark) * 1000))

        chapter_of = info["chapters"]
        gold = [chapter_of.get(idx, idx) for idx in question.gold]
        chapters = [item.chunk.chapter_idx for item in result.scored]
        hits = [chapter in set(gold) for chapter in chapters]
        outcome = QuestionResult(
            arm=label,
            question=question,
            chapters=chapters,
            recall={k: recall_at(chapters, gold, k) for k in RECALL_AT},
            mrr=reciprocal_rank(chapters, gold),
            first_rank=(hits.index(True) + 1) if any(hits) else None,
            iou=token_iou(
                question.evidence, "\n\n".join(block.text for block in result.blocks)
            ),
            confidence=float(result.top_score),
            retrieve_ms=retrieve_ms,
            repaired=repaired,
            query=query,
        )
        if not answers:
            return outcome

        mark = time.monotonic()
        try:
            outcome.answer = self.ai.chat(
                build_messages(info["title"], result.context, query),
                purpose="answer",
                model=self.settings.llm_model,
                temperature=ANSWER_TEMPERATURE,
            )
        except Exception as exc:  # noqa: BLE001 - one bad call must not end a run
            outcome.error = str(exc)
            self._log("  ! {0}: {1}".format(question.id, exc))
        outcome.answer_ms = int(round((time.monotonic() - mark) * 1000))

        rate, complete = substring_score(outcome.answer, question.substrings)
        outcome.substring_rate = None if question.expect_refusal else rate
        outcome.substring_all = None if question.expect_refusal else complete
        outcome.refused = looks_like_refusal(outcome.answer)
        outcome.refusal_correct = outcome.refused == question.expect_refusal
        return outcome

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr)


def _reranker_name(reranker: Optional[Any]) -> str:
    if reranker is None:
        return "none"
    return str(getattr(reranker, "model_name", reranker.__class__.__name__))


# -------------------------------------------------------------- aggregation

METRIC_COLUMNS = (
    ("recall@1", "recall@1"),
    ("recall@3", "recall@3"),
    ("recall@5", "recall@5"),
    ("recall@10", "recall@10"),
    ("mrr", "MRR"),
    ("iou", "token-IoU"),
    ("answer", "answer match"),
    ("answer_all", "answer full"),
    ("refusal", "refusal ok"),
)
#: Added to every per-category table: the abstain gate's input (SPEC §14.6).
#: Only an absolute quantity when a reranker ran — without one it is a
#: normalized fused rank, i.e. 1.0 for any question that matched anything.
CONFIDENCE_COLUMN = ("confidence", "confidence")
#: Added to the ASR table only: how often the phonetic repair restored the name.
REPAIR_COLUMN = ("asr_repair", "repair rate")
RANK_BUCKETS = ("1", "2", "3", "4–5", "6–10", "miss")


def aggregate(results: Sequence[QuestionResult]) -> Dict[str, Optional[float]]:
    """Mean of each metric over the questions for which it is defined."""
    retrieval = [item for item in results if item.question.scored_for_retrieval]
    summary: Dict[str, Optional[float]] = {"n": float(len(results))}
    for k in RECALL_AT:
        summary["recall@{0}".format(k)] = _mean([item.recall[k] for item in retrieval])
    summary["mrr"] = _mean([item.mrr for item in retrieval])
    summary["iou"] = _mean([item.iou for item in retrieval if item.iou is not None])
    summary["answer"] = _mean(
        [item.substring_rate for item in results if item.substring_rate is not None]
    )
    summary["answer_all"] = _mean(
        [1.0 if item.substring_all else 0.0 for item in results if item.substring_all is not None]
    )
    summary["refusal"] = _mean(
        [1.0 if item.refusal_correct else 0.0 for item in results if item.refusal_correct is not None]
    )
    # Over every question, not only the ones with a gold chapter: the whole
    # point of the abstain signal is what it reads on a question the book does
    # not cover.
    summary["confidence"] = _mean([item.confidence for item in results])
    summary["retrieve_ms"] = _mean([float(item.retrieve_ms) for item in results])
    repairs = [item for item in results if item.repaired is not None]
    summary["asr_repair"] = _mean([1.0 if item.repaired else 0.0 for item in repairs])
    return summary


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    usable = [float(value) for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def rank_histogram(results: Sequence[QuestionResult]) -> "OrderedDict[str, int]":
    """Where the first correct chunk landed — the shape behind the averages."""
    buckets: "OrderedDict[str, int]" = OrderedDict((name, 0) for name in RANK_BUCKETS)
    for item in results:
        if not item.question.scored_for_retrieval:
            continue
        rank = item.first_rank
        if rank is None:
            buckets["miss"] += 1
        elif rank <= 3:
            buckets[str(rank)] += 1
        elif rank <= 5:
            buckets["4–5"] += 1
        else:
            buckets["6–10"] += 1
    return buckets


# ------------------------------------------------------------------ reporting


def _fmt(value: Optional[float], places: int = 3) -> str:
    return "—" if value is None else "{0:.{1}f}".format(value, places)


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def arm_table(runs: Sequence[ArmRun]) -> str:
    headers = ["Arm", "Configuration", "chunks"] + [label for _, label in METRIC_COLUMNS]
    rows = []
    for run in runs:
        summary = aggregate(run.results)
        rows.append(
            ["**{0}** {1}".format(run.arm.key, run.arm.title), run.arm.detail, str(run.chunks)]
            + [_fmt(summary[key]) for key, _ in METRIC_COLUMNS]
        )
    return table(headers, rows)


def category_table(runs: Sequence[ArmRun], category: str) -> str:
    columns = list(METRIC_COLUMNS) + [CONFIDENCE_COLUMN]
    if category == CATEGORY_ASR:
        columns.append(REPAIR_COLUMN)
    headers = ["Arm", "n"] + [label for _, label in columns]
    rows = []
    for run in runs:
        subset = [item for item in run.results if item.question.category == category]
        if not subset:
            continue
        summary = aggregate(subset)
        rows.append(
            ["**{0}** {1}".format(run.arm.key, run.arm.title), str(int(summary["n"]))]
            + [_fmt(summary[key]) for key, _ in columns]
        )
    return table(headers, rows)


def rank_table(runs: Sequence[ArmRun]) -> str:
    rows = []
    for run in runs:
        histogram = rank_histogram(run.results)
        rows.append(
            ["**{0}** {1}".format(run.arm.key, run.arm.title)]
            + [str(histogram[name]) for name in RANK_BUCKETS]
        )
    return table(["Arm"] + ["rank " + name for name in RANK_BUCKETS], rows)


def sweep_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = ["k", "dense", "bm25", "recall@1", "recall@5", "recall@10", "MRR", "token-IoU"]
    body = [
        [
            str(row["k"]),
            "{0:.1f}".format(row["dense"]),
            "{0:.1f}".format(row["bm25"]),
            _fmt(row["recall@1"]),
            _fmt(row["recall@5"]),
            _fmt(row["recall@10"]),
            _fmt(row["mrr"]),
            _fmt(row["iou"]),
        ]
        for row in rows
    ]
    return table(headers, body)


FAKE_WARNING = """> **This run used `--fake`.** `FakeAIClient` seeds its embeddings from
> `sha1(text)`, so similar texts get unrelated vectors: dense retrieval is noise
> by construction and the dense-only arms (A, B) are expected to score near
> zero. Its chat completion is a canned sentence, so every answer metric —
> substring match, refusal correctness — measures the fake, not the pipeline.
> What a green `--fake` run proves is that the harness, the six index
> configurations and the six retrieval configurations all execute end to end.
> Numbers worth quoting come from a real-key run."""


def write_report(
    path: Path,
    runs: Sequence[ArmRun],
    golden: Dict[str, Any],
    meta: Dict[str, Any],
    sweep: Sequence[Dict[str, Any]],
) -> None:
    parts: List[str] = ["# BookTalk retrieval evaluation", ""]
    parts.append(
        "Generated {0} · {1} · fixture «{2}» ({3} pages, {4} chapters) · "
        "{5} questions · retrieval depth {6}".format(
            meta["timestamp"],
            "**offline (`--fake`)**" if meta["fake"] else "real models",
            golden["book"]["title"],
            golden["book"]["pages"],
            len(golden["book"]["chapters"]),
            meta["questions"],
            RANK_DEPTH,
        )
    )
    parts.append("")
    if meta["fake"]:
        parts.extend([FAKE_WARNING, ""])
    else:
        parts.append(
            "Models: chat `{0}`, embeddings `{1}`. Reranker: `{2}`. "
            "RRF k={3}, weights dense {4} / BM25 {5}.".format(
                meta["llm_model"],
                meta["embed_model"],
                meta["reranker"],
                meta["rrf_k"],
                meta["weights"][0],
                meta["weights"][1],
            )
        )
        parts.append("")

    if runs:
        parts.extend(["## Ablation arms", "", arm_table(runs), ""])
        parts.append(
            "Each arm adds one thing to the arm above it. `chunks` is the size of "
            "the index that arm retrieved from; `answer match` is the share of "
            "expected substrings found in the generated answer and `answer full` "
            "the share of questions where every one of them was; `refusal ok` "
            "scores both directions — refusing the out-of-book questions and *not* "
            "refusing the rest."
        )
        parts.append("")
        parts.append(
            "Read `token-IoU` against `recall`, never on its own. It is a "
            "precision signal — evidence tokens over total context tokens — so "
            "arm A scores well on it partly by retrieving *less*: naive chunks "
            "carry no parent window, so a naive context is a third the size of a "
            "structured one even when it contains nothing the question asked for."
        )
        parts.append("")
        parts.extend(["### Index build", "", _build_table(runs), ""])
        for category in CATEGORIES:
            body = category_table(runs, category)
            if body:
                parts.extend(["### Category: {0}".format(category), "", body, ""])
                note = CATEGORY_NOTES.get(category)
                if note:
                    parts.extend([note, ""])
        parts.extend(["### First-hit rank distribution", "", rank_table(runs), ""])

    if sweep:
        parts.extend(
            [
                "## RRF sweep",
                "",
                "Fusion in isolation: arm C's index, no reranker, retrieval only, "
                "over every question with a gold chapter.",
                "",
                sweep_table(sweep),
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


CATEGORY_NOTES = {
    CATEGORY_ABSENT: (
        "No gold chapter exists, so the retrieval columns are blank by design — "
        "this category only asks whether the pipeline declines to answer."
    ),
    CATEGORY_ASR: (
        "The question text is what a transcription model heard, with the island's "
        "name mangled into a phonetic near-miss. It is repaired against the "
        "book's own vocabulary before retrieval (SPEC §14.9); `repair rate` is "
        "how often the original name was restored exactly, and it varies between "
        "arms because the vocabulary is harvested from whatever chunks that index "
        "was built from."
    ),
}


def _build_table(runs: Sequence[ArmRun]) -> str:
    seen: Dict[str, ArmRun] = OrderedDict()
    for run in runs:
        seen.setdefault(run.arm.ingest.key, run)
    rows = [
        [
            run.arm.ingest.key,
            run.arm.ingest.mode,
            "yes" if run.arm.ingest.contextual else "no",
            "yes" if run.arm.ingest.summaries else "no",
            str(run.chunks),
            str(run.summaries),
            str(run.contextualized),
            "{0:.1f}".format(run.ingest_s),
        ]
        for run in seen.values()
    ]
    return table(
        ["index", "mode", "contextual", "summaries", "chunks", "summary nodes",
         "enriched", "build s"],
        rows,
    )


# ----------------------------------------------------------------------- CSV

PER_QUESTION_COLUMNS = (
    "arm", "question_id", "category", "question", "query", "gold_chapters",
    "retrieved_chapters", "first_rank", "recall@1", "recall@3", "recall@5",
    "recall@10", "mrr", "token_iou", "confidence", "answer_substring_rate",
    "answer_all_substrings", "expect_refusal", "refused", "refusal_correct",
    "asr_repaired", "retrieve_ms", "answer_ms", "answer", "error",
)


def write_per_question(path: Path, runs: Sequence[ArmRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PER_QUESTION_COLUMNS)
        for run in runs:
            for item in run.results:
                writer.writerow(
                    [
                        item.arm,
                        item.question.id,
                        item.question.category,
                        item.question.text,
                        item.query,
                        " ".join(str(idx) for idx in item.question.gold),
                        " ".join(str(idx) for idx in item.chapters),
                        "" if item.first_rank is None else item.first_rank,
                        item.recall[1], item.recall[3], item.recall[5], item.recall[10],
                        round(item.mrr, 6),
                        "" if item.iou is None else round(item.iou, 6),
                        round(item.confidence, 6),
                        "" if item.substring_rate is None else round(item.substring_rate, 6),
                        "" if item.substring_all is None else int(item.substring_all),
                        int(item.question.expect_refusal),
                        "" if item.refused is None else int(item.refused),
                        "" if item.refusal_correct is None else int(item.refusal_correct),
                        "" if item.repaired is None else int(item.repaired),
                        item.retrieve_ms,
                        item.answer_ms,
                        " ".join(item.answer.split()),
                        item.error,
                    ]
                )


PER_CATEGORY_COLUMNS = (
    ("n", "n"),
    ("recall@1", "recall@1"),
    ("recall@3", "recall@3"),
    ("recall@5", "recall@5"),
    ("recall@10", "recall@10"),
    ("mrr", "mrr"),
    ("iou", "token_iou"),
    ("answer", "answer_substring_rate"),
    ("answer_all", "answer_all_substrings"),
    ("refusal", "refusal_correct"),
    ("confidence", "confidence"),
    ("asr_repair", "asr_repair_rate"),
    ("retrieve_ms", "retrieve_ms"),
)


def write_per_category(path: Path, runs: Sequence[ArmRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "arm_title", "category"] + [name for _, name in PER_CATEGORY_COLUMNS])
        for run in runs:
            groups: "OrderedDict[str, List[QuestionResult]]" = OrderedDict()
            groups["all"] = list(run.results)
            for category in CATEGORIES:
                subset = [i for i in run.results if i.question.category == category]
                if subset:
                    groups[category] = subset
            for category, subset in groups.items():
                summary = aggregate(subset)
                writer.writerow(
                    [run.arm.key, run.arm.title, category]
                    + [_csv_number(summary[key]) for key, _ in PER_CATEGORY_COLUMNS]
                )


def write_sweep(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        columns = ["k", "dense", "bm25", "recall@1", "recall@3", "recall@5",
                   "recall@10", "mrr", "iou"]
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_csv_number(row[name]) for name in columns])


def _csv_number(value: Optional[float]) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return round(float(value), 6) if isinstance(value, float) else value


# ------------------------------------------------------------------------ run


def build_settings(fake: bool, data_dir: Path) -> Settings:
    """Configuration for the run, with the data directory redirected.

    In real mode this is the application's own loader, so a run picks up
    ``backend/.env`` exactly as the server does. The key it carries is never
    printed, logged or written to any artefact this harness produces.
    """
    from dataclasses import replace

    base = Settings() if fake else load_settings()
    return replace(base, data_dir=data_dir, flashrank=not fake)


def build_client(settings: Settings, fake: bool) -> Any:
    if fake:
        return FakeAIClient(dims=settings.embed_dims)
    if not settings.openai_configured:
        raise SystemExit(NO_KEY_MESSAGE)
    from app.services.openai_client import OpenAIClient

    return OpenAIClient(settings)


def build_reranker(settings: Settings, fake: bool) -> Optional[Any]:
    if fake:
        return LexicalReranker()
    return get_reranker(settings, allow_download=True)


def parse_arms(raw: str) -> List[Arm]:
    keys = [item.strip().upper() for item in raw.split(",") if item.strip()]
    unknown = [key for key in keys if key not in ARMS_BY_KEY]
    if unknown:
        raise SystemExit("unknown arm(s): {0}".format(", ".join(unknown)))
    return [ARMS_BY_KEY[key] for key in keys]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fake", action="store_true", help="run offline with FakeAIClient")
    parser.add_argument("--arms", default=",".join(ARMS_BY_KEY), help="arms to run, e.g. A,C,F")
    parser.add_argument("--sweep", action="store_true", help="also run the RRF (k, weight) sweep")
    parser.add_argument("--sweep-only", action="store_true", help="run only the RRF sweep")
    parser.add_argument("--no-answers", action="store_true", help="retrieval metrics only")
    parser.add_argument("--limit", type=int, default=0, help="max questions per category")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--data-dir", type=Path, default=None, help="where indexes are built")
    parser.add_argument("--reuse", action="store_true", help="keep indexes from a previous run")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.pdf.is_file() or not args.golden.is_file():
        raise SystemExit(
            "fixture missing; run:  python eval/make_fixture_book.py"
        )

    questions, golden = load_questions(args.golden, limit=args.limit or None)
    arms = [] if args.sweep_only else parse_arms(args.arms)
    run_sweep = args.sweep or args.sweep_only

    temporary = args.data_dir is None and not args.reuse
    data_dir = Path(args.data_dir) if args.data_dir else (
        Path(tempfile.mkdtemp(prefix="booktalk-eval-")) if temporary else DEFAULT_WORK
    )
    settings = build_settings(args.fake, data_dir)
    settings.ensure_dirs()
    init_db(settings.db_path)

    ai = build_client(settings, args.fake)
    reranker = build_reranker(settings, args.fake)
    harness = Harness(
        ai,
        Store(settings.db_path),
        settings,
        args.pdf,
        reranker,
        answers=not args.no_answers,
        reuse=args.reuse,
        verbose=not args.quiet,
    )

    started = time.monotonic()
    runs = [harness.run_arm(arm, questions) for arm in arms]
    sweep_rows = _sweep(harness, questions, run_sweep, not args.quiet)

    meta = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "fake": args.fake,
        "questions": len(questions),
        "llm_model": settings.llm_model,
        "embed_model": settings.embed_model,
        "reranker": _reranker_name(reranker),
        "rrf_k": settings.rrf_k,
        "weights": settings.rrf_weights,
    }
    write_report(args.report, runs, golden, meta, sweep_rows)
    if runs:
        write_per_question(args.results / "per_question.csv", runs)
        write_per_category(args.results / "per_category.csv", runs)
    if sweep_rows:
        write_sweep(args.results / "sweep.csv", sweep_rows)

    if runs:
        print(arm_table(runs))
        print()
    if sweep_rows:
        print(sweep_table(sweep_rows))
        print()
    print("report  {0}".format(args.report))
    if runs:
        print("csv     {0}".format(args.results / "per_question.csv"))
        print("        {0}".format(args.results / "per_category.csv"))
    if sweep_rows:
        print("        {0}".format(args.results / "sweep.csv"))
    print("elapsed {0:.1f}s".format(time.monotonic() - started))

    if temporary:
        shutil.rmtree(data_dir, ignore_errors=True)
    else:
        print("indexes {0}".format(data_dir))
    return 0


def _sweep(
    harness: Harness,
    questions: Sequence[Question],
    enabled: bool,
    verbose: bool,
) -> List[Dict[str, Any]]:
    """The (k, weight) grid, on the hybrid index with fusion doing all the work."""
    if not enabled:
        return []
    scored = [question for question in questions if question.scored_for_retrieval]
    rows: List[Dict[str, Any]] = []
    for k in SWEEP_K:
        for dense_weight, bm25_weight in SWEEP_WEIGHTS:
            if verbose:
                print(
                    "sweep k={0} weights={1}/{2}".format(k, dense_weight, bm25_weight),
                    file=sys.stderr,
                )
            results = harness.retrieval_only(
                "sweep-k{0}-{1}".format(k, dense_weight),
                scored,
                STRUCTURED,
                rrf_k=k,
                weights=(dense_weight, bm25_weight),
            )
            summary = aggregate(results)
            rows.append(
                {
                    "k": k,
                    "dense": dense_weight,
                    "bm25": bm25_weight,
                    "recall@1": summary["recall@1"],
                    "recall@3": summary["recall@3"],
                    "recall@5": summary["recall@5"],
                    "recall@10": summary["recall@10"],
                    "mrr": summary["mrr"],
                    "iou": summary["iou"],
                }
            )
    return rows


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
