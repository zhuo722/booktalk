"""Hybrid retrieval and context packing (SPEC §5.2 step 2, revised by §14.6).

    dense top-40 ┐                        ┌ rerank top-20 children ┐
                 ├─ weighted RRF (k=20) ──┤                        ├─ top-8
    BM25  top-40 ┘                        └ (cross-encoder, ~8 ms) ┘
                                                     │
              ≤5 sections, ≤4500 tokens ── reading order ── parent expansion
                                                              (score-gated)

Children are small and precise, so they are what gets *scored*; parents are the
surrounding section text, so they are what the LLM actually *reads*. Four
changes over v1, each with a reason:

* **Recall first, precision second.** 40 per leg instead of 24: the fusion head
  can only reorder what the legs surfaced, and the reranker only sharpens what
  fusion kept. Widening costs a slice of numpy and a BM25 argsort.
* **k and the leg weights are configuration, not constants** — see
  :mod:`app.index.fusion`. Defaults (20, 0.5/0.5) are a starting point for the
  eval sweep, not a measured optimum.
* **Parent expansion is gated.** A weak match that drags in 1600 tokens of
  section text buys noise. Chunks within a third of the best hit's score earn
  their parent window; the rest appear as bare children.
* **Blocks are assembled in reading order.** Sections are chosen best-first,
  then re-sorted by chapter and chunk ordinal and merged where they are
  contiguous, so the model reads the book's own sequence rather than a
  relevance-ordered shuffle. Gaps are marked, never silently elided.

Every kept chunk carries normalized per-signal scores (dense / BM25 / RRF /
rerank), and the result carries the best final score: that number drives the
eval reports as `confidence`; the UI warns from citation coverage instead,
which this score proved unable to stand in for.
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.index import dense
from app.index.fusion import DEFAULT_K, reciprocal_rank_fusion
from app.index.rerank import Reranker
from app.index.store import BookIndex, ChunkRecord
from app.ingest.chunker import count_tokens
from app.rag.prompts import (
    CONTEXT_BLOCK_TEMPLATE,
    NO_CONTEXT_PLACEHOLDER,
    RERANK_SYSTEM,
    RERANK_USER_TEMPLATE,
)
from app.services.openai_client import AIClient

logger = logging.getLogger(__name__)

DENSE_TOP_K = 40
BM25_TOP_K = 40
RERANK_CANDIDATES = 20
CHILD_TOP_K = 8
MAX_PARENTS = 5
MAX_CONTEXT_TOKENS = 4500
PREVIEW_CHARS = 200
DEFAULT_WEIGHTS = (0.5, 0.5)
# Share of the best hit's score a chunk must reach to earn its 1600-token
# parent window. Expressed as a *ratio* so it means the same thing whether the
# final score is a cross-encoder probability or a normalized fused rank, and so
# the top chunk always expands: no question can end up with child-only context
# by accident. Below a third of the best match a chunk reads as "related, not
# relevant" — worth quoting, not worth a section of the book.
PARENT_SCORE_THRESHOLD = 0.3
# Marks omitted text between two non-adjacent chunks of one section, so the
# model cannot mistake a jump for continuous prose.
GAP_MARKER = "[…]"


@dataclass
class Scores:
    """Normalized retrieval signals for one chunk, all in ``0..1``.

    ``dense`` is the raw cosine (clamped); ``bm25`` and ``rrf`` are scaled by
    the best value in their own leg, since neither has a meaningful absolute
    scale; ``rerank`` is the cross-encoder's own probability, or ``None`` when
    no reranker ran. ``final`` is what ranking and the parent-expansion
    gate use: the reranker when present, fused rank otherwise.
    """

    dense: float = 0.0
    bm25: float = 0.0
    rrf: float = 0.0
    rerank: Optional[float] = None
    final: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dense": round(self.dense, 6),
            "bm25": round(self.bm25, 6),
            "rrf": round(self.rrf, 6),
            "rerank": None if self.rerank is None else round(self.rerank, 6),
            "final": round(self.final, 6),
        }


@dataclass
class ScoredChunk:
    """A retrieved child chunk with its index row and its scores."""

    chunk: ChunkRecord
    row: int
    scores: Scores

    @property
    def expanded_text(self) -> str:
        """Parent window when it exists, else the child's own text."""
        return self.chunk.parent_text.strip() or self.chunk.text.strip()


@dataclass
class ContextBlock:
    """One numbered excerpt in the prompt: a section, assembled in order."""

    chapter_idx: int
    chapter_title: str
    section_title: Optional[str]
    page_start: int
    page_end: int
    ord: int
    text: str
    chunk_ids: List[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class Source:
    """A citation for one retrieved child chunk, as sent to the client."""

    chunk_id: str
    chapter_idx: int
    chapter_title: str
    section_title: Optional[str]
    pages: Tuple[int, int]
    preview: str
    #: The passage in full. The preview is what a card shows at rest; without
    #: this, "Full passage" had nothing more to open.
    text: str = ""
    scores: Optional[Scores] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chapter_idx": self.chapter_idx,
            "chapter_title": self.chapter_title,
            "section_title": self.section_title,
            "pages": [self.pages[0], self.pages[1]],
            "preview": self.preview,
            "text": self.text,
            "scores": self.scores.to_dict() if self.scores else None,
        }


@dataclass
class Retrieval:
    """Everything the generation step needs from retrieval."""

    context: str
    sources: List[Source]
    children: List[ChunkRecord]
    scored: List[ScoredChunk] = field(default_factory=list)
    blocks: List[ContextBlock] = field(default_factory=list)
    #: Best ``Scores.final`` in the result — what the eval reports as `confidence`. Only an
    #: absolute quantity when a reranker ran: fused ranks are normalized
    #: against the best hit, so without one this is 1.0 for any question that
    #: matched anything at all.
    top_score: float = 0.0

    @property
    def rows(self) -> List[int]:
        """Index rows of the retrieved children (for embedding lookups)."""
        return [item.row for item in self.scored]


def retrieve(
    ai: AIClient,
    index: BookIndex,
    question: str,
    dense_only: bool = False,
    rerank: bool = False,
    model: Optional[str] = None,
    dense_top_k: int = DENSE_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    child_top_k: int = CHILD_TOP_K,
    max_parents: int = MAX_PARENTS,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    rrf_k: int = DEFAULT_K,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    reranker: Optional[Reranker] = None,
    rerank_candidates: int = RERANK_CANDIDATES,
    parent_score_threshold: float = PARENT_SCORE_THRESHOLD,
) -> Retrieval:
    """Run hybrid retrieval for one question and pack the LLM context.

    ``reranker`` is injected, never loaded here: loading it may download a
    model, and retrieval sits on the latency-critical path (see
    :func:`app.index.rerank.prewarm`). ``rerank`` is the separate, off-by-default
    single-call LLM rerank from SPEC §12.
    """
    if not index.chunks or not question.strip():
        return Retrieval(context=NO_CONTEXT_PLACEHOLDER, sources=[], children=[])

    dense_hits: List[Tuple[int, float]] = []
    bm25_hits: List[Tuple[int, float]] = []
    rankings: List[List[int]] = []
    leg_weights: List[float] = []
    dense_weight, bm25_weight = _leg_weights(weights)

    if index.matrix.size:
        query_vector = ai.embed([question])[0]
        dense_hits = dense.top_k(index.matrix, query_vector, dense_top_k)
        rankings.append([row for row, _ in dense_hits])
        leg_weights.append(dense_weight)
    if not dense_only:
        bm25_hits = index.bm25.top_k(question, bm25_top_k)
        rankings.append([row for row, _ in bm25_hits])
        leg_weights.append(bm25_weight)

    fused = reciprocal_rank_fusion(rankings, k=rrf_k, weights=leg_weights)
    if not fused:
        return Retrieval(context=NO_CONTEXT_PLACEHOLDER, sources=[], children=[])

    head = fused[: max(rerank_candidates, child_top_k)]
    rows = [int(row) for row, _ in head]
    rrf_scores = [float(score) for _, score in head]
    rerank_scores = _rerank_scores(reranker, question, index, rows)
    if rerank_scores is not None:
        order = sorted(range(len(rows)), key=lambda i: (-rerank_scores[i], i))
        rows = [rows[i] for i in order]
        rrf_scores = [rrf_scores[i] for i in order]
        rerank_scores = [rerank_scores[i] for i in order]

    selected = _score_selection(
        index,
        rows[:child_top_k],
        rrf_scores[:child_top_k],
        None if rerank_scores is None else rerank_scores[:child_top_k],
        dense_hits,
        bm25_hits,
        best_rrf=max((score for _, score in fused), default=0.0),
    )
    top_score = max((item.scores.final for item in selected), default=0.0)

    if rerank and len(selected) > 1:
        selected = _llm_rerank(ai, question, selected, model)

    blocks = _assemble(
        selected, max_parents, max_context_tokens, parent_score_threshold
    )
    kept_ids = {chunk_id for block in blocks for chunk_id in block.chunk_ids}
    return Retrieval(
        context=_format_context(blocks),
        sources=[_source(item) for item in selected if item.chunk.id in kept_ids],
        children=[item.chunk for item in selected],
        scored=selected,
        blocks=blocks,
        top_score=top_score,
    )


def _leg_weights(weights: Sequence[float]) -> Tuple[float, float]:
    """Read the dense/BM25 weight pair, tolerating a short or absent value."""
    values = list(weights or ())
    dense_weight = float(values[0]) if len(values) > 0 else DEFAULT_WEIGHTS[0]
    bm25_weight = float(values[1]) if len(values) > 1 else DEFAULT_WEIGHTS[1]
    return dense_weight, bm25_weight


def _rerank_scores(
    reranker: Optional[Reranker],
    question: str,
    index: BookIndex,
    rows: Sequence[int],
) -> Optional[List[float]]:
    """Cross-encoder scores for the fused head, or ``None`` to keep that order.

    Only child text is scored: the model truncates at 512 tokens, so a parent
    window would be judged on its opening third. A single candidate is scored
    too — there is nothing to reorder, but the score is still reported.
    """
    if reranker is None or not rows:
        return None
    passages = [index.chunks[row].text for row in rows]
    try:
        scores = [float(value) for value in reranker.score(question, passages)]
    except Exception:  # noqa: BLE001 - fusion order is a fine fallback
        logger.warning("reranker failed; keeping fusion order", exc_info=True)
        return None
    if len(scores) != len(passages):
        logger.warning(
            "reranker returned %d scores for %d passages; keeping fusion order",
            len(scores),
            len(passages),
        )
        return None
    return scores


def _score_selection(
    index: BookIndex,
    rows: Sequence[int],
    rrf_scores: Sequence[float],
    rerank_scores: Optional[Sequence[float]],
    dense_hits: Sequence[Tuple[int, float]],
    bm25_hits: Sequence[Tuple[int, float]],
    best_rrf: float,
) -> List[ScoredChunk]:
    """Attach normalized per-signal scores to the chosen children."""
    dense_by_row = {int(row): score for row, score in dense_hits}
    bm25_by_row = {int(row): score for row, score in bm25_hits}
    best_bm25 = max(bm25_by_row.values(), default=0.0)

    selected: List[ScoredChunk] = []
    for position, row in enumerate(rows):
        rerank_score = (
            float(rerank_scores[position]) if rerank_scores is not None else None
        )
        rrf_normalized = _ratio(rrf_scores[position], best_rrf)
        scores = Scores(
            dense=_clamp(dense_by_row.get(row, 0.0)),
            bm25=_ratio(bm25_by_row.get(row, 0.0), best_bm25),
            rrf=rrf_normalized,
            rerank=None if rerank_score is None else _clamp(rerank_score),
            final=rrf_normalized if rerank_score is None else _clamp(rerank_score),
        )
        selected.append(ScoredChunk(chunk=index.chunks[row], row=row, scores=scores))
    return selected


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _ratio(value: float, best: float) -> float:
    if best <= 0.0:
        return 0.0
    return _clamp(float(value) / float(best))


def _assemble(
    selected: Sequence[ScoredChunk],
    max_parents: int,
    max_context_tokens: int,
    parent_score_threshold: float,
) -> List[ContextBlock]:
    """Group children into section blocks, then put them in reading order.

    Sections compete on their best child's score (the budget must protect the
    strongest evidence), but the surviving blocks are handed to the model in
    the order the book prints them.
    """
    groups: "OrderedDict[Tuple[str, bool], List[ScoredChunk]]" = OrderedDict()
    seen_ids: set = set()
    for item in selected:
        if item.chunk.id in seen_ids:
            continue
        seen_ids.add(item.chunk.id)
        groups.setdefault(_group_key(item.chunk), []).append(item)

    # The gate is relative to the best hit, so it means the same thing whether
    # `final` is a cross-encoder probability or a normalized fused rank — and
    # the top chunk always earns its parent window.
    best = max((item.scores.final for item in selected), default=0.0)
    cutoff = parent_score_threshold * best if best > 0 else float("inf")

    blocks: List[ContextBlock] = []
    budget = 0
    for members in groups.values():
        block = _block(members, cutoff)
        tokens = count_tokens(block.text)
        if blocks and budget + tokens > max_context_tokens:
            continue
        blocks.append(block)
        budget += tokens
        if len(blocks) >= max_parents:
            break
    blocks.sort(key=lambda block: (block.chapter_idx, block.ord))
    return blocks


def _group_key(chunk: ChunkRecord) -> Tuple[str, bool]:
    """Which block a chunk belongs to: its parent section, summaries apart.

    A chapter synopsis (SPEC §14.5) shares a chapter with body prose but is not
    continuous with it — merging the two into one excerpt would present a
    summary as if it were the page it summarizes.
    """
    return (chunk.parent_key, bool(getattr(chunk, "is_summary", False)))


def _block(members: Sequence[ScoredChunk], cutoff: float) -> ContextBlock:
    """Build one section block from every selected child of that section."""
    ordered = sorted(members, key=lambda item: item.chunk.ord)
    first = ordered[0].chunk
    return ContextBlock(
        chapter_idx=first.chapter_idx,
        chapter_title=first.chapter_title,
        section_title=first.section_title,
        page_start=min(item.chunk.page_start for item in ordered),
        page_end=max(item.chunk.page_end for item in ordered),
        ord=first.ord,
        text=_merge_runs(ordered, cutoff),
        chunk_ids=[item.chunk.id for item in ordered],
        score=max(item.scores.final for item in ordered),
    )


def _merge_runs(ordered: Sequence[ScoredChunk], cutoff: float) -> str:
    """Concatenate the members of one section, deduping and marking gaps.

    Working at paragraph granularity does three jobs at once: it collapses the
    overlap two neighbouring children share, it collapses the much larger
    overlap of two parent windows cut from the same section, and it leaves a
    marker wherever the selected chunks are not adjacent.
    """
    paragraphs: List[str] = []
    seen: set = set()
    previous_ord: Optional[int] = None
    for item in ordered:
        expand = item.scores.final >= cutoff
        segment = item.expanded_text if expand else item.chunk.text.strip()
        gap = previous_ord is not None and item.chunk.ord > previous_ord + 1
        for paragraph in segment.split("\n\n"):
            text = paragraph.strip()
            if not text or text in seen:
                continue
            if gap and paragraphs:
                paragraphs.append(GAP_MARKER)
            gap = False
            seen.add(text)
            paragraphs.append(text)
        previous_ord = item.chunk.ord
    return "\n\n".join(paragraphs)


def _format_context(blocks: Sequence[ContextBlock]) -> str:
    """Render the numbered excerpt blocks the answer prompt embeds."""
    if not blocks:
        return NO_CONTEXT_PLACEHOLDER
    rendered = [
        CONTEXT_BLOCK_TEMPLATE.format(
            position=position,
            chapter_idx=block.chapter_idx,
            chapter_title=block.chapter_title or "Untitled chapter",
            section=" — {0}".format(block.section_title) if block.section_title else "",
            page_start=block.page_start,
            page_end=block.page_end,
            text=block.text.strip(),
        )
        for position, block in enumerate(blocks, start=1)
    ]
    return "\n\n".join(rendered)


def _source(item: ScoredChunk) -> Source:
    child = item.chunk
    full = " ".join(child.text.split())
    return Source(
        chunk_id=child.id,
        chapter_idx=child.chapter_idx,
        chapter_title=child.chapter_title,
        section_title=child.section_title,
        pages=(child.page_start, child.page_end),
        preview=full[:PREVIEW_CHARS],
        text=full,
        scores=item.scores,
    )


def _llm_rerank(
    ai: AIClient,
    question: str,
    selected: Sequence[ScoredChunk],
    model: Optional[str],
) -> List[ScoredChunk]:
    """Optional single-call rerank of the selected children (``BOOKTALK_RERANK=1``).

    Off by default: it adds a full round trip before the first token, which is
    the latency we are optimizing for, and the cross-encoder now covers the
    same ground for ~8 ms. Any malformed answer leaves the existing order
    untouched.
    """
    excerpts = "\n\n".join(
        "[{0}] {1}".format(position, " ".join(item.chunk.text.split())[:400])
        for position, item in enumerate(selected)
    )
    messages = [
        {"role": "system", "content": RERANK_SYSTEM},
        {
            "role": "user",
            "content": RERANK_USER_TEMPLATE.format(question=question, excerpts=excerpts),
        },
    ]
    try:
        raw = ai.chat(messages, purpose="rerank", temperature=0, model=model)
        order = _parse_order(raw, len(selected))
    except Exception:  # noqa: BLE001 - fusion order is a fine fallback
        logger.warning("LLM rerank failed; keeping fusion order", exc_info=True)
        return list(selected)
    if not order:
        return list(selected)
    ranked = [selected[position] for position in order]
    ranked.extend(item for index, item in enumerate(selected) if index not in order)
    return ranked


def _parse_order(raw: str, count: int) -> List[int]:
    """Extract a list of distinct in-range indices from a model reply."""
    match = re.search(r"\[.*\]", raw or "", re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        return []
    order: List[int] = []
    for item in parsed if isinstance(parsed, list) else []:
        try:
            position = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= position < count and position not in order:
            order.append(position)
    return order
