"""Hybrid retrieval, reranking and context assembly (SPEC §14.6).

Every test builds its own :class:`BookIndex` in memory: the dense side is
one-hot vectors so the cosine ordering is whatever the test says it is, and the
lexical side is real BM25 over the chunk headers. Nothing here touches the
database, the network, or a real cross-encoder.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest

from app.index.bm25 import BM25Index
from app.index.dense import normalize
from app.index.store import BookIndex, ChunkRecord
from app.rag import retriever
from app.rag.retriever import GAP_MARKER, retrieve

# --------------------------------------------------------------------- doubles


class StubAI:
    """An AI client whose query embedding (and rerank reply) the test chooses."""

    def __init__(
        self, query_vector: Sequence[float] = (1.0,), chat_reply: str = "[]"
    ) -> None:
        self.query_vector = [float(value) for value in query_vector]
        self.chat_reply = chat_reply
        self.embed_calls: List[List[str]] = []
        self.chat_calls: List[List[Dict[str, str]]] = []

    def embed(self, texts: List[str]) -> List[List[float]]:
        self.embed_calls.append(list(texts))
        return [list(self.query_vector) for _ in texts]

    def chat(self, messages: List[Dict[str, str]], **kw: Any) -> str:
        self.chat_calls.append(list(messages))
        return self.chat_reply


class FakeRanker:
    """Deterministic stand-in for FlashRank: a score per passage substring."""

    def __init__(
        self, scores: Optional[Dict[str, float]] = None, fail: bool = False,
        truncate: bool = False,
    ) -> None:
        self.scores = scores or {}
        self.fail = fail
        self.truncate = truncate
        self.calls: List[Sequence[str]] = []

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        self.calls.append(list(passages))
        if self.fail:
            raise RuntimeError("onnx session died")
        values = [
            max(
                [score for marker, score in self.scores.items() if marker in passage]
                or [0.0]
            )
            for passage in passages
        ]
        return values[:-1] if self.truncate else values


# -------------------------------------------------------------------- builders


def make_chunk(
    row: int,
    chapter_idx: int = 1,
    section: Optional[str] = None,
    text: Optional[str] = None,
    parent: Optional[str] = None,
    ordinal: Optional[int] = None,
    pages: Sequence[int] = (1, 1),
) -> ChunkRecord:
    body = text if text is not None else "chunk {0} body prose".format(row)
    return ChunkRecord(
        id="c{0}".format(row),
        book_id="book",
        chapter_idx=chapter_idx,
        chapter_title="Chapter {0}".format(chapter_idx),
        section_title=section,
        page_start=pages[0],
        page_end=pages[1],
        text=body,
        header=body,
        parent_text=body if parent is None else parent,
        ord=row if ordinal is None else ordinal,
    )


def make_index(
    chunks: Sequence[ChunkRecord], vectors: Optional[Sequence[Sequence[float]]] = None
) -> BookIndex:
    """A retrieval bundle with one-hot embeddings unless told otherwise."""
    rows = np.eye(len(chunks), dtype=np.float32) if vectors is None else np.asarray(
        vectors, dtype=np.float32
    )
    return BookIndex(
        book_id="book",
        chunks=list(chunks),
        matrix=normalize(rows),
        bm25=BM25Index([chunk.header for chunk in chunks]),
    )


def descending(count: int) -> List[float]:
    """A query vector that ranks one-hot chunks in index order."""
    return [float(count - position) for position in range(count)]


def two_leg_index() -> BookIndex:
    """Five chunks: dense favours c0, BM25 sees only c4 ("zephyr")."""
    texts = [
        "alpha harbour prose about lantern keepers",
        "bravo orchard prose about granite walls",
        "charlie ferry prose about tide charts",
        "delta beacon prose about reef markers",
        "zephyr ledger prose about quarry books",
    ]
    return make_index([make_chunk(row, text=texts[row]) for row in range(5)])


# ------------------------------------------------------------------ RRF wiring


def test_weights_decide_which_leg_wins():
    index = two_leg_index()
    ai = StubAI(descending(5))

    dense_led = retrieve(ai, index, "zephyr", weights=(1.0, 0.0), rrf_k=60)
    lexical_led = retrieve(ai, index, "zephyr", weights=(0.0, 1.0), rrf_k=60)

    assert dense_led.children[0].id == "c0"
    assert lexical_led.children[0].id == "c4"


def test_rrf_k_arbitrates_between_a_first_place_and_agreement():
    # c0 is dense rank 1 only; c4 is dense rank 5 *and* BM25 rank 1. A small k
    # makes rank one nearly decisive, a large k flattens the curve until the
    # agreement of two legs outweighs it.
    index = two_leg_index()
    ai = StubAI(descending(5))

    sharp = retrieve(ai, index, "zephyr", rrf_k=1, weights=(1.0, 0.35))
    flat = retrieve(ai, index, "zephyr", rrf_k=60, weights=(1.0, 0.35))

    assert sharp.children[0].id == "c0"
    assert flat.children[0].id == "c4"


def test_default_k_comes_from_the_fusion_module():
    from app.index.fusion import DEFAULT_K

    # SPEC §14.6: TREC's 60 assumes million-doc corpora; a book is four orders
    # smaller, so the default moved into the small-corpus range.
    assert DEFAULT_K == 20


def test_the_retrieval_knobs_are_configuration_not_constants(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    from app.config import load_settings

    monkeypatch.setenv("BOOKTALK_RRF_K", "10")
    monkeypatch.setenv("BOOKTALK_DENSE_WEIGHT", "0.8")
    monkeypatch.setenv("BOOKTALK_BM25_WEIGHT", "0.2")

    settings = load_settings(env_file=tmp_path / "absent.env")

    assert settings.rrf_k == 10
    assert settings.rrf_weights == (0.8, 0.2)
    # And the retriever accepts exactly that pair, in leg order.
    index = two_leg_index()
    result = retrieve(
        StubAI(descending(5)), index, "zephyr",
        rrf_k=settings.rrf_k, weights=settings.rrf_weights,
    )
    assert result.children[0].id == "c0"


def test_unparseable_knobs_fall_back_to_the_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    from app.config import load_settings

    monkeypatch.setenv("BOOKTALK_DENSE_WEIGHT", "not-a-number")

    settings = load_settings(env_file=tmp_path / "absent.env")

    assert settings.rrf_weights == (0.5, 0.5)


def test_both_legs_are_widened_to_forty(monkeypatch: pytest.MonkeyPatch):
    index = two_leg_index()
    ai = StubAI(descending(5))
    seen: Dict[str, int] = {}

    real_dense = retriever.dense.top_k

    def spy_dense(matrix, query, k):
        seen["dense"] = k
        return real_dense(matrix, query, k)

    real_bm25 = index.bm25.top_k

    def spy_bm25(query, k):
        seen["bm25"] = k
        return real_bm25(query, k)

    monkeypatch.setattr(retriever.dense, "top_k", spy_dense)
    monkeypatch.setattr(index.bm25, "top_k", spy_bm25)

    retrieve(ai, index, "zephyr")

    assert seen == {"dense": 40, "bm25": 40}


def test_dense_only_skips_the_lexical_leg():
    index = two_leg_index()
    ai = StubAI(descending(5))

    result = retrieve(ai, index, "zephyr", dense_only=True, weights=(0.5, 0.5))

    assert result.children[0].id == "c0"
    assert all(item.scores.bm25 == 0.0 for item in result.scored)


# -------------------------------------------------------------------- reranker


def test_reranker_reorders_the_fused_head():
    index = two_leg_index()
    ai = StubAI(descending(5))
    ranker = FakeRanker({"delta": 0.95, "alpha": 0.10})

    result = retrieve(ai, index, "zephyr", reranker=ranker, weights=(1.0, 0.0))

    assert [chunk.id for chunk in result.children[:2]] == ["c3", "c0"]
    assert result.scored[0].scores.rerank == pytest.approx(0.95)
    assert result.scored[0].scores.final == pytest.approx(0.95)
    assert result.top_score == pytest.approx(0.95)


def test_reranker_scores_child_text_never_parents():
    # The cross-encoder truncates at 512 tokens; a parent window would be
    # judged on its opening third.
    chunks = [
        make_chunk(0, text="alpha child", parent="alpha child plus a long parent window"),
        make_chunk(1, text="bravo child", parent="bravo child plus a long parent window"),
    ]
    ranker = FakeRanker({"alpha": 0.9})

    retrieve(StubAI(descending(2)), make_index(chunks), "alpha", reranker=ranker)

    assert ranker.calls == [["alpha child", "bravo child"]]


def test_reranker_sees_a_wider_window_than_it_returns():
    chunks = [make_chunk(row, text="passage {0} prose".format(row)) for row in range(12)]
    ranker = FakeRanker({"passage 9 ": 0.99})

    result = retrieve(
        StubAI(descending(12)),
        make_index(chunks),
        "passage",
        child_top_k=3,
        rerank_candidates=10,
        reranker=ranker,
    )

    assert len(ranker.calls[0]) == 10
    # A chunk fusion ranked tenth is promoted to first by the cross-encoder.
    assert result.children[0].id == "c9"
    assert len(result.children) == 3


def test_a_failing_reranker_leaves_the_fusion_order_alone():
    index = two_leg_index()
    result = retrieve(
        StubAI(descending(5)),
        index,
        "zephyr",
        reranker=FakeRanker(fail=True),
        weights=(1.0, 0.0),
    )

    assert result.children[0].id == "c0"
    assert result.scored[0].scores.rerank is None
    assert result.scored[0].scores.final == pytest.approx(result.scored[0].scores.rrf)


def test_a_reranker_that_drops_a_score_is_ignored():
    index = two_leg_index()
    result = retrieve(
        StubAI(descending(5)),
        index,
        "zephyr",
        reranker=FakeRanker({"delta": 0.95}, truncate=True),
        weights=(1.0, 0.0),
    )

    assert result.children[0].id == "c0"
    assert result.scored[0].scores.rerank is None


def test_no_reranker_means_no_rerank_score():
    result = retrieve(StubAI(descending(5)), two_leg_index(), "zephyr")

    assert all(item.scores.rerank is None for item in result.scored)


# ----------------------------------------------------------- parent expansion


def test_parent_expansion_is_gated_by_score():
    chunks = [
        make_chunk(
            0, chapter_idx=1, section="One",
            text="alpha child", parent="alpha child SURROUNDING-ONE",
        ),
        make_chunk(
            1, chapter_idx=2, section="Two",
            text="bravo child", parent="bravo child SURROUNDING-TWO",
        ),
    ]
    ranker = FakeRanker({"alpha": 0.90, "bravo": 0.10})

    result = retrieve(
        StubAI(descending(2)), make_index(chunks), "alpha",
        reranker=ranker, parent_score_threshold=0.3,
    )

    assert "SURROUNDING-ONE" in result.context
    assert "SURROUNDING-TWO" not in result.context
    assert "bravo child" in result.context


def test_a_lone_candidate_is_still_scored_by_the_reranker():
    # Nothing to reorder, but the abstain gate needs an absolute number: fused
    # ranks are normalized against the best hit, so they are always 1.0 here.
    chunks = [make_chunk(0, text="alpha child")]
    ranker = FakeRanker({"alpha": 0.04})

    result = retrieve(StubAI(descending(1)), make_index(chunks), "alpha", reranker=ranker)

    assert result.top_score == pytest.approx(0.04)
    assert len(ranker.calls) == 1


def test_the_best_chunk_always_expands():
    # Scores are normalized against the best hit, so the top chunk sits at 1.0
    # and no question can end up with child-only context by accident.
    chunks = [make_chunk(0, text="alpha child", parent="alpha child SURROUNDING")]

    result = retrieve(StubAI(descending(1)), make_index(chunks), "alpha")

    assert "SURROUNDING" in result.context
    assert result.scored[0].scores.final == pytest.approx(1.0)


def test_a_threshold_above_one_disables_expansion_entirely():
    chunks = [make_chunk(0, text="alpha child", parent="alpha child SURROUNDING")]

    result = retrieve(
        StubAI(descending(1)), make_index(chunks), "alpha",
        parent_score_threshold=1.5,
    )

    assert "SURROUNDING" not in result.context


# --------------------------------------------------------------- assembly


def test_blocks_are_emitted_in_reading_order_not_relevance_order():
    chunks = [
        make_chunk(0, chapter_idx=3, section="Late", text="delta prose", ordinal=30),
        make_chunk(1, chapter_idx=1, section="Early", text="alpha prose", ordinal=10),
        make_chunk(2, chapter_idx=1, section="Middle", text="bravo prose", ordinal=20),
    ]
    ranker = FakeRanker({"delta": 0.99, "bravo": 0.80, "alpha": 0.60})

    result = retrieve(StubAI(descending(3)), make_index(chunks), "prose", reranker=ranker)

    assert [chunk.id for chunk in result.children] == ["c0", "c2", "c1"]
    assert [(block.chapter_idx, block.ord) for block in result.blocks] == [
        (1, 10), (1, 20), (3, 30)
    ]
    assert result.context.index("alpha prose") < result.context.index("delta prose")


def test_contiguous_chunks_of_one_section_merge_and_dedupe():
    shared = "the shared overlap paragraph"
    chunks = [
        make_chunk(
            0, section="One", ordinal=0,
            text="first paragraph\n\n" + shared,
            parent="first paragraph\n\n" + shared,
        ),
        make_chunk(
            1, section="One", ordinal=1,
            text=shared + "\n\nsecond paragraph",
            parent=shared + "\n\nsecond paragraph",
        ),
    ]

    result = retrieve(StubAI(descending(2)), make_index(chunks), "paragraph")

    assert len(result.blocks) == 1
    assert result.blocks[0].chunk_ids == ["c0", "c1"]
    assert result.context.count(shared) == 1
    assert GAP_MARKER not in result.context


def test_a_gap_between_chunks_of_one_section_is_marked():
    chunks = [
        make_chunk(0, section="One", ordinal=0, text="opening paragraph"),
        make_chunk(1, section="One", ordinal=7, text="much later paragraph"),
    ]

    result = retrieve(StubAI(descending(2)), make_index(chunks), "paragraph")

    assert len(result.blocks) == 1
    assert GAP_MARKER in result.blocks[0].text
    assert result.blocks[0].text.index("opening") < result.blocks[0].text.index("later")


def test_a_chapter_summary_never_merges_into_the_prose_it_summarizes():
    from app.rag.retriever import _group_key

    class Body:
        parent_key = "1::"
        is_summary = False

    class Synopsis:
        parent_key = "1::"
        is_summary = True

    # Same chapter, same (absent) section — but a synopsis is not continuous
    # with the pages it describes, so it must not be spliced into them.
    assert _group_key(Body()) != _group_key(Synopsis())


def test_chunks_without_a_summary_flag_still_group():
    from app.rag.retriever import _group_key

    class Plain:
        parent_key = "2::Tides"

    assert _group_key(Plain()) == ("2::Tides", False)


def test_sections_are_capped_and_the_token_budget_is_respected():
    long_parent = " ".join(["paragraph"] * 400)
    chunks = [
        make_chunk(
            row, chapter_idx=row, section="S{0}".format(row),
            text="body {0}".format(row), parent="body {0} ".format(row) + long_parent,
        )
        for row in range(6)
    ]

    capped = retrieve(StubAI(descending(6)), make_index(chunks), "body", max_parents=2)
    budgeted = retrieve(
        StubAI(descending(6)), make_index(chunks), "body", max_context_tokens=900
    )

    assert len(capped.blocks) == 2
    assert 0 < len(budgeted.blocks) < 6


def test_sources_only_describe_chunks_that_reached_the_context():
    long_parent = " ".join(["paragraph"] * 400)
    chunks = [
        make_chunk(
            row, chapter_idx=row, section="S{0}".format(row),
            text="body {0}".format(row), parent=long_parent,
        )
        for row in range(6)
    ]

    result = retrieve(StubAI(descending(6)), make_index(chunks), "body", max_parents=2)

    kept = {chunk_id for block in result.blocks for chunk_id in block.chunk_ids}
    assert {source.chunk_id for source in result.sources} == kept


# ------------------------------------------------------------------- scores


def test_every_source_carries_normalized_signal_scores():
    result = retrieve(StubAI(descending(5)), two_leg_index(), "zephyr")

    payload = result.sources[0].to_dict()
    assert set(payload["scores"]) == {"dense", "bm25", "rrf", "rerank", "final"}
    for item in result.scored:
        for value in (item.scores.dense, item.scores.bm25, item.scores.rrf, item.scores.final):
            assert 0.0 <= value <= 1.0
    # The lexical leg only matched c4, so exactly one chunk carries BM25 mass.
    assert [item.chunk.id for item in result.scored if item.scores.bm25 > 0] == ["c4"]


def test_top_score_is_the_best_final_score():
    result = retrieve(StubAI(descending(5)), two_leg_index(), "zephyr")

    assert result.top_score == pytest.approx(
        max(item.scores.final for item in result.scored)
    )
    assert result.top_score == pytest.approx(1.0)


def test_rows_expose_the_index_positions_for_embedding_lookups():
    result = retrieve(StubAI(descending(5)), two_leg_index(), "zephyr", weights=(1.0, 0.0))

    assert result.rows[0] == 0
    assert [result.scored[0].chunk.id] == ["c0"]


# -------------------------------------------------------------- empty inputs


def test_an_empty_question_retrieves_nothing():
    result = retrieve(StubAI(descending(5)), two_leg_index(), "   ")

    assert result.sources == [] and result.children == []
    assert result.top_score == 0.0
    assert result.context.startswith("(no excerpts")


def test_an_empty_book_retrieves_nothing():
    index = BookIndex(book_id="book", chunks=[], matrix=normalize(np.zeros((0, 3), dtype=np.float32)), bm25=BM25Index([]))

    result = retrieve(StubAI([1.0, 0.0, 0.0]), index, "anything")

    assert result.children == [] and result.top_score == 0.0


def test_a_question_matching_nothing_at_all_retrieves_nothing():
    # A book ingested without embeddings (dense leg absent) and a query whose
    # tokens appear nowhere: both legs come back empty, so must the result.
    chunks = [make_chunk(0, text="harbour prose")]
    index = make_index(chunks)
    index.matrix = np.zeros((1, 0), dtype=np.float32)

    result = retrieve(StubAI([1.0]), index, "zzzz")

    assert result.children == []
    assert result.context.startswith("(no excerpts")


# ------------------------------------------------------- legacy LLM rerank


def test_the_legacy_llm_rerank_still_reorders_children():
    index = two_leg_index()
    ai = StubAI(descending(5), chat_reply="[2, 0]")

    result = retrieve(ai, index, "zephyr", rerank=True, weights=(1.0, 0.0))

    assert [chunk.id for chunk in result.children[:2]] == ["c2", "c0"]
    assert len(result.children) == 5


def test_a_malformed_llm_rerank_reply_is_ignored():
    index = two_leg_index()
    ai = StubAI(descending(5), chat_reply="sorry, no")

    result = retrieve(ai, index, "zephyr", rerank=True, weights=(1.0, 0.0))

    assert result.children[0].id == "c0"
