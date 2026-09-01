"""Post-hoc sentence → chunk attribution (SPEC §14.7).

The embedder is a stub that returns whatever vector the test names, so every
similarity in here is exact arithmetic rather than a guess about what a real
embedding model would do. That is the only way to pin down a rule as specific
as "0.63, times 0.8, never below 0.3".
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pytest

from app.index.bm25 import BM25Index
from app.index.dense import normalize
from app.index.store import BookIndex, ChunkRecord
from app.rag.citations import (
    CitationChunk,
    attribute_citations,
    citation_chunks,
    split_answer_sentence_spans,
    split_answer_sentences,
)
from app.rag.retriever import Retrieval, ScoredChunk, Scores, Source

DIMS = 4
MATCH = [1.0, 0.0, 0.0, 0.0]
OTHER = [0.0, 1.0, 0.0, 0.0]


def source(record: ChunkRecord) -> Source:
    """The ``sources`` entry a chunk gets once it reaches the model."""
    return Source(
        chunk_id=record.id,
        chapter_idx=record.chapter_idx,
        chapter_title=record.chapter_title,
        section_title=record.section_title,
        pages=(record.page_start, record.page_end),
        preview=record.text[:200],
    )


class StubEmbedder:
    """Returns a caller-supplied vector per text (default: orthogonal to all)."""

    def __init__(self, vectors: Dict[str, Sequence[float]], default: Sequence[float] = OTHER):
        self.vectors = vectors
        self.default = list(default)
        self.calls: List[List[str]] = []

    def embed(self, texts: List[str]) -> List[List[float]]:
        self.calls.append(list(texts))
        return [list(self.vectors.get(text, self.default)) for text in texts]


def partial(cosine: float) -> List[float]:
    """A unit vector whose cosine against ``MATCH`` is exactly ``cosine``."""
    return [cosine, (1.0 - cosine ** 2) ** 0.5, 0.0, 0.0]


def chunk(chunk_id: str, text: str, embedding: Sequence[float] = MATCH) -> CitationChunk:
    return CitationChunk(chunk_id=chunk_id, text=text, embedding=list(embedding))


# ------------------------------------------------------------------ matching


def test_a_sentence_is_attributed_to_the_chunk_it_came_from():
    answer = "The lighthouse was raised in eighteen forty one."
    ai = StubEmbedder({answer: MATCH})
    chunks = [chunk("a", "granite and mortar"), chunk("b", "ferry timetables", OTHER)]

    assert attribute_citations(ai, answer, chunks) == [
        {"sentence_idx": 0, "span": [0, len(answer)], "chunk_ids": ["a"]}
    ]


def test_an_ungrounded_sentence_gets_no_citation():
    answer = "Nothing in these excerpts speaks to that."
    ai = StubEmbedder({}, default=[0.0, 0.0, 1.0, 0.0])
    chunks = [chunk("a", "granite and mortar"), chunk("b", "ferry timetables")]

    assert attribute_citations(ai, answer, chunks) == []


def test_no_chunks_means_no_citations():
    ai = StubEmbedder({})

    assert attribute_citations(ai, "Some finished answer here.", []) == []


def test_an_empty_answer_costs_no_embedding_call():
    ai = StubEmbedder({})

    assert attribute_citations(ai, "   ", [chunk("a", "granite")]) == []
    assert ai.calls == []


def test_each_sentence_is_attributed_independently():
    first = "The keeper's name was Elsa Marren."
    second = "The ferry runs twice a day in winter."
    ai = StubEmbedder({second: MATCH}, default=[0.0, 0.0, 1.0, 0.0])
    chunks = [chunk("a", "timetables and tides")]

    joined = first + " " + second
    assert attribute_citations(ai, joined, chunks) == [
        {
            "sentence_idx": 1,
            "span": [len(first) + 1, len(joined)],
            "chunk_ids": ["a"],
        }
    ]


def test_the_whole_answer_is_embedded_in_one_batch():
    answer = "First sentence of the answer. Second sentence of the answer."
    ai = StubEmbedder({})

    attribute_citations(ai, answer, [chunk("a", "granite")])

    assert len(ai.calls) == 1
    assert len(ai.calls[0]) == 2


def test_a_failing_embedder_leaves_the_answer_uncited():
    class Broken:
        def embed(self, texts):
            raise RuntimeError("embeddings are down")

    assert attribute_citations(Broken(), "A grounded sentence here.", [chunk("a", "x")]) == []


# ----------------------------------------------------------- threshold ladder


def test_the_threshold_descends_until_something_matches():
    # cosine 0.5 → 0.45 hybrid: under the opening 0.63, over 0.63×0.8×0.8.
    answer = "A paraphrase that only loosely tracks the source."
    ai = StubEmbedder({answer: partial(0.5)})

    assert attribute_citations(ai, answer, [chunk("a", "granite and mortar")]) == [
        {"sentence_idx": 0, "span": [0, len(answer)], "chunk_ids": ["a"]}
    ]


def test_similarity_below_the_floor_stays_uncited():
    # cosine 0.3 → 0.27 hybrid, under 0.63×0.8³ = 0.3226, and the next rung
    # (0.258) is below the 0.3 floor, so the ladder stops.
    answer = "A sentence that merely rhymes with the source."
    ai = StubEmbedder({answer: partial(0.3)})

    assert attribute_citations(ai, answer, [chunk("a", "granite and mortar")]) == []


def test_a_stricter_floor_rejects_what_the_default_accepts():
    answer = "A paraphrase that only loosely tracks the source."
    ai = StubEmbedder({answer: partial(0.5)})
    chunks = [chunk("a", "granite and mortar")]

    assert attribute_citations(ai, answer, chunks, floor=0.6) == []


def test_a_decay_that_never_descends_is_refused():
    answer = "A paraphrase that only loosely tracks the source."
    ai = StubEmbedder({answer: partial(0.5)})

    assert attribute_citations(ai, answer, [chunk("a", "granite")], decay=1.0) == []


# --------------------------------------------------------------- the two signals


def test_token_overlap_breaks_a_tie_between_equal_vectors():
    answer = "The quarry at Marren Cove supplied basalt."
    ai = StubEmbedder({answer: MATCH})
    chunks = [
        chunk("no-overlap", "granite ferries and lantern keepers"),
        chunk("overlap", "the quarry at marren cove supplied basalt to the harbour"),
    ]

    result = attribute_citations(ai, answer, chunks)

    assert result[0]["chunk_ids"][0] == "overlap"


def test_token_overlap_alone_cannot_manufacture_a_citation():
    # Capped at weight 0.1, lexical agreement can never clear the 0.3 floor by
    # itself — otherwise every sentence would cite every chunk on stopwords.
    answer = "The quarry at Marren Cove supplied basalt."
    ai = StubEmbedder({}, default=[0.0, 0.0, 1.0, 0.0])
    chunks = [chunk("overlap", "the quarry at marren cove supplied basalt", OTHER)]

    assert attribute_citations(ai, answer, chunks) == []


def test_a_chunk_without_an_embedding_can_still_be_scored():
    answer = "The quarry at Marren Cove supplied basalt."
    ai = StubEmbedder({answer: MATCH})
    chunks = [
        CitationChunk(chunk_id="vectorless", text=answer.lower(), embedding=[]),
        chunk("vectored", "granite ferries", MATCH),
    ]

    result = attribute_citations(ai, answer, chunks)

    # The vectorless chunk never wins on cosine, but it is not crashed on.
    assert result[0]["chunk_ids"] == ["vectored"]


# ------------------------------------------------------------------- limits


def test_at_most_four_chunks_are_cited_per_sentence():
    answer = "One sentence supported by the whole retrieved set."
    ai = StubEmbedder({answer: MATCH})
    chunks = [chunk("c{0}".format(index), "granite {0}".format(index)) for index in range(6)]

    result = attribute_citations(ai, answer, chunks)

    assert len(result[0]["chunk_ids"]) == 4


def test_the_source_limit_is_configurable():
    answer = "One sentence supported by the whole retrieved set."
    ai = StubEmbedder({answer: MATCH})
    chunks = [chunk("c{0}".format(index), "granite {0}".format(index)) for index in range(6)]

    result = attribute_citations(ai, answer, chunks, max_sources=2)

    assert len(result[0]["chunk_ids"]) == 2


# ---------------------------------------------------------- sentence splitting


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("One. Two. Three.", ["One.", "Two.", "Three."]),
        ("The tide rose 3.5 metres overnight.", ["The tide rose 3.5 metres overnight."]),
        ("Ask Dr. Marren about it.", ["Ask Dr. Marren about it."]),
        ("Really? Yes! Truly.", ["Really?", "Yes!", "Truly."]),
        ('He said "go north." Then he left.', ['He said "go north."', "Then he left."]),
        ("第一章讲灯塔。第二章讲渡轮。", ["第一章讲灯塔。", "第二章讲渡轮。"]),
        ("No terminator here", ["No terminator here"]),
        ("", []),
    ],
)
def test_split_answer_sentences(answer, expected):
    assert split_answer_sentences(answer) == expected


def test_short_sentences_are_not_merged_away():
    # Unlike the TTS splitter, which merges anything under 25 characters into
    # the next fragment, citations need one display unit per sentence.
    assert split_answer_sentences("Yes. The keeper was Elsa Marren.") == [
        "Yes.",
        "The keeper was Elsa Marren.",
    ]


# ------------------------------------------------------------ retrieval glue


def test_citation_chunks_read_embeddings_out_of_the_book_index():
    records = [
        ChunkRecord(
            id="c{0}".format(row), book_id="book", chapter_idx=1,
            chapter_title="Chapter 1", section_title=None, page_start=1, page_end=1,
            text="body {0}".format(row), header="body {0}".format(row),
            parent_text="body {0}".format(row), ord=row,
        )
        for row in range(2)
    ]
    index = BookIndex(
        book_id="book",
        chunks=records,
        matrix=normalize(np.eye(2, dtype=np.float32)),
        bm25=BM25Index([record.header for record in records]),
    )
    retrieval = Retrieval(
        context="",
        sources=[source(record) for record in records],
        children=records,
        scored=[
            ScoredChunk(chunk=records[1], row=1, scores=Scores(final=1.0)),
            ScoredChunk(chunk=records[0], row=0, scores=Scores(final=0.5)),
        ],
    )

    candidates = citation_chunks(index, retrieval)

    assert [candidate.chunk_id for candidate in candidates] == ["c1", "c0"]
    assert list(candidates[0].embedding) == pytest.approx([0.0, 1.0])


def test_citation_chunks_skip_blocks_the_context_budget_dropped():
    """A chunk the model never read is not a source the answer can have used.

    It is also unrenderable: ``sources`` is the whole id vocabulary the client
    was given, so a citation naming anything else is a chip it cannot draw.
    """
    records = [
        ChunkRecord(
            id="c{0}".format(row), book_id="book", chapter_idx=1,
            chapter_title="Chapter 1", section_title=None, page_start=1, page_end=1,
            text="body {0}".format(row), header="body {0}".format(row),
            parent_text="body {0}".format(row), ord=row,
        )
        for row in range(2)
    ]
    index = BookIndex(
        book_id="book",
        chunks=records,
        matrix=normalize(np.eye(2, dtype=np.float32)),
        bm25=BM25Index([record.header for record in records]),
    )
    retrieval = Retrieval(
        context="",
        # Only the first chunk's block fitted the token budget.
        sources=[source(records[0])],
        children=records,
        scored=[
            ScoredChunk(chunk=records[0], row=0, scores=Scores(final=1.0)),
            ScoredChunk(chunk=records[1], row=1, scores=Scores(final=0.5)),
        ],
    )

    assert [candidate.chunk_id for candidate in citation_chunks(index, retrieval)] == [
        "c0"
    ]


def test_citation_chunks_skip_rows_the_index_no_longer_has():
    record = ChunkRecord(
        id="c0", book_id="book", chapter_idx=1, chapter_title="Chapter 1",
        section_title=None, page_start=1, page_end=1, text="body",
        header="body", parent_text="body", ord=0,
    )
    index = BookIndex(
        book_id="book", chunks=[record],
        matrix=normalize(np.eye(1, dtype=np.float32)), bm25=BM25Index(["body"]),
    )
    retrieval = Retrieval(
        context="", sources=[source(record)], children=[record],
        scored=[ScoredChunk(chunk=record, row=7, scores=Scores(final=1.0))],
    )

    assert citation_chunks(index, retrieval) == []


def test_sentence_spans_slice_the_answer_exactly():
    """Spans must index the raw answer so the client can highlight by slicing
    — decimals and abbreviations were exactly where a client-side re-split
    used to disagree with this splitter."""
    answer = "Its lantern burned for 4.5 hours. The keeper, Dr. Halvard, trimmed the wick."
    spans = split_answer_sentence_spans(answer)
    assert [s for s, _, _ in spans] == [
        "Its lantern burned for 4.5 hours.",
        "The keeper, Dr. Halvard, trimmed the wick.",
    ]
    assert all(answer[a:b] == s for s, a, b in spans)
