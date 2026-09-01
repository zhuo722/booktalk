"""Contextual enrichment and summary nodes (SPEC §14.4, §14.5)."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from app.ingest.chunker import (
    KIND_BOOK_SUMMARY,
    KIND_CHAPTER_SUMMARY,
    KIND_TEXT,
    LEVEL_BOOK_SUMMARY,
    LEVEL_CHAPTER_SUMMARY,
    Chunk,
    build_header,
)
from app.ingest.enrich import chapter_texts, contextualize, summary_chunks
from app.ingest.structure import BookStructure, Chapter

BOOK = "A Field Guide to the Meridian Archipelago"


def chunk(ordinal: int, chapter_idx: int, chapter_title: str, text: str) -> Chunk:
    path = [chapter_title]
    return Chunk(
        id="book-{0:05d}".format(ordinal),
        ord=ordinal,
        chapter_idx=chapter_idx,
        chapter_title=chapter_title,
        section_title=None,
        page_start=chapter_idx,
        page_end=chapter_idx,
        text=text,
        header=build_header(BOOK, path, text),
        parent_text=text,
        char_span=(0, len(text)),
        heading_path=tuple(path),
    )


def sample_chunks() -> List[Chunk]:
    return [
        chunk(0, 1, "Chapter 1. Isle of Basalt", "the ferry leaves the quay at dawn"),
        chunk(1, 1, "Chapter 1. Isle of Basalt", "the lighthouse keeper trims the wick"),
        chunk(2, 2, "Chapter 2. Isle of Granite", "the quarry closed in 1911"),
    ]


def sample_structure() -> BookStructure:
    return BookStructure(
        title=BOOK,
        chapters=[
            Chapter(idx=1, title="Chapter 1. Isle of Basalt", page_start=1, page_end=1, anchor=(1, 0.0)),
            Chapter(idx=2, title="Chapter 2. Isle of Granite", page_start=2, page_end=2, anchor=(2, 0.0)),
        ],
        source="toc",
    )


def prompts(client) -> List[str]:
    return [
        message["content"]
        for call in client.chat_calls
        for message in call
        if message["role"] == "user"
    ]


# --------------------------------------------------------------- §14.4


def test_contextual_enrichment_rewrites_what_gets_indexed(fake_ai):
    chunks = sample_chunks()

    enriched = contextualize(fake_ai, chunks, book_title=BOOK, workers=1)

    assert enriched == len(chunks)
    for item in chunks:
        assert item.context
        assert item.header == "{0}\n\n{1}".format(item.context, item.text)
        assert item.header.endswith(item.text)  # the chunk itself is still indexed
    assert len(fake_ai.chat_calls) == len(chunks)


def test_the_context_prompt_is_scoped_to_the_chunks_own_chapter(fake_ai):
    chunks = sample_chunks()

    contextualize(fake_ai, chunks, book_title=BOOK, workers=1)

    first = prompts(fake_ai)[0]
    assert "the lighthouse keeper trims the wick" in first  # its own chapter
    assert "the quarry closed in 1911" not in first  # never the whole book


def test_concurrent_enrichment_keeps_every_chunk_matched_to_its_own_context(fake_ai):
    chunks = sample_chunks()

    contextualize(fake_ai, chunks, book_title=BOOK, workers=4)

    for item in chunks:
        opening = " ".join(item.text.split()[:12])
        assert opening in item.context


def test_concurrent_progress_is_reported_in_counter_order():
    """The library card's progress bar must never run backwards.

    Counting under a lock is not enough on its own: two workers can take 6 and
    7 and then race into the callback in either order, so the poller sees
    7-of-8 followed by 6-of-8. The callback here is slow enough (a millisecond,
    about what the real one's SQLite write costs) to make that reordering
    certain if the report is ever let out of the lock.
    """
    from concurrent.futures import ThreadPoolExecutor

    from app.ingest.enrich import _Counter

    ticks = 40
    reported: List[int] = []

    def progress(done: int, total: int) -> None:
        time.sleep(0.001)
        reported.append(done)

    counter = _Counter(ticks, progress)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: counter.tick(), range(ticks)))

    assert reported == list(range(1, ticks + 1))


class _FlakyClient:
    """Fails for one chunk, succeeds for the rest."""

    def __init__(self, failing: str) -> None:
        self.failing = failing
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, **kw):
        self.calls.append(kw)
        content = messages[-1]["content"]
        if self.failing in content.split("<excerpt>")[-1]:
            raise RuntimeError("upstream is having a moment")
        return "a situating sentence"


def test_a_failed_call_leaves_the_breadcrumb_header_in_place():
    chunks = sample_chunks()
    breadcrumbs = [item.header for item in chunks]

    enriched = contextualize(
        _FlakyClient("the quarry closed in 1911"), chunks, book_title=BOOK, workers=1
    )

    assert enriched == 2
    assert chunks[2].header == breadcrumbs[2]
    assert chunks[2].context is None
    assert chunks[0].header.startswith("a situating sentence")


def test_chapter_texts_are_capped():
    chunks = [chunk(index, 1, "Chapter 1", "harbour " * 200) for index in range(5)]

    texts = chapter_texts(chunks, max_tokens=100)

    from app.ingest.chunker import count_tokens

    assert count_tokens(texts[1]) <= 101  # the ellipsis costs a token


# --------------------------------------------------------------- §14.5


def test_one_summary_node_per_chapter_plus_one_for_the_book(fake_ai):
    chunks = sample_chunks()

    nodes = summary_chunks(
        fake_ai, chunks, sample_structure(), "book", book_title=BOOK, workers=1
    )

    assert [node.kind for node in nodes] == [
        KIND_CHAPTER_SUMMARY,
        KIND_CHAPTER_SUMMARY,
        KIND_BOOK_SUMMARY,
    ]
    assert [node.level for node in nodes] == [
        LEVEL_CHAPTER_SUMMARY,
        LEVEL_CHAPTER_SUMMARY,
        LEVEL_BOOK_SUMMARY,
    ]
    assert [node.chapter_idx for node in nodes] == [1, 2, 0]
    assert len({node.id for node in nodes}) == 3
    assert [node.ord for node in nodes] == [3, 4, 5]  # continues the leaf ordinals
    assert all(node.text.strip() for node in nodes)
    assert all(node.header.endswith(node.text) for node in nodes)
    assert all(node.kind != KIND_TEXT for node in nodes)


def test_a_chapter_summary_sees_only_its_own_chapter(fake_ai):
    summary_chunks(
        fake_ai, sample_chunks(), sample_structure(), "book", book_title=BOOK, workers=1
    )

    first = prompts(fake_ai)[0]
    assert "the ferry leaves the quay at dawn" in first
    assert "the quarry closed in 1911" not in first


def test_the_book_summary_is_built_from_the_chapter_summaries(fake_ai):
    nodes = summary_chunks(
        fake_ai, sample_chunks(), sample_structure(), "book", book_title=BOOK, workers=1
    )

    book_prompt = prompts(fake_ai)[-1]
    assert nodes[0].text[:40] in book_prompt
    assert nodes[1].text[:40] in book_prompt
    assert nodes[-1].page_start == 1 and nodes[-1].page_end == 2


def test_summaries_are_skipped_without_a_client():
    assert summary_chunks(None, sample_chunks(), sample_structure(), "book") == []
    assert contextualize(None, sample_chunks()) == 0
