"""Chunking invariants for the three-pass hybrid chunker (SPEC §14.3)."""

from __future__ import annotations

from typing import List, Optional, Sequence

import pytest
from conftest import FixtureChapter, lorem

from app.ingest.chunker import (
    CHILD_MAX_TOKENS,
    CHILD_MIN_TOKENS,
    CONTENT_FLOOR_TOKENS,
    HEADER_MAX_FRACTION,
    PARENT_MAX_TOKENS,
    breadcrumb,
    build_header,
    chunk_book,
    count_tokens,
    header_budget,
    naive_chunks,
    split_text,
)
from app.ingest.markdown import KIND_HEADING, KIND_TEXT, Block, extract_blocks
from app.ingest.pdf import PdfDocument
from app.ingest.structure import BookStructure, Chapter, Section, detect_structure

BOOK_ID = "bookid"
BOOK_TITLE = "Meridian Fixture"


# ------------------------------------------------------------ synthetic input


def heading(text: str, level: int = 1, page: int = 1) -> Block:
    return Block(
        kind=KIND_HEADING, level=level, text=text, page_start=page, page_end=page
    )


def body(text: str, page: int = 1) -> Block:
    return Block(kind=KIND_TEXT, level=0, text=text, page_start=page, page_end=page)


def structure_of(
    chapters: Sequence[str],
    sections: Optional[Sequence[Sequence[str]]] = None,
    pages_each: int = 4,
    title: str = BOOK_TITLE,
) -> BookStructure:
    """A hand-built structure, so block tests do not need a PDF."""
    built: List[Chapter] = []
    for index, chapter_title in enumerate(chapters):
        start = 1 + index * pages_each
        end = start + pages_each - 1
        chapter = Chapter(
            idx=index + 1,
            title=chapter_title,
            page_start=start,
            page_end=end,
            anchor=(start, 0.0),
        )
        for section_title in (sections or [[]] * len(chapters))[index]:
            chapter.sections.append(
                Section(
                    title=section_title,
                    page_start=start,
                    page_end=end,
                    anchor=(start, 10.0),
                )
            )
        built.append(chapter)
    return BookStructure(title=title, chapters=built, source="toc")


def words(seed: str, count: int) -> str:
    return lorem(seed, count)


# ---------------------------------------------------------------- PDF fixture


def parse_book(path, use_toc: bool = True):
    """Parse a fixture PDF exactly the way the ingestor does."""
    with PdfDocument(path) as doc:
        pages = list(doc.iter_pages())
        toc = doc.toc() if use_toc else []
        structure = detect_structure(pages, toc, title=BOOK_TITLE, page_count=doc.page_count)
    blocks = extract_blocks(path, structure, pages=pages)
    return pages, structure, blocks


def fixture_chapters(count: int = 3, paragraphs: int = 6, sections: bool = False):
    return [
        FixtureChapter(
            title="Chapter {0}. Isle {0}".format(index),
            paragraphs=[words("c{0}p{1}".format(index, n), 90) for n in range(paragraphs)],
            sections=(
                [("Harbour notes {0}".format(index), [words("s{0}".format(index), 120)])]
                if sections
                else []
            ),
        )
        for index in range(1, count + 1)
    ]


@pytest.fixture
def chunked(book_pdf_factory):
    """A parsed fixture book plus its chunks."""
    path = book_pdf_factory(fixture_chapters(sections=True), title=BOOK_TITLE)
    _, structure, blocks = parse_book(path)
    return structure, chunk_book(blocks, structure, BOOK_ID, book_title=structure.title)


# --------------------------------------------------------------- basic shape


def test_chunks_are_non_empty_and_fit_the_contextualized_budget(chunked):
    _, chunks = chunked

    assert chunks
    for chunk in chunks:
        assert chunk.text.strip()
        # §14.3 budgets by the *contextualized* count: header + body together.
        assert count_tokens(chunk.header) <= CHILD_MAX_TOKENS


def test_chunk_sizes_land_in_the_target_band(chunked):
    _, chunks = chunked

    # Trailing chunks of a section may be short; the bulk must sit in the band.
    sized = [count_tokens(chunk.text) for chunk in chunks]
    assert max(sized) <= CHILD_MAX_TOKENS
    assert sum(1 for size in sized if size >= CHILD_MIN_TOKENS) >= len(sized) // 2


def test_chunk_identifiers_and_order_are_stable(chunked):
    _, chunks = chunked

    assert [chunk.ord for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert chunks[0].id.startswith(BOOK_ID)


def test_metadata_matches_the_detected_structure(chunked):
    structure, chunks = chunked
    by_idx = {chapter.idx: chapter for chapter in structure.chapters}

    for chunk in chunks:
        chapter = by_idx[chunk.chapter_idx]
        assert chunk.chapter_title == chapter.title
        assert chunk.page_start <= chunk.page_end
        assert chunk.page_start >= chapter.page_start
        assert chunk.page_end <= chapter.page_end
        if chunk.section_title is not None:
            assert chunk.section_title in {
                section.title for section in chapter.sections
            }


# ------------------------------------------------------------- breadcrumbs


def test_breadcrumb_carries_the_full_heading_path():
    assert breadcrumb("Guide", ["Chapter 2", "Harbours"]) == (
        "«Guide» — Chapter 2 — Harbours"
    )
    assert build_header("Guide", ["Chapter 2"], "body") == "«Guide» — Chapter 2\n\nbody"
    assert build_header("Guide", [], "body") == "«Guide»\n\nbody"


def test_header_is_omitted_on_a_sections_first_chunk(chunked):
    _, chunks = chunked

    firsts = [chunk for chunk in chunks if chunk.header == chunk.text]
    assert firsts, "every section's first chunk indexes its own heading line"
    for chunk in firsts:
        opening = chunk.text.split("\n\n")[0]
        assert opening in (chunk.chapter_title, chunk.section_title)
    for chunk in chunks:
        if chunk.header != chunk.text:
            assert chunk.header.startswith("«{0}» — ".format(BOOK_TITLE))
            assert chunk.header.endswith(chunk.text)
            assert chunk.chapter_title in chunk.header.split("\n\n")[0]
            if chunk.section_title:
                assert chunk.section_title in chunk.header.split("\n\n")[0]


def test_header_budget_caps_the_header_and_trims_from_the_head():
    verbose = "Part One " + " ".join(["of the Meridian Survey"] * 30)
    path = [verbose, "Chapter 2. Isle 2", "Harbour notes"]

    kept, available = header_budget(BOOK_TITLE, path, max_tokens=CHILD_MAX_TOKENS)

    assert list(kept) == path[1:]  # the outermost level is dropped first
    assert kept[-1] == path[-1]  # the most specific level survives
    ceiling = int(CHILD_MAX_TOKENS * HEADER_MAX_FRACTION)
    assert count_tokens(breadcrumb(BOOK_TITLE, kept)) <= ceiling
    assert available == CHILD_MAX_TOKENS - count_tokens(breadcrumb(BOOK_TITLE, kept))
    assert available >= CONTENT_FLOOR_TOKENS


def test_header_is_dropped_rather_than_starving_the_body():
    long_title = " ".join(["lighthouse"] * 60)

    kept, available = header_budget(long_title, [long_title], max_tokens=CONTENT_FLOOR_TOKENS)

    assert kept == ()
    assert available == CONTENT_FLOOR_TOKENS


# ----------------------------------------------------- the three-pass rules


def test_chunks_have_zero_overlap_and_reconstruct_the_source():
    blocks = [heading("Chapter 1. Isle 1")] + [
        body(words("p{0}".format(index), 90)) for index in range(8)
    ]
    structure = structure_of(["Chapter 1. Isle 1"])

    chunks = chunk_book(blocks, structure, BOOK_ID, book_title=BOOK_TITLE)

    assert len(chunks) >= 2
    rebuilt = "\n\n".join(chunk.text for chunk in chunks)
    expected = "\n\n".join([blocks[0].text] + [block.text for block in blocks[1:]])
    assert rebuilt == expected  # nothing duplicated, nothing dropped
    for previous, following in zip(chunks, chunks[1:]):
        assert previous.text.split("\n\n")[-1] != following.text.split("\n\n")[0]


def test_undersized_chunks_merge_only_within_one_heading_path():
    blocks = [
        heading("Chapter 1. Isle 1", 1, 1),
        body(words("a", 20), 1),
        heading("Ferries", 2, 1),
        body(words("b", 20), 1),
        heading("Harbours", 2, 2),
        body(words("c", 20), 2),
    ]
    structure = structure_of(["Chapter 1. Isle 1"], [["Ferries", "Harbours"]])

    chunks = chunk_book(blocks, structure, BOOK_ID, book_title=BOOK_TITLE)

    # Three tiny paragraphs, three different heading paths: no merging at all.
    assert len(chunks) == 3
    assert [chunk.heading_path for chunk in chunks] == [
        ("Chapter 1. Isle 1",),
        ("Chapter 1. Isle 1", "Ferries"),
        ("Chapter 1. Isle 1", "Harbours"),
    ]
    assert [chunk.section_title for chunk in chunks] == [None, "Ferries", "Harbours"]


def test_two_chapters_with_the_same_title_are_not_merged():
    """Identical heading paths still belong to different sections of the book."""
    blocks = [
        heading("Notes", 1, 1),
        body(words("a", 20), 1),
        heading("Notes", 1, 2),
        body(words("b", 20), 2),
    ]
    structure = structure_of(["Notes", "Notes"], pages_each=1)

    chunks = chunk_book(blocks, structure, BOOK_ID, book_title=BOOK_TITLE)

    assert len(chunks) == 2
    assert [chunk.chapter_idx for chunk in chunks] == [1, 2]
    assert [chunk.page_start for chunk in chunks] == [1, 2]


def test_undersized_neighbours_in_the_same_path_are_merged():
    blocks = [heading("Chapter 1. Isle 1")] + [
        body(words("p{0}".format(index), 30)) for index in range(4)
    ]
    structure = structure_of(["Chapter 1. Isle 1"])

    chunks = chunk_book(blocks, structure, BOOK_ID, book_title=BOOK_TITLE)

    assert len(chunks) == 1
    assert count_tokens(chunks[0].text) >= CHILD_MIN_TOKENS // 2
    assert all(block.text in chunks[0].text for block in blocks)


def test_an_oversized_paragraph_is_leaf_split_below_the_budget():
    giant = " ".join(words("giant{0}".format(index), 120) for index in range(12))
    blocks = [heading("Chapter 1. Deluge"), body(giant)]
    structure = structure_of(["Chapter 1. Deluge"])

    chunks = chunk_book(blocks, structure, BOOK_ID, book_title=BOOK_TITLE)

    assert len(chunks) >= 4
    assert all(count_tokens(chunk.header) <= CHILD_MAX_TOKENS for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)


def test_a_bare_heading_never_becomes_its_own_chunk():
    blocks = [
        heading("Chapter 1. Isle 1", 1, 1),
        heading("Ferries", 2, 1),  # a heading immediately under another heading
        body(words("a", 40), 1),
    ]
    structure = structure_of(["Chapter 1. Isle 1"], [["Ferries"]])

    chunks = chunk_book(blocks, structure, BOOK_ID, book_title=BOOK_TITLE)

    assert len(chunks) == 1
    assert chunks[0].text.startswith("Chapter 1. Isle 1\n\nFerries\n\n")
    assert not any(chunk.text.strip() in {"Ferries", "Chapter 1. Isle 1"} for chunk in chunks)


def test_split_text_respects_the_budget_without_overlap():
    text = "\n\n".join(words("s{0}".format(index), 80) for index in range(6))

    pieces = split_text(text, 120)

    assert len(pieces) >= 4
    assert all(count_tokens(piece) <= 120 for piece in pieces)
    # Every word appears exactly once: the leaf splitter cuts, it never repeats.
    assert " ".join(pieces).split() == text.split()


# ------------------------------------------------------------------ parents


def test_parent_context_contains_the_child_and_respects_the_cap(chunked):
    _, chunks = chunked

    for chunk in chunks:
        assert count_tokens(chunk.parent_text) <= PARENT_MAX_TOKENS
        first_paragraph = chunk.text.split("\n\n")[0]
        assert first_paragraph in chunk.parent_text


# ----------------------------------------------------------- extraction ties


def test_running_headers_and_footers_are_dropped(book_pdf_factory):
    header = "A Field Guide to the Meridian Archipelago"
    path = book_pdf_factory(
        fixture_chapters(count=3),
        title=BOOK_TITLE,
        running_header=header,
        running_footer=True,
    )
    _, structure, blocks = parse_book(path)

    chunks = chunk_book(blocks, structure, BOOK_ID, book_title=structure.title)

    body_text = "\n".join(chunk.text for chunk in chunks)
    assert header not in body_text
    assert "Page 1" not in body_text
    assert "Page 2" not in body_text


def test_a_chinese_book_chunks_without_inserted_spaces(book_pdf_factory):
    chapters = [
        FixtureChapter(
            title="第{0}章 灯塔与港口".format(numeral),
            paragraphs=["这是关于岛屿的段落，记录了灯塔、渡轮与市集的日常。" * 6],
            heading_size=11.0,
        )
        for numeral in ("一", "二", "三")
    ]
    path = book_pdf_factory(
        chapters, name="cjk.pdf", title="群岛指南", font="china-s", bold_font="china-s"
    )
    _, structure, blocks = parse_book(path, use_toc=False)

    chunks = chunk_book(blocks, structure, BOOK_ID, book_title=structure.title)

    assert len(chunks) == 3
    assert [chunk.chapter_idx for chunk in chunks] == [1, 2, 3]
    for chunk in chunks:
        assert chunk.text.startswith("第")
        assert " " not in chunk.text.split("\n\n")[1]  # no latin spacing in CJK prose
        assert count_tokens(chunk.header) <= CHILD_MAX_TOKENS


def test_wrapped_lines_are_rejoined_into_paragraphs(chunked):
    _, chunks = chunked

    for chunk in chunks:
        for paragraph in chunk.text.split("\n\n"):
            assert "\n" not in paragraph
    # The prose lorem() generates ends every paragraph with a full stop.
    assert any(chunk.text.rstrip().endswith(".") for chunk in chunks)


# --------------------------------------------------------------- baselines


def test_naive_baseline_is_fixed_size_and_headerless(book_pdf_factory):
    path = book_pdf_factory(fixture_chapters(count=2), title=BOOK_TITLE)
    pages, structure, _ = parse_book(path)

    chunks = naive_chunks(pages, structure, BOOK_ID, chars=800)

    assert len(chunks) >= 2
    assert all(chunk.header == chunk.text for chunk in chunks)
    assert all(chunk.parent_text == chunk.text for chunk in chunks)
    assert all(len(chunk.text) <= 800 for chunk in chunks)
    assert {chunk.chapter_idx for chunk in chunks} == {
        chapter.idx for chapter in structure.chapters
    }


def test_count_tokens_is_positive_and_monotonic():
    assert count_tokens("") == 0
    short = count_tokens("the harbour")
    long = count_tokens("the harbour " * 50)
    assert 0 < short < long
