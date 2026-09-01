"""The structure detection ladder over synthetic PDFs (SPEC §4.2, §14.2)."""

from __future__ import annotations

from conftest import FixtureChapter, PdfLine, lorem

from app.ingest.pdf import TocEntry, read_pages
from app.ingest.structure import (
    FALLBACK_CHAPTER_TITLE,
    SOURCE_CONTENTS,
    SOURCE_FALLBACK,
    SOURCE_HEADINGS,
    SOURCE_TOC,
    SOURCE_VOTING,
    detect_structure,
    leaf_units,
)


def _chapters(path, use_toc=True, ai=None):
    """Parse a PDF and detect its structure the way the ingestor does."""
    from app.ingest.pdf import PdfDocument

    with PdfDocument(path) as doc:
        pages = list(doc.iter_pages())
        toc = doc.toc() if use_toc else []
        page_count = doc.page_count
    return detect_structure(
        pages, toc, title="Fixture Book", page_count=page_count, ai=ai
    )


def _book(prefix="Chapter", heading_size=20.0, count=3, sections=False):
    return [
        FixtureChapter(
            title="{0} {1}. Island {1}".format(prefix, index),
            paragraphs=[lorem("c{0}p{1}".format(index, n), 80) for n in range(4)],
            heading_size=heading_size,
            sections=(
                [("Tides and ferries {0}".format(index), [lorem("s{0}".format(index), 70)])]
                if sections
                else []
            ),
        )
        for index in range(1, count + 1)
    ]


def test_embedded_toc_takes_priority(book_pdf_factory):
    structure = _chapters(book_pdf_factory(_book(sections=True), toc=True))

    assert structure.source == "toc"
    assert [chapter.idx for chapter in structure.chapters] == [1, 2, 3]
    assert all(len(chapter.sections) == 1 for chapter in structure.chapters)
    for chapter in structure.chapters:
        assert chapter.page_start <= chapter.page_end
        assert chapter.sections[0].page_start >= chapter.page_start


def test_toc_page_ranges_do_not_overlap(book_pdf_factory):
    structure = _chapters(book_pdf_factory(_book(count=4), toc=True))

    starts = [chapter.page_start for chapter in structure.chapters]
    ends = [chapter.page_end for chapter in structure.chapters]
    assert starts == sorted(starts)
    for position in range(len(structure.chapters) - 1):
        assert ends[position] < starts[position + 1]


def test_too_short_toc_falls_through_to_heuristics(book_pdf_factory):
    path = book_pdf_factory(_book(count=2), toc=True, name="short-toc.pdf")

    structure = _chapters(path)

    assert structure.source == "headings"
    assert len(structure.chapters) == 2


def test_heading_family_voting_finds_chapters_at_body_size(book_pdf_factory):
    """A book with no TOC at all: the "Chapter N" family wins the vote (§14.2)."""
    path = book_pdf_factory(_book(heading_size=11.0), name="regex.pdf")

    structure = _chapters(path, use_toc=False)

    assert structure.source == SOURCE_VOTING
    assert len(structure.chapters) == 3
    assert structure.chapters[0].title.startswith("Chapter 1")


def test_voting_keeps_typographic_subheadings_as_sections(book_pdf_factory):
    """The elected family sets the chapter level; larger type below it is sections."""
    path = book_pdf_factory(
        _book(heading_size=11.0, sections=True), name="voted-sections.pdf"
    )

    structure = _chapters(path, use_toc=False)

    assert structure.source == SOURCE_VOTING
    assert len(structure.chapters) == 3
    assert all(len(chapter.sections) == 1 for chapter in structure.chapters)
    assert structure.chapters[0].sections[0].title.startswith("Tides and ferries")


def test_printed_contents_page_is_parsed_when_there_is_no_outline(pdf_lines_factory):
    """Rung 2: the book's own contents page, validated by monotonic pages."""
    contents = [
        PdfLine(text="Contents", size=16.0, bold=True),
        PdfLine(text="Isle of Basalt . . . . . . . . 2"),
        PdfLine(text="Isle of Granite . . . . . . . . 3"),
        PdfLine(text="Isle of Salt . . . . . . . . 4"),
    ]
    pages = [contents] + [
        [
            PdfLine(text=title, size=20.0, bold=True),
            PdfLine(text=lorem("body{0}".format(index), 40)),
        ]
        for index, title in enumerate(("Isle of Basalt", "Isle of Granite", "Isle of Salt"))
    ]
    path = pdf_lines_factory(pages, name="contents.pdf")

    structure = _chapters(path, use_toc=False)

    assert structure.source == SOURCE_CONTENTS
    assert [chapter.title for chapter in structure.chapters] == [
        "Isle of Basalt",
        "Isle of Granite",
        "Isle of Salt",
    ]
    assert [chapter.page_start for chapter in structure.chapters] == [2, 3, 4]


def test_a_flat_toc_gets_its_levels_from_one_llm_call(book_pdf_factory, fake_ai):
    """Rung 1 with no usable levels: RAGFlow's assign_toc_levels (§14.2)."""
    path = book_pdf_factory(_book(count=1), name="flat-toc.pdf")
    flat = [
        TocEntry(level=1, title="PART ONE", page=1),
        TocEntry(level=1, title="Chapter 1. Isle 1", page=1),
        TocEntry(level=1, title="Chapter 2. Isle 2", page=1),
    ]
    pages = read_pages(path)

    without_llm = detect_structure(pages, flat, title="Fixture Book", page_count=len(pages))
    with_llm = detect_structure(
        pages, flat, title="Fixture Book", page_count=len(pages), ai=fake_ai
    )

    assert without_llm.source == SOURCE_TOC
    assert [chapter.title for chapter in without_llm.chapters] == [
        "PART ONE",
        "Chapter 1. Isle 1",
        "Chapter 2. Isle 2",
    ]
    assert with_llm.source == SOURCE_TOC
    assert [chapter.title for chapter in with_llm.chapters] == ["PART ONE"]
    assert [section.title for section in with_llm.chapters[0].sections] == [
        "Chapter 1. Isle 1",
        "Chapter 2. Isle 2",
    ]


def test_font_size_clusters_split_chapters_from_sections(book_pdf_factory):
    """With no "Chapter" keyword, the larger font becomes the chapter level."""
    path = book_pdf_factory(
        _book(prefix="Isle of", sections=True), name="sized.pdf"
    )

    structure = _chapters(path, use_toc=False)

    assert structure.source == "headings"
    assert len(structure.chapters) == 3
    assert all(len(chapter.sections) == 1 for chapter in structure.chapters)
    assert structure.chapters[0].sections[0].title.startswith("Tides and ferries")


def test_no_headings_falls_back_to_one_chapter(pdf_lines_factory):
    pages = [[PdfLine(text=lorem("plain{0}".format(index), 40))] for index in range(3)]
    path = pdf_lines_factory(pages, name="plain.pdf")

    structure = _chapters(path)

    assert structure.source == "fallback"
    assert len(structure.chapters) == 1
    assert structure.chapters[0].title == FALLBACK_CHAPTER_TITLE
    assert structure.chapters[0].page_start == 1
    assert structure.chapters[0].page_end == 3


def test_running_headers_are_not_mistaken_for_chapters(book_pdf_factory):
    path = book_pdf_factory(
        _book(count=4),
        name="running.pdf",
        running_header="A Field Guide to the Meridian Archipelago",
        running_footer=True,
    )

    structure = _chapters(path, use_toc=False)

    titles = [chapter.title for chapter in structure.chapters]
    assert len(titles) == 4
    assert "A Field Guide to the Meridian Archipelago" not in titles
    assert not any(title.startswith("Page ") for title in titles)


def test_cjk_chapter_headings(book_pdf_factory):
    chapters = [
        FixtureChapter(
            title="第{0}章 灯塔与港口".format(numeral),
            paragraphs=["这是关于岛屿的段落，记录了灯塔、渡轮与市集的日常。" * 4],
            heading_size=11.0,
        )
        for numeral in ("一", "二", "三")
    ]
    path = book_pdf_factory(
        chapters, name="cjk.pdf", font="china-s", bold_font="china-s"
    )

    structure = _chapters(path, use_toc=False)

    assert len(structure.chapters) == 3
    assert structure.chapters[0].title.startswith("第一章")


def test_leaf_units_are_ordered_and_cover_every_section(book_pdf_factory):
    structure = _chapters(book_pdf_factory(_book(sections=True), toc=True))

    units = leaf_units(structure)

    assert [unit.anchor for unit in units] == sorted(unit.anchor for unit in units)
    assert len(units) == 6  # three chapters, each with one section
    assert [unit.section_title for unit in units][:2] == [
        None,
        "Tides and ferries 1",
    ]


def test_book_title_falls_back_when_metadata_is_empty(book_pdf_factory):
    pages = read_pages(book_pdf_factory(_book(count=1), name="untitled.pdf"))

    structure = detect_structure(pages, [], title="", page_count=len(pages))

    assert structure.title == "Untitled"
