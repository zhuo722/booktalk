"""The pymupdf4llm extraction layer and its block stream (SPEC §14.1)."""

from __future__ import annotations

from typing import List, Sequence

from conftest import PdfLine, lorem

from app.ingest.markdown import KIND_HEADING, KIND_TEXT, extract_blocks
from app.ingest.pdf import (
    MIN_CHARS_PER_PAGE,
    chars_per_page,
    looks_scanned,
    read_pages,
)
from app.ingest.structure import BookStructure, Chapter, Section

SENTENCE = "the ferry leaves the quay at dawn and returns"


def structure_of(titles: Sequence[str], pages: Sequence[int], title: str = "Guide"):
    """A structure whose chapters are anchored at the given pages."""
    chapters = [
        Chapter(
            idx=index + 1,
            title=chapter_title,
            page_start=page_no,
            page_end=page_no,
            anchor=(page_no, 0.0),
        )
        for index, (chapter_title, page_no) in enumerate(zip(titles, pages))
    ]
    return BookStructure(title=title, chapters=chapters, source="toc")


def blocks_of(path, structure):
    return extract_blocks(path, structure, pages=read_pages(path))


def texts(blocks, kind: str) -> List[str]:
    return [block.text for block in blocks if block.kind == kind]


def test_detected_headings_drive_the_markdown_levels(pdf_lines_factory):
    """Our structure ladder, not pymupdf4llm's font ranking, marks the headings."""
    path = pdf_lines_factory(
        [
            [
                PdfLine(text="Isle of Basalt", size=11.0),  # body-size heading
                PdfLine(text=SENTENCE),
            ],
            [
                PdfLine(text="Isle of Granite", size=11.0),
                PdfLine(text="Quarries", size=11.0),
                PdfLine(text="the quarry closed in 1911."),
            ],
        ],
        name="levels.pdf",
    )
    structure = structure_of(["Isle of Basalt", "Isle of Granite"], [1, 2])
    structure.chapters[1].sections.append(
        Section(title="Quarries", page_start=2, page_end=2, anchor=(2, 120.0))
    )

    blocks = blocks_of(path, structure)

    headings = [(block.text, block.level) for block in blocks if block.kind == KIND_HEADING]
    assert headings == [
        ("Isle of Basalt", 1),
        ("Isle of Granite", 1),
        ("Quarries", 2),  # marked at the level our structure assigned it
    ]
    assert all(block.text.strip() for block in blocks)


def test_wrapped_lines_become_one_paragraph_block(pdf_lines_factory):
    path = pdf_lines_factory(
        [
            [
                PdfLine(text="Isle of Basalt", size=11.0),
                PdfLine(text=SENTENCE),
                PdfLine(text="before the tide turns at the north cove."),
            ]
        ],
        name="wrapped.pdf",
    )

    blocks = blocks_of(path, structure_of(["Isle of Basalt"], [1]))

    assert texts(blocks, KIND_TEXT) == [
        SENTENCE + " before the tide turns at the north cove."
    ]


def test_end_of_line_hyphenation_is_undone(pdf_lines_factory):
    """``light-\\nhouse`` must index as ``lighthouse`` — a direct BM25 win."""
    path = pdf_lines_factory(
        [
            [
                PdfLine(text="Isle of Basalt", size=11.0),
                PdfLine(text="the keeper walked out to the light-"),
                PdfLine(text="house before the tide turned."),
            ]
        ],
        name="hyphen.pdf",
    )

    blocks = blocks_of(path, structure_of(["Isle of Basalt"], [1]))

    body = " ".join(texts(blocks, KIND_TEXT))
    assert body == "the keeper walked out to the lighthouse before the tide turned."


def test_a_word_hyphenated_across_a_page_turn_is_rejoined(pdf_lines_factory):
    path = pdf_lines_factory(
        [
            [
                PdfLine(text="Isle of Basalt", size=11.0),
                PdfLine(text="the keeper walked out to the light-"),
            ],
            [PdfLine(text="house before the tide turned.")],
        ],
        name="hyphen-page.pdf",
    )

    blocks = blocks_of(path, structure_of(["Isle of Basalt"], [1]))

    assert texts(blocks, KIND_TEXT) == [
        "the keeper walked out to the lighthouse before the tide turned."
    ]


def test_a_paragraph_that_spans_a_page_turn_stays_one_block(pdf_lines_factory):
    path = pdf_lines_factory(
        [
            [PdfLine(text="Isle of Basalt", size=11.0), PdfLine(text=SENTENCE)],
            [PdfLine(text="before the tide turns.")],
        ],
        name="spanning.pdf",
    )

    blocks = blocks_of(path, structure_of(["Isle of Basalt"], [1]))

    body = [block for block in blocks if block.kind == KIND_TEXT]
    assert len(body) == 1
    assert body[0].text.endswith("before the tide turns.")
    assert (body[0].page_start, body[0].page_end) == (1, 2)


def test_running_heads_and_feet_are_stripped_by_frequency(pdf_lines_factory):
    header = "A Field Guide to the Meridian Archipelago"
    pages = [
        [
            PdfLine(text=header, size=9.0),
            PdfLine(text=lorem("page{0}".format(index), 10)),
            PdfLine(text="Page {0}".format(index + 1), size=9.0),
        ]
        for index in range(6)
    ]
    path = pdf_lines_factory(pages, name="running.pdf")

    blocks = blocks_of(path, structure_of(["Full text"], [1]))

    body = " ".join(texts(blocks, KIND_TEXT))
    assert header not in body
    assert "Page 3" not in body
    assert "page3" in body  # the real content of that page survived


def test_a_heading_the_page_never_spells_out_is_still_inserted(pdf_lines_factory):
    """An outline can name a chapter whose title is drawn as an image."""
    path = pdf_lines_factory(
        [
            [PdfLine(text="the ferry leaves the quay at dawn.")],
            [PdfLine(text="the quarry closed in 1911.")],
        ],
        name="missing-heading.pdf",
    )

    blocks = blocks_of(path, structure_of(["Isle of Basalt", "Isle of Granite"], [1, 2]))

    headings = [
        (block.text, block.page_start) for block in blocks if block.kind == KIND_HEADING
    ]
    assert headings == [("Isle of Basalt", 1), ("Isle of Granite", 2)]
    assert blocks[0].kind == KIND_HEADING  # inserted before the page's first block


def test_the_raw_line_stream_takes_over_if_the_markdown_layer_fails(
    pdf_lines_factory, monkeypatch
):
    """A book that trips pymupdf4llm still ingests, just without its cleanup."""
    import pymupdf4llm

    def explode(*args, **kwargs):
        raise RuntimeError("this document confuses the markdown layer")

    monkeypatch.setattr(pymupdf4llm, "to_markdown", explode)
    path = pdf_lines_factory(
        [[PdfLine(text="Isle of Basalt", size=11.0), PdfLine(text=SENTENCE)]],
        name="fallback.pdf",
    )

    blocks = blocks_of(path, structure_of(["Isle of Basalt"], [1]))

    assert [(block.kind, block.text) for block in blocks] == [
        (KIND_HEADING, "Isle of Basalt"),
        (KIND_TEXT, SENTENCE),
    ]


# ------------------------------------------------------------------ preflight


def test_a_born_digital_page_measures_far_above_the_scan_threshold(pdf_lines_factory):
    path = pdf_lines_factory(
        [[PdfLine(text=lorem("page{0}".format(index), 20))] for index in range(3)],
        name="digital.pdf",
    )

    pages = read_pages(path)

    assert chars_per_page(pages) > MIN_CHARS_PER_PAGE * 5
    assert not looks_scanned(pages)


def test_an_image_only_pdf_is_flagged_as_scanned(pdf_lines_factory):
    path = pdf_lines_factory([[], [], []], name="scanned.pdf")

    pages = read_pages(path)

    assert chars_per_page(pages) == 0.0
    assert looks_scanned(pages)
