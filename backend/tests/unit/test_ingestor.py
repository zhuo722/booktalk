"""Ingestion orchestration end to end, offline (SPEC §14.1, §14.4, §14.5)."""

from __future__ import annotations

import pytest
from conftest import FixtureChapter, PdfLine, lorem

from app.index.store import STATUS_FAILED, STATUS_READY
from app.ingest.chunker import KIND_BOOK_SUMMARY, KIND_CHAPTER_SUMMARY, KIND_TEXT
from app.ingest.ingestor import IngestionError, ingest_book

BOOK_ID = "book-under-test"


@pytest.fixture(autouse=True)
def book_row(store):
    """The API creates the row before handing the file to the ingestor."""
    return store.create_book(BOOK_ID, "pending", "meridian.pdf", "sha256", "parsing")


def fixture_book(book_pdf_factory, count: int = 3):
    return book_pdf_factory(
        [
            FixtureChapter(
                title="Chapter {0}. Isle {0}".format(index),
                paragraphs=[lorem("c{0}p{1}".format(index, n), 90) for n in range(4)],
            )
            for index in range(1, count + 1)
        ],
        title="Meridian Fixture",
    )


def test_a_full_ingestion_indexes_chunks_and_summary_nodes(store, fake_ai, book_pdf_factory):
    path = fixture_book(book_pdf_factory)

    result = ingest_book(store, fake_ai, BOOK_ID, path, "meridian.pdf")

    book = store.get_book(BOOK_ID)
    assert book is not None and book.status == STATUS_READY and book.progress == 1.0
    assert result.chapters == 3
    assert result.summaries == 4  # one per chapter plus the book-level node
    assert result.contextualized > 0
    assert result.structure_source == "voting"

    index = store.load_index(BOOK_ID)
    kinds = [record.kind for record in index.chunks]
    assert kinds.count(KIND_CHAPTER_SUMMARY) == 3
    assert kinds.count(KIND_BOOK_SUMMARY) == 1
    assert kinds.count(KIND_TEXT) == len(index.chunks) - 4
    assert index.matrix.shape[0] == len(index.chunks)  # summaries are retrievable
    assert [record.level for record in index.chunks if record.is_summary] == [1, 1, 1, 2]

    book_node = [r for r in index.chunks if r.kind == KIND_BOOK_SUMMARY][0]
    assert book_node.chapter_title == "Meridian Fixture"  # it belongs to no chapter
    assert book_node.section_title == "Book summary"


def test_chapter_summaries_are_readable_for_the_toc(store, fake_ai, book_pdf_factory):
    ingest_book(store, fake_ai, BOOK_ID, fixture_book(book_pdf_factory), "meridian.pdf")

    summaries = store.get_summaries(BOOK_ID)

    assert [record.chapter_idx for record in summaries] == [1, 2, 3]
    assert all(record.text.strip() for record in summaries)
    assert all(record.section_title == "Chapter summary" for record in summaries)


def test_contextual_enrichment_is_switchable(store, fake_ai, book_pdf_factory):
    path = fixture_book(book_pdf_factory, count=1)

    ingest_book(store, fake_ai, BOOK_ID, path, "meridian.pdf", contextual=False)
    without = [record.header for record in store.load_index(BOOK_ID).chunks]

    ingest_book(store, fake_ai, BOOK_ID, path, "meridian.pdf", contextual=True)
    with_context = [record.header for record in store.load_index(BOOK_ID).chunks]

    # Off: the static breadcrumb (or, for a section's first chunk, the text).
    assert any(header.startswith("«Meridian Fixture» — ") for header in without)
    assert not any(header.startswith("This excerpt sits") for header in without)
    # On: an LLM-written situating sentence in front of every chunk.
    assert any(header.startswith("This excerpt sits") for header in with_context)


def test_a_scanned_pdf_fails_with_an_actionable_message(store, fake_ai, pdf_lines_factory):
    path = pdf_lines_factory([[], [], []], name="scan.pdf")

    with pytest.raises(IngestionError):
        ingest_book(store, fake_ai, BOOK_ID, path, "scan.pdf")

    book = store.get_book(BOOK_ID)
    assert book is not None and book.status == STATUS_FAILED
    assert "no text layer" in (book.error or "")
    assert "OCR" in (book.error or "")
    assert store.count_chunks(BOOK_ID) == 0


def test_a_book_with_no_api_key_fails_before_any_parsing(store):
    with pytest.raises(IngestionError):
        ingest_book(store, None, BOOK_ID, "unused.pdf", "unused.pdf")

    book = store.get_book(BOOK_ID)
    assert book is not None and book.status == STATUS_FAILED
    assert "OPENAI_API_KEY" in (book.error or "")


def test_the_naive_arm_skips_enrichment_entirely(store, fake_ai, book_pdf_factory):
    path = fixture_book(book_pdf_factory, count=2)

    result = ingest_book(store, fake_ai, BOOK_ID, path, "meridian.pdf", mode="naive")

    assert result.summaries == 0
    assert result.contextualized == 0
    index = store.load_index(BOOK_ID)
    assert all(record.kind == KIND_TEXT for record in index.chunks)
    assert all(record.header == record.text for record in index.chunks)


def test_the_title_and_page_count_are_written_back(store, fake_ai, book_pdf_factory):
    ingest_book(store, fake_ai, BOOK_ID, fixture_book(book_pdf_factory), "meridian.pdf")

    book = store.get_book(BOOK_ID)

    assert book is not None
    assert book.title == "Meridian Fixture"  # from the PDF metadata, not the filename
    assert book.pages and book.pages >= 3
    assert book.error is None
