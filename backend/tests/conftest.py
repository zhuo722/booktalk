"""Shared fixtures: isolated settings, the fake AI client, synthetic PDFs.

Nothing here touches the network or the developer's real data directory, so the
whole suite runs offline with no ``OPENAI_API_KEY`` set.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import fitz
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.index.store import Store
from app.main import create_app
from app.services.openai_client import FakeAIClient

PAGE_WIDTH = 595.0
PAGE_HEIGHT = 842.0
MARGIN_LEFT = 72.0
MARGIN_TOP = 96.0
MARGIN_BOTTOM = 770.0
LINE_SPACING = 1.6
PARAGRAPH_SPACING = 2.7
HEADER_Y = 48.0
FOOTER_Y = 806.0

BOOKTALK_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "BOOKTALK_LLM_MODEL",
    "BOOKTALK_ASR_MODEL",
    "BOOKTALK_TTS_MODEL",
    "BOOKTALK_TTS_VOICE",
    "BOOKTALK_EMBED_MODEL",
    "BOOKTALK_MAX_UPLOAD_MB",
    "BOOKTALK_DATA_DIR",
    "BOOKTALK_CONTEXTUAL",
    "BOOKTALK_RERANK",
    "BOOKTALK_RRF_K",
    "BOOKTALK_DENSE_WEIGHT",
    "BOOKTALK_BM25_WEIGHT",
    "BOOKTALK_FLASHRANK",
    "BOOKTALK_FLASHRANK_MODEL",
    "BOOKTALK_CORS_ORIGINS",
    "BOOKTALK_STATIC_DIR",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test as if the machine had no BookTalk configuration."""
    for name in BOOKTALK_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    assert "OPENAI_API_KEY" not in os.environ


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointing at a throwaway data directory (and no API key).

    ``flashrank=False`` is explicit rather than incidental: the app factory only
    pre-warms the cross-encoder when an API key is present, but the suite must
    be offline by construction, not by coincidence.
    """
    return Settings(data_dir=tmp_path / "data", max_upload_mb=2, flashrank=False)


@pytest.fixture
def fake_ai() -> FakeAIClient:
    """The deterministic offline stand-in for OpenAI."""
    return FakeAIClient()


@pytest.fixture
def store(settings: Settings) -> Store:
    """A store on a fresh SQLite file with the schema applied."""
    from app.db import init_db

    settings.ensure_dirs()
    init_db(settings.db_path)
    return Store(settings.db_path)


@pytest.fixture
def client(settings: Settings, fake_ai: FakeAIClient) -> Iterator[TestClient]:
    """TestClient wired to the fake AI client."""
    with TestClient(create_app(settings=settings, ai=fake_ai)) as test_client:
        yield test_client


@pytest.fixture
def client_without_key(settings: Settings) -> Iterator[TestClient]:
    """TestClient for an app with no AI client at all (no API key)."""
    with TestClient(create_app(settings=settings)) as test_client:
        yield test_client


# --------------------------------------------------------------- PDF builders


@dataclass
class PdfLine:
    """One line to draw, with the typography the parser reads back."""

    text: str
    size: float = 11.0
    bold: bool = False
    indent: float = 0.0
    spacing: float = LINE_SPACING


@dataclass
class FixtureChapter:
    """A chapter of a synthetic book: a heading plus paragraphs of prose."""

    title: str
    paragraphs: List[str]
    heading_size: float = 20.0
    sections: List[Tuple[str, List[str]]] = field(default_factory=list)
    # 15pt clears the detector's 1.3x-body threshold for 11pt body text.
    section_heading_size: float = 15.0


def _wrap(text: str, words_per_line: int = 12) -> List[str]:
    words = text.split()
    return [
        " ".join(words[start : start + words_per_line])
        for start in range(0, len(words), words_per_line)
    ] or [""]


def _write_pdf(
    path: Path,
    pages: Sequence[Sequence[PdfLine]],
    title: Optional[str] = None,
    toc: Optional[Sequence[Tuple[int, str, int]]] = None,
    font: str = "helv",
    bold_font: str = "hebo",
) -> Path:
    """Render explicit lines, one list per page."""
    doc = fitz.open()
    for page_lines in pages:
        page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        cursor = MARGIN_TOP
        for line in page_lines:
            if not line.text:
                cursor += line.size * line.spacing
                continue
            page.insert_text(
                (MARGIN_LEFT + line.indent, cursor),
                line.text,
                fontsize=line.size,
                fontname=bold_font if line.bold else font,
            )
            cursor += line.size * line.spacing
    if title:
        doc.set_metadata({"title": title})
    if toc:
        doc.set_toc([list(entry) for entry in toc])
    doc.save(str(path))
    doc.close()
    return path


def _flow_pdf(
    path: Path,
    chapters: Sequence[FixtureChapter],
    title: Optional[str],
    toc: bool,
    running_header: Optional[str],
    running_footer: bool,
    body_size: float,
    words_per_line: int,
    font: str,
    bold_font: str,
) -> Path:
    """Lay chapters out over as many pages as they need, like a real book."""
    doc = fitz.open()
    toc_entries: List[List[object]] = []
    state = {"page": None, "cursor": MARGIN_TOP, "number": 0}

    def new_page() -> None:
        page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        state["page"] = page
        state["number"] += 1
        state["cursor"] = MARGIN_TOP
        if running_header:
            page.insert_text(
                (MARGIN_LEFT, HEADER_Y), running_header, fontsize=9, fontname=font
            )
        if running_footer:
            page.insert_text(
                (MARGIN_LEFT, FOOTER_Y),
                "Page {0}".format(state["number"]),
                fontsize=9,
                fontname=font,
            )

    def draw(text: str, size: float, bold: bool, spacing: float) -> None:
        if state["page"] is None or state["cursor"] + size * spacing > MARGIN_BOTTOM:
            new_page()
        state["page"].insert_text(
            (MARGIN_LEFT, state["cursor"]),
            text,
            fontsize=size,
            fontname=bold_font if bold else font,
        )
        state["cursor"] += size * spacing

    def heading(text: str, size: float, level: int) -> None:
        # Headings always start a page: that is what real books do, and it keeps
        # them inside the "top 25% of the page" window the detector looks at.
        new_page()
        toc_entries.append([level, text, state["number"]])
        draw(text, size, True, PARAGRAPH_SPACING)

    def paragraphs(items: Sequence[str]) -> None:
        for index, paragraph in enumerate(items):
            lines = _wrap(paragraph, words_per_line)
            for position, line in enumerate(lines):
                last = position == len(lines) - 1
                draw(
                    line,
                    body_size,
                    False,
                    PARAGRAPH_SPACING if last and index < len(items) - 1 else LINE_SPACING,
                )

    for chapter in chapters:
        heading(chapter.title, chapter.heading_size, 1)
        paragraphs(chapter.paragraphs)
        for section_title, section_paragraphs in chapter.sections:
            heading(section_title, chapter.section_heading_size, 2)
            paragraphs(section_paragraphs)

    if title:
        doc.set_metadata({"title": title})
    if toc and toc_entries:
        doc.set_toc(toc_entries)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def pdf_lines_factory(tmp_path: Path) -> Callable[..., Path]:
    """Build a PDF from explicit per-page lines (precise typography control)."""

    def build(
        pages: Sequence[Sequence[PdfLine]],
        name: str = "lines.pdf",
        title: Optional[str] = None,
        toc: Optional[Sequence[Tuple[int, str, int]]] = None,
        font: str = "helv",
        bold_font: str = "hebo",
    ) -> Path:
        return _write_pdf(
            tmp_path / name, pages, title=title, toc=toc, font=font, bold_font=bold_font
        )

    return build


@pytest.fixture
def book_pdf_factory(tmp_path: Path) -> Callable[..., Path]:
    """Build a multi-page book PDF with headings, prose and running heads."""

    def build(
        chapters: Sequence[FixtureChapter],
        name: str = "book.pdf",
        title: Optional[str] = None,
        toc: bool = False,
        running_header: Optional[str] = None,
        running_footer: bool = False,
        body_size: float = 11.0,
        words_per_line: int = 12,
        font: str = "helv",
        bold_font: str = "hebo",
    ) -> Path:
        return _flow_pdf(
            tmp_path / name,
            chapters,
            title=title,
            toc=toc,
            running_header=running_header,
            running_footer=running_footer,
            body_size=body_size,
            words_per_line=words_per_line,
            font=font,
            bold_font=bold_font,
        )

    return build


# ------------------------------------------------------------- ingested books

BOOK_TITLE = "A Field Guide to the Meridian Archipelago"
BOOK_TERMS = [
    ("Kestrel", 14),
    ("Meridian", 11),
    ("Thornbury", 9),
    ("Halvard", 7),
    ("Sable", 5),
]
BOOK_CHUNKS = [
    (
        1,
        "Kestrel Cove",
        "The harbour at Kestrel Cove was built in 1873 by the Halvard family. "
        "It still guards the northern approach to the island.",
    ),
    (
        1,
        "Kestrel Cove",
        "Every evening the keeper walked out to the lantern house and lit the "
        "wick, a duty the Halvard family kept for four generations.",
    ),
    (
        2,
        "Thornbury Reach",
        "Thornbury Reach is a shallow channel, three and a half metres deep at "
        "low tide, crossed by the Sable ferry twice a day.",
    ),
]


@pytest.fixture
def ready_book(store: Store, fake_ai: FakeAIClient) -> str:
    """A small book already in the database, ready to answer questions.

    Built by writing the rows an ingestion would have written rather than by
    running one: these tests are about the query path, and a real PDF ingest
    would make them slow and about something else.
    """
    from app.index.store import STATUS_READY
    from app.ingest.chunker import Chunk
    from app.ingest.structure import BookStructure, Chapter

    book_id = "book-meridian"
    store.create_book(
        book_id=book_id,
        title=BOOK_TITLE,
        filename="meridian.pdf",
        sha256="sha-meridian",
        status=STATUS_READY,
    )
    store.replace_chapters(
        book_id,
        BookStructure(
            title=BOOK_TITLE,
            chapters=[
                Chapter(
                    idx=index,
                    title=title,
                    page_start=index * 10,
                    page_end=index * 10 + 9,
                    anchor=0,
                    sections=[],
                )
                for index, title in ((1, "Kestrel Cove"), (2, "Thornbury Reach"))
            ],
            source="toc",
        ),
    )
    chunks = [
        Chunk(
            id="{0}-c{1}".format(book_id, position),
            ord=position,
            chapter_idx=chapter_idx,
            chapter_title=section,
            section_title=section,
            page_start=chapter_idx * 10,
            page_end=chapter_idx * 10,
            text=text,
            header="«{0}» — {1}\n\n{2}".format(BOOK_TITLE, section, text),
            parent_text=text,
            char_span=(0, len(text)),
        )
        for position, (chapter_idx, section, text) in enumerate(BOOK_CHUNKS)
    ]
    store.replace_chunks(
        book_id, chunks, fake_ai.embed([chunk.header for chunk in chunks])
    )
    store.replace_vocabulary(book_id, BOOK_TERMS)
    return book_id


VOCABULARY = (
    "harbour lantern basalt ferry orchard granite tide keeper market salt chart "
    "island cove beacon fisher reef sail storm quarry bridge windmill cistern "
    "almanac ledger cooper tanner vineyard cliff gull anchor rope compass "
    "meridian archipelago causeway lighthouse sextant barometer estuary"
).split()


def lorem(seed: str, words: int = 90) -> str:
    """Deterministic but non-repeating filler prose, ending in a full stop.

    Seeded per paragraph so fixtures are reproducible, yet no two paragraphs
    share a line — repeated lines would be stripped as running headers.
    """
    generator = random.Random(seed)
    body = [generator.choice(VOCABULARY) for _ in range(max(1, words - 1))]
    return " ".join([seed] + body) + "."
