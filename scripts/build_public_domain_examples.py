#!/usr/bin/env python3
"""Build the two English public-domain example books as BookTalk-friendly PDFs.

The Chinese examples are built by ``build_simplified_chinese_examples.py``.

The source texts are the complete UTF-8 transcriptions published by Project
Gutenberg.  This script only changes their presentation: it removes the
Gutenberg wrapper, lays the original text out on A4 pages, and adds PDF outline
entries at the work's real chapter boundaries so BookTalk can detect structure.

Download the source files listed in ``BOOKS`` into ``--source-dir`` first, then
run this script with the backend virtual environment::

    backend/.venv/bin/python scripts/build_public_domain_examples.py
"""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = Path("/tmp/booktalk_public_domain")
DEFAULT_OUTPUT_DIR = ROOT / "examples" / "books"
CHINESE_NUMERALS = "一二三四五六七八九十百千零〇"


@dataclass(frozen=True)
class Book:
    source_name: str
    output_name: str
    title: str
    author: str
    language: str
    ebook_id: int
    body_start: str
    heading_pattern: str
    chapter_titles: Tuple[str, ...] = ()
    body_start_occurrence: int = 1

    @property
    def source_url(self) -> str:
        return "https://www.gutenberg.org/cache/epub/{0}/pg{0}.txt".format(
            self.ebook_id
        )


BOOKS: Tuple[Book, ...] = (
    Book(
        source_name="alice.txt",
        output_name="04-Alices-Adventures-in-Wonderland-Lewis-Carroll.pdf",
        title="Alice's Adventures in Wonderland",
        author="Lewis Carroll",
        language="en",
        ebook_id=11,
        body_start="CHAPTER I.",
        heading_pattern=r"^CHAPTER [IVXLCDM]+\.$",
        chapter_titles=(
            "Down the Rabbit-Hole",
            "The Pool of Tears",
            "A Caucus-Race and a Long Tale",
            "The Rabbit Sends in a Little Bill",
            "Advice from a Caterpillar",
            "Pig and Pepper",
            "A Mad Tea-Party",
            "The Queen's Croquet-Ground",
            "The Mock Turtle's Story",
            "The Lobster Quadrille",
            "Who Stole the Tarts?",
            "Alice's Evidence",
        ),
    ),
    Book(
        source_name="jekyll_hyde.txt",
        output_name="05-Dr-Jekyll-and-Mr-Hyde-Robert-Louis-Stevenson.pdf",
        title="The Strange Case of Dr. Jekyll and Mr. Hyde",
        author="Robert Louis Stevenson",
        language="en",
        ebook_id=43,
        body_start="STORY OF THE DOOR",
        heading_pattern=(
            r"^(?:STORY OF THE DOOR|SEARCH FOR MR\. HYDE|"
            r"DR\. JEKYLL WAS QUITE AT EASE|THE CAREW MURDER CASE|"
            r"INCIDENT OF THE LETTER|INCIDENT OF DR\. LANYON|"
            r"INCIDENT AT THE WINDOW|THE LAST NIGHT|"
            r"DR\. LANYON’S NARRATIVE|"
            r"HENRY JEKYLL’S FULL STATEMENT OF THE CASE)$"
        ),
        body_start_occurrence=2,
    ),
)


class BookTemplate(BaseDocTemplate):
    """Document template that turns marked paragraphs into PDF bookmarks."""

    def __init__(self, *args, running_title: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.running_title = running_title
        self._outline_index = 0

    def afterFlowable(self, flowable: Flowable) -> None:
        if not getattr(flowable, "is_chapter_heading", False):
            return
        self._outline_index += 1
        key = "chapter-{0:03d}".format(self._outline_index)
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(flowable.getPlainText(), key, level=0)


def strip_gutenberg_wrapper(text: str) -> str:
    start = re.search(
        r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*\s*$",
        text,
        flags=re.MULTILINE,
    )
    end = re.search(
        r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*\s*$",
        text,
        flags=re.MULTILINE,
    )
    if not start or not end or end.start() <= start.end():
        raise ValueError("Project Gutenberg start/end markers were not found")
    return text[start.end() : end.start()].strip()


def from_exact_line(text: str, marker: str, occurrence: int = 1) -> str:
    """Drop title/contents matter and begin at an exact body marker."""
    lines = text.splitlines()
    seen = 0
    for index, line in enumerate(lines):
        if line.strip() == marker:
            seen += 1
            if seen == occurrence:
                return "\n".join(lines[index:])
    raise ValueError("Body marker not found: {0}".format(marker))


def blocks(book: Book, text: str) -> List[Tuple[str, str]]:
    """Return ``(kind, text)`` blocks while preserving original wording."""
    body = from_exact_line(
        strip_gutenberg_wrapper(text),
        book.body_start,
        occurrence=book.body_start_occurrence,
    )
    heading_re = re.compile(book.heading_pattern)
    result: List[Tuple[str, str]] = []
    paragraph_lines: List[str] = []
    chapter_index = 0

    def flush() -> None:
        if not paragraph_lines:
            return
        separator = "" if book.language == "zh" else " "
        paragraph = separator.join(piece.strip() for piece in paragraph_lines)
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            result.append(("body", paragraph))
        paragraph_lines.clear()

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if heading_re.fullmatch(line):
            flush()
            heading = line
            if book.chapter_titles and chapter_index < len(book.chapter_titles):
                heading = "{0}  {1}".format(
                    heading, book.chapter_titles[chapter_index]
                )
            result.append(("heading", heading))
            chapter_index += 1
        elif not line:
            flush()
        else:
            paragraph_lines.append(line)
    flush()

    if chapter_index < 2:
        raise ValueError(
            "Only {0} chapter headings detected for {1}".format(
                chapter_index, book.title
            )
        )
    return result


def styles(book: Book) -> Tuple[ParagraphStyle, ParagraphStyle, ParagraphStyle]:
    if book.language == "zh":
        font = "STSong-Light"
        title_font = font
        leading = 18
    else:
        font = "Times-Roman"
        title_font = "Times-Bold"
        leading = 15
    body = ParagraphStyle(
        "Body-{0}".format(book.language),
        fontName=font,
        fontSize=10.5,
        leading=leading,
        firstLineIndent=18,
        alignment=TA_JUSTIFY,
        spaceAfter=7,
    )
    heading = ParagraphStyle(
        "Heading-{0}".format(book.language),
        fontName=title_font,
        fontSize=17,
        leading=23,
        spaceBefore=18,
        spaceAfter=12,
    )
    title = ParagraphStyle(
        "Title-{0}".format(book.language),
        fontName=title_font,
        fontSize=26,
        leading=34,
        alignment=TA_CENTER,
        spaceAfter=20,
    )
    return body, heading, title


def decorate(canvas, doc) -> None:
    if canvas.getPageNumber() <= 1:
        return
    canvas.saveState()
    canvas.setFillColorRGB(0.35, 0.35, 0.35)
    # Helvetica has no CJK glyphs: a Chinese running title needs the CID font.
    title_font = "Helvetica" if doc.running_title.isascii() else "STSong-Light"
    canvas.setFont(title_font, 8)
    canvas.drawString(48, A4[1] - 33, doc.running_title)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(A4[0] / 2, 28, str(canvas.getPageNumber()))
    canvas.restoreState()


def render(book: Book, source_path: Path, output_path: Path) -> None:
    text = source_path.read_text(encoding="utf-8-sig")
    content = blocks(book, text)
    body_style, heading_style, title_style = styles(book)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = Frame(48, 45, A4[0] - 96, A4[1] - 92, id="content")
    doc = BookTemplate(
        str(output_path),
        pagesize=A4,
        title=book.title,
        author=book.author,
        subject="Public-domain BookTalk example",
        running_title=book.title,
    )
    doc.addPageTemplates([PageTemplate(id="book", frames=[frame], onPage=decorate)])

    note_style = ParagraphStyle(
        "Source-note",
        fontName="STSong-Light" if book.language == "zh" else "Times-Roman",
        fontSize=9,
        leading=14,
        alignment=TA_CENTER,
        textColor="#555555",
    )
    story: List[Flowable] = [
        Spacer(1, 165),
        Paragraph(html.escape(book.title), title_style),
        Paragraph(html.escape(book.author), note_style),
        Spacer(1, 36),
        Paragraph(
            html.escape(
                "Public-domain text source: Project Gutenberg eBook #{0}\n{1}".format(
                    book.ebook_id, book.source_url
                )
            ).replace("\n", "<br/>")
            ,
            note_style,
        ),
        PageBreak(),
    ]
    for kind, value in content:
        style = heading_style if kind == "heading" else body_style
        paragraph = Paragraph(html.escape(value), style)
        if kind == "heading":
            paragraph.is_chapter_heading = True
        story.append(paragraph)
    doc.build(story)


def validate(outputs: Iterable[Tuple[Book, Path]]) -> None:
    import fitz

    for book, path in outputs:
        with fitz.open(path) as document:
            extracted = "".join(page.get_text() for page in document)
            toc = document.get_toc()
            if len(document) < 3:
                raise ValueError("Rendered book is unexpectedly short: {0}".format(path))
            if len(toc) < 2:
                raise ValueError("Rendered book has no usable outline: {0}".format(path))
            if len(extracted.strip()) < 1000:
                raise ValueError("Rendered text did not extract correctly: {0}".format(path))
            print(
                "{0}: {1} pages, {2} chapters, {3:,} extracted characters".format(
                    path.name, len(document), len(toc), len(extracted)
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registerFont(UnicodeCIDFont("STSong-Light"))
    outputs: List[Tuple[Book, Path]] = []
    for book in BOOKS:
        source_path = args.source_dir / book.source_name
        if not source_path.is_file():
            raise FileNotFoundError(
                "Missing {0}; download {1}".format(source_path, book.source_url)
            )
        output_path = args.output_dir / book.output_name
        render(book, source_path, output_path)
        outputs.append((book, output_path))
    validate(outputs)


if __name__ == "__main__":
    main()
