"""PyMuPDF text extraction (SPEC §4.1, revised by §14.1).

Pages are parsed lazily and immediately reduced to small ``Line`` records
(text + font size + bold + position). The full ``get_text("dict")`` payload for a
page is discarded before the next page is read, so peak memory stays flat even
for a 1000-page book.

These geometric records are what *structure detection* reads: font sizes, the
top-of-page test and the cross-page repeated-line detector all need coordinates,
which markdown has thrown away. The text that actually gets indexed comes from
:mod:`app.ingest.markdown` instead — pymupdf4llm, driven by the structure we
detect here (SPEC §14.1).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Set, Tuple, Union

import fitz

_BOLD_FLAG = 1 << 4


@dataclass
class Line:
    """One visual line of text with the typography needed for heading detection."""

    text: str
    size: float
    bold: bool
    y: float
    x0: float


@dataclass
class PageText:
    """All text lines of a single page, in reading order."""

    page_no: int  # 1-based, matching what a reader sees and what the TOC uses
    lines: List[Line] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0


@dataclass
class TocEntry:
    """One entry of a PDF's embedded outline."""

    level: int
    title: str
    page: int  # 1-based


class PdfDocument:
    """Thin wrapper over ``fitz.Document`` exposing only what ingestion needs."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self._doc = fitz.open(str(self.path))

    def __enter__(self) -> "PdfDocument":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._doc.close()

    @property
    def page_count(self) -> int:
        return self._doc.page_count

    @property
    def metadata_title(self) -> str:
        metadata = self._doc.metadata or {}
        return (metadata.get("title") or "").strip()

    def toc(self) -> List[TocEntry]:
        """Embedded outline entries with in-range page numbers."""
        entries: List[TocEntry] = []
        for level, title, page in self._doc.get_toc(simple=True) or []:
            clean = (title or "").strip()
            if not clean or page < 1 or page > self.page_count:
                continue
            entries.append(TocEntry(level=int(level), title=clean, page=int(page)))
        return entries

    def iter_pages(self) -> Iterator[PageText]:
        """Yield one :class:`PageText` per page, parsing lazily."""
        for index in range(self._doc.page_count):
            page = self._doc.load_page(index)
            yield _page_text(page, index + 1)


def _page_text(page: "fitz.Page", page_no: int) -> PageText:
    rect = page.rect
    result = PageText(page_no=page_no, width=rect.width, height=rect.height)
    raw = page.get_text("dict")
    for block in raw.get("blocks", []):
        if block.get("type") != 0:  # 0 = text block; images are ignored
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            size = max((float(span.get("size", 0.0)) for span in spans), default=0.0)
            bold = any(int(span.get("flags", 0)) & _BOLD_FLAG for span in spans)
            bbox = line.get("bbox") or (0.0, 0.0, 0.0, 0.0)
            result.lines.append(
                Line(text=text, size=round(size, 2), bold=bold, y=float(bbox[1]), x0=float(bbox[0]))
            )
    result.lines.sort(key=lambda item: (round(item.y, 1), item.x0))
    return result


def read_pages(path: Union[str, Path]) -> List[PageText]:
    """Eagerly parse a whole document (used by tests and small documents)."""
    with PdfDocument(path) as doc:
        return list(doc.iter_pages())


SCANNED_PREFLIGHT_PAGES = 10
MIN_CHARS_PER_PAGE = 20.0


def chars_per_page(
    pages: Sequence[PageText], limit: int = SCANNED_PREFLIGHT_PAGES
) -> float:
    """Mean extractable characters per page over the first ``limit`` pages.

    The scanned-PDF preflight (SPEC §14.1): a born-digital book measures ~1400+,
    an image-only scan ~0, and nothing real sits in between.
    """
    sample = list(pages)[:limit]
    if not sample:
        return 0.0
    total = sum(len(line.text) for page in sample for line in page.lines)
    return total / float(len(sample))


def looks_scanned(
    pages: Sequence[PageText],
    limit: int = SCANNED_PREFLIGHT_PAGES,
    minimum: float = MIN_CHARS_PER_PAGE,
) -> bool:
    """True when a PDF carries no usable text layer and would need OCR."""
    return chars_per_page(pages, limit) < minimum


def body_font_size(pages: Sequence[PageText]) -> float:
    """The modal line size across the document — the body text size."""
    counter: Counter = Counter()
    for page in pages:
        for line in page.lines:
            counter[round(line.size, 1)] += 1
    if not counter:
        return 0.0
    return max(counter.items(), key=lambda item: (item[1], -item[0]))[0]


def repeated_line_texts(
    pages: Sequence[PageText],
    threshold: float = 0.3,
    min_pages: int = 4,
    min_occurrences: int = 4,
) -> Set[str]:
    """Texts that appear on more than ``threshold`` of pages: running heads/feet.

    Three refinements keep real content out of the result:

    * digits are normalized away, so ``"Page 12"`` and ``"Page 13"`` count as
      the same running footer;
    * lines set larger than the body text are never candidates — that is what
      separates a display "Chapter 2" from a running head, which is always body
      size or smaller;
    * a *numbered heading* set at body size ("Chapter 2. Isle 2") normalizes to
      the same key as its neighbours and would otherwise look like a running
      head in a short book. It is spared unless one of its exact strings repeats
      across pages, which is what a genuine running head does and what a chapter
      opening never does.

    The rule is skipped entirely for documents too short for the frequency to
    mean anything.
    """
    if len(pages) < min_pages:
        return set()
    from app.ingest.headings import any_family_level

    body_size = body_font_size(pages)
    seen_on_pages: Dict[str, Set[int]] = {}
    exact_on_pages: Dict[Tuple[str, str], Set[int]] = {}
    numbered: Set[str] = set()
    for page in pages:
        for line in page.lines:
            if body_size and line.size > body_size + 0.5:
                continue
            text = " ".join(line.text.split())
            key = _normalize_repeat_key(text)
            if not key:
                continue
            seen_on_pages.setdefault(key, set()).add(page.page_no)
            exact_on_pages.setdefault((key, text), set()).add(page.page_no)
            if any_family_level(text) is not None:
                numbered.add(key)

    verbatim: Dict[str, int] = {}
    for (key, _), page_nos in exact_on_pages.items():
        verbatim[key] = max(verbatim.get(key, 0), len(page_nos))

    limit = threshold * len(pages)
    repeated: Set[str] = set()
    for key, page_nos in seen_on_pages.items():
        if len(page_nos) <= limit or len(page_nos) < min_occurrences:
            continue
        if key in numbered and verbatim.get(key, 0) < 2:
            continue
        repeated.add(key)
    return repeated


_DIGIT_RUN = re.compile(r"\d+")


def _normalize_repeat_key(text: str) -> str:
    """Collapse digit runs and whitespace so paginated headers compare equal.

    A *run* of digits, not each digit: "Page 9" and "Page 10" have to land on
    the same key, or a book's footer splits into a one-digit group and a
    two-digit group and neither reaches the repetition threshold.
    """
    stripped = " ".join(text.split()).strip()
    if not stripped:
        return ""
    return _DIGIT_RUN.sub("#", stripped).casefold()


def is_repeated(text: str, repeated: Set[str]) -> bool:
    """True when ``text`` matches one of the running header/footer keys."""
    return _normalize_repeat_key(text) in repeated


