"""Chapter/section detection — the structure ladder (SPEC §4.2, §14.2).

Five rungs, tried in order, each one a weaker but broader signal than the last:

1. **embedded TOC** — ``doc.get_toc()``. The publisher told us the answer.
2. **printed contents page** — the book's own "Contents", parsed and validated
   by monotonic page numbers (:mod:`app.ingest.toc`).
3. **heading-family voting** — score numbering conventions across every line and
   keep the argmax (:mod:`app.ingest.headings`). Finds "Chapter 7" and "第七章"
   in books that are typographically flat.
4. **font histogram** — the v1 heuristic: lines set well above body size, near
   the top of a page, clustered into chapter and section levels.
5. **fallback** — one synthetic "Full text" chapter, so the rest of the pipeline
   always has a structure to work with.

Rungs 1 and 2 can produce a flat list with no hierarchy; when they do and an AI
client is available, one strict-JSON LLM call assigns the levels (RAGFlow
``assign_toc_levels``).

Every unit carries an ``anchor`` — the ``(page_no, y)`` of the line where it
starts — and chapter ``idx`` is 1-based, matching the "Chapter N" a reader sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from app.ingest.headings import Vote, normalize_levels, vote
from app.ingest.pdf import PageText, TocEntry, body_font_size, is_repeated, repeated_line_texts
from app.ingest.toc import assign_toc_levels, parse_contents_pages

Anchor = Tuple[int, float]

HEADING_SIZE_RATIO = 1.3
HEADING_MAX_CHARS = 80
REGEX_HEADING_MAX_CHARS = 120
TOP_OF_PAGE_FRACTION = 0.25
MIN_TOC_ENTRIES = 3
MIN_VOTED_CHAPTERS = 2
FALLBACK_CHAPTER_TITLE = "Full text"

SOURCE_TOC = "toc"
SOURCE_CONTENTS = "contents"
SOURCE_VOTING = "voting"
SOURCE_HEADINGS = "headings"
SOURCE_FALLBACK = "fallback"


@dataclass
class Section:
    """A subdivision of a chapter."""

    title: str
    page_start: int
    page_end: int
    anchor: Anchor


@dataclass
class Chapter:
    """A top-level unit of the book. ``idx`` is 1-based."""

    idx: int
    title: str
    page_start: int
    page_end: int
    anchor: Anchor
    sections: List[Section] = field(default_factory=list)


@dataclass
class BookStructure:
    """The detected outline of a book, plus which rung of the ladder found it."""

    title: str
    chapters: List[Chapter]
    source: str  # toc | contents | voting | headings | fallback


@dataclass
class _Candidate:
    """A line that looks like a heading, and why."""

    text: str
    size: float
    page_no: int
    y: float
    family_level: Optional[int] = None  # level within the winning heading family
    by_style: bool = False

    @property
    def anchor(self) -> Anchor:
        # Rounded to match the position key the chunker computes per line.
        return (self.page_no, round(self.y, 2))


def detect_structure(
    pages: Sequence[PageText],
    toc: Optional[Sequence[TocEntry]] = None,
    title: str = "",
    page_count: Optional[int] = None,
    ai: Optional[object] = None,
    model: Optional[str] = None,
) -> BookStructure:
    """Detect the chapter/section outline of a parsed book.

    ``ai`` is optional: when present it is used for one TOC-level assignment
    call, and when absent the ladder simply degrades to a flat outline.
    """
    total_pages = page_count if page_count is not None else (len(pages) or 1)
    book_title = title.strip() or "Untitled"

    chapters = _from_toc(_leveled(toc or [], ai, book_title, model), total_pages)
    source = SOURCE_TOC
    if not chapters:
        printed = parse_contents_pages(pages, total_pages)
        chapters = _from_toc(_leveled(printed, ai, book_title, model), total_pages)
        source = SOURCE_CONTENTS
    if not chapters:
        chapters = _from_voting(pages, total_pages)
        source = SOURCE_VOTING
    if not chapters:
        chapters = _from_font_histogram(pages, total_pages)
        source = SOURCE_HEADINGS
    if not chapters:
        chapters = [
            Chapter(
                idx=1,
                title=FALLBACK_CHAPTER_TITLE,
                page_start=1,
                page_end=total_pages,
                anchor=(1, 0.0),
            )
        ]
        source = SOURCE_FALLBACK
    return BookStructure(title=book_title, chapters=chapters, source=source)


# ------------------------------------------------------------ rungs 1 and 2


def _leveled(
    entries: Sequence[TocEntry],
    ai: Optional[object],
    book_title: str,
    model: Optional[str],
) -> List[TocEntry]:
    """Give a flat TOC a hierarchy, with one LLM call if one is available."""
    listing = list(entries)
    if len(listing) < MIN_TOC_ENTRIES:
        return listing
    if len({entry.level for entry in listing}) > 1 or ai is None:
        return listing
    levels = assign_toc_levels(ai, [entry.title for entry in listing], book_title, model)
    if len(levels) != len(listing) or len(set(levels)) < 2:
        return listing
    return [
        TocEntry(level=level, title=entry.title, page=entry.page)
        for entry, level in zip(listing, levels)
    ]


def _from_toc(toc: Sequence[TocEntry], page_count: int) -> List[Chapter]:
    """Turn TOC entries (embedded or printed) into chapters and sections."""
    if len(toc) < MIN_TOC_ENTRIES:
        return []
    base = min(entry.level for entry in toc)
    chapters: List[Chapter] = []
    for entry in toc:
        if entry.level == base:
            chapters.append(
                Chapter(
                    idx=len(chapters) + 1,
                    title=entry.title,
                    page_start=entry.page,
                    page_end=entry.page,
                    anchor=(entry.page, 0.0),
                )
            )
        elif entry.level == base + 1 and chapters:
            chapters[-1].sections.append(
                Section(
                    title=entry.title,
                    page_start=entry.page,
                    page_end=entry.page,
                    anchor=(entry.page, 0.0),
                )
            )
    if not chapters:
        return []
    _close_ranges(chapters, page_count)
    return chapters


# ------------------------------------------------------------------- rung 3


def _from_voting(pages: Sequence[PageText], page_count: int) -> List[Chapter]:
    """Heading-family voting: the book's own numbering convention wins."""
    repeated = repeated_line_texts(pages)
    texts = [
        line.text
        for page in pages
        for line in page.lines
        if not is_repeated(line.text, repeated)
    ]
    winner = vote(texts)
    if winner is None:
        return []

    candidates = _heading_candidates(pages, winner)
    family = [c for c in candidates if c.family_level is not None]
    if not family:
        return []
    levels = normalize_levels([c.family_level or 1 for c in family])
    for candidate, level in zip(family, levels):
        candidate.family_level = level

    chapter_level = _chapter_family_level(levels)
    anchors = {c.anchor for c in family if (c.family_level or 1) <= chapter_level}
    if len(anchors) < MIN_VOTED_CHAPTERS:
        return []
    return _assemble(candidates, anchors, page_count)


def _chapter_family_level(levels: Sequence[int]) -> int:
    """The shallowest family level that names more than one unit.

    A single "PART ONE" above twelve chapters must not turn the book into one
    chapter: the level that repeats is the one that carries the structure.
    """
    counts: Dict[int, int] = {}
    for level in levels:
        counts[level] = counts.get(level, 0) + 1
    for level in sorted(counts):
        if counts[level] >= MIN_VOTED_CHAPTERS:
            return level
    return min(counts) if counts else 1


# ------------------------------------------------------------------- rung 4


def _from_font_histogram(pages: Sequence[PageText], page_count: int) -> List[Chapter]:
    """Typography only: cluster the distinct heading sizes into two levels."""
    candidates = [c for c in _heading_candidates(pages, None) if c.by_style]
    if not candidates:
        return []
    return _assemble(candidates, _size_split(candidates), page_count)


def _size_split(candidates: Sequence[_Candidate]) -> Set[Anchor]:
    """Cut the distinct font sizes at their largest gap; above it = chapters."""
    sizes = sorted({round(c.size, 1) for c in candidates}, reverse=True)
    if len(sizes) == 1:
        return {c.anchor for c in candidates}
    gaps = [(sizes[i] - sizes[i + 1], i) for i in range(len(sizes) - 1)]
    best_gap, split_at = max(gaps, key=lambda item: (item[0], -item[1]))
    if best_gap < 0.5:
        return {c.anchor for c in candidates}
    cutoff = sizes[split_at]
    return {c.anchor for c in candidates if round(c.size, 1) >= cutoff}


# --------------------------------------------------------------- shared bits


def _heading_candidates(
    pages: Sequence[PageText], winner: Optional[Vote]
) -> List[_Candidate]:
    """Lines matching the winning heading family or standing out typographically."""
    body_size = body_font_size(pages)
    repeated = repeated_line_texts(pages)
    candidates: List[_Candidate] = []
    for page in pages:
        top_limit = page.height * TOP_OF_PAGE_FRACTION if page.height else float("inf")
        for line in page.lines:
            text = line.text.strip()
            if not text or is_repeated(text, repeated):
                continue
            family_level = (
                winner.level_of(text)
                if winner is not None and len(text) <= REGEX_HEADING_MAX_CHARS
                else None
            )
            by_style = (
                body_size > 0
                and line.size >= HEADING_SIZE_RATIO * body_size
                and len(text) <= HEADING_MAX_CHARS
                and line.y <= top_limit
            )
            if family_level is not None or by_style:
                candidates.append(
                    _Candidate(
                        text=text,
                        size=line.size,
                        page_no=page.page_no,
                        y=line.y,
                        family_level=family_level,
                        by_style=by_style,
                    )
                )
    candidates.sort(key=lambda c: c.anchor)
    return candidates


def _assemble(
    candidates: Sequence[_Candidate], chapter_anchors: Set[Anchor], page_count: int
) -> List[Chapter]:
    """Fold ordered heading candidates into chapters with their sections."""
    chapters: List[Chapter] = []
    for candidate in candidates:
        if candidate.anchor in chapter_anchors:
            if chapters and _same_title(chapters[-1].title, candidate.text):
                continue
            chapters.append(
                Chapter(
                    idx=len(chapters) + 1,
                    title=candidate.text,
                    page_start=candidate.page_no,
                    page_end=candidate.page_no,
                    anchor=candidate.anchor,
                )
            )
        elif chapters:
            if chapters[-1].sections and _same_title(
                chapters[-1].sections[-1].title, candidate.text
            ):
                continue
            chapters[-1].sections.append(
                Section(
                    title=candidate.text,
                    page_start=candidate.page_no,
                    page_end=candidate.page_no,
                    anchor=candidate.anchor,
                )
            )
        else:
            # A section-level heading before any chapter heading is promoted,
            # so its text is never orphaned.
            chapters.append(
                Chapter(
                    idx=1,
                    title=candidate.text,
                    page_start=candidate.page_no,
                    page_end=candidate.page_no,
                    anchor=candidate.anchor,
                )
            )
    if not chapters:
        return []
    _close_ranges(chapters, page_count)
    return chapters


def _close_ranges(chapters: List[Chapter], page_count: int) -> None:
    """Fill in ``page_end`` for every chapter and section from the next unit."""
    for position, chapter in enumerate(chapters):
        next_start = (
            chapters[position + 1].page_start
            if position + 1 < len(chapters)
            else page_count + 1
        )
        chapter.page_end = max(chapter.page_start, min(page_count, next_start - 1))
        for s_pos, section in enumerate(chapter.sections):
            section_next = (
                chapter.sections[s_pos + 1].page_start
                if s_pos + 1 < len(chapter.sections)
                else chapter.page_end + 1
            )
            section.page_end = max(
                section.page_start, min(chapter.page_end, section_next - 1)
            )


def _same_title(left: str, right: str) -> bool:
    return " ".join(left.split()).casefold() == " ".join(right.split()).casefold()


@dataclass
class LeafUnit:
    """A chunking unit: a section, or a chapter's text outside any section."""

    chapter_idx: int
    chapter_title: str
    section_title: Optional[str]
    anchor: Anchor
    level: int = 1  # 1 = chapter, 2 = section


def leaf_units(structure: BookStructure) -> List[LeafUnit]:
    """Flatten the structure into ordered leaf units for the chunker.

    Every chapter contributes one unit for the text between its heading and its
    first section, plus one unit per section.
    """
    units: List[LeafUnit] = []
    for chapter in structure.chapters:
        units.append(
            LeafUnit(
                chapter_idx=chapter.idx,
                chapter_title=chapter.title,
                section_title=None,
                anchor=chapter.anchor,
                level=1,
            )
        )
        for section in chapter.sections:
            units.append(
                LeafUnit(
                    chapter_idx=chapter.idx,
                    chapter_title=chapter.title,
                    section_title=section.title,
                    anchor=section.anchor,
                    level=2,
                )
            )
    units.sort(key=lambda unit: unit.anchor)
    return units


__all__ = [
    "Anchor",
    "BookStructure",
    "Chapter",
    "FALLBACK_CHAPTER_TITLE",
    "LeafUnit",
    "Section",
    "SOURCE_CONTENTS",
    "SOURCE_FALLBACK",
    "SOURCE_HEADINGS",
    "SOURCE_TOC",
    "SOURCE_VOTING",
    "detect_structure",
    "leaf_units",
]
