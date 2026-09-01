"""The book's own contents page, and LLM level assignment (SPEC §14.2).

Two things live here, both about tables of contents that the *PDF* does not
provide as an outline:

* :func:`parse_contents_pages` — a state machine over the pages a reader would
  call "Contents": find the marker, read ``title … page`` entries until the
  entries stop, validate them by monotonic page numbers, then map the printed
  numbers onto physical pages (the two differ by the front matter offset).
* :func:`assign_toc_levels` — RAGFlow's ingest-time trick for a flat contents
  list: one strict-JSON LLM call that assigns an outline level per title. Only
  used when no offline signal (the heading families) can supply the hierarchy.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional, Sequence, Tuple

from app.ingest.headings import family_levels
from app.ingest.pdf import PageText, TocEntry
from app.ingest.prompts import TOC_LEVELS_SYSTEM, TOC_LEVELS_USER_TEMPLATE

logger = logging.getLogger(__name__)

MIN_CONTENTS_ENTRIES = 3
MAX_SCAN_PAGES = 30
LEADER_CHARS = " \t.·•‧∙…_-–—~"

CONTENTS_MARKER = re.compile(
    r"^\s*(?:table\s+of\s+contents|contents|目\s*[录錄]|次\s*第|目\s*次)\s*$",
    re.IGNORECASE,
)
_TRAILING_PAGE = re.compile(r"(\d{1,4})\s*$")
_HAS_LETTER = re.compile(r"[0-9A-Za-z一-鿿]")


def parse_contents_pages(
    pages: Sequence[PageText],
    page_count: int,
    max_scan_pages: int = MAX_SCAN_PAGES,
    min_entries: int = MIN_CONTENTS_ENTRIES,
) -> List[TocEntry]:
    """Read the book's printed contents page into TOC entries.

    Returns entries with **physical** page numbers, or ``[]`` when the book has
    no contents page, too few usable entries, or entries whose page numbers do
    not increase (the signature of a false positive — a running list, an index).
    """
    start = _marker_page(pages, max_scan_pages)
    if start is None:
        return []
    raw = _collect_entries(pages, start, max_scan_pages)
    raw = _monotonic(raw)
    if len(raw) < min_entries:
        return []
    resolved = _to_physical(raw, pages, start, page_count)
    if len(resolved) < min_entries:
        return []
    levels = family_levels([title for title, _ in resolved]) or [1] * len(resolved)
    return [
        TocEntry(level=level, title=title, page=page)
        for (title, page), level in zip(resolved, levels)
    ]


def _marker_page(pages: Sequence[PageText], max_scan_pages: int) -> Optional[int]:
    """Index into ``pages`` of the page holding the "Contents" heading."""
    for position, page in enumerate(pages[:max_scan_pages]):
        for line in page.lines:
            if CONTENTS_MARKER.match(line.text.strip()):
                return position
    return None


def _collect_entries(
    pages: Sequence[PageText], start: int, max_scan_pages: int
) -> List[Tuple[str, int]]:
    """Walk forward from the contents page, reading entries until they stop."""
    entries: List[Tuple[str, int]] = []
    for page in pages[start : start + max_scan_pages]:
        found = 0
        for line in page.lines:
            text = line.text.strip()
            if CONTENTS_MARKER.match(text):
                continue
            entry = _parse_entry(text)
            if entry is None:
                continue
            entries.append(entry)
            found += 1
        if entries and not found:
            break  # the contents list ended on the previous page
    return entries


def _parse_entry(text: str) -> Optional[Tuple[str, int]]:
    """Split ``"Chapter 3. Ferries ..... 41"`` into its title and page number."""
    match = _TRAILING_PAGE.search(text)
    if not match:
        return None
    page = int(match.group(1))
    title = text[: match.start()].rstrip(LEADER_CHARS).strip()
    if page < 1 or len(title) < 2 or not _HAS_LETTER.search(title):
        return None
    if title == text.strip():  # nothing separated title from number
        return None
    return title, page


def _monotonic(entries: Sequence[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Drop entries whose page number goes backwards."""
    kept: List[Tuple[str, int]] = []
    highest = 0
    for title, page in entries:
        if page < highest:
            continue
        kept.append((title, page))
        highest = page
    return kept


def _to_physical(
    entries: Sequence[Tuple[str, int]],
    pages: Sequence[PageText],
    contents_index: int,
    page_count: int,
) -> List[Tuple[str, int]]:
    """Shift printed page numbers onto physical ones.

    The offset is measured, not guessed: every title that can be found as a line
    of the body votes for ``physical - printed``, and the median wins. With no
    match at all the printed numbers are used as-is, but only if they all fit
    inside the document.
    """
    body = list(pages)[contents_index + 1 :]
    offsets: List[int] = []
    for title, printed in entries:
        located = _find_title_page(body, title)
        if located is not None:
            offsets.append(located - printed)
    if offsets:
        offsets.sort()
        offset = offsets[len(offsets) // 2]
    elif max(page for _, page in entries) <= page_count:
        offset = 0
    else:
        return []
    resolved = [
        (title, min(page_count, max(1, printed + offset)) if page_count else printed)
        for title, printed in entries
    ]
    return _monotonic(resolved)


def _find_title_page(pages: Sequence[PageText], title: str) -> Optional[int]:
    """Page number where ``title`` appears as a line of its own, if it does."""
    needle = _normalize(title)
    if len(needle) < 4:
        return None
    for page in pages:
        for line in page.lines:
            candidate = _normalize(line.text)
            if candidate and (candidate.startswith(needle) or needle.startswith(candidate)):
                if abs(len(candidate) - len(needle)) <= max(4, len(needle) // 4):
                    return page.page_no
    return None


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def assign_toc_levels(
    ai: object,
    titles: Sequence[str],
    book_title: str = "",
    model: Optional[str] = None,
) -> List[int]:
    """Ask the LLM for one outline level per title; ``[]`` on any doubt.

    Strict JSON in, strict JSON out, order preserved (RAGFlow
    ``assign_toc_levels``). Anything unexpected — wrong length, unparseable
    reply, a level outside 1–4 — is discarded wholesale rather than partially
    trusted, because a half-applied hierarchy is worse than a flat one.
    """
    if ai is None or len(titles) < 2:
        return []
    listing = "\n".join(
        "{0}. {1}".format(position, title)
        for position, title in enumerate(titles, start=1)
    )
    messages = [
        {"role": "system", "content": TOC_LEVELS_SYSTEM},
        {
            "role": "user",
            "content": TOC_LEVELS_USER_TEMPLATE.format(
                book_title=book_title or "Untitled", titles=listing
            ),
        },
    ]
    try:
        raw = ai.chat(messages, purpose="toc_levels", temperature=0, model=model)
        levels = _parse_levels(raw, len(titles))
    except Exception:  # noqa: BLE001 - a flat TOC is a fine fallback
        logger.warning("TOC level assignment failed; keeping flat levels", exc_info=True)
        return []
    return levels


def _parse_levels(raw: str, expected: int) -> List[int]:
    """Extract ``expected`` levels from a JSON array of ``{level, title}``."""
    match = re.search(r"\[.*\]", raw or "", re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        return []
    if not isinstance(parsed, list) or len(parsed) != expected:
        return []
    levels: List[int] = []
    for item in parsed:
        if not isinstance(item, dict):
            return []
        try:
            level = int(item.get("level", 0))
        except (TypeError, ValueError):
            return []
        if not 1 <= level <= 4:
            return []
        levels.append(level)
    return levels


__all__ = [
    "MIN_CONTENTS_ENTRIES",
    "assign_toc_levels",
    "parse_contents_pages",
]
