"""pymupdf4llm extraction, driven by our own structure detection (SPEC §14.1).

``pymupdf4llm.to_markdown`` is the text layer: it undoes ligatures and end-of-line
hyphenation that raw ``get_text()`` leaves behind (measured on a 244-page book:
129 ligature characters and 68 hyphen breaks with the raw call, 0 and 0 here) —
de-hyphenation alone is a direct BM25 win, because ``light-\\nhouse`` is not the
token ``lighthouse``.

What we do *not* take from it is its idea of what a heading is. Its default
``IdentifyHeaders`` ranks font sizes; we already ran a five-rung structure ladder
over the same document, so we feed that back in through the documented
``hdr_info=`` hook. The markdown that comes out therefore carries *our* ``#``
levels, and the block stream below is a faithful, cheap serialization of the
detected outline.

Running heads and feet are removed by the cross-page repeated-line detector, not
by cropping margins: a margin crop also eats the first line of text on a page
whose header sits low, and misses a header that sits high on a short page.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from app.ingest.pdf import PageText, is_repeated, repeated_line_texts
from app.ingest.structure import BookStructure, LeafUnit, leaf_units

logger = logging.getLogger(__name__)

KIND_HEADING = "heading"
KIND_TEXT = "text"

_EMPHASIS = re.compile(r"(\*\*|__|\*|`)")
_HEADING_LINE = re.compile(r"^(#{1,6})\s+(.*)$")
_RULE_LINE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_BLANK_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"[.!?。！？…”\"')\]]$")
_CJK_RANGE = re.compile(r"[㐀-䶿一-鿿　-〿＀-￯]")
_CONTINUES = re.compile(r"^[a-z0-9(\[“\"']")


@dataclass
class Block:
    """One document element: a heading, or a paragraph of body text."""

    kind: str
    level: int  # heading level (1-based); 0 for text
    text: str
    page_start: int
    page_end: int

    @property
    def is_heading(self) -> bool:
        return self.kind == KIND_HEADING


class StructureHeadings:
    """The ``hdr_info=`` hook: marks the headings *we* detected.

    Mirrors pymupdf4llm's own ``TocHeaders`` contract — ``get_header_id(span,
    page=...)`` returns the markdown prefix for a span — but answers from the
    structure ladder's output instead of from a font-size table.
    """

    def __init__(self, structure: BookStructure) -> None:
        self._by_page: Dict[int, List[Tuple[str, int]]] = {}
        for unit in leaf_units(structure):
            title = _clean(_title_of(unit))
            if not title:
                continue
            self._by_page.setdefault(unit.anchor[0], []).append((title, unit.level))

    @property
    def titles(self) -> List[str]:
        return [title for entries in self._by_page.values() for title, _ in entries]

    def level_for(self, text: str, page_no: int) -> int:
        """Outline level of ``text`` on that page, or 0 when it is body text."""
        clean = _clean(text)
        if not clean:
            return 0
        for title, level in self._by_page.get(page_no, []):
            if _same_heading(clean, title):
                return max(1, level)
        return 0

    def get_header_id(self, span: dict, page: object = None) -> str:
        page_no = getattr(page, "number", None)
        if page_no is None:
            return ""
        level = self.level_for(span.get("text", ""), int(page_no) + 1)
        return "#" * level + " " if level else ""


def extract_blocks(
    path: Union[str, Path],
    structure: BookStructure,
    pages: Optional[Sequence[PageText]] = None,
    repeated: Optional[Set[str]] = None,
) -> List[Block]:
    """Extract one book as an ordered stream of heading and text blocks.

    ``pages`` is the already-parsed geometric view of the same file. It is not
    re-read for its text — only for the two things markdown cannot answer: which
    lines repeat as running heads, and which words were broken by an
    end-of-line hyphen (see :func:`_hyphen_joins`).
    """
    hook = StructureHeadings(structure)
    if repeated is None:
        repeated = repeated_line_texts(pages) if pages else set()
    joins = _hyphen_joins(pages or [])
    markdown = _markdown_pages(path, hook)
    if not markdown and pages:
        markdown = _pages_as_markdown(pages, hook)
    blocks = _blocks_from_markdown(markdown, repeated, joins)
    return _insert_missing_headings(blocks, structure)


def _hyphen_joins(pages: Sequence[PageText]) -> Set[Tuple[str, str]]:
    """Word pairs that a line-end hyphen split apart, e.g. ``(light, house)``.

    pymupdf4llm deletes the hyphen *and* the newline it sat on, which is the
    de-hyphenation win — but when the break fell on a block boundary the two
    halves arrive as separate paragraphs with nothing left to say they belong
    together. The raw line geometry still knows, so we read the evidence there
    and rejoin exactly those pairs, never guessing.
    """
    joins: Set[Tuple[str, str]] = set()
    # One flat stream: a word broken at the foot of a page is the very case a
    # per-page loop would miss.
    lines = [line.text for page in pages for line in page.lines]
    for index, text in enumerate(lines[:-1]):
        text = text.rstrip()
        if not text.endswith("-") or len(text) < 2:
            continue
        stem = _last_word(text[:-1])
        head = _first_word(lines[index + 1])
        if stem and head and head[:1].islower():
            joins.add((stem, head))
    return joins


def _last_word(text: str) -> str:
    match = re.search(r"([\w’']+)\s*$", text, re.UNICODE)
    return match.group(1).casefold() if match else ""


def _first_word(text: str) -> str:
    match = re.match(r"\s*([\w’']+)", text, re.UNICODE)
    return match.group(1).casefold() if match else ""


def _rejoins(left: str, right: str, joins: Set[Tuple[str, str]]) -> bool:
    """True when ``left`` and ``right`` are the two halves of one hyphenated word."""
    return bool(joins) and (_last_word(left), _first_word(right)) in joins


def _markdown_pages(path: Union[str, Path], hook: StructureHeadings) -> List[Tuple[int, str]]:
    """Per-page markdown, in reading order; empty if the layer cannot read it."""
    try:
        import pymupdf4llm

        chunks = pymupdf4llm.to_markdown(
            str(path),
            page_chunks=True,
            hdr_info=hook,
            margins=0,  # running heads are removed by frequency, not by cropping
            table_strategy=None,
            ignore_code=True,
            show_progress=False,
        )
    except Exception:  # noqa: BLE001 - the raw line stream is a usable fallback
        logger.warning("pymupdf4llm extraction failed; using raw lines", exc_info=True)
        return []
    pages: List[Tuple[int, str]] = []
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        page_no = int(metadata.get("page", len(pages) + 1))
        pages.append((page_no, chunk.get("text") or ""))
    return pages


def _pages_as_markdown(
    pages: Sequence[PageText], hook: StructureHeadings
) -> List[Tuple[int, str]]:
    """The same shape as :func:`_markdown_pages`, built from the raw lines.

    The fallback when pymupdf4llm cannot read a file: one line per markdown
    paragraph, headings marked from the same hook. Everything downstream —
    repeated-line stripping, paragraph rejoining, de-hyphenation — is unchanged,
    so a book that trips the markdown layer still ingests, just without the
    ligature and hyphen cleanup.
    """
    result: List[Tuple[int, str]] = []
    for page in pages:
        lines: List[str] = []
        for line in page.lines:
            text = _clean(line.text)
            if not text:
                continue
            level = hook.level_for(text, page.page_no)
            lines.append("{0} {1}".format("#" * level, text) if level else text)
        result.append((page.page_no, "\n\n".join(lines)))
    return result


def _blocks_from_markdown(
    pages: Sequence[Tuple[int, str]],
    repeated: Set[str],
    joins: Optional[Set[Tuple[str, str]]] = None,
) -> List[Block]:
    """Fold per-page markdown into blocks, rejoining split paragraphs."""
    joins = joins or set()
    blocks: List[Block] = []
    for page_no, text in pages:
        for fragment in _BLANK_SPLIT.split(text):
            block = _fragment_block(fragment, page_no, repeated, joins)
            if block is None:
                continue
            if _merges_into(blocks, block, joins):
                continue
            blocks.append(block)
    return blocks


def _fragment_block(
    fragment: str, page_no: int, repeated: Set[str], joins: Set[Tuple[str, str]]
) -> Optional[Block]:
    """One markdown paragraph → a block, or ``None`` when it is furniture."""
    lines = [
        _clean(line)
        for line in fragment.splitlines()
        if line.strip() and not _RULE_LINE.match(line)
    ]
    lines = [line for line in lines if line]
    if not lines:
        return None

    heading = _HEADING_LINE.match(lines[0])
    if heading:
        title = _clean(heading.group(2))
        if not title:
            return None
        return Block(
            kind=KIND_HEADING,
            level=len(heading.group(1)),
            text=title,
            page_start=page_no,
            page_end=page_no,
        )

    lines = [line for line in lines if not is_repeated(line, repeated)]
    if not lines:
        return None
    text = _join_lines(lines, joins)
    if not text:
        return None
    return Block(
        kind=KIND_TEXT, level=0, text=text, page_start=page_no, page_end=page_no
    )


def _merges_into(
    blocks: List[Block], block: Block, joins: Set[Tuple[str, str]]
) -> bool:
    """Append ``block`` to the previous one when it continues its sentence.

    PDF paragraphs are routinely split into several blocks — by a column break,
    a page turn, or simply by how the file was produced. A fragment that starts
    mid-sentence after a fragment that did not end one belongs to it.
    """
    if block.is_heading or not blocks or blocks[-1].is_heading:
        return False
    previous = blocks[-1]
    tail = previous.text.rstrip()
    head = block.text.lstrip()
    if not tail or not head:
        return False
    if tail.endswith("-") and head[:1].isalpha() and head[:1].islower():
        previous.text = tail[:-1] + head  # an end-of-line hyphenation
    elif _rejoins(tail, head, joins):
        previous.text = tail + head  # ditto, with the hyphen already deleted
    elif _SENTENCE_END.search(tail):
        return False
    elif _CONTINUES.match(head) or _CJK_RANGE.match(head[:1]):
        previous.text = tail + _separator(tail, head) + head
    else:
        return False
    previous.page_end = max(previous.page_end, block.page_end)
    return True


def _insert_missing_headings(
    blocks: Sequence[Block], structure: BookStructure
) -> List[Block]:
    """Guarantee every detected unit opens a block, even if its line was not found.

    An embedded TOC can name a chapter with a title that never appears verbatim
    on the page (or appears as an image). Without a heading block the chapter's
    text would be filed under its predecessor, so a synthetic heading is inserted
    at the first block of the page the outline points at.
    """
    seen = [block.text for block in blocks if block.is_heading]
    pending: Dict[int, List[LeafUnit]] = {}
    for unit in leaf_units(structure):
        title = _clean(_title_of(unit))
        # The same forgiving comparison the hook used, so a heading it matched
        # by prefix is not then inserted a second time verbatim.
        if not title or any(_same_heading(title, text) for text in seen):
            continue
        pending.setdefault(unit.anchor[0], []).append(unit)
    if not pending:
        return list(blocks)
    logger.debug(
        "inserting %d heading(s) the page text never spelled out",
        sum(len(units) for units in pending.values()),
    )

    result: List[Block] = []
    placed: Set[int] = set()
    for block in blocks:
        for unit in pending.get(block.page_start, []):
            if id(unit) in placed:
                continue
            placed.add(id(unit))
            result.append(
                Block(
                    kind=KIND_HEADING,
                    level=unit.level,
                    text=_clean(_title_of(unit)),
                    page_start=block.page_start,
                    page_end=block.page_start,
                )
            )
        result.append(block)
    return result


def _title_of(unit: LeafUnit) -> str:
    return unit.section_title if unit.section_title is not None else unit.chapter_title


def _clean(text: str) -> str:
    """Strip markdown emphasis and collapse whitespace."""
    return " ".join(_EMPHASIS.sub("", text or "").split())


def _normalize(text: str) -> str:
    return _clean(text).casefold()


def _same_heading(span_text: str, title: str) -> bool:
    """Forgiving span↔title match: exact matches are rare in real PDFs."""
    left, right = _normalize(span_text), _normalize(title)
    if not left or not right:
        return False
    return left.startswith(right) or right.startswith(left)


def _join_lines(lines: Sequence[str], joins: Optional[Set[Tuple[str, str]]] = None) -> str:
    """Rejoin the wrapped lines of one paragraph."""
    joins = joins or set()
    text = lines[0]
    for line in lines[1:]:
        if text.endswith("-") and line[:1].isalpha() and line[:1].islower():
            text = text[:-1] + line
        elif _rejoins(text, line, joins):
            text = text + line
        else:
            text = text + _separator(text, line) + line
    return text.strip()


def _separator(left: str, right: str) -> str:
    """No space between two CJK characters; one space otherwise."""
    if not left or not right:
        return ""
    if _CJK_RANGE.search(left[-1]) and _CJK_RANGE.search(right[:1]):
        return ""
    return " "


__all__ = [
    "Block",
    "KIND_HEADING",
    "KIND_TEXT",
    "StructureHeadings",
    "extract_blocks",
]
