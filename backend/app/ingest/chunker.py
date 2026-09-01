"""Hybrid chunking, three passes (SPEC §4.3 as revised by §14.3).

The algorithm is Docling's ``HybridChunker`` (IBM, MIT) ported onto our block
stream, with ``semchunk`` as the leaf splitter:

1. **assemble** — walk the block stream keeping ``heading_by_level``, popping
   every key at or below an incoming heading's level, so each base chunk carries
   the full ancestor breadcrumb it was written under. A heading is never a chunk
   of its own: it opens the body of the first chunk beneath it.
2. **split** — any base chunk whose *contextualized* token count (breadcrumb
   header included: ``available = max_tokens - header_tokens``) exceeds the cap
   is split by ``semchunk``, which recurses through paragraph, sentence and word
   separators to cut at the most semantic boundary that fits.
3. **merge** — adjacent undersized chunks are merged, but only when their
   heading paths are identical, so a chunk never straddles two sections. A final
   split pass guarantees the cap survives merging.

Children are 256–320 tokens with **zero overlap**: Chroma's chunking study
(5 corpora, 472 queries) measured recursive 200/0 at 88.1% recall, and every
overlap arm scored worse — overlap buys duplication, not recall.

The *parent* window (what the LLM reads) is still the surrounding section text
capped at 1600 tokens; the child is what retrieval scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from app.ingest.markdown import Block
from app.ingest.pdf import PageText
from app.ingest.structure import BookStructure, Chapter, LeafUnit, leaf_units

CHILD_MAX_TOKENS = 320
CHILD_MIN_TOKENS = 256
CHUNK_OVERLAP = 0
PARENT_MAX_TOKENS = 1600
HEADER_MAX_FRACTION = 0.25
CONTENT_FLOOR_TOKENS = 96

HEADER_TEMPLATE = "{breadcrumb}\n\n{text}"
BOOK_TEMPLATE = "«{0}»"
PATH_SEPARATOR = " — "
PARAGRAPH_SEPARATOR = "\n\n"

KIND_TEXT = "text"
KIND_CHAPTER_SUMMARY = "chapter_summary"
KIND_BOOK_SUMMARY = "book_summary"

LEVEL_LEAF = 0
LEVEL_CHAPTER_SUMMARY = 1
LEVEL_BOOK_SUMMARY = 2


@dataclass
class Chunk:
    """A child chunk plus everything needed to index, cite and expand it.

    ``header`` is the full indexed document — the breadcrumb (or, when
    contextual enrichment is on, the LLM-written situating sentence) followed by
    the chunk text. It is what gets embedded and BM25-indexed, while ``text`` is
    what is shown as a source preview and ``parent_text`` is what the LLM reads.
    """

    id: str
    ord: int
    chapter_idx: int
    chapter_title: str
    section_title: Optional[str]
    page_start: int
    page_end: int
    text: str
    header: str
    parent_text: str
    char_span: Tuple[int, int]
    kind: str = KIND_TEXT
    level: int = LEVEL_LEAF
    heading_path: Tuple[str, ...] = ()
    context: Optional[str] = None


_ENCODER = None
_ENCODER_LOADED = False


def _encoder():
    """Load ``cl100k_base`` once; ``None`` when tiktoken is unavailable."""
    global _ENCODER, _ENCODER_LOADED
    if not _ENCODER_LOADED:
        _ENCODER_LOADED = True
        try:
            import tiktoken

            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover - offline / missing model cache
            _ENCODER = None
    return _ENCODER


def count_tokens(text: str) -> int:
    """Token count with ``cl100k_base``, falling back to ``len(text) // 4``."""
    if not text:
        return 0
    encoder = _encoder()
    if encoder is None:
        return max(1, len(text) // 4)
    return len(encoder.encode(text, disallowed_special=()))


_SPLITTERS: Dict[int, Callable[[str], List[str]]] = {}


def _splitter(budget: int) -> Callable[[str], List[str]]:
    """A ``semchunk`` leaf splitter for one token budget, built once."""
    if budget not in _SPLITTERS:
        import semchunk

        _SPLITTERS[budget] = semchunk.chunkerify(count_tokens, budget)
    return _SPLITTERS[budget]


def split_text(text: str, budget: int) -> List[str]:
    """Split ``text`` into pieces of at most ``budget`` tokens, no overlap."""
    if budget <= 0 or count_tokens(text) <= budget:
        return [text] if text.strip() else []
    pieces = _splitter(budget)(text)
    return [piece.strip() for piece in pieces if piece and piece.strip()]


# ------------------------------------------------------------------- headers


def breadcrumb(book_title: str, path: Sequence[str]) -> str:
    """``«Book» — Chapter — Section``: where an isolated chunk comes from."""
    parts = [BOOK_TEMPLATE.format(book_title.strip())] if book_title.strip() else []
    parts.extend(part.strip() for part in path if part and part.strip())
    return PATH_SEPARATOR.join(parts)


def build_header(book_title: str, path: Sequence[str], text: str) -> str:
    """Compose the contextual header + chunk text that gets indexed."""
    line = breadcrumb(book_title, path)
    if not line:
        return text
    return HEADER_TEMPLATE.format(breadcrumb=line, text=text)


def header_budget(
    book_title: str,
    path: Sequence[str],
    max_tokens: int = CHILD_MAX_TOKENS,
    header_max_fraction: float = HEADER_MAX_FRACTION,
    content_floor: int = CONTENT_FLOOR_TOKENS,
) -> Tuple[Tuple[str, ...], int]:
    """Trim the heading path until the header fits its share of the budget.

    Two guardrails, both from §14.3: the header may not exceed
    ``header_max_fraction`` of a chunk, and what is left for the body may not
    fall under ``content_floor``. Trimming takes from the *head* of the path —
    the book title and outermost part — because the most specific level is the
    one that actually distinguishes this chunk. When even a bare path cannot
    meet the guardrails the header is dropped rather than the body starved.
    """
    ceiling = int(max_tokens * header_max_fraction)
    trimmed = [part for part in path if part and part.strip()]
    while True:
        tokens = count_tokens(breadcrumb(book_title, trimmed))
        available = max_tokens - tokens
        if tokens <= ceiling and available >= content_floor:
            return tuple(trimmed), available
        if not trimmed:
            return (), max_tokens
        trimmed.pop(0)


# ------------------------------------------------------------------ assembly


@dataclass
class _Piece:
    """A base chunk: body text under one heading path."""

    text: str
    path: Tuple[str, ...]
    unit: LeafUnit
    page_start: int
    page_end: int
    opens_section: bool = False
    group: int = 0


@dataclass
class _Group:
    """Consecutive pieces sharing one heading path — a section's own text."""

    key: Tuple[object, ...]
    pieces: List[_Piece] = field(default_factory=list)


class _UnitResolver:
    """Walks the structure's leaf units in step with the heading blocks."""

    def __init__(self, structure: BookStructure) -> None:
        self.units = leaf_units(structure) or [
            LeafUnit(
                chapter_idx=structure.chapters[0].idx if structure.chapters else 1,
                chapter_title=structure.chapters[0].title if structure.chapters else "",
                section_title=None,
                anchor=(1, 0.0),
            )
        ]
        # -1, not 0: the very first heading block must still be able to match
        # the first unit. Text before any heading falls back to it anyway.
        self._position = -1

    @property
    def current(self) -> LeafUnit:
        position = min(max(self._position, 0), len(self.units) - 1)
        return self.units[position]

    def advance(self, title: str) -> LeafUnit:
        """Move to the next unit whose title matches ``title``; stay if none."""
        target = _normalize(title)
        for position in range(self._position + 1, len(self.units)):
            unit = self.units[position]
            candidate = _normalize(_unit_title(unit))
            if candidate and (
                candidate.startswith(target) or target.startswith(candidate)
            ):
                self._position = position
                return unit
        return self.current


def chunk_book(
    blocks: Sequence[Block],
    structure: BookStructure,
    book_id: str,
    book_title: Optional[str] = None,
    max_tokens: int = CHILD_MAX_TOKENS,
    min_tokens: int = CHILD_MIN_TOKENS,
    parent_max_tokens: int = PARENT_MAX_TOKENS,
) -> List[Chunk]:
    """Turn an extracted block stream into indexed child chunks."""
    title = (book_title or structure.title or "Untitled").strip()
    pieces = _assemble(blocks, structure)
    pieces = _split_pass(pieces, title, max_tokens)
    pieces = _merge_pass(pieces, title, max_tokens, min_tokens)
    pieces = _split_pass(pieces, title, max_tokens)
    return _finalize(pieces, structure, book_id, title, max_tokens, parent_max_tokens)


def _assemble(blocks: Sequence[Block], structure: BookStructure) -> List[_Piece]:
    """Pass 1: base chunks carrying their full ancestor breadcrumb."""
    resolver = _UnitResolver(structure)
    heading_by_level: Dict[int, str] = {}
    pending: List[str] = []  # headings not yet attached to any body text
    pieces: List[_Piece] = []
    group = 0
    opened = False  # a heading has been seen since the last body block

    for block in blocks:
        if block.is_heading:
            # Pop every level at or below this one: an incoming h2 ends the
            # previous h2 and everything under it, but not its h1 ancestor.
            for level in [key for key in heading_by_level if key >= block.level]:
                heading_by_level.pop(level)
            heading_by_level[block.level] = block.text
            resolver.advance(block.text)
            pending.append(block.text)
            opened = True
            continue
        path = tuple(heading_by_level[level] for level in sorted(heading_by_level))
        text = block.text
        opens = bool(pending)
        if pending:
            # A bare heading never becomes its own chunk: it opens the body of
            # the first chunk written under it.
            text = PARAGRAPH_SEPARATOR.join(pending + [text])
            pending = []
        if opened:
            # Every heading starts a new section, even one that repeats the
            # title of the section before it.
            group += 1
            opened = False
        pieces.append(
            _Piece(
                text=text,
                path=path,
                unit=resolver.current,
                page_start=block.page_start,
                page_end=block.page_end,
                opens_section=opens,
                group=group,
            )
        )
    if pending and pieces:
        # Trailing headings with no body at all: fold them into the last chunk
        # rather than emitting a chunk that is only a title.
        pieces[-1].text = PARAGRAPH_SEPARATOR.join([pieces[-1].text] + pending)
    return pieces


def _available(piece: _Piece, book_title: str, max_tokens: int) -> int:
    """Room for body text once this chunk's header is accounted for.

    A chunk that opens a section carries its heading in the body and is indexed
    without a header, so it gets the whole budget.
    """
    if piece.opens_section:
        return max_tokens
    return header_budget(book_title, piece.path, max_tokens)[1]


def _split_pass(
    pieces: Sequence[_Piece], book_title: str, max_tokens: int
) -> List[_Piece]:
    """Pass 2: leaf-split anything over the contextualized budget."""
    result: List[_Piece] = []
    for piece in pieces:
        available = _available(piece, book_title, max_tokens)
        if count_tokens(piece.text) <= available:
            result.append(piece)
            continue
        parts = split_text(piece.text, available)
        for position, part in enumerate(parts):
            clone = _Piece(
                text=part,
                path=piece.path,
                unit=piece.unit,
                page_start=piece.page_start,
                page_end=piece.page_end,
                opens_section=piece.opens_section and position == 0,
                group=piece.group,
            )
            result.append(clone)
    return result


def _merge_pass(
    pieces: Sequence[_Piece], book_title: str, max_tokens: int, min_tokens: int
) -> List[_Piece]:
    """Pass 3: merge undersized neighbours that share an identical heading path."""
    result: List[_Piece] = []
    for piece in pieces:
        if not result:
            result.append(piece)
            continue
        previous = result[-1]
        if (
            previous.path == piece.path
            and previous.group == piece.group
            and count_tokens(previous.text) < min_tokens
        ):
            merged = previous.text + PARAGRAPH_SEPARATOR + piece.text
            if count_tokens(merged) <= _available(previous, book_title, max_tokens):
                previous.text = merged
                previous.page_end = max(previous.page_end, piece.page_end)
                continue
        result.append(piece)
    return result


def _finalize(
    pieces: Sequence[_Piece],
    structure: BookStructure,
    book_id: str,
    book_title: str,
    max_tokens: int,
    parent_max_tokens: int,
) -> List[Chunk]:
    """Attach headers, parent windows, page ranges and identifiers."""
    groups = _group(pieces)
    chapters = {chapter.idx: chapter for chapter in structure.chapters}
    chunks: List[Chunk] = []
    for group in groups:
        offsets = _offsets([piece.text for piece in group.pieces])
        for position, piece in enumerate(group.pieces):
            if not piece.text.strip():
                continue
            ordinal = len(chunks)
            path, _ = header_budget(book_title, piece.path, max_tokens)
            # The first chunk under a heading already opens with that heading:
            # repeating it in the header would index the same words twice.
            header = (
                piece.text
                if piece.opens_section
                else build_header(book_title, path, piece.text)
            )
            page_start, page_end = _page_range(piece, chapters.get(piece.unit.chapter_idx))
            chunks.append(
                Chunk(
                    id="{0}-{1:05d}".format(book_id, ordinal),
                    ord=ordinal,
                    chapter_idx=piece.unit.chapter_idx,
                    chapter_title=piece.unit.chapter_title,
                    section_title=_section_of(piece),
                    page_start=page_start,
                    page_end=page_end,
                    text=piece.text,
                    header=header,
                    parent_text=_parent_window(group.pieces, position, parent_max_tokens),
                    char_span=(
                        offsets[position],
                        offsets[position] + len(piece.text),
                    ),
                    heading_path=piece.path,
                )
            )
    return chunks


def _group(pieces: Sequence[_Piece]) -> List[_Group]:
    """Consecutive pieces with one heading path form one parent section."""
    groups: List[_Group] = []
    for piece in pieces:
        key = (piece.group, piece.path)
        if not groups or groups[-1].key != key:
            groups.append(_Group(key=key))
        groups[-1].pieces.append(piece)
    return groups


def _offsets(texts: Sequence[str]) -> List[int]:
    """Character offset of each piece within the joined group text."""
    offsets: List[int] = []
    cursor = 0
    for index, text in enumerate(texts):
        if index:
            cursor += len(PARAGRAPH_SEPARATOR)
        offsets.append(cursor)
        cursor += len(text)
    return offsets


def _parent_window(
    pieces: Sequence[_Piece], position: int, parent_max_tokens: int
) -> str:
    """Expand around one child over its section until the token cap is reached."""
    left = right = position
    total = count_tokens(pieces[position].text)
    while True:
        grew = False
        if left > 0:
            tokens = count_tokens(pieces[left - 1].text)
            if total + tokens <= parent_max_tokens:
                left -= 1
                total += tokens
                grew = True
        if right + 1 < len(pieces):
            tokens = count_tokens(pieces[right + 1].text)
            if total + tokens <= parent_max_tokens:
                right += 1
                total += tokens
                grew = True
        if not grew:
            break
    return PARAGRAPH_SEPARATOR.join(
        piece.text for piece in pieces[left : right + 1]
    ).strip()


def _page_range(piece: _Piece, chapter: Optional[Chapter]) -> Tuple[int, int]:
    """The piece's pages, clamped to the chapter it was filed under."""
    start, end = min(piece.page_start, piece.page_end), max(piece.page_start, piece.page_end)
    if chapter is None:
        return start, end
    start = min(max(start, chapter.page_start), chapter.page_end)
    end = min(max(end, start), chapter.page_end)
    return start, end


def _section_of(piece: _Piece) -> Optional[str]:
    """The deepest heading below chapter level, which the schema stores."""
    if len(piece.path) > 1:
        return piece.path[-1]
    return piece.unit.section_title


def _unit_title(unit: LeafUnit) -> str:
    return unit.section_title if unit.section_title is not None else unit.chapter_title


def _normalize(text: str) -> str:
    return " ".join((text or "").split()).casefold()


# ------------------------------------------------------------ naive baseline

NAIVE_CHUNK_CHARS = 800


def naive_chunks(
    pages: Sequence[PageText],
    structure: BookStructure,
    book_id: str,
    chars: int = NAIVE_CHUNK_CHARS,
) -> List[Chunk]:
    """Fixed-size character chunking: no structure, no header, no parent.

    This is the ablation baseline the eval harness measures the real pipeline
    against (SPEC §10, arm A). Chunks still carry the chapter they fall in —
    otherwise chapter-level recall could not be compared between the two
    configurations — but nothing about the *content* of a chunk depends on the
    book's structure. It reads the raw PDF lines rather than the markdown layer
    and keeps running heads, so §14.1's extraction win is one of the things the
    ablation measures.
    """
    chapter_of_page = _chapter_by_page(structure)
    chunks: List[Chunk] = []
    buffer = ""
    buffer_pages: List[int] = []

    def emit(body: str, page_nos: Sequence[int]) -> None:
        body = body.strip()
        if not body:
            return
        ordinal = len(chunks)
        page_start, page_end = min(page_nos), max(page_nos)
        chapter_idx = chapter_of_page.get(page_start, structure.chapters[0].idx)
        chunks.append(
            Chunk(
                id="{0}-{1:05d}".format(book_id, ordinal),
                ord=ordinal,
                chapter_idx=chapter_idx,
                chapter_title="",
                section_title=None,
                page_start=page_start,
                page_end=page_end,
                text=body,
                header=body,
                parent_text=body,
                char_span=(0, len(body)),
            )
        )

    for page in pages:
        for line in page.lines:
            text = line.text.strip()
            if not text:
                continue
            buffer = "{0} {1}".format(buffer, text) if buffer else text
            buffer_pages.append(page.page_no)
            while len(buffer) >= chars:
                emit(buffer[:chars], buffer_pages)
                buffer = buffer[chars:].strip()
                buffer_pages = [page.page_no]
    if buffer.strip():
        emit(buffer, buffer_pages or [pages[-1].page_no if pages else 1])
    return chunks


def _chapter_by_page(structure: BookStructure) -> Dict[int, int]:
    """Map every page number covered by the structure to its chapter index."""
    mapping: Dict[int, int] = {}
    for chapter in structure.chapters:
        for page_no in range(chapter.page_start, chapter.page_end + 1):
            mapping[page_no] = chapter.idx
    return mapping


__all__ = [
    "CHILD_MAX_TOKENS",
    "CHILD_MIN_TOKENS",
    "CHUNK_OVERLAP",
    "CONTENT_FLOOR_TOKENS",
    "Chunk",
    "HEADER_MAX_FRACTION",
    "KIND_BOOK_SUMMARY",
    "KIND_CHAPTER_SUMMARY",
    "KIND_TEXT",
    "LEVEL_BOOK_SUMMARY",
    "LEVEL_CHAPTER_SUMMARY",
    "LEVEL_LEAF",
    "PARENT_MAX_TOKENS",
    "breadcrumb",
    "build_header",
    "chunk_book",
    "count_tokens",
    "header_budget",
    "naive_chunks",
    "split_text",
]
