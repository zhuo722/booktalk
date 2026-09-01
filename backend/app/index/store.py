"""Persistence for books, chapters and chunks + the per-book runtime bundle.

Everything lives in one SQLite file; embeddings are float32 BLOBs. On the first
query for a book the rows are loaded once into a *bundle* — a normalized numpy
matrix, a BM25 index and the chunk metadata — and that bundle is cached for the
three most recently used books (SPEC §4.5).
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from app.db import session
from app.index.bm25 import BM25Index
from app.index.dense import normalize, to_matrix
from app.ingest.chunker import (
    KIND_BOOK_SUMMARY,
    KIND_CHAPTER_SUMMARY,
    KIND_TEXT,
    Chunk,
)
from app.ingest.structure import BookStructure

MAX_CACHED_BOOKS = 3

STATUS_PARSING = "parsing"
STATUS_STRUCTURING = "structuring"
STATUS_CHUNKING = "chunking"
# The two LLM enrichment stages of SPEC §14.4/§14.5; both are skippable, so the
# UI must treat the stage list as open-ended rather than as a fixed sequence.
STATUS_CONTEXTUALIZING = "contextualizing"
STATUS_SUMMARIZING = "summarizing"
STATUS_EMBEDDING = "embedding"
STATUS_READY = "ready"
STATUS_FAILED = "failed"

_CJK_RANGE = re.compile(r"[㐀-䶿一-鿿　-〿＀-￯]")
#: How much prose makes a passage the chapter's opening rather than a title
#: page or a stray fragment — counted in words where words are separated and
#: in characters where they are not.
MIN_OPENING_WORDS = 40
MIN_OPENING_CHARS = 60


def _flatten(text: str) -> str:
    """Collapse the extractor's line breaks without inventing word gaps.

    Joining every line with a space is right for a language written with
    spaces and wrong for Chinese, where it would open a gap mid-sentence at
    whatever point the typesetter broke the line. The join is decided per
    boundary, so a mixed passage keeps both conventions.
    """
    parts = text.split()
    if not parts:
        return ""
    out = parts[0]
    for part in parts[1:]:
        joined = _CJK_RANGE.match(out[-1]) and _CJK_RANGE.match(part[0])
        out += part if joined else " " + part
    return out


def _substantial(text: str) -> bool:
    """Long enough to be a chapter's opening passage.

    A word count does not survive the trip to Chinese: 200 characters of it
    come back as one or two whitespace-separated tokens, which would reject
    every chapter in the book and leave the reader printing its empty state
    on a chapter that has plenty of text.
    """
    if _CJK_RANGE.search(text):
        return len(_CJK_RANGE.findall(text)) >= MIN_OPENING_CHARS
    return len(text.split()) >= MIN_OPENING_WORDS


@dataclass
class BookRecord:
    """A row of the ``books`` table."""

    id: str
    title: str
    filename: str
    sha256: str
    pages: Optional[int]
    status: str
    progress: float
    error: Optional[str]
    created_at: str


@dataclass
class ChapterRecord:
    idx: int
    title: str
    page_start: Optional[int]
    page_end: Optional[int]


@dataclass
class ChunkRecord:
    """A chunk as retrieved from the database, with its chapter title joined in."""

    id: str
    book_id: str
    chapter_idx: int
    chapter_title: str
    section_title: Optional[str]
    page_start: int
    page_end: int
    text: str
    header: str
    parent_text: str
    ord: int
    kind: str = KIND_TEXT
    level: int = 0

    @property
    def parent_key(self) -> str:
        """Identity of the parent section this chunk was cut from."""
        return "{0}::{1}".format(self.chapter_idx, self.section_title or "")

    @property
    def is_summary(self) -> bool:
        """True for the RAPTOR-lite chapter/book synopsis nodes (SPEC §14.5)."""
        return self.kind in (KIND_CHAPTER_SUMMARY, KIND_BOOK_SUMMARY)


@dataclass
class BookIndex:
    """The runtime retrieval bundle for one book."""

    book_id: str
    chunks: List[ChunkRecord]
    matrix: np.ndarray
    bm25: BM25Index

    def __len__(self) -> int:
        return len(self.chunks)


def utc_now() -> str:
    """ISO-8601 UTC timestamp used for ``created_at``."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Store:
    """All SQLite access for the app, plus the bundle cache."""

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self._cache: "OrderedDict[str, BookIndex]" = OrderedDict()
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- books

    def create_book(
        self, book_id: str, title: str, filename: str, sha256: str, status: str
    ) -> BookRecord:
        record = BookRecord(
            id=book_id,
            title=title,
            filename=filename,
            sha256=sha256,
            pages=None,
            status=status,
            progress=0.0,
            error=None,
            created_at=utc_now(),
        )
        with session(self.db_path) as conn:
            conn.execute(
                "INSERT INTO books (id, title, filename, sha256, pages, status,"
                " progress, error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.title,
                    record.filename,
                    record.sha256,
                    record.pages,
                    record.status,
                    record.progress,
                    record.error,
                    record.created_at,
                ),
            )
        return record

    def get_book(self, book_id: str) -> Optional[BookRecord]:
        with session(self.db_path) as conn:
            row = conn.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()
        return _book_from_row(row) if row else None

    def get_book_by_sha256(self, sha256: str) -> Optional[BookRecord]:
        with session(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM books WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return _book_from_row(row) if row else None

    def list_books(self) -> List[BookRecord]:
        with session(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM books ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [_book_from_row(row) for row in rows]

    def update_book(
        self,
        book_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        error: Optional[str] = None,
        pages: Optional[int] = None,
        title: Optional[str] = None,
    ) -> None:
        """Patch the mutable columns of a book row (ingestion progress)."""
        fields: List[str] = []
        values: List[object] = []
        for column, value in (
            ("status", status),
            ("progress", progress),
            ("pages", pages),
            ("title", title),
        ):
            if value is not None:
                fields.append("{0} = ?".format(column))
                values.append(value)
        if status is not None:
            # Clearing the error on any status change keeps failed → retry clean.
            fields.append("error = ?")
            values.append(error)
        elif error is not None:
            fields.append("error = ?")
            values.append(error)
        if not fields:
            return
        values.append(book_id)
        with session(self.db_path) as conn:
            conn.execute(
                "UPDATE books SET {0} WHERE id = ?".format(", ".join(fields)), values
            )

    def delete_book(self, book_id: str) -> bool:
        """Remove a book and everything attached to it. True when it existed."""
        with session(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM books WHERE id = ?", (book_id,))
            conn.execute("DELETE FROM chapters WHERE book_id = ?", (book_id,))
            conn.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
            conn.execute("DELETE FROM vocab WHERE book_id = ?", (book_id,))
            deleted = cursor.rowcount > 0
        self.invalidate(book_id)
        return deleted

    # ------------------------------------------------------------- chapters

    def replace_chapters(self, book_id: str, structure: BookStructure) -> None:
        rows = [
            (book_id, chapter.idx, chapter.title, chapter.page_start, chapter.page_end)
            for chapter in structure.chapters
        ]
        with session(self.db_path) as conn:
            conn.execute("DELETE FROM chapters WHERE book_id = ?", (book_id,))
            conn.executemany(
                "INSERT INTO chapters (book_id, idx, title, page_start, page_end)"
                " VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def get_chapters(self, book_id: str) -> List[ChapterRecord]:
        with session(self.db_path) as conn:
            rows = conn.execute(
                "SELECT idx, title, page_start, page_end FROM chapters"
                " WHERE book_id = ? ORDER BY idx",
                (book_id,),
            ).fetchall()
        return [
            ChapterRecord(
                idx=row["idx"],
                title=row["title"],
                page_start=row["page_start"],
                page_end=row["page_end"],
            )
            for row in rows
        ]

    # --------------------------------------------------------------- chunks

    def replace_chunks(
        self,
        book_id: str,
        chunks: Sequence[Chunk],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        """Write a book's chunks, replacing anything previously stored."""
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("embeddings must line up with chunks")
        rows = []
        for position, chunk in enumerate(chunks):
            blob = (
                encode_embedding(embeddings[position]) if embeddings is not None else None
            )
            rows.append(
                (
                    chunk.id,
                    book_id,
                    chunk.chapter_idx,
                    chunk.section_title,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.text,
                    chunk.header,
                    chunk.parent_text,
                    blob,
                    chunk.ord,
                    getattr(chunk, "kind", KIND_TEXT),
                    int(getattr(chunk, "level", 0)),
                )
            )
        with session(self.db_path) as conn:
            conn.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
            conn.executemany(
                "INSERT INTO chunks (id, book_id, chapter_idx, section_title,"
                " page_start, page_end, text, header, parent_text, embedding, ord,"
                " kind, level)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        self.invalidate(book_id)

    def count_chunks(self, book_id: str) -> int:
        with session(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE book_id = ?", (book_id,)
            ).fetchone()
        return int(row["n"]) if row else 0

    # ----------------------------------------------------------- vocabulary

    def replace_vocabulary(
        self, book_id: str, terms: Sequence[Tuple[str, int]]
    ) -> None:
        """Store the book's proper nouns in rank order (SPEC §14.9)."""
        rows = [
            (book_id, term, int(count), rank)
            for rank, (term, count) in enumerate(terms)
            if term
        ]
        with session(self.db_path) as conn:
            conn.execute("DELETE FROM vocab WHERE book_id = ?", (book_id,))
            conn.executemany(
                "INSERT INTO vocab (book_id, term, count, ord)"
                " VALUES (?, ?, ?, ?)",
                rows,
            )

    def get_vocabulary(self, book_id: str, limit: Optional[int] = None) -> List[str]:
        """The book's proper nouns, most frequent first."""
        sql = "SELECT term FROM vocab WHERE book_id = ? ORDER BY ord"
        params: List[object] = [book_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with session(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [row["term"] for row in rows]

    # ---------------------------------------------------------- book bundle

    def load_index(self, book_id: str) -> BookIndex:
        """Return the runtime bundle for a book, building it on first use."""
        with self._lock:
            cached = self._cache.get(book_id)
            if cached is not None:
                self._cache.move_to_end(book_id)
                return cached
        bundle = self._build_index(book_id)
        with self._lock:
            self._cache[book_id] = bundle
            self._cache.move_to_end(book_id)
            while len(self._cache) > MAX_CACHED_BOOKS:
                self._cache.popitem(last=False)
        return bundle

    def invalidate(self, book_id: str) -> None:
        """Drop a book's cached bundle (after re-ingestion or deletion)."""
        with self._lock:
            self._cache.pop(book_id, None)

    def get_summaries(self, book_id: str, kind: str = KIND_CHAPTER_SUMMARY) -> List[ChunkRecord]:
        """The stored synopsis nodes for a book (SPEC §14.5).

        The chapter synopses double as TOC blurbs in the UI, so they are exposed
        directly and not only through retrieval.
        """
        titles = self._chapter_titles(book_id)
        with session(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, book_id, chapter_idx, section_title, page_start, page_end,"
                " text, header, parent_text, ord, kind, level FROM chunks"
                " WHERE book_id = ? AND kind = ? ORDER BY ord",
                (book_id, kind),
            ).fetchall()
        return [_chunk_from_row(row, titles) for row in rows]

    def get_chapter_openings(
        self, book_id: str, chars: int = 900
    ) -> Dict[int, str]:
        """The first body passage of each chapter, by chapter index.

        The reader shows these on the fanned chapter pages, so it is the book's
        own prose rather than a generated blurb. Summary nodes are excluded —
        they are the synopsis, not the opening.
        """
        titles = self._chapter_titles(book_id)
        with session(self.db_path) as conn:
            rows = conn.execute(
                "SELECT chapter_idx, text FROM chunks"
                " WHERE book_id = ? AND kind = ? ORDER BY ord",
                (book_id, KIND_TEXT),
            ).fetchall()
        openings: Dict[int, str] = {}
        for row in rows:
            idx = row["chapter_idx"]
            if idx in openings:
                continue
            text = _flatten(row["text"] or "")
            # A title page or a stray fragment is not the chapter's opening;
            # the first chapter of a book routinely starts behind one.
            if not _substantial(text):
                continue
            # The heading opens the first chunk beneath it, so it has to come
            # off here. Both sides go through _flatten first because the stored
            # title keeps the book's own separators — an ideographic space in a
            # Chinese chapter title — which the flattened body no longer has.
            title = _flatten(titles.get(idx) or "")
            if title and text.lower().startswith(title.lower()):
                text = text[len(title) :].lstrip(" .—-–:　")
            if text:
                openings[idx] = text[:chars].rstrip()
        return openings

    def _chapter_titles(self, book_id: str) -> Dict[int, str]:
        """Chapter titles by index, plus the book title under index 0.

        The book-level summary node (SPEC §14.5) belongs to no chapter and is
        stored under index 0; labelling it with the book's own title is what a
        reader expects to see on its source chip.
        """
        titles = {chapter.idx: chapter.title for chapter in self.get_chapters(book_id)}
        book = self.get_book(book_id)
        titles.setdefault(0, book.title if book else "")
        return titles

    def _build_index(self, book_id: str) -> BookIndex:
        titles = self._chapter_titles(book_id)
        with session(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, book_id, chapter_idx, section_title, page_start, page_end,"
                " text, header, parent_text, embedding, ord, kind, level FROM chunks"
                " WHERE book_id = ? ORDER BY ord",
                (book_id,),
            ).fetchall()
        chunks: List[ChunkRecord] = []
        vectors: List[np.ndarray] = []
        for row in rows:
            chunks.append(_chunk_from_row(row, titles))
            vectors.append(decode_embedding(row["embedding"]))
        # A chunk whose embedding is missing or of the wrong width becomes a zero
        # row: it can still be found by BM25, it just never wins on cosine.
        dims = max((int(vector.shape[0]) for vector in vectors), default=0)
        rows_of_floats = [
            vector.tolist() if int(vector.shape[0]) == dims else [0.0] * dims
            for vector in vectors
        ]
        matrix = normalize(to_matrix(rows_of_floats, dims=dims))
        return BookIndex(
            book_id=book_id,
            chunks=chunks,
            matrix=matrix,
            bm25=BM25Index([chunk.header for chunk in chunks]),
        )


def _chunk_from_row(row: sqlite3.Row, titles: Dict[int, str]) -> ChunkRecord:
    """Build a :class:`ChunkRecord`, joining in its chapter title."""
    keys = row.keys()
    return ChunkRecord(
        id=row["id"],
        book_id=row["book_id"],
        chapter_idx=row["chapter_idx"],
        chapter_title=titles.get(row["chapter_idx"], ""),
        section_title=row["section_title"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        text=row["text"],
        header=row["header"],
        parent_text=row["parent_text"],
        ord=row["ord"],
        kind=row["kind"] if "kind" in keys else KIND_TEXT,
        level=int(row["level"] or 0) if "level" in keys else 0,
    )


def encode_embedding(vector: Sequence[float]) -> bytes:
    """Pack a vector into the float32 BLOB stored in ``chunks.embedding``."""
    return np.asarray(vector, dtype=np.float32).tobytes()


def decode_embedding(blob: Optional[bytes]) -> np.ndarray:
    """Unpack a float32 BLOB (missing embeddings decode to an empty vector)."""
    if not blob:
        return np.zeros((0,), dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32)


def _book_from_row(row: sqlite3.Row) -> BookRecord:
    return BookRecord(
        id=row["id"],
        title=row["title"],
        filename=row["filename"],
        sha256=row["sha256"],
        pages=row["pages"],
        status=row["status"],
        progress=float(row["progress"] or 0.0),
        error=row["error"],
        created_at=row["created_at"],
    )
