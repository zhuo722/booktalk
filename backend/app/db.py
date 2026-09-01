"""SQLite helpers: connection factory, schema DDL and migrate-on-start.

Connections are created per call (never shared across threads) because ingestion
runs on a background thread while the API serves requests; WAL mode plus a busy
timeout is enough concurrency for a single-process local app.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS books (
        id         TEXT PRIMARY KEY,
        title      TEXT NOT NULL,
        filename   TEXT NOT NULL,
        sha256     TEXT UNIQUE,
        pages      INTEGER,
        status     TEXT NOT NULL,
        progress   REAL NOT NULL DEFAULT 0,
        error      TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chapters (
        book_id    TEXT NOT NULL,
        idx        INTEGER NOT NULL,
        title      TEXT NOT NULL,
        page_start INTEGER,
        page_end   INTEGER,
        PRIMARY KEY (book_id, idx)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id            TEXT PRIMARY KEY,
        book_id       TEXT NOT NULL,
        chapter_idx   INTEGER NOT NULL,
        section_title TEXT,
        page_start    INTEGER,
        page_end      INTEGER,
        text          TEXT NOT NULL,
        header        TEXT NOT NULL,
        parent_text   TEXT NOT NULL,
        embedding     BLOB,
        ord           INTEGER NOT NULL,
        kind          TEXT NOT NULL DEFAULT 'text',
        level         INTEGER NOT NULL DEFAULT 0
    )
    """,
    # The book's proper nouns (SPEC §14.9), ranked at ingest: an ASR decoding
    # hint and the target set for phonetic repair. ``ord`` is the rank, so the
    # top N can be read without re-sorting by count.
    """
    CREATE TABLE IF NOT EXISTS vocab (
        book_id TEXT NOT NULL,
        term    TEXT NOT NULL,
        count   INTEGER NOT NULL DEFAULT 0,
        ord     INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (book_id, term)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_book ON chunks(book_id, ord)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_kind ON chunks(book_id, kind)",
    "CREATE INDEX IF NOT EXISTS idx_books_sha ON books(sha256)",
    "CREATE INDEX IF NOT EXISTS idx_vocab_rank ON vocab(book_id, ord)",
)

# Columns added after the first release, applied to databases created before
# they existed (SPEC §14.5 tags summary nodes with a kind and a level).
ADDED_COLUMNS = (
    ("chunks", "kind", "TEXT NOT NULL DEFAULT 'text'"),
    ("chunks", "level", "INTEGER NOT NULL DEFAULT 0"),
)


def connect(db_path: Union[str, Path]) -> sqlite3.Connection:
    """Open a connection with row access by name and WAL journaling."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def session(db_path: Union[str, Path]) -> Iterator[sqlite3.Connection]:
    """Transactional connection: commits on success, rolls back on error."""
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Union[str, Path]) -> None:
    """Create the schema if needed. Safe to call on every app start.

    Order matters: tables first, then column migrations, then indexes —
    an index over a migrated column (idx_chunks_kind) would otherwise fail
    against a database created before that column existed.
    """
    with session(db_path) as conn:
        for statement in SCHEMA_STATEMENTS:
            if not statement.lstrip().startswith("CREATE INDEX"):
                conn.execute(statement)
        _add_missing_columns(conn)
        for statement in SCHEMA_STATEMENTS:
            if statement.lstrip().startswith("CREATE INDEX"):
                conn.execute(statement)


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Migrate-on-start: add columns a previously created database lacks."""
    for table, column, definition in ADDED_COLUMNS:
        existing = {
            row["name"] for row in conn.execute("PRAGMA table_info({0})".format(table))
        }
        if column not in existing:
            conn.execute(
                "ALTER TABLE {0} ADD COLUMN {1} {2}".format(table, column, definition)
            )
