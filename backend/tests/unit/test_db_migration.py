"""Migrate-on-start against a database created before kind/level existed."""

import sqlite3

from app.db import init_db


def _v1_chunks_table(db_path):
    """Create a chunks table shaped like the first release (no kind/level)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE chunks (
            id          TEXT PRIMARY KEY,
            book_id     TEXT NOT NULL,
            chapter_idx INTEGER NOT NULL,
            section_title TEXT,
            page_start  INTEGER NOT NULL,
            page_end    INTEGER NOT NULL,
            text        TEXT NOT NULL,
            header      TEXT NOT NULL,
            parent_text TEXT NOT NULL,
            embedding   BLOB,
            ord         INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def test_init_db_migrates_a_v1_database(tmp_path):
    db_path = tmp_path / "old.db"
    _v1_chunks_table(db_path)

    init_db(db_path)  # must not raise on the pre-kind schema

    conn = sqlite3.connect(str(db_path))
    columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(chunks)")}
    conn.close()
    assert {"kind", "level"} <= columns
    assert "idx_chunks_kind" in indexes


def test_init_db_still_creates_a_fresh_database(tmp_path):
    db_path = tmp_path / "fresh.db"
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
    conn.close()
    assert {"kind", "level"} <= columns
