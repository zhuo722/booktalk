#!/usr/bin/env python3
"""Build the seed library that the hosted demo boots with.

Copies the local database (backend/data/booktalk.db) keeping only the books
whose PDF is byte-identical to one in examples/books/, and writes it with
those PDFs to deploy/seed/. The Docker entrypoint copies the seed into an
empty data directory, so a fresh container serves five ready, already-indexed
books without spending any API credit on ingestion.

Ingest the example books through the app first, then run:

    python3 scripts/make_demo_seed.py
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DB = ROOT / "backend" / "data" / "booktalk.db"
SOURCE_BOOKS = ROOT / "backend" / "data" / "books"
EXAMPLES = ROOT / "examples" / "books"
OUT = ROOT / "deploy" / "seed"
TABLES_WITH_BOOK_ID = ("chapters", "chunks", "vocab")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if not SOURCE_DB.is_file():
        print("no database at {0}; ingest the example books first".format(SOURCE_DB))
        return 1
    wanted = {sha256(pdf) for pdf in EXAMPLES.glob("*.pdf")}

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "books").mkdir(parents=True)
    target_db = OUT / "booktalk.db"

    with sqlite3.connect(str(SOURCE_DB)) as src, sqlite3.connect(str(target_db)) as dst:
        src.backup(dst)
    conn = sqlite3.connect(str(target_db))
    rows = conn.execute("SELECT id, title, sha256, status FROM books").fetchall()
    keep = [(i, t) for i, t, digest, status in rows if digest in wanted and status == "ready"]
    keep_ids = {book_id for book_id, _ in keep}
    drop = [row[0] for row in rows if row[0] not in keep_ids]
    for book_id in drop:
        conn.execute("DELETE FROM books WHERE id = ?", (book_id,))
        for table in TABLES_WITH_BOOK_ID:
            conn.execute("DELETE FROM {0} WHERE book_id = ?".format(table), (book_id,))
    conn.commit()
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("VACUUM")
    conn.close()

    for book_id, title in keep:
        shutil.copy2(SOURCE_BOOKS / "{0}.pdf".format(book_id), OUT / "books")
        print("kept  {0}  {1}".format(book_id, title))
    missing = len(wanted) - len(keep)
    if missing:
        print("warning: {0} example book(s) are not ingested locally".format(missing))
    size_mb = target_db.stat().st_size / 1e6
    print("wrote {0} ({1:.1f} MB, {2} books)".format(OUT, size_mb, len(keep)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
