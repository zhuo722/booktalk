"""Book endpoints: upload, list, status, toc, delete (SPEC §6)."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile, status

from app.api import get_settings, get_store
from app.config import Settings
from app.index.store import (
    STATUS_FAILED,
    STATUS_PARSING,
    STATUS_READY,
    BookRecord,
    ChapterRecord,
    Store,
)
from app.ingest.ingestor import start_ingestion
from app.models import (
    BookListResponse,
    BookSummary,
    StatusResponse,
    TocChapter,
    TocResponse,
    UploadResponse,
)

router = APIRouter(prefix="/api/books", tags=["books"])

PDF_MAGIC = b"%PDF"
READ_CHUNK = 1024 * 1024
NOT_A_PDF = "Only PDF files are accepted."
CORRUPT_PDF = "That file is not a valid PDF (missing %PDF header)."
UNKNOWN_BOOK = "Unknown book."


@router.post("", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
def upload_book(
    request: Request, response: Response, file: UploadFile = File(...)
) -> UploadResponse:
    """Accept a PDF and start ingesting it in the background.

    Uploading a file that is already ingested is a no-op: the same content
    (sha256) returns the existing book with 200 instead of re-ingesting it.
    """
    settings = get_settings(request)
    store = get_store(request)
    filename = (file.filename or "book.pdf").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=NOT_A_PDF)

    temp_path, digest = _spool_upload(file, settings)
    try:
        existing = store.get_book_by_sha256(digest)
        if existing is not None and existing.status != STATUS_FAILED:
            response.status_code = status.HTTP_200_OK
            return UploadResponse(book_id=existing.id, status=existing.status)

        book_id = existing.id if existing is not None else uuid.uuid4().hex
        target = settings.books_dir / "{0}.pdf".format(book_id)
        shutil.move(str(temp_path), str(target))
    finally:
        if temp_path.exists():
            temp_path.unlink()

    if existing is None:
        store.create_book(
            book_id=book_id,
            title=Path(filename).stem or "Untitled",
            filename=filename,
            sha256=digest,
            status=STATUS_PARSING,
        )
    else:
        store.update_book(book_id, status=STATUS_PARSING, progress=0.0)

    start_ingestion(store, request.app.state.ai, book_id, target, filename)
    return UploadResponse(book_id=book_id, status=STATUS_PARSING)


@router.get("", response_model=BookListResponse)
def list_books(request: Request) -> BookListResponse:
    """Every uploaded book, newest first."""
    store = get_store(request)
    return BookListResponse(
        books=[
            BookSummary(
                book_id=record.id,
                title=record.title,
                pages=record.pages,
                status=record.status,
                progress=record.progress,
                created_at=record.created_at,
            )
            for record in store.list_books()
        ]
    )


@router.get("/{book_id}/status", response_model=StatusResponse)
def book_status(request: Request, book_id: str) -> StatusResponse:
    """Ingestion stage and progress, polled by the library card."""
    record = _require_book(get_store(request), book_id)
    return StatusResponse(
        status=record.status, progress=record.progress, error=record.error
    )


@router.get(
    "/{book_id}/toc", response_model=TocResponse, response_model_exclude_unset=True
)
def book_toc(request: Request, book_id: str) -> TocResponse:
    """The detected chapter list, each entry carrying its synopsis when it has one.

    The chapter summary nodes of SPEC §14.5 are written for retrieval, but they
    are also the only prose description of a chapter this app ever has, so the
    TOC hands them to the UI directly. Summarizing is skippable (no key, an LLM
    failure, a chapter too short to summarize), which is why the field is
    per-chapter and optional rather than a promise the response makes.
    """
    store = get_store(request)
    record = _require_book(store, book_id)
    synopses = {
        node.chapter_idx: node.text.strip()
        for node in store.get_summaries(book_id)
        if node.text.strip()
    }
    openings = store.get_chapter_openings(book_id)
    return TocResponse(
        title=record.title,
        chapters=[
            _toc_chapter(
                chapter, synopses.get(chapter.idx), openings.get(chapter.idx)
            )
            for chapter in store.get_chapters(book_id)
        ],
    )


@router.delete("/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_book(request: Request, book_id: str) -> Response:
    """Delete a book, its chunks and its stored PDF."""
    settings = get_settings(request)
    store = get_store(request)
    _require_book(store, book_id)
    store.delete_book(book_id)
    pdf_path = settings.books_dir / "{0}.pdf".format(book_id)
    if pdf_path.exists():
        pdf_path.unlink()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _spool_upload(file: UploadFile, settings: Settings) -> Tuple[Path, str]:
    """Stream the upload to a temp file, enforcing size and magic bytes.

    Returns the temp path and the content sha256. The file is read in 1 MB
    slices so a 200 MB book never sits in memory in one piece.
    """
    settings.ensure_dirs()
    digest = hashlib.sha256()
    total = 0
    first_block = b""
    handle = tempfile.NamedTemporaryFile(
        delete=False, dir=str(settings.books_dir), suffix=".part"
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            while True:
                block = file.file.read(READ_CHUNK)
                if not block:
                    break
                if not first_block:
                    first_block = block[: len(PDF_MAGIC)]
                total += len(block)
                if total > settings.max_upload_bytes:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="File is larger than the {0} MB limit.".format(
                            settings.max_upload_mb
                        ),
                    )
                digest.update(block)
                handle.write(block)
        if not first_block.startswith(PDF_MAGIC):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=CORRUPT_PDF)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path, digest.hexdigest()


def _toc_chapter(
    chapter: ChapterRecord,
    synopsis: Optional[str],
    opening: Optional[str] = None,
) -> TocChapter:
    """One TOC entry, whose ``synopsis`` key exists only when there is one.

    Left unset rather than set to ``None``: paired with the route's
    ``response_model_exclude_unset``, that is what makes the field absent from
    the JSON instead of null, while ``page_start``/``page_end`` — always passed,
    sometimes null for a book with no page numbers — keep their keys.
    """
    fields: Dict[str, object] = {
        "idx": chapter.idx,
        "title": chapter.title,
        "page_start": chapter.page_start,
        "page_end": chapter.page_end,
    }
    if synopsis:
        fields["synopsis"] = synopsis
    if opening:
        fields["opening"] = opening
    return TocChapter(**fields)


def _require_book(store: Store, book_id: str) -> BookRecord:
    record = store.get_book(book_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=UNKNOWN_BOOK)
    return record


def ready_or_error(record: BookRecord) -> Optional[str]:
    """Reason a book cannot answer questions yet, or ``None`` when ready."""
    if record.status == STATUS_READY:
        return None
    if record.status == STATUS_FAILED:
        return record.error or "Ingestion of this book failed."
    return "This book is still being processed (status: {0}).".format(record.status)
