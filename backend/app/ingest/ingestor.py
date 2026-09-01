"""Ingestion orchestration: parse → structure → extract → chunk → enrich → embed.

Runs on a background thread (SPEC §4.4, extended by §14.1–§14.5) and reports
progress into the ``books`` row, which ``GET /api/books/{id}/status`` polls.
Progress is a single 0…1 float split across the stages so the UI's bar always
moves forward:

    parsing 0.00–0.25 · structuring 0.25 · chunking 0.30
    contextualizing 0.35–0.60 · summarizing 0.60–0.70 · embedding 0.70–1.00

The two enrichment stages are skipped when they are switched off or when there
is no client to run them, and the bar simply jumps — a stage that did not happen
should not fake a duration.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from app.config import contextual_enabled
from app.index.store import (
    STATUS_CHUNKING,
    STATUS_CONTEXTUALIZING,
    STATUS_EMBEDDING,
    STATUS_FAILED,
    STATUS_PARSING,
    STATUS_READY,
    STATUS_STRUCTURING,
    STATUS_SUMMARIZING,
    Store,
)
from app.ingest.chunker import Chunk, chunk_book, naive_chunks
from app.ingest.enrich import contextualize, summary_chunks
from app.ingest.markdown import extract_blocks
from app.ingest.pdf import (
    MIN_CHARS_PER_PAGE,
    PageText,
    PdfDocument,
    TocEntry,
    chars_per_page,
)
from app.ingest.structure import BookStructure, detect_structure
from app.ingest.vocab import vocabulary_of
from app.services.openai_client import AIClient, batched

logger = logging.getLogger(__name__)

EMBED_BATCH = 128
PARSE_PROGRESS_SHARE = 0.25
PARSE_PROGRESS_EVERY = 25
STRUCTURE_PROGRESS = 0.25
CHUNK_PROGRESS = 0.30
CONTEXT_PROGRESS_START = 0.35
SUMMARY_PROGRESS_START = 0.60
EMBED_PROGRESS_START = 0.70
PROGRESS_EVERY = 16

MODE_FULL = "full"
MODE_NAIVE = "naive"

NO_KEY_MESSAGE = (
    "OPENAI_API_KEY is not configured, so chunk embeddings cannot be computed. "
    "Set it in backend/.env and re-upload the book."
)
NO_TEXT_MESSAGE = (
    "No extractable text found in this PDF. Scanned books need OCR first."
)
SCANNED_MESSAGE = (
    "This PDF has no text layer — it looks like a scan of printed pages "
    "({0:.0f} characters per page over the first pages, a readable book has "
    "well over {1:.0f}). Run OCR on it first; BookTalk does not do OCR."
)
UNREADABLE_MESSAGE = "This PDF could not be opened; the file looks corrupted."


class IngestionError(RuntimeError):
    """Raised when a book cannot be ingested; the message reaches the UI."""


@dataclass
class IngestResult:
    """What one ingestion produced."""

    book_id: str
    title: str
    pages: int
    chapters: int
    chunks: int
    summaries: int = 0
    contextualized: int = 0
    structure_source: str = ""
    #: Proper nouns stored for ASR biasing and repair (SPEC §14.9).
    vocabulary: int = 0


def ingest_book(
    store: Store,
    ai: Optional[AIClient],
    book_id: str,
    pdf_path: Union[str, Path],
    filename: str,
    mode: str = MODE_FULL,
    embed_batch: int = EMBED_BATCH,
    contextual: Optional[bool] = None,
    summaries: bool = True,
    model: Optional[str] = None,
) -> IngestResult:
    """Ingest one PDF, updating its ``books`` row as it goes.

    ``mode`` selects the chunking strategy: ``"full"`` (this spec) or
    ``"naive"`` (the 800-character ablation baseline used by the eval harness).
    ``contextual`` defaults to ``BOOKTALK_CONTEXTUAL`` (on).
    Raises :class:`IngestionError` after marking the book failed.
    """
    use_context = contextual_enabled() if contextual is None else contextual
    try:
        if ai is None:
            raise IngestionError(NO_KEY_MESSAGE)

        store.update_book(book_id, status=STATUS_PARSING, progress=0.0)
        pages, page_count, title, toc = _parse(store, book_id, pdf_path, filename)
        _preflight(pages)

        store.update_book(
            book_id,
            status=STATUS_STRUCTURING,
            progress=STRUCTURE_PROGRESS,
            pages=page_count,
            title=title,
        )
        structure = detect_structure(
            pages, toc, title=title, page_count=page_count, ai=ai, model=model
        )
        store.replace_chapters(book_id, structure)

        store.update_book(book_id, status=STATUS_CHUNKING, progress=CHUNK_PROGRESS)
        chunks = _chunks(pages, structure, book_id, pdf_path, title, mode)
        if not chunks:
            raise IngestionError(NO_TEXT_MESSAGE)

        enriched = 0
        if mode == MODE_FULL and use_context:
            enriched = _contextualize(store, book_id, ai, chunks, title, model)
        nodes: List[Chunk] = []
        if mode == MODE_FULL and summaries:
            nodes = _summarize(store, book_id, ai, chunks, structure, title, model)
        indexed = list(chunks) + nodes

        embeddings = _embed(store, book_id, ai, indexed, embed_batch)
        store.replace_chunks(book_id, indexed, embeddings)
        # The book's own names, harvested from the leaf chunks only: summary
        # nodes are the model's prose, not the author's, and a hallucinated
        # name must never end up in the vocabulary we repair transcripts
        # against (SPEC §14.9).
        vocabulary = vocabulary_of(chunks)
        store.replace_vocabulary(book_id, vocabulary)
        store.update_book(book_id, status=STATUS_READY, progress=1.0)
        return IngestResult(
            book_id=book_id,
            title=title,
            pages=page_count,
            chapters=len(structure.chapters),
            chunks=len(indexed),
            summaries=len(nodes),
            contextualized=enriched,
            structure_source=structure.source,
            vocabulary=len(vocabulary),
        )
    except Exception as exc:  # noqa: BLE001 - the reason is surfaced to the user
        message = str(exc) or exc.__class__.__name__
        store.update_book(book_id, status=STATUS_FAILED, progress=0.0, error=message)
        raise


def _parse(
    store: Store, book_id: str, pdf_path: Union[str, Path], filename: str
) -> Tuple[List[PageText], int, str, List[TocEntry]]:
    """Read every page, reporting progress; returns pages, count, title, toc."""
    pages: List[PageText] = []
    try:
        document = PdfDocument(pdf_path)
    except Exception as exc:  # noqa: BLE001 - PyMuPDF errors name internal paths
        raise IngestionError(UNREADABLE_MESSAGE) from exc
    with document as doc:
        page_count = doc.page_count
        title = doc.metadata_title or Path(filename).stem or "Untitled"
        for page in doc.iter_pages():
            pages.append(page)
            if page.page_no % PARSE_PROGRESS_EVERY == 0 and page_count:
                store.update_book(
                    book_id,
                    status=STATUS_PARSING,
                    progress=PARSE_PROGRESS_SHARE * page.page_no / page_count,
                )
        toc = doc.toc()
    return pages, page_count, title, toc


def _preflight(pages: Sequence[PageText]) -> None:
    """Reject an image-only PDF before spending a single model call (§14.1)."""
    density = chars_per_page(pages)
    if density < MIN_CHARS_PER_PAGE:
        raise IngestionError(SCANNED_MESSAGE.format(density, MIN_CHARS_PER_PAGE))


def _chunks(
    pages: Sequence[PageText],
    structure: BookStructure,
    book_id: str,
    pdf_path: Union[str, Path],
    title: str,
    mode: str,
) -> List[Chunk]:
    """Chunk the book, either through the markdown layer or the naive baseline."""
    if mode == MODE_NAIVE:
        return naive_chunks(pages, structure, book_id)
    blocks = extract_blocks(pdf_path, structure, pages=pages)
    return chunk_book(blocks, structure, book_id, book_title=title)


def _contextualize(
    store: Store,
    book_id: str,
    ai: AIClient,
    chunks: Sequence[Chunk],
    title: str,
    model: Optional[str],
) -> int:
    """Per-chunk situating context, with the progress bar following along."""
    store.update_book(
        book_id, status=STATUS_CONTEXTUALIZING, progress=CONTEXT_PROGRESS_START
    )
    span = SUMMARY_PROGRESS_START - CONTEXT_PROGRESS_START

    def progress(done: int, total: int) -> None:
        if total and (done % PROGRESS_EVERY == 0 or done == total):
            store.update_book(
                book_id,
                status=STATUS_CONTEXTUALIZING,
                progress=CONTEXT_PROGRESS_START + span * done / total,
            )

    return contextualize(ai, chunks, book_title=title, model=model, progress=progress)


def _summarize(
    store: Store,
    book_id: str,
    ai: AIClient,
    chunks: Sequence[Chunk],
    structure: BookStructure,
    title: str,
    model: Optional[str],
) -> List[Chunk]:
    """One synopsis per chapter plus one for the book (SPEC §14.5)."""
    store.update_book(
        book_id, status=STATUS_SUMMARIZING, progress=SUMMARY_PROGRESS_START
    )
    span = EMBED_PROGRESS_START - SUMMARY_PROGRESS_START

    def progress(done: int, total: int) -> None:
        if total:
            store.update_book(
                book_id,
                status=STATUS_SUMMARIZING,
                progress=SUMMARY_PROGRESS_START + span * done / total,
            )

    return summary_chunks(
        ai,
        chunks,
        structure,
        book_id,
        book_title=title,
        model=model,
        progress=progress,
        first_ord=len(chunks),
    )


def _embed(
    store: Store,
    book_id: str,
    ai: AIClient,
    chunks: Sequence[Chunk],
    embed_batch: int,
) -> List[Sequence[float]]:
    """Embed every indexed document (leaf chunks and summary nodes alike)."""
    store.update_book(book_id, status=STATUS_EMBEDDING, progress=EMBED_PROGRESS_START)
    embeddings: List[Sequence[float]] = []
    for batch in batched([chunk.header for chunk in chunks], embed_batch):
        embeddings.extend(ai.embed(batch))
        store.update_book(
            book_id,
            status=STATUS_EMBEDDING,
            progress=EMBED_PROGRESS_START
            + (1.0 - EMBED_PROGRESS_START) * len(embeddings) / max(1, len(chunks)),
        )
    return embeddings


def start_ingestion(
    store: Store,
    ai: Optional[AIClient],
    book_id: str,
    pdf_path: Union[str, Path],
    filename: str,
    mode: str = MODE_FULL,
    contextual: Optional[bool] = None,
) -> threading.Thread:
    """Run :func:`ingest_book` on a daemon thread and return it."""

    def run() -> None:
        try:
            ingest_book(
                store, ai, book_id, pdf_path, filename, mode=mode, contextual=contextual
            )
        except Exception:  # noqa: BLE001 - already recorded on the book row
            logger.exception("ingestion failed for book %s", book_id)

    thread = threading.Thread(
        target=run, name="ingest-{0}".format(book_id[:8]), daemon=True
    )
    thread.start()
    return thread


__all__ = [
    "EMBED_BATCH",
    "IngestResult",
    "IngestionError",
    "MODE_FULL",
    "MODE_NAIVE",
    "NO_TEXT_MESSAGE",
    "SCANNED_MESSAGE",
    "ingest_book",
    "start_ingestion",
]
