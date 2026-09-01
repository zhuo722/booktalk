"""LLM enrichment at ingest time (SPEC §14.4, §14.5).

Two passes over the chunks a book produced, both optional and both behind the
:class:`~app.services.openai_client.AIClient` seam:

* **contextual enrichment** (§14.4) — Anthropic's contextual retrieval: a short
  LLM-written sentence saying where each chunk sits, prepended to what gets
  embedded and BM25-indexed (published effect: 5.7% → 2.9% retrieval failures).
  The prompt is scoped to the chunk's *chapter*, not the whole book, which is
  what makes it affordable: one chapter of context per call instead of one book.
  With ``BOOKTALK_CONTEXTUAL=0`` the static breadcrumb header stays in place —
  that is the ablation arm the eval compares against.

* **summary nodes** (§14.5, RAPTOR-lite) — one synopsis per chapter plus one for
  the book, indexed as ordinary retrievable chunks but tagged with a ``kind``
  and a ``level``. They answer the thematic questions ("what is chapter 7
  about") that flat chunks provably miss, and double as TOC synopses in the UI.
  Full RAPTOR clustering was rejected: the chapter tree already exists.

Calls run concurrently through a small thread pool — they are IO-bound, and a
600-chunk book would otherwise serialize ten minutes of round trips.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence

from app.ingest.chunker import (
    KIND_BOOK_SUMMARY,
    KIND_CHAPTER_SUMMARY,
    LEVEL_BOOK_SUMMARY,
    LEVEL_CHAPTER_SUMMARY,
    Chunk,
    build_header,
    count_tokens,
)
from app.ingest.prompts import (
    BOOK_SUMMARY_LABEL,
    BOOK_SUMMARY_SYSTEM,
    BOOK_SUMMARY_USER_TEMPLATE,
    CHAPTER_SUMMARY_LABEL,
    CHAPTER_SUMMARY_SYSTEM,
    CHAPTER_SUMMARY_USER_TEMPLATE,
    CONTEXT_SYSTEM,
    CONTEXT_USER_TEMPLATE,
)
from app.ingest.structure import BookStructure

logger = logging.getLogger(__name__)

CONTEXT_WORKERS = 8
CHAPTER_CONTEXT_TOKENS = 1500
CHAPTER_SUMMARY_TOKENS = 3000
BOOK_SUMMARY_TOKENS = 3000
CONTEXT_TEMPLATE = "{context}\n\n{text}"
BOOK_SUMMARY_CHAPTER_IDX = 0

Progress = Optional[Callable[[int, int], None]]


class _Counter:
    """Thread-safe "n of m done", so the progress bar never runs backwards."""

    def __init__(self, total: int, progress: Progress) -> None:
        self.total = total
        self._progress = progress
        self._done = 0
        self._lock = threading.Lock()

    def tick(self) -> int:
        """Count one unit of work and report it, both under the same lock.

        A monotonic counter is not enough: two workers finishing together take
        6 and 7, then race into the callback in either order, and the poller
        sees 7-of-8 followed by 6-of-8. Reporting inside the lock is what makes
        the promise in the class docstring true.
        """
        with self._lock:
            self._done += 1
            done = self._done
            if self._progress is not None:
                self._progress(done, self.total)
        return done


def chapter_texts(chunks: Sequence[Chunk], max_tokens: int) -> Dict[int, str]:
    """One capped text blob per chapter, assembled from its own chunks."""
    collected: Dict[int, List[str]] = {}
    for chunk in chunks:
        collected.setdefault(chunk.chapter_idx, []).append(chunk.text)
    return {
        idx: _truncate("\n\n".join(texts), max_tokens)
        for idx, texts in collected.items()
    }


def contextualize(
    ai: object,
    chunks: Sequence[Chunk],
    book_title: str = "",
    model: Optional[str] = None,
    workers: int = CONTEXT_WORKERS,
    progress: Progress = None,
) -> int:
    """Rewrite each chunk's indexed text with an LLM-written situating context.

    Returns how many chunks were successfully enriched. A chunk whose call fails
    keeps its breadcrumb header, so a flaky minute during ingestion degrades the
    index rather than failing the book.
    """
    targets = [chunk for chunk in chunks if chunk.text.strip()]
    if not targets or ai is None:
        return 0
    chapters = chapter_texts(targets, CHAPTER_CONTEXT_TOKENS)
    counter = _Counter(len(targets), progress)

    def enrich(chunk: Chunk) -> bool:
        try:
            context = _context_for(ai, chunk, chapters, model)
        except Exception:  # noqa: BLE001 - the breadcrumb header still works
            logger.warning("contextual enrichment failed for %s", chunk.id, exc_info=True)
            context = ""
        if context:
            chunk.context = context
            chunk.header = CONTEXT_TEMPLATE.format(context=context, text=chunk.text)
        counter.tick()
        return bool(context)

    if workers <= 1:
        return sum(int(enrich(chunk)) for chunk in targets)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(int(result) for result in pool.map(enrich, targets))


def _context_for(
    ai: object, chunk: Chunk, chapters: Dict[int, str], model: Optional[str]
) -> str:
    messages = [
        {"role": "system", "content": CONTEXT_SYSTEM},
        {
            "role": "user",
            "content": CONTEXT_USER_TEMPLATE.format(
                chapter_title=chunk.chapter_title or "Untitled chapter",
                chapter_text=chapters.get(chunk.chapter_idx, chunk.text),
                chunk_text=chunk.text,
            ),
        },
    ]
    reply = ai.chat(messages, purpose="context", temperature=0, model=model)
    return " ".join((reply or "").split())


def summary_chunks(
    ai: object,
    chunks: Sequence[Chunk],
    structure: BookStructure,
    book_id: str,
    book_title: str = "",
    model: Optional[str] = None,
    workers: int = CONTEXT_WORKERS,
    progress: Progress = None,
    first_ord: Optional[int] = None,
) -> List[Chunk]:
    """One summary node per chapter, plus one for the book (RAPTOR-lite)."""
    if ai is None or not chunks:
        return []
    title = (book_title or structure.title or "Untitled").strip()
    texts = chapter_texts(chunks, CHAPTER_SUMMARY_TOKENS)
    chapters = [
        chapter for chapter in structure.chapters if texts.get(chapter.idx, "").strip()
    ]
    if not chapters:
        return []

    counter = _Counter(len(chapters) + 1, progress)

    def summarize(chapter) -> str:
        try:
            text = _chapter_summary(ai, title, chapter.title, texts[chapter.idx], model)
        except Exception:  # noqa: BLE001 - a missing synopsis is not fatal
            logger.warning("chapter summary failed for %s", chapter.title, exc_info=True)
            text = ""
        counter.tick()
        return text

    if workers <= 1:
        summaries = [summarize(chapter) for chapter in chapters]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            summaries = list(pool.map(summarize, chapters))

    base = first_ord if first_ord is not None else len(chunks)
    nodes: List[Chunk] = []
    for chapter, summary in zip(chapters, summaries):
        if not summary.strip():
            continue
        nodes.append(
            _summary_chunk(
                book_id=book_id,
                ordinal=base + len(nodes),
                suffix="ch{0:04d}".format(chapter.idx),
                book_title=title,
                chapter_idx=chapter.idx,
                chapter_title=chapter.title,
                label=CHAPTER_SUMMARY_LABEL,
                page_start=chapter.page_start,
                page_end=chapter.page_end,
                text=summary.strip(),
                kind=KIND_CHAPTER_SUMMARY,
                level=LEVEL_CHAPTER_SUMMARY,
            )
        )

    book_summary = ""
    if nodes:
        try:
            book_summary = _book_summary(ai, title, nodes, model)
        except Exception:  # noqa: BLE001
            logger.warning("book summary failed", exc_info=True)
    counter.tick()
    if book_summary.strip():
        pages = [chapter.page_start for chapter in structure.chapters] or [1]
        ends = [chapter.page_end for chapter in structure.chapters] or [1]
        nodes.append(
            _summary_chunk(
                book_id=book_id,
                ordinal=base + len(nodes),
                suffix="book",
                book_title=title,
                chapter_idx=BOOK_SUMMARY_CHAPTER_IDX,
                chapter_title=title,
                label=BOOK_SUMMARY_LABEL,
                page_start=min(pages),
                page_end=max(ends),
                text=book_summary.strip(),
                kind=KIND_BOOK_SUMMARY,
                level=LEVEL_BOOK_SUMMARY,
            )
        )
    return nodes


def _summary_chunk(
    book_id: str,
    ordinal: int,
    suffix: str,
    book_title: str,
    chapter_idx: int,
    chapter_title: str,
    label: str,
    page_start: int,
    page_end: int,
    text: str,
    kind: str,
    level: int,
) -> Chunk:
    path = [part for part in (chapter_title, label) if part]
    return Chunk(
        id="{0}-sum-{1}".format(book_id, suffix),
        ord=ordinal,
        chapter_idx=chapter_idx,
        chapter_title=chapter_title,
        section_title=label,
        page_start=page_start,
        page_end=page_end,
        text=text,
        header=build_header(book_title, path, text),
        parent_text=text,
        char_span=(0, len(text)),
        kind=kind,
        level=level,
        heading_path=tuple(path),
    )


def _chapter_summary(
    ai: object, book_title: str, chapter_title: str, text: str, model: Optional[str]
) -> str:
    messages = [
        {"role": "system", "content": CHAPTER_SUMMARY_SYSTEM},
        {
            "role": "user",
            "content": CHAPTER_SUMMARY_USER_TEMPLATE.format(
                book_title=book_title,
                chapter_title=chapter_title or "Untitled chapter",
                chapter_text=text,
            ),
        },
    ]
    return (ai.chat(messages, purpose="summary", temperature=0, model=model) or "").strip()


def _book_summary(
    ai: object, book_title: str, nodes: Sequence[Chunk], model: Optional[str]
) -> str:
    joined = _truncate(
        "\n\n".join(
            "{0}: {1}".format(node.chapter_title or "Chapter", node.text)
            for node in nodes
        ),
        BOOK_SUMMARY_TOKENS,
    )
    messages = [
        {"role": "system", "content": BOOK_SUMMARY_SYSTEM},
        {
            "role": "user",
            "content": BOOK_SUMMARY_USER_TEMPLATE.format(
                book_title=book_title, summaries=joined
            ),
        },
    ]
    return (ai.chat(messages, purpose="summary", temperature=0, model=model) or "").strip()


def _truncate(text: str, max_tokens: int) -> str:
    """Cut a blob to roughly ``max_tokens``, on a whitespace boundary."""
    if max_tokens <= 0 or not text:
        return text
    tokens = count_tokens(text)
    if tokens <= max_tokens:
        return text
    budget = max(1, int(len(text) * max_tokens / tokens))
    cut = text[:budget]
    space = cut.rfind(" ")
    if space > budget // 2:
        cut = cut[:space]
    return cut.rstrip() + "…"


__all__ = [
    "BOOK_SUMMARY_CHAPTER_IDX",
    "CHAPTER_CONTEXT_TOKENS",
    "CONTEXT_WORKERS",
    "chapter_texts",
    "contextualize",
    "summary_chunks",
]
