"""Ingest-time prompt templates (SPEC §14.2, §14.4, §14.5).

Query-time prompts live in :mod:`app.rag.prompts`; these three run once per book
during ingestion and never on the answer path.
"""

from __future__ import annotations

TOC_LEVELS_SYSTEM = (
    "You organize book tables of contents. Given the entry titles in reading "
    "order, assign each one an outline level: 1 for a top-level unit (a part or "
    "a chapter), 2 for a subdivision of the entry above it, 3 for a deeper "
    "subdivision. Front matter such as a preface or an index is level 1. "
    "Reply with JSON only — an array of objects with the keys \"level\" and "
    "\"title\", one per entry, in the order given, and nothing else."
)

TOC_LEVELS_USER_TEMPLATE = "Book: {book_title}\n\nEntries:\n{titles}"

CONTEXT_SYSTEM = (
    "You situate an excerpt inside the chapter it was taken from, so that the "
    "excerpt can be found by search on its own. Reply with one or two plain "
    "sentences naming what the excerpt is about and where it sits in the "
    "chapter. No preamble, no quotes, no markdown. Answer in the language of "
    "the excerpt."
)

CONTEXT_USER_TEMPLATE = (
    "<chapter title=\"{chapter_title}\">\n{chapter_text}\n</chapter>\n\n"
    "Here is the excerpt we want to situate within that chapter:\n"
    "<excerpt>\n{chunk_text}\n</excerpt>\n\n"
    "Give the short situating context for this excerpt and nothing else."
)

CHAPTER_SUMMARY_SYSTEM = (
    "You write the synopsis of one chapter of a book. Cover what the chapter is "
    "about, the topics it treats and the names, places and figures it "
    "introduces, in 4 to 6 plain sentences. Stay strictly inside the text you "
    "are given; never add outside knowledge. No markdown, no lists, no "
    "preamble. Answer in the language of the chapter."
)

CHAPTER_SUMMARY_USER_TEMPLATE = (
    "Book: {book_title}\nChapter: {chapter_title}\n\n{chapter_text}"
)

BOOK_SUMMARY_SYSTEM = (
    "You write the synopsis of a whole book from its chapter synopses. Say what "
    "the book covers overall and how it is organized, in 5 to 8 plain "
    "sentences. Stay strictly inside the text you are given. No markdown, no "
    "lists, no preamble. Answer in the language of the book."
)

BOOK_SUMMARY_USER_TEMPLATE = "Book: {book_title}\n\nChapter synopses:\n{summaries}"

CHAPTER_SUMMARY_LABEL = "Chapter summary"
BOOK_SUMMARY_LABEL = "Book summary"

__all__ = [
    "BOOK_SUMMARY_LABEL",
    "BOOK_SUMMARY_SYSTEM",
    "BOOK_SUMMARY_USER_TEMPLATE",
    "CHAPTER_SUMMARY_LABEL",
    "CHAPTER_SUMMARY_SYSTEM",
    "CHAPTER_SUMMARY_USER_TEMPLATE",
    "CONTEXT_SYSTEM",
    "CONTEXT_USER_TEMPLATE",
    "TOC_LEVELS_SYSTEM",
    "TOC_LEVELS_USER_TEMPLATE",
]
