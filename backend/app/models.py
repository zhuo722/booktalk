"""Pydantic request/response models for the HTTP contract in SPEC §6.

These types exist only at the API boundary; internal layers pass dataclasses.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    ok: bool = True
    openai_configured: bool


class UploadResponse(BaseModel):
    book_id: str
    status: str


class BookSummary(BaseModel):
    book_id: str
    title: str
    pages: Optional[int] = None
    status: str
    progress: float = 0.0
    created_at: str


class BookListResponse(BaseModel):
    books: List[BookSummary]


class StatusResponse(BaseModel):
    status: str
    progress: float = 0.0
    error: Optional[str] = None


class TocChapter(BaseModel):
    idx: int
    title: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    #: The chapter's synopsis node (SPEC §14.5), when one was generated. The
    #: endpoint sets this key only when there is a synopsis to carry, and is
    #: served with ``exclude_unset``, so an unsummarized chapter has no
    #: ``synopsis`` field at all rather than a null one.
    synopsis: Optional[str] = None
    #: The chapter's own opening prose, straight from the book, shown on the
    #: reader's chapter page. Absent for a chapter with no indexed body text.
    opening: Optional[str] = None


class TocResponse(BaseModel):
    title: str
    chapters: List[TocChapter]


class AsrResponse(BaseModel):
    text: str
    duration_ms: int


class ChatRequest(BaseModel):
    book_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


class CancelRequest(BaseModel):
    """What the client reports when it interrupts a spoken answer (§14.8)."""

    #: The last audio segment the browser finished playing; ``-1`` when the
    #: user cut in before a single word was spoken.
    played_seq: int = Field(default=-1, ge=-1)


class CancelResponse(BaseModel):
    turn_id: str
    cancelled: bool
    played_seq: int
    #: The truncated transcript now stored for this turn — what was heard.
    answer: str = ""
