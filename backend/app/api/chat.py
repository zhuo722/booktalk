"""The chat endpoint: one question in, an SSE answer stream out (SPEC §5.2).

SSE is written by hand over ``StreamingResponse`` — the wire format is four
lines of code and one fewer dependency than sse-starlette.

Alongside it lives the cancel endpoint of SPEC §14.8. A voice answer cannot be
stopped by closing the socket alone: the browser can abort its read, but the
server would keep synthesizing sentences nobody will hear, and the stored
history would claim the assistant said things that were never spoken. So the
client aborts *and* posts the last segment it finished playing, which stops the
work and rewrites the transcript to match what was heard.
"""

from __future__ import annotations

import json
from typing import Iterator, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.api import get_ai, get_sessions, get_settings, get_store, get_turns
from app.api.books import UNKNOWN_BOOK, ready_or_error
from app.models import CancelRequest, CancelResponse, ChatRequest
from app.rag.pipeline import SSEEvent, answer_stream
from app.speech.turns import NOTHING_PLAYED

router = APIRouter(prefix="/api", tags=["chat"])

EMPTY_QUESTION = "The question is empty."
UNKNOWN_TURN = "Unknown or expired turn."
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # don't let a proxy sit on the stream
}


@router.post("/chat")
def chat(request: Request, payload: ChatRequest) -> StreamingResponse:
    """Answer a question about one book, streaming text and audio as they come."""
    ai = get_ai(request)
    store = get_store(request)
    sessions = get_sessions(request)
    settings = get_settings(request)
    turns = get_turns(request)

    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail=EMPTY_QUESTION
        )

    book = store.get_book(payload.book_id)
    if book is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=UNKNOWN_BOOK)
    not_ready = ready_or_error(book)
    if not_ready:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=not_ready)

    events = answer_stream(
        ai=ai,
        store=store,
        sessions=sessions,
        settings=settings,
        book_id=payload.book_id,
        session_id=payload.session_id,
        question=question,
        turns=turns,
        # Loaded once at startup (SPEC §14.6); ``None`` on a machine that has
        # no model cached, in which case retrieval keeps its fusion order.
        reranker=getattr(request.app.state, "reranker", None),
    )
    return StreamingResponse(
        _encode(events), media_type="text/event-stream", headers=SSE_HEADERS
    )


@router.post("/chat/turns/{turn_id}/cancel", response_model=CancelResponse)
def cancel_turn(
    request: Request, turn_id: str, payload: Optional[CancelRequest] = None
) -> CancelResponse:
    """Stop a turn and truncate its history to what was actually played.

    Idempotent, and deliberately forgiving: a client that cancels a turn which
    has already finished still gets its history corrected, because "the user
    stopped listening after sentence two" is true whether or not the server had
    already written sentence six.
    """
    turns = get_turns(request)
    sessions = get_sessions(request)

    played_seq = payload.played_seq if payload is not None else NOTHING_PLAYED
    turn = turns.cancel(turn_id, played_seq)
    if turn is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=UNKNOWN_TURN)

    spoken = turn.spoken_answer()
    sessions.record(turn.session_id, turn.book_id, turn.turn_id, turn.question, spoken)
    return CancelResponse(
        turn_id=turn.turn_id, cancelled=True, played_seq=turn.played_seq, answer=spoken
    )


def _encode(events: Iterator[SSEEvent]) -> Iterator[str]:
    for event in events:
        yield format_sse(event)


def format_sse(event: SSEEvent) -> str:
    """Serialize one event in the ``event:``/``data:`` wire format."""
    return "event: {0}\ndata: {1}\n\n".format(
        event.event, json.dumps(event.data, ensure_ascii=False)
    )
