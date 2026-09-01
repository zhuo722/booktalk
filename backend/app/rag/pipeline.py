"""The answer pipeline: retrieve → stream tokens → speak sentences (SPEC §5.2).

The latency trick lives here. Instead of waiting for the whole answer before
calling TTS, the two-phase splitter hands the first *fragment* to synthesis the
moment it is worth speaking — often before the model has finished its opening
sentence — and every later sentence as it completes. Segments carry a ``seq``
and are emitted strictly in order, so the client plays them back-to-back with no
gaps.

SPEC §14.7/§14.8 add four things around that core, all of them about the gap
between "the text is done" and "the answer has been heard":

* **Raw PCM.** Audio events are headerless 24 kHz s16le mono, so the browser
  schedules samples instead of decoding mp3 frames — and a byte count is a
  duration, which the back-pressure estimate needs.
* **Back-pressure.** Synthesis runs ahead of playback on purpose, but only so
  far: the TTS queue holds one sentence behind what is in flight, and synthesis
  pauses entirely while more than five seconds of emitted audio cannot have been
  played yet. A turn cancelled after two sentences should not have paid for six.
* **A turn id on every event, and a cancel path.** See :mod:`app.speech.turns`.
  Nothing is emitted for a turn after it has been cancelled — including the
  segment that was already in flight when the cancel arrived.
* **Post-hoc citations.** Attribution runs on its own thread while the last
  sentences are still being synthesized, so per-sentence provenance costs
  nothing against time-to-first-audio, and its event precedes ``done``.
"""

from __future__ import annotations

import base64
import logging
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional

from app.config import Settings
from app.index.rerank import Reranker
from app.index.store import BookIndex, Store
from app.rag.citations import attribute_citations, citation_chunks
from app.rag.prompts import ANSWER_SYSTEM, ANSWER_USER_TEMPLATE
from app.rag.retriever import Retrieval, retrieve
from app.rag.rewrite import rewrite_question
from app.services.openai_client import AIClient
from app.sessions import SessionStore
from app.speech.pcm import AUDIO_FORMAT, SAMPLE_RATE
from app.speech.sentences import SentenceSplitter
from app.speech.turns import (
    TTS_QUEUE_BOUND,
    AudioPacer,
    TurnRegistry,
    TurnState,
)

logger = logging.getLogger(__name__)

TTS_CONCURRENCY = 2
ANSWER_TEMPERATURE = 0.2

#: How long to wait between back-pressure checks while synthesis is paused.
PACING_POLL_S = 0.05
#: Absolute cap on time spent waiting for playback in one turn. Back-pressure
#: is an estimate, and an estimate must never be able to wedge a stream.
MAX_PACING_WAIT_S = 30.0

STAGE_REWRITING = "rewriting"
STAGE_RETRIEVING = "retrieving"
STAGE_GENERATING = "generating"
STAGE_SPEAKING = "speaking"

GENERIC_ERROR = "The answer service failed. Please try again."


@dataclass
class SSEEvent:
    """One server-sent event: a name and a JSON-serializable payload."""

    event: str
    data: Dict[str, Any] = field(default_factory=dict)


def answer_stream(
    ai: AIClient,
    store: Store,
    sessions: SessionStore,
    settings: Settings,
    book_id: str,
    session_id: str,
    question: str,
    turns: Optional[TurnRegistry] = None,
    reranker: Optional[Reranker] = None,
) -> Iterator[SSEEvent]:
    """Yield the full SSE event sequence for one question.

    Two invariants are enforced here rather than at each ``yield`` inside the
    body, so neither can be forgotten by a later edit: every event carries the
    turn id, and no event escapes once the turn has been cancelled.
    """
    registry = turns if turns is not None else TurnRegistry()
    turn = registry.create(session_id, book_id, question)
    events = _run(ai, store, sessions, settings, turn, reranker)
    try:
        for event in events:
            if turn.cancelled:
                return
            event.data["turn_id"] = turn.turn_id
            yield event
    finally:
        events.close()
        _truncate_history(sessions, turn)


def _run(
    ai: AIClient,
    store: Store,
    sessions: SessionStore,
    settings: Settings,
    turn: TurnState,
    reranker: Optional[Reranker],
) -> Iterator[SSEEvent]:
    """The pipeline proper. Stamping and cancellation are the caller's job."""
    started = time.monotonic()
    timings: Dict[str, int] = {
        "rewrite_ms": 0,
        "retrieve_ms": 0,
        "first_token_ms": 0,
        "first_audio_ms": 0,
        "total_ms": 0,
    }
    tts_pool = ThreadPoolExecutor(max_workers=TTS_CONCURRENCY, thread_name_prefix="tts")
    aux_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cite")
    speech = _SpeechQueue(ai, tts_pool, turn, timings, started)
    answer_parts: List[str] = []
    speaking = False

    try:
        book = store.get_book(turn.book_id)
        book_title = book.title if book else "the book"
        index = store.load_index(turn.book_id)

        # The stream always opens with a status event; "rewriting" only appears
        # when there is history to rewrite against.
        history = sessions.history(turn.session_id, turn.book_id)
        yield SSEEvent(
            "status", {"stage": STAGE_REWRITING if history else STAGE_RETRIEVING}
        )
        mark = time.monotonic()
        search_question = rewrite_question(
            ai, turn.question, history, model=settings.llm_model
        )
        timings["rewrite_ms"] = _ms_since(mark)
        yield SSEEvent("question", {"text": search_question})

        if history:
            yield SSEEvent("status", {"stage": STAGE_RETRIEVING})
        mark = time.monotonic()
        result = retrieve(
            ai,
            index,
            search_question,
            rerank=settings.rerank,
            model=settings.llm_model,
            rrf_k=settings.rrf_k,
            weights=settings.rrf_weights,
            reranker=reranker,
        )
        timings["retrieve_ms"] = _ms_since(mark)
        yield SSEEvent(
            "sources",
            {
                "sources": [source.to_dict() for source in result.sources],
                # Drives the low-confidence banner (SPEC §14.6). Only an
                # absolute quantity when a cross-encoder ran; without one it is
                # a normalized fused rank, i.e. 1.0 for anything that matched.
                "confidence": round(float(result.top_score), 6),
            },
        )

        yield SSEEvent("status", {"stage": STAGE_GENERATING})
        messages = build_messages(book_title, result.context, turn.question)
        splitter = SentenceSplitter()

        for delta in ai.chat_stream(
            messages, model=settings.llm_model, temperature=ANSWER_TEMPERATURE
        ):
            if turn.cancelled:
                break
            if not delta:
                continue
            if not answer_parts:
                timings["first_token_ms"] = _ms_since(started)
            answer_parts.append(delta)
            yield SSEEvent("delta", {"text": delta})

            for sentence in splitter.feed(delta):
                speech.offer(sentence)
                if not speaking:
                    speaking = True
                    yield SSEEvent("status", {"stage": STAGE_SPEAKING})
            for event in speech.drain(block=False):
                yield event

        if not turn.cancelled:
            for sentence in splitter.flush():
                speech.offer(sentence)
                if not speaking:
                    speaking = True
                    yield SSEEvent("status", {"stage": STAGE_SPEAKING})

        answer = "".join(answer_parts).strip()
        # Attribution overlaps the TTS tail: by the time the last sentence has
        # been synthesized the citation map is usually already waiting. A
        # cancelled turn never starts it — nobody is going to read the answer.
        citations: Optional["Future[List[Dict[str, Any]]]"] = None
        if not turn.cancelled:
            citations = aux_pool.submit(_attribute, ai, answer, index, result)

        for event in speech.finish():
            yield event

        if turn.cancelled:
            return
        timings["total_ms"] = _ms_since(started)
        if answer:
            sessions.record(
                turn.session_id, turn.book_id, turn.turn_id, turn.question, answer
            )
        yield SSEEvent("citations", {"map": _result(citations)})
        yield SSEEvent("done", {"answer": answer, "timings": timings})
    except Exception as exc:  # noqa: BLE001 - reported to the client, then closed
        logger.exception("answer stream failed for book %s", turn.book_id)
        yield SSEEvent("error", {"message": str(exc) or GENERIC_ERROR})
    finally:
        tts_pool.shutdown(wait=False, cancel_futures=True)
        aux_pool.shutdown(wait=False, cancel_futures=True)


def build_messages(book_title: str, context: str, question: str) -> List[Dict[str, str]]:
    """Assemble the answer prompt."""
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {
            "role": "user",
            "content": ANSWER_USER_TEMPLATE.format(
                book_title=book_title, context=context, question=question.strip()
            ),
        },
    ]


# ------------------------------------------------------------- speech queueing


class _SpeechQueue:
    """Sentence → TTS → ordered ``audio`` events, with back-pressure.

    Sentences arrive faster than they can be spoken, which is the point, but
    only two things may be in flight and one may wait behind them
    (:data:`~app.speech.turns.TTS_QUEUE_BOUND`) — and nothing at all is
    submitted while the client is estimated to be more than five seconds behind.
    Segments are released strictly in ``seq`` order, because the client
    schedules them onto one timeline and cannot reorder.
    """

    def __init__(
        self,
        ai: AIClient,
        executor: ThreadPoolExecutor,
        turn: TurnState,
        timings: Dict[str, int],
        started: float,
        pacer: Optional[AudioPacer] = None,
        concurrency: int = TTS_CONCURRENCY,
        queue_bound: int = TTS_QUEUE_BOUND,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._ai = ai
        self._executor = executor
        self._turn = turn
        self._timings = timings
        self._started = started
        self._pacer = pacer or AudioPacer()
        self._capacity = max(1, concurrency + queue_bound)
        self._sleep = sleep
        self._backlog: Deque[str] = deque()
        self._pending: Dict[int, "Future[Optional[bytes]]"] = {}
        self._submitted = 0
        self._next_seq = 0
        self._waited = 0.0

    def offer(self, sentence: str) -> None:
        """Queue one sentence for synthesis (it may not start immediately)."""
        text = sentence.strip()
        if text:
            self._backlog.append(text)

    def drain(self, block: bool, limit: Optional[int] = None) -> List[SSEEvent]:
        """Submit what capacity allows, then release finished segments in order.

        ``limit`` caps how many segments one call may release. :meth:`finish`
        passes 1: a blocking drain that released the whole chain would hold
        every finished segment until the *last* one was synthesized, which
        delays playback of the tail, starts the pacer's clock on bytes the
        client has not been sent, and leaves a cancel arriving mid-tail with
        nothing left to stop.
        """
        self._pump()
        events: List[SSEEvent] = []
        released = 0
        while self._next_seq in self._pending:
            if self._turn.cancelled or (limit is not None and released >= limit):
                break
            future = self._pending[self._next_seq]
            if not block and not future.done():
                break
            audio = future.result()
            del self._pending[self._next_seq]
            seq = self._next_seq
            self._next_seq += 1
            released += 1
            if audio:
                self._pacer.emitted(audio)
                if not self._timings["first_audio_ms"]:
                    self._timings["first_audio_ms"] = _ms_since(self._started)
                events.append(
                    SSEEvent(
                        "audio",
                        {
                            "seq": seq,
                            "format": AUDIO_FORMAT,
                            "sample_rate": SAMPLE_RATE,
                            "b64": base64.b64encode(audio).decode("ascii"),
                        },
                    )
                )
            self._pump()
        return events

    def finish(self) -> Iterator[SSEEvent]:
        """Block until the backlog is spoken, releasing segments as they land.

        A generator, not a list: each segment has to reach the caller — and
        through it the socket, and the cancellation check that guards every
        yield in :func:`answer_stream` — the moment it is synthesized, not once
        the whole tail is.
        """
        while (self._backlog or self._pending) and not self._turn.cancelled:
            before = (len(self._backlog), len(self._pending))
            for event in self.drain(block=True, limit=1):
                yield event
            if (len(self._backlog), len(self._pending)) == before:
                # Nothing moved: capacity is free but the pacer is holding the
                # queue closed, so wait for playback rather than spinning.
                if not self._wait_for_playback():
                    break

    @property
    def pacer(self) -> AudioPacer:
        return self._pacer

    def _pump(self) -> None:
        """Submit sentences while there is room and the client is keeping up."""
        while (
            self._backlog
            and self._outstanding() < self._capacity
            and not self._turn.cancelled
            and not self._pacer.should_pause()
        ):
            sentence = self._backlog.popleft()
            seq = self._submitted
            self._submitted += 1
            self._turn.add_sentence(seq, sentence)
            self._pending[seq] = self._executor.submit(
                _speak, self._ai, sentence, self._turn
            )

    def _outstanding(self) -> int:
        """Segments submitted but not yet synthesized — in flight plus queued."""
        return sum(1 for future in self._pending.values() if not future.done())

    def _wait_for_playback(self) -> bool:
        """Sleep until back-pressure lifts. False when we gave up waiting."""
        if not self._pacer.should_pause():
            return not self._turn.cancelled
        if self._waited >= MAX_PACING_WAIT_S:
            logger.warning("back-pressure wait exceeded %.0fs; resuming", MAX_PACING_WAIT_S)
            self._pacer.acknowledge(self._pacer.emitted_seconds)
            return True
        self._sleep(PACING_POLL_S)
        self._waited += PACING_POLL_S
        return not self._turn.cancelled


def _speak(ai: AIClient, sentence: str, turn: TurnState) -> Optional[bytes]:
    """Synthesize one sentence; ``None`` when it is not worth speaking.

    Checked against the turn immediately before the call, which is what makes
    "a cancelled turn wastes one sentence of spend" true: anything still queued
    when the cancel lands is never paid for. A TTS failure degrades that one
    sentence to text-only rather than killing an answer the reader can see.
    """
    text = sentence.strip()
    if not text or turn.cancelled:
        return None
    try:
        return ai.tts(text)
    except Exception:  # noqa: BLE001 - the answer text is unaffected
        logger.warning("TTS failed for one sentence", exc_info=True)
        return None


# ---------------------------------------------------------------- citations


def _attribute(
    ai: AIClient, answer: str, index: BookIndex, result: Retrieval
) -> List[Dict[str, Any]]:
    """Post-hoc sentence → chunk attribution (SPEC §14.7), off the hot path."""
    if not answer:
        return []
    try:
        return attribute_citations(ai, answer, citation_chunks(index, result))
    except Exception:  # noqa: BLE001 - an uncited answer is still an answer
        logger.warning("citation attribution failed", exc_info=True)
        return []


def _result(
    future: Optional["Future[List[Dict[str, Any]]]"],
) -> List[Dict[str, Any]]:
    """The citation map, or an empty one if the thread failed or was cancelled."""
    if future is None:
        return []
    try:
        return future.result()
    except Exception:  # noqa: BLE001 - never let attribution break the stream
        logger.warning("citation attribution unavailable", exc_info=True)
        return []


# ------------------------------------------------------------------ history


def _truncate_history(sessions: SessionStore, turn: TurnState) -> None:
    """Rewrite a cancelled turn's history to the sentences actually played.

    The vocode pattern: a follow-up question must be resolved against what the
    listener heard, not against sentences that were synthesized into a stream
    nobody was reading any more. Recording an empty answer drops the exchange
    entirely, which is the right outcome when the user cut in before a word of
    it was spoken.
    """
    if not turn.cancelled:
        return
    sessions.record(
        turn.session_id,
        turn.book_id,
        turn.turn_id,
        turn.question,
        turn.spoken_answer(),
    )


def _ms_since(mark: float) -> int:
    return int(round((time.monotonic() - mark) * 1000))
