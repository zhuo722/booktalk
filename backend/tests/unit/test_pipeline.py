"""The answer stream's SSE contract (SPEC §5.2, revised by §14.7 and §14.8).

Everything here runs against the real pipeline with :class:`FakeAIClient`, so
the assertions are about event order, payload shape and lifecycle — the things a
client actually depends on — rather than about model output.
"""

from __future__ import annotations

import base64
import threading
import time
from typing import Any, Dict, Iterator, List

import pytest

from app.config import Settings
from app.index.store import Store
from app.rag import pipeline
from app.rag.pipeline import TTS_CONCURRENCY, SSEEvent, answer_stream
from app.services.openai_client import FakeAIClient
from app.sessions import SessionStore
from app.speech.pcm import AUDIO_FORMAT, SAMPLE_RATE
from app.speech.turns import TTS_QUEUE_BOUND, TurnRegistry


def collect(events: Iterator[SSEEvent]) -> List[SSEEvent]:
    return list(events)


def run(
    store: Store,
    ai: FakeAIClient,
    book_id: str,
    settings: Settings,
    question: str = "Who built the harbour at Kestrel Cove?",
    sessions: SessionStore = None,
    turns: TurnRegistry = None,
) -> List[SSEEvent]:
    return collect(
        answer_stream(
            ai=ai,
            store=store,
            sessions=sessions or SessionStore(),
            settings=settings,
            book_id=book_id,
            session_id="session-1",
            question=question,
            turns=turns,
        )
    )


def named(events: List[SSEEvent], name: str) -> List[Dict[str, Any]]:
    return [event.data for event in events if event.event == name]


# ------------------------------------------------------------- event sequence


def test_the_stream_yields_the_documented_event_sequence(store, fake_ai, ready_book, settings):
    events = run(store, fake_ai, ready_book, settings)

    order = [event.event for event in events]
    assert order[0] == "status"
    assert order[-1] == "done"
    assert order.index("sources") < order.index("delta")
    assert "audio" in order
    # §14.7: citations always precede done, so a client can render provenance
    # in the same paint as the finished answer.
    assert order.index("citations") == len(order) - 2


def test_every_event_carries_the_same_turn_id(store, fake_ai, ready_book, settings):
    events = run(store, fake_ai, ready_book, settings)

    ids = {event.data.get("turn_id") for event in events}
    assert len(ids) == 1
    assert ids.pop()


def test_the_turn_id_is_registered_and_findable(store, fake_ai, ready_book, settings):
    turns = TurnRegistry()

    events = run(store, fake_ai, ready_book, settings, turns=turns)

    turn_id = events[0].data["turn_id"]
    assert turns.get(turn_id) is not None
    assert turns.get(turn_id).question == "Who built the harbour at Kestrel Cove?"


# -------------------------------------------------------------------- audio


def test_audio_events_are_pcm_at_24khz_in_sequence(store, fake_ai, ready_book, settings):
    audio = named(run(store, fake_ai, ready_book, settings), "audio")

    assert audio, "the answer should produce at least one spoken segment"
    assert [item["seq"] for item in audio] == list(range(len(audio)))
    for item in audio:
        assert item["format"] == AUDIO_FORMAT == "pcm"
        assert item["sample_rate"] == SAMPLE_RATE == 24000
        assert base64.b64decode(item["b64"])


def test_each_spoken_segment_is_the_audio_of_one_sentence(
    store, fake_ai, ready_book, settings
):
    events = run(store, fake_ai, ready_book, settings)

    audio = named(events, "audio")
    assert len(fake_ai.tts_calls) == len(audio)
    for item, sentence in zip(audio, fake_ai.tts_calls):
        assert base64.b64decode(item["b64"]) == fake_ai.tts(sentence)


def test_a_failing_tts_call_degrades_that_sentence_to_text_only(
    store, fake_ai, ready_book, settings, monkeypatch
):
    def explode(text: str) -> bytes:
        raise RuntimeError("tts is down")

    monkeypatch.setattr(fake_ai, "tts", explode)

    events = run(store, fake_ai, ready_book, settings)

    assert named(events, "audio") == []
    assert named(events, "done")[0]["answer"]


# ------------------------------------------------------------------ sources


def test_the_sources_event_carries_a_confidence_score(
    store, fake_ai, ready_book, settings
):
    sources = named(run(store, fake_ai, ready_book, settings), "sources")[0]

    assert 0.0 <= sources["confidence"] <= 1.0
    assert sources["sources"]
    assert "scores" in sources["sources"][0]


# ---------------------------------------------------------------- citations


def test_citations_map_sentences_onto_retrieved_chunks(
    store, fake_ai, ready_book, settings
):
    events = run(store, fake_ai, ready_book, settings)

    citation = named(events, "citations")[0]
    assert isinstance(citation["map"], list)
    known = {source["chunk_id"] for source in named(events, "sources")[0]["sources"]}
    for entry in citation["map"]:
        assert entry["sentence_idx"] >= 0
        assert set(entry["chunk_ids"]) <= known


def test_a_broken_citation_pass_still_lets_the_answer_finish(
    store, fake_ai, ready_book, settings, monkeypatch
):
    monkeypatch.setattr(
        pipeline, "attribute_citations", lambda *a, **k: 1 / 0
    )

    events = run(store, fake_ai, ready_book, settings)

    assert named(events, "citations")[0]["map"] == []
    assert events[-1].event == "done"


def test_the_answer_prompt_no_longer_asks_for_inline_chapter_markers():
    """§14.7: spoken prose must not contain "(Chapter 3)"."""
    from app.rag.prompts import ANSWER_SYSTEM

    assert "(Chapter 3)" not in ANSWER_SYSTEM
    assert "Cite chapters inline" not in ANSWER_SYSTEM


# ------------------------------------------------------------------- history


def test_a_completed_answer_is_stored_under_its_turn_id(
    store, fake_ai, ready_book, settings
):
    sessions = SessionStore()
    turns = TurnRegistry()

    events = run(store, fake_ai, ready_book, settings, sessions=sessions, turns=turns)

    turn_id = events[0].data["turn_id"]
    history = sessions.history("session-1", ready_book)
    assert len(history) == 1
    assert history[0].turn_id == turn_id
    assert history[0].answer == named(events, "done")[0]["answer"]


# -------------------------------------------------------------- cancellation


def test_no_event_escapes_after_a_turn_is_cancelled(
    store, fake_ai, ready_book, settings
):
    """The point of the turn id: a dead turn goes silent immediately."""
    turns = TurnRegistry()
    seen: List[SSEEvent] = []
    events = answer_stream(
        ai=fake_ai,
        store=store,
        sessions=SessionStore(),
        settings=settings,
        book_id=ready_book,
        session_id="session-1",
        question="Who built the harbour?",
        turns=turns,
    )

    for event in events:
        seen.append(event)
        if event.event == "delta" and len(seen) > 4:
            turns.cancel(seen[0].data["turn_id"], played_seq=0)

    assert [event.event for event in seen[-1:]] == ["delta"]
    assert "done" not in [event.event for event in seen]
    assert "citations" not in [event.event for event in seen]


def test_cancelling_truncates_history_to_what_was_played(
    store, fake_ai, ready_book, settings
):
    """A follow-up question must see what was heard, not what was generated."""
    question = "Who built the harbour?"
    uninterrupted = named(
        run(store, fake_ai, ready_book, settings, question=question), "done"
    )[0]["answer"]

    turns = TurnRegistry()
    sessions = SessionStore()
    events = answer_stream(
        ai=fake_ai,
        store=store,
        sessions=sessions,
        settings=settings,
        book_id=ready_book,
        session_id="session-1",
        question=question,
        turns=turns,
    )

    turn_id = None
    for event in events:
        turn_id = turn_id or event.data.get("turn_id")
        if event.event == "audio" and event.data["seq"] == 0:
            turns.cancel(turn_id, played_seq=0)

    turn = turns.get(turn_id)
    history = sessions.history("session-1", ready_book)
    assert len(history) == 1
    assert history[0].answer == turn.sentences[0]
    assert uninterrupted.startswith(history[0].answer)
    assert len(history[0].answer) < len(uninterrupted)


def test_cancelling_before_any_audio_leaves_no_history_at_all(
    store, fake_ai, ready_book, settings
):
    turns = TurnRegistry()
    sessions = SessionStore()
    events = answer_stream(
        ai=fake_ai,
        store=store,
        sessions=sessions,
        settings=settings,
        book_id=ready_book,
        session_id="session-1",
        question="Who built the harbour?",
        turns=turns,
    )

    for event in events:
        turns.cancel(event.data["turn_id"], played_seq=-1)

    assert sessions.history("session-1", ready_book) == []


def test_a_cancelled_turn_stops_paying_for_synthesis(
    store, fake_ai, ready_book, settings
):
    """At most the in-flight work is wasted, never the rest of the answer."""
    turns = TurnRegistry()
    events = answer_stream(
        ai=fake_ai,
        store=store,
        sessions=SessionStore(),
        settings=settings,
        book_id=ready_book,
        session_id="session-1",
        question="Who built the harbour?",
        turns=turns,
    )

    spoken_at_cancel = None
    for event in events:
        if event.event == "audio" and spoken_at_cancel is None:
            turns.cancel(event.data["turn_id"], played_seq=0)
            spoken_at_cancel = len(fake_ai.tts_calls)

    assert spoken_at_cancel is not None
    # Nothing new is submitted after the cancel; only what was already queued
    # inside the executor can still run.
    assert len(fake_ai.tts_calls) <= spoken_at_cancel + TTS_CONCURRENCY


# ----------------------------------------------------------- back-pressure


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_synthesis_pauses_while_the_client_is_behind(fake_ai):
    """The queue holds one sentence behind what is in flight, and stops when
    more than five seconds of audio cannot have been played yet."""
    from concurrent.futures import ThreadPoolExecutor

    from app.speech.turns import AudioPacer, TurnState

    clock = Clock()
    pacer = AudioPacer(clock=clock)
    pacer.emitted(b"\x00" * (SAMPLE_RATE * 2 * 9))  # nine seconds outstanding
    turn = TurnState("t", "s", "b", "q")
    with ThreadPoolExecutor(max_workers=TTS_CONCURRENCY) as pool:
        queue = pipeline._SpeechQueue(
            fake_ai, pool, turn, {"first_audio_ms": 0}, time.monotonic(), pacer=pacer
        )
        for index in range(4):
            queue.offer("Sentence number {0} of the answer.".format(index))

        assert queue.drain(block=False) == []
        assert fake_ai.tts_calls == []

        clock.advance(9.0)
        # ``finish`` releases each segment as it lands, so it is a generator.
        events = list(queue.finish())

    assert len(events) == 4
    assert len(fake_ai.tts_calls) == 4


def test_the_queue_bound_limits_work_in_flight(fake_ai):
    from concurrent.futures import ThreadPoolExecutor

    from app.speech.turns import TurnState

    started = threading.Semaphore(0)
    release = threading.Event()
    seen: List[str] = []

    def slow_tts(text: str) -> bytes:
        seen.append(text)
        started.release()
        release.wait(5.0)
        return b"\x00" * 2400

    fake_ai.tts = slow_tts
    turn = TurnState("t", "s", "b", "q")
    with ThreadPoolExecutor(max_workers=TTS_CONCURRENCY) as pool:
        queue = pipeline._SpeechQueue(
            fake_ai, pool, turn, {"first_audio_ms": 0}, time.monotonic()
        )
        for index in range(8):
            queue.offer("Sentence {0}.".format(index))
        queue.drain(block=False)
        for _ in range(TTS_CONCURRENCY):
            assert started.acquire(timeout=5.0)

        # Two workers are busy and one sentence waits behind them; the other
        # five are still ours to cancel for free.
        assert len(seen) <= TTS_CONCURRENCY + TTS_QUEUE_BOUND

        release.set()
        list(queue.finish())


def test_the_spoken_tail_is_released_segment_by_segment(fake_ai):
    """A finished segment goes out now, not when the last one is synthesized.

    The tail of an answer is almost always still being synthesized after the
    model has stopped writing, so holding it until the slowest call returns
    would delay playback, start the pacer's clock on bytes nobody has received,
    and leave a mid-tail cancel with nothing left to stop.
    """
    from concurrent.futures import ThreadPoolExecutor

    from app.speech.turns import TurnState

    release = threading.Event()

    def gated_tts(text: str) -> bytes:
        if text.startswith("Sentence 3"):
            release.wait(10.0)
        return b"\x00" * 4800

    fake_ai.tts = gated_tts
    turn = TurnState("t", "s", "b", "q")
    with ThreadPoolExecutor(max_workers=TTS_CONCURRENCY) as pool:
        queue = pipeline._SpeechQueue(
            fake_ai, pool, turn, {"first_audio_ms": 0}, time.monotonic()
        )
        for index in range(4):
            queue.offer("Sentence {0} of the answer, spoken aloud.".format(index))

        stream = queue.finish()
        # The last sentence is still inside a blocked TTS call right now.
        assert next(stream).data["seq"] == 0
        release.set()
        assert [event.data["seq"] for event in stream] == [1, 2, 3]


# ---------------------------------------------------------------- error path


def test_a_model_failure_becomes_an_error_event(store, fake_ai, ready_book, settings):
    def explode(*args: Any, **kwargs: Any):
        raise RuntimeError("the model is down")

    fake_ai.chat_stream = explode

    events = run(store, fake_ai, ready_book, settings)

    assert events[-1].event == "error"
    assert "the model is down" in events[-1].data["message"]
    assert events[-1].data["turn_id"]


def test_an_unknown_book_still_produces_a_well_formed_stream(store, fake_ai, settings):
    events = run(store, fake_ai, "missing-book", settings)

    assert events[-1].event in {"done", "error"}
    assert all(event.data.get("turn_id") for event in events)


@pytest.mark.parametrize("question", ["", "   "])
def test_an_empty_question_retrieves_nothing(store, fake_ai, ready_book, settings, question):
    events = run(store, fake_ai, ready_book, settings, question=question)

    assert named(events, "sources")[0]["sources"] == []
