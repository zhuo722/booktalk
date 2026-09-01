"""Turn identity, cancellation and TTS back-pressure (SPEC §14.8).

A spoken answer is not finished when the text is finished — it is finished when
the audio has been heard. That gap is where three problems live, and this module
owns all three.

**Identity.** Every SSE event of an answer carries the same ``turn_id``, so a
client that has moved on can drop anything belonging to a turn it interrupted,
and so a cancel request has something to name.

**Cancellation.** ``POST /api/chat/turns/{turn_id}/cancel`` flips
:attr:`TurnState.cancelled`. The generator checks it between events and stops
emitting — a late audio segment for an abandoned turn must never reach the
client. It also reports ``played_seq``: the last segment the browser actually
finished playing. That number is what makes the *history* honest — the assistant
said what was heard, not what was generated, so a follow-up question
("what about its harbour?") is resolved against a transcript the user recognises
(the vocode pattern).

**Back-pressure.** Synthesis runs ahead of playback by design — that is the
latency win — but running too far ahead is pure waste: a turn cancelled after
two sentences should not have paid for six. :class:`AudioPacer` estimates how
much of the emitted audio is still unplayed from the PCM byte count and the wall
clock, and the pipeline stops feeding the TTS queue while that estimate is over
five seconds.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from app.speech.pcm import duration_seconds

#: How far synthesis may run ahead of playback before it pauses.
MAX_UNPLAYED_SECONDS = 5.0
#: Sentences allowed to sit in the TTS queue *behind* the in-flight ones.
TTS_QUEUE_BOUND = 1
#: Turns remembered per process; only the live one is ever cancelled, the rest
#: are kept so a cancel that races the end of the stream still finds its turn.
MAX_TURNS = 64

#: ``played_seq`` when the client has played nothing at all.
NOTHING_PLAYED = -1


@dataclass
class TurnState:
    """One in-flight (or just-finished) answer."""

    turn_id: str
    session_id: str
    book_id: str
    question: str
    cancelled: bool = False
    played_seq: int = NOTHING_PLAYED
    #: Sentence texts in ``seq`` order — index i is what audio segment i says.
    sentences: List[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_sentence(self, seq: int, text: str) -> None:
        """Record what audio segment ``seq`` will say."""
        with self._lock:
            while len(self.sentences) <= seq:
                self.sentences.append("")
            self.sentences[seq] = text

    def cancel(self, played_seq: int = NOTHING_PLAYED) -> None:
        """Mark the turn dead and record how much of it was heard."""
        with self._lock:
            self.cancelled = True
            self.played_seq = int(played_seq)

    def spoken_answer(self) -> str:
        """The part of the answer the listener actually heard.

        Joined from the sentences up to and including ``played_seq``. An
        uncancelled turn has no truncation point, so callers store the full
        answer text instead of asking this.
        """
        with self._lock:
            if self.played_seq < 0:
                return ""
            spoken = [text for text in self.sentences[: self.played_seq + 1] if text]
        return " ".join(spoken).strip()


class TurnRegistry:
    """Thread-safe LRU of turns, addressed by ``turn_id``.

    Small and in-memory for the same reason :class:`~app.sessions.SessionStore`
    is: a turn is only interesting while someone might still cancel it.
    """

    def __init__(self, max_turns: int = MAX_TURNS) -> None:
        self.max_turns = max_turns
        self._turns: "OrderedDict[str, TurnState]" = OrderedDict()
        self._lock = threading.Lock()

    def create(self, session_id: str, book_id: str, question: str) -> TurnState:
        turn = TurnState(
            turn_id=uuid.uuid4().hex,
            session_id=session_id,
            book_id=book_id,
            question=question,
        )
        with self._lock:
            self._turns[turn.turn_id] = turn
            while len(self._turns) > self.max_turns:
                self._turns.popitem(last=False)
        return turn

    def get(self, turn_id: str) -> Optional[TurnState]:
        with self._lock:
            turn = self._turns.get(turn_id)
            if turn is not None:
                self._turns.move_to_end(turn_id)
            return turn

    def cancel(self, turn_id: str, played_seq: int = NOTHING_PLAYED) -> Optional[TurnState]:
        """Cancel a turn by id; ``None`` when the id is unknown or expired."""
        turn = self.get(turn_id)
        if turn is None:
            return None
        turn.cancel(played_seq)
        return turn

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()


class AudioPacer:
    """Estimates how much emitted audio the client has not played yet.

    There is no ack channel during a turn — the browser only reports
    ``played_seq`` when it cancels — so the estimate is the honest one available
    to a push-only stream: audio starts playing when the first segment is
    emitted and plays at real time thereafter, so

        unplayed = (seconds emitted) − (seconds since the first emission)

    which is LiveKit's ``SentenceStreamPacer(min_remaining_audio=...)``. It errs
    towards *over*-estimating the backlog (the client cannot play faster than
    real time, but it can start later), so the worst case is synthesizing a
    little later than necessary — never further ahead than intended.

    ``clock`` is injectable so the tests do not sleep.
    """

    def __init__(
        self,
        max_unplayed: float = MAX_UNPLAYED_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_unplayed = max_unplayed
        self._clock = clock
        self._emitted_seconds = 0.0
        self._started: Optional[float] = None

    def emitted(self, audio: bytes) -> None:
        """Record one audio segment as sent to the client."""
        if not audio:
            return
        if self._started is None:
            self._started = self._clock()
        self._emitted_seconds += duration_seconds(audio)

    def acknowledge(self, played_seconds: float) -> None:
        """Pull the playback clock forward when the client tells us where it is."""
        if self._started is None:
            return
        self._started = self._clock() - max(0.0, played_seconds)

    @property
    def emitted_seconds(self) -> float:
        return self._emitted_seconds

    def unplayed_seconds(self) -> float:
        """Seconds of emitted audio that cannot have been played yet."""
        if self._started is None:
            return 0.0
        elapsed = max(0.0, self._clock() - self._started)
        return max(0.0, self._emitted_seconds - elapsed)

    def should_pause(self) -> bool:
        """True while the client is more than :attr:`max_unplayed` seconds behind."""
        return self.unplayed_seconds() > self.max_unplayed
