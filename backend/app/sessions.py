"""In-memory chat history, keyed by ``(session_id, book_id)`` (SPEC §5.2 step 5).

Deliberately not persisted: history only exists to resolve follow-up questions
("what about its harbour?"), and a local single-user app has no reason to keep
transcripts on disk. LRU-capped at 100 sessions × 20 turns.

Turns are written by ``turn_id`` rather than blindly appended (SPEC §14.8). A
cancelled answer has to be rewritten to the sentences the listener actually
heard, and the cancel request can arrive either before the stream stored
anything or after it stored the whole thing — so :meth:`SessionStore.record` is
an upsert, and recording an empty answer removes the turn instead of leaving a
question with a phantom reply attached to it.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

MAX_SESSIONS = 100
MAX_TURNS = 20


@dataclass
class Turn:
    """One completed question/answer exchange."""

    question: str
    answer: str
    #: Identity of the streaming turn that produced it, so a later cancel can
    #: find and truncate exactly this exchange.
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class SessionStore:
    """Thread-safe LRU of conversation histories."""

    def __init__(self, max_sessions: int = MAX_SESSIONS, max_turns: int = MAX_TURNS) -> None:
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self._sessions: "OrderedDict[Tuple[str, str], List[Turn]]" = OrderedDict()
        self._lock = threading.Lock()

    def history(self, session_id: str, book_id: str) -> List[Turn]:
        """Turns so far, oldest first (a copy — callers may not mutate ours)."""
        key = (session_id, book_id)
        with self._lock:
            turns = self._sessions.get(key)
            if turns is None:
                return []
            self._sessions.move_to_end(key)
            return list(turns)

    def append(self, session_id: str, book_id: str, question: str, answer: str) -> Turn:
        """Record an exchange, evicting the oldest session when full."""
        return self.record(session_id, book_id, uuid.uuid4().hex, question, answer)

    def record(
        self,
        session_id: str,
        book_id: str,
        turn_id: str,
        question: str,
        answer: str,
    ) -> Optional[Turn]:
        """Upsert one exchange by ``turn_id``; an empty answer deletes it.

        Idempotent on purpose: the answer stream and a racing cancel request may
        both write the same turn, and whichever runs second must leave one row
        holding the truncated text — not two rows, and not the untruncated one.
        """
        key = (session_id, book_id)
        text = (answer or "").strip()
        with self._lock:
            turns = self._sessions.setdefault(key, [])
            existing = next((turn for turn in turns if turn.turn_id == turn_id), None)
            if not text:
                if existing is not None:
                    turns.remove(existing)
                if not turns:
                    self._sessions.pop(key, None)
                    return None
                self._touch(key)
                return None
            if existing is not None:
                existing.question = question
                existing.answer = text
                self._touch(key)
                return existing
            turn = Turn(question=question, answer=text, turn_id=turn_id)
            turns.append(turn)
            del turns[: max(0, len(turns) - self.max_turns)]
            self._touch(key)
            return turn

    def clear(self) -> None:
        """Forget every session (used between tests)."""
        with self._lock:
            self._sessions.clear()

    def _touch(self, key: Tuple[str, str]) -> None:
        """Mark a session as most recently used; caller holds the lock."""
        self._sessions.move_to_end(key)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)
