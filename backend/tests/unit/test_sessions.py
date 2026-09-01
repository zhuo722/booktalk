"""In-memory chat history, including turn-keyed truncation (SPEC §14.8)."""

from __future__ import annotations

from app.sessions import SessionStore


def test_history_is_returned_oldest_first_and_is_a_copy():
    sessions = SessionStore()
    sessions.append("s", "b", "first?", "first answer")
    sessions.append("s", "b", "second?", "second answer")

    history = sessions.history("s", "b")
    history.clear()

    assert [turn.question for turn in sessions.history("s", "b")] == [
        "first?",
        "second?",
    ]


def test_history_is_scoped_to_the_session_and_the_book():
    sessions = SessionStore()
    sessions.append("s1", "b1", "q", "a")

    assert sessions.history("s2", "b1") == []
    assert sessions.history("s1", "b2") == []


def test_turns_are_capped_per_session():
    sessions = SessionStore(max_turns=2)
    for index in range(4):
        sessions.append("s", "b", "q{0}".format(index), "a{0}".format(index))

    assert [turn.question for turn in sessions.history("s", "b")] == ["q2", "q3"]


def test_the_oldest_session_is_evicted_first():
    sessions = SessionStore(max_sessions=2)
    sessions.append("s1", "b", "q", "a")
    sessions.append("s2", "b", "q", "a")
    sessions.append("s3", "b", "q", "a")

    assert sessions.history("s1", "b") == []
    assert sessions.history("s3", "b") != []


def test_recording_the_same_turn_twice_rewrites_it_in_place():
    """The stream and a racing cancel both write; one row must survive."""
    sessions = SessionStore()
    sessions.record("s", "b", "turn-1", "why?", "The full six-sentence answer.")

    sessions.record("s", "b", "turn-1", "why?", "The first sentence only.")

    history = sessions.history("s", "b")
    assert len(history) == 1
    assert history[0].answer == "The first sentence only."


def test_recording_an_empty_answer_drops_the_exchange():
    """Cut off before a word was spoken: the assistant said nothing."""
    sessions = SessionStore()
    sessions.record("s", "b", "turn-1", "why?", "An answer nobody heard.")

    sessions.record("s", "b", "turn-1", "why?", "")

    assert sessions.history("s", "b") == []


def test_truncating_a_turn_leaves_the_earlier_ones_alone():
    sessions = SessionStore()
    sessions.append("s", "b", "earlier?", "earlier answer")
    sessions.record("s", "b", "turn-2", "later?", "later answer")

    sessions.record("s", "b", "turn-2", "later?", "")

    assert [turn.question for turn in sessions.history("s", "b")] == ["earlier?"]


def test_appended_turns_get_distinct_ids():
    sessions = SessionStore()
    first = sessions.append("s", "b", "q1", "a1")
    second = sessions.append("s", "b", "q2", "a2")

    assert first.turn_id != second.turn_id
