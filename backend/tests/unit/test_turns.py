"""Turn identity, cancellation and TTS back-pressure (SPEC §14.8)."""

from __future__ import annotations

import pytest

from app.speech.pcm import BYTES_PER_SECOND, SAMPLE_RATE, duration_seconds, seconds_to_bytes
from app.speech.turns import (
    MAX_UNPLAYED_SECONDS,
    NOTHING_PLAYED,
    AudioPacer,
    TurnRegistry,
    TurnState,
)


class Clock:
    """A hand-cranked monotonic clock, so pacing tests never sleep."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def pcm(seconds: float) -> bytes:
    return b"\x00" * seconds_to_bytes(seconds)


# ------------------------------------------------------------------- duration


def test_pcm_byte_length_is_a_duration():
    assert duration_seconds(pcm(2.5)) == pytest.approx(2.5)
    assert duration_seconds(b"") == 0.0
    assert BYTES_PER_SECOND == SAMPLE_RATE * 2


def test_an_odd_trailing_byte_is_not_half_a_sample():
    assert duration_seconds(b"\x00" * 3) == pytest.approx(1.0 / SAMPLE_RATE)


# ---------------------------------------------------------------------- turns


def test_a_turn_records_what_each_segment_says():
    turn = TurnState("t1", "session", "book", "why?")

    turn.add_sentence(0, "The harbour was built in 1873.")
    turn.add_sentence(1, "It guards the northern approach.")

    assert turn.sentences == [
        "The harbour was built in 1873.",
        "It guards the northern approach.",
    ]


def test_history_is_truncated_to_the_sentences_actually_played():
    turn = TurnState("t1", "session", "book", "why?")
    for seq, sentence in enumerate(["First one.", "Second one.", "Third one."]):
        turn.add_sentence(seq, sentence)

    turn.cancel(played_seq=1)

    assert turn.cancelled is True
    assert turn.spoken_answer() == "First one. Second one."


def test_a_turn_cancelled_before_a_word_was_spoken_has_no_answer():
    turn = TurnState("t1", "session", "book", "why?")
    turn.add_sentence(0, "Never heard.")

    turn.cancel(played_seq=NOTHING_PLAYED)

    assert turn.spoken_answer() == ""


def test_the_registry_hands_out_unique_ids_and_finds_them_again():
    registry = TurnRegistry()

    first = registry.create("s", "b", "one?")
    second = registry.create("s", "b", "two?")

    assert first.turn_id != second.turn_id
    assert registry.get(first.turn_id) is first
    assert registry.get("nope") is None


def test_cancelling_an_unknown_turn_is_not_an_error():
    assert TurnRegistry().cancel("missing", played_seq=2) is None


def test_the_registry_evicts_the_oldest_turns():
    registry = TurnRegistry(max_turns=2)

    first = registry.create("s", "b", "one?")
    registry.create("s", "b", "two?")
    registry.create("s", "b", "three?")

    assert registry.get(first.turn_id) is None


# --------------------------------------------------------------- back-pressure


def test_the_pacer_pauses_once_the_client_is_five_seconds_behind():
    clock = Clock()
    pacer = AudioPacer(clock=clock)

    pacer.emitted(pcm(4.0))
    assert pacer.should_pause() is False

    pacer.emitted(pcm(3.0))
    assert pacer.unplayed_seconds() == pytest.approx(7.0)
    assert pacer.should_pause() is True
    assert MAX_UNPLAYED_SECONDS == 5.0


def test_the_backlog_drains_as_the_wall_clock_advances():
    clock = Clock()
    pacer = AudioPacer(clock=clock)
    pacer.emitted(pcm(7.0))

    clock.advance(3.0)

    assert pacer.unplayed_seconds() == pytest.approx(4.0)
    assert pacer.should_pause() is False


def test_the_pacer_is_idle_before_any_audio_is_emitted():
    pacer = AudioPacer(clock=Clock())

    assert pacer.unplayed_seconds() == 0.0
    assert pacer.should_pause() is False


def test_playback_never_runs_negative():
    clock = Clock()
    pacer = AudioPacer(clock=clock)
    pacer.emitted(pcm(1.0))

    clock.advance(60.0)

    assert pacer.unplayed_seconds() == 0.0


def test_an_acknowledgement_moves_the_playback_clock():
    clock = Clock()
    pacer = AudioPacer(clock=clock)
    pacer.emitted(pcm(9.0))
    assert pacer.should_pause() is True

    pacer.acknowledge(9.0)

    assert pacer.unplayed_seconds() == pytest.approx(0.0)
