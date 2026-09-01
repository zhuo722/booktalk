"""The speech half of the AIClient seam: PCM synthesis and biased ASR (§14.8/§14.9).

The live client is exercised against a stubbed SDK object — constructing an
:class:`OpenAIClient` makes no network call, so replacing its ``_client`` is
enough to see exactly which parameters would go over the wire.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.config import Settings
from app.services.openai_client import (
    TTS_RESPONSE_FORMAT,
    FakeAIClient,
    OpenAIClient,
    fake_pcm,
)
from app.speech.pcm import SAMPLE_RATE, duration_seconds


class StubSpeech:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        return type("Response", (), {"content": b"\x00\x01" * 100})()


class StubTranscriptions:
    """Records calls; optionally rejects any request carrying a prompt."""

    def __init__(self, reject_prompt=None, text: str = "heard it") -> None:
        self.calls: List[Dict[str, Any]] = []
        self.reject_prompt = reject_prompt
        self.text = text

    def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        if self.reject_prompt is not None and "prompt" in kwargs:
            raise self.reject_prompt
        return type("Response", (), {"text": self.text})()


def live_client(speech=None, transcriptions=None) -> OpenAIClient:
    client = OpenAIClient(Settings(openai_api_key="test-key-not-used"))
    audio = type(
        "Audio",
        (),
        {"speech": speech or StubSpeech(), "transcriptions": transcriptions or StubTranscriptions()},
    )()
    client._client = type("SDK", (), {"audio": audio})()
    return client


# --------------------------------------------------------------------- TTS


def test_synthesis_asks_for_raw_pcm():
    speech = StubSpeech()

    live_client(speech=speech).tts("The ferry leaves at dawn.")

    assert speech.calls[0]["response_format"] == TTS_RESPONSE_FORMAT == "pcm"
    assert speech.calls[0]["input"] == "The ferry leaves at dawn."


def test_the_fake_client_returns_deterministic_pcm():
    first = FakeAIClient()
    second = FakeAIClient()

    assert first.tts("The ferry leaves at dawn.") == second.tts(
        "The ferry leaves at dawn."
    )
    assert first.tts("A different sentence entirely.") != first.tts(
        "The ferry leaves at dawn."
    )


def test_fake_pcm_length_tracks_the_text_so_pacing_can_be_tested():
    short = fake_pcm("Short one.")
    long = fake_pcm("A considerably longer sentence than the one before it, by far.")

    assert len(long) > len(short)
    assert len(short) % 2 == 0
    assert duration_seconds(short) == pytest.approx(len("Short one.") / 100.0, abs=1e-4)
    assert SAMPLE_RATE == 24000


# --------------------------------------------------------------------- ASR


def test_a_vocabulary_prompt_is_passed_to_the_transcription_call():
    transcriptions = StubTranscriptions()

    live_client(transcriptions=transcriptions).transcribe(
        b"audio", "clip.webm", prompt="Kestrel, Meridian"
    )

    assert transcriptions.calls[0]["prompt"] == "Kestrel, Meridian"
    assert transcriptions.calls[0]["file"] == ("clip.webm", b"audio")


def test_no_prompt_key_is_sent_when_there_is_no_vocabulary():
    transcriptions = StubTranscriptions()

    live_client(transcriptions=transcriptions).transcribe(b"audio", "clip.webm")

    assert "prompt" not in transcriptions.calls[0]


def test_an_sdk_without_the_prompt_keyword_is_retried_without_it():
    """The prompt parameter is advisory; losing it must not lose the turn."""
    transcriptions = StubTranscriptions(
        reject_prompt=TypeError("create() got an unexpected keyword argument 'prompt'")
    )

    text = live_client(transcriptions=transcriptions).transcribe(
        b"audio", "clip.webm", prompt="Kestrel"
    )

    assert text == "heard it"
    assert [("prompt" in call) for call in transcriptions.calls] == [True, False]


def test_the_api_rejecting_a_prompt_counts_as_unsupported():
    from openai import BadRequestError

    from app.services.openai_client import _PROMPT_UNSUPPORTED_ERRORS

    assert BadRequestError in _PROMPT_UNSUPPORTED_ERRORS
    assert TypeError in _PROMPT_UNSUPPORTED_ERRORS


def test_a_failure_without_a_prompt_is_not_swallowed():
    """Only the prompt is optional; a genuinely broken call must still raise."""

    class Broken(StubTranscriptions):
        def create(self, **kwargs: Any):
            raise TypeError("something else entirely")

    with pytest.raises(TypeError):
        live_client(transcriptions=Broken()).transcribe(b"audio", "clip.webm")


def test_the_fake_client_records_the_prompt_it_was_given(fake_ai):
    fake_ai.transcribe(b"audio", "clip.webm", prompt="Kestrel")
    fake_ai.transcribe(b"audio", "clip.webm")

    assert fake_ai.transcribe_prompts == ["Kestrel", None]
