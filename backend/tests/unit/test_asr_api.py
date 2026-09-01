"""The transcription endpoint, including the §14.9 vocabulary hardening."""

from __future__ import annotations

import pytest

from app.services.openai_client import FakeAIClient

AUDIO = ("clip.webm", b"fake-opus-bytes", "audio/webm")


def post(client, *, book_id=None, audio=AUDIO):
    data = {"book_id": book_id} if book_id is not None else None
    return client.post("/api/asr", files={"audio": audio}, data=data)


def test_a_clip_is_transcribed(client):
    response = post(client)

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == FakeAIClient().transcript
    assert body["duration_ms"] >= 0


def test_without_a_book_no_vocabulary_hint_is_sent(client, fake_ai):
    post(client)

    assert fake_ai.transcribe_prompts == [None]


def test_naming_a_book_sends_its_proper_nouns_as_a_prompt(
    client, fake_ai, ready_book
):
    post(client, book_id=ready_book)

    prompt = fake_ai.transcribe_prompts[0]
    assert prompt is not None
    assert "Kestrel" in prompt
    assert "Meridian" in prompt


def test_an_unknown_book_degrades_to_plain_transcription(client, fake_ai):
    """A bad book id must never cost the user their question."""
    response = post(client, book_id="no-such-book")

    assert response.status_code == 200
    assert fake_ai.transcribe_prompts == [None]


def test_a_mistranscribed_name_is_repaired_against_the_book(
    settings, store, ready_book
):
    from fastapi.testclient import TestClient

    from app.main import create_app

    ai = FakeAIClient(transcript="Who founded Castrel Cove on Meridien?")
    with TestClient(create_app(settings=settings, ai=ai)) as client:
        response = post(client, book_id=ready_book)

    assert response.json()["text"] == "Who founded Kestrel Cove on Meridian?"


def test_the_transcript_is_untouched_without_a_book(settings, store, ready_book):
    from fastapi.testclient import TestClient

    from app.main import create_app

    ai = FakeAIClient(transcript="Who founded Castrel Cove on Meridien?")
    with TestClient(create_app(settings=settings, ai=ai)) as client:
        response = post(client)

    assert response.json()["text"] == "Who founded Castrel Cove on Meridien?"


@pytest.mark.parametrize(
    "filename,status_code",
    [("clip.webm", 200), ("clip.m4a", 200), ("clip.txt", 415)],
)
def test_container_formats(client, filename, status_code):
    response = post(client, audio=(filename, b"bytes", "application/octet-stream"))

    assert response.status_code == status_code


def test_an_empty_recording_is_rejected(client):
    response = post(client, audio=("clip.webm", b"", "audio/webm"))

    assert response.status_code == 422


def test_an_oversized_recording_is_rejected(client, settings):
    payload = b"x" * (settings.max_upload_bytes + 1)

    response = post(client, audio=("clip.webm", payload, "audio/webm"))

    assert response.status_code == 413


def test_a_silent_recording_is_rejected(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(settings=settings, ai=FakeAIClient(transcript=" "))) as client:
        response = post(client)

    assert response.status_code == 422


def test_a_transcription_failure_is_a_502(client, fake_ai):
    def explode(*args, **kwargs):
        raise RuntimeError("upstream is down")

    fake_ai.transcribe = explode

    assert post(client).status_code == 502


def test_asr_without_a_key_reports_503(client_without_key):
    assert post(client_without_key).status_code == 503
