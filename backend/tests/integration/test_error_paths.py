"""Everything that can go wrong, and what the reader is told when it does.

Two kinds of failure live here and they are answered differently on purpose.

*Rejections* — the wrong file type, a truncated download, something far too big
— are decided while the upload is still streaming, before a byte is kept or a
model call is made, and come back as a 4xx with a sentence a person can act on.

*Ingestion failures* — a corrupt PDF, a scan with no text layer, a missing API
key — cannot be known until the file is being read, so the upload is accepted
and the book is marked ``failed`` with its reason. That reason is the only thing
the library card has to show, so each of these tests asserts on the words the
reader will actually see, not on a status code alone.
"""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

from _support import (  # noqa: F401 - fixtures are used by injection
    ask,
    book_pdf,
    ingested_book,
    names,
    payloads,
    poll_status,
    upload,
    upload_bytes,
)

from app.main import create_app
from app.services.openai_client import FakeAIClient

QUESTION = "Who kept the lantern at Kestrel Cove?"


def chat(client, book_id, question=QUESTION, session_id="s1"):
    return client.post(
        "/api/chat",
        json={"book_id": book_id, "session_id": session_id, "question": question},
    )


# ------------------------------------------------------- rejected at the door


def test_a_file_that_is_not_a_pdf_is_refused(client):
    response = client.post(
        "/api/books", files={"file": ("notes.txt", b"%PDF-1.7 lying", "text/plain")}
    )

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]
    assert client.get("/api/books").json()["books"] == []


def test_a_pdf_without_the_magic_bytes_is_refused(client, settings):
    response = upload_bytes(client, b"GIF89a not a pdf at all", filename="book.pdf")

    assert response.status_code == 400
    assert "%PDF" in response.json()["detail"]
    # Nothing was kept: a rejected upload leaves no book and no spooled file.
    assert client.get("/api/books").json()["books"] == []
    assert list(settings.books_dir.iterdir()) == []


def test_an_empty_upload_is_refused(client):
    response = upload_bytes(client, b"", filename="book.pdf")

    assert response.status_code == 400


def test_a_file_over_the_size_limit_is_refused(client, settings):
    payload = b"%PDF-1.7\n" + b"0" * settings.max_upload_bytes

    response = upload_bytes(client, payload)

    assert response.status_code == 413
    assert str(settings.max_upload_mb) in response.json()["detail"]
    assert list(settings.books_dir.iterdir()) == []


# ------------------------------------------------------- failed during ingest


def test_a_corrupt_pdf_is_accepted_then_fails_with_a_reason(client):
    response = upload_bytes(client, b"%PDF-1.7\nnot really a pdf\n%%EOF\n")
    assert response.status_code == 202

    book_id = response.json()["book_id"]
    _, final = poll_status(client, book_id)

    assert final["status"] == "failed"
    assert "corrupted" in final["error"]
    assert client.app.state.store.count_chunks(book_id) == 0


def test_a_scanned_pdf_is_rejected_with_an_actionable_message(
    client, pdf_lines_factory
):
    """SPEC §14.1: an image-only book is refused before a model call is spent."""
    scan = pdf_lines_factory([[], [], []], name="scan.pdf")

    book_id = upload(client, scan).json()["book_id"]
    _, final = poll_status(client, book_id)

    assert final["status"] == "failed"
    assert "no text layer" in final["error"]
    assert "OCR" in final["error"]


def test_a_failed_book_can_be_re_uploaded_and_tried_again(client):
    broken = upload_bytes(client, b"%PDF-1.7\nnot really a pdf\n%%EOF\n")
    book_id = broken.json()["book_id"]
    _, final = poll_status(client, book_id)
    assert final["status"] == "failed"

    # The same bytes again: a ready book would short-circuit with 200, a failed
    # one is worth another attempt.
    again = upload_bytes(client, b"%PDF-1.7\nnot really a pdf\n%%EOF\n")

    assert again.status_code == 202
    assert again.json()["book_id"] == book_id
    assert poll_status(client, book_id)[1]["status"] == "failed"


def test_without_a_key_a_book_fails_with_configuration_instructions(
    client_without_key, book_pdf
):
    book_id = upload(client_without_key, book_pdf).json()["book_id"]

    _, final = poll_status(client_without_key, book_id)

    assert final["status"] == "failed"
    assert "OPENAI_API_KEY" in final["error"]
    assert ".env" in final["error"]


# ------------------------------------------------------------- unknown books


def test_unknown_books_are_404_everywhere(client):
    assert client.get("/api/books/nope/status").status_code == 404
    assert client.get("/api/books/nope/toc").status_code == 404
    assert client.delete("/api/books/nope").status_code == 404
    assert chat(client, "nope").status_code == 404
    assert chat(client, "nope").json()["detail"] == "Unknown book."


# -------------------------------------------------------------- not yet ready


class BlockingAIClient(FakeAIClient):
    """A fake that parks inside ``embed`` until the test lets it go.

    It makes "the book is not ready yet" a state the test *holds* rather than
    one it races towards: without it, asking a question mid-ingestion is a
    coin flip on how fast the machine is.
    """

    def __init__(self) -> None:
        super().__init__()
        self.reached = threading.Event()
        self.release = threading.Event()

    def embed(self, texts):
        self.reached.set()
        self.release.wait(30.0)
        return super().embed(texts)


def test_a_book_that_is_still_ingesting_answers_409(settings, book_pdf):
    ai = BlockingAIClient()
    with TestClient(create_app(settings=settings, ai=ai)) as client:
        try:
            book_id = upload(client, book_pdf).json()["book_id"]
            assert ai.reached.wait(30.0), "ingestion never reached the embedding stage"

            status = client.get("/api/books/{0}/status".format(book_id)).json()
            response = chat(client, book_id)

            assert status["status"] == "embedding"
            assert response.status_code == 409
            assert "still being processed" in response.json()["detail"]
            assert status["status"] in response.json()["detail"]
        finally:
            ai.release.set()

        assert poll_status(client, book_id)[1]["status"] == "ready"
        assert chat(client, book_id).status_code == 200


def test_a_failed_book_answers_409_with_the_ingestion_error(client):
    book_id = upload_bytes(client, b"%PDF-1.7\nnot really a pdf\n%%EOF\n").json()[
        "book_id"
    ]
    _, final = poll_status(client, book_id)

    response = chat(client, book_id)

    assert final["status"] == "failed"
    assert response.status_code == 409
    assert response.json()["detail"] == final["error"]


# ------------------------------------------------------------- bad questions


def test_a_blank_question_is_refused_before_the_stream_opens(client, ingested_book):
    for question in ("", "   ", "\n\t"):
        response = chat(client, ingested_book, question=question)
        assert response.status_code == 422, question


def test_a_malformed_chat_request_is_refused(client, ingested_book):
    assert client.post("/api/chat", json={"book_id": ingested_book}).status_code == 422
    assert client.post("/api/chat", json={}).status_code == 422


# ------------------------------------------------------------------- no key


def test_without_a_key_the_model_endpoints_report_503(client_without_key):
    health = client_without_key.get("/api/health").json()
    chat_response = chat(client_without_key, "any-book")
    asr_response = client_without_key.post(
        "/api/asr", files={"audio": ("clip.webm", b"bytes", "audio/webm")}
    )

    assert health == {"ok": True, "openai_configured": False}
    assert chat_response.status_code == 503
    assert asr_response.status_code == 503
    for response in (chat_response, asr_response):
        detail = response.json()["detail"]
        assert "OPENAI_API_KEY" in detail and ".env" in detail
    # The rest of the app still works without a key.
    assert client_without_key.get("/api/books").status_code == 200


# --------------------------------------------------------- failure mid-stream


class BrokenAIClient(FakeAIClient):
    """A fake whose generation call dies after retrieval has already succeeded."""

    def chat_stream(self, messages, **kw):
        raise RuntimeError("the model is down")


def test_a_model_failure_mid_stream_becomes_an_error_event(settings, ingested_book):
    """SPEC §5.3: the stream has already begun, so the failure is *in* it.

    The status code was committed the moment the first event was written, which
    is why the client is told about this in-band and the connection is closed
    rather than reset.
    """
    with TestClient(create_app(settings=settings, ai=BrokenAIClient())) as client:
        events = ask(client, ingested_book, QUESTION)

    order = names(events)
    assert order[-1] == "error"
    assert "done" not in order
    assert "the model is down" in payloads(events, "error")[0]["message"]
    assert payloads(events, "error")[0]["turn_id"]
    # Retrieval had already succeeded, so the sources the reader can see are
    # still on screen when the retry button appears.
    assert payloads(events, "sources")[0]["sources"]


# --------------------------------------------------------------- bad audio


def test_an_unsupported_audio_container_is_refused(client):
    response = client.post(
        "/api/asr", files={"audio": ("clip.txt", b"bytes", "text/plain")}
    )

    assert response.status_code == 415
    assert ".txt" in response.json()["detail"]


def test_an_oversized_recording_is_refused(client, settings):
    payload = b"0" * (settings.max_upload_bytes + 1)

    response = client.post(
        "/api/asr", files={"audio": ("clip.webm", payload, "audio/webm")}
    )

    assert response.status_code == 413


def test_a_recording_with_no_speech_is_refused(settings):
    with TestClient(create_app(settings=settings, ai=FakeAIClient(transcript="  "))) as client:
        response = client.post(
            "/api/asr", files={"audio": ("clip.webm", b"bytes", "audio/webm")}
        )

    assert response.status_code == 422
    assert "No speech" in response.json()["detail"]
