"""The chat HTTP surface: the SSE wire format and the cancel endpoint (§14.8)."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Tuple

from app.speech.pcm import SAMPLE_RATE


def parse_sse(body: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Parse the wire format back into ``(event, data)`` pairs."""
    events: List[Tuple[str, Dict[str, Any]]] = []
    for block in body.split("\n\n"):
        name = payload = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                payload = line[len("data: ") :]
        if name and payload is not None:
            events.append((name, json.loads(payload)))
    return events


def ask(client, book_id: str, question: str = "Who built the harbour?"):
    response = client.post(
        "/api/chat",
        json={"book_id": book_id, "session_id": "s1", "question": question},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


def test_the_wire_stream_carries_turn_id_pcm_and_citations(client, ready_book):
    events = ask(client, ready_book)

    names = [name for name, _ in events]
    assert names[0] == "status"
    assert names[-1] == "done"
    assert names[-2] == "citations"

    turn_ids = {data["turn_id"] for _, data in events}
    assert len(turn_ids) == 1

    audio = [data for name, data in events if name == "audio"]
    assert audio
    assert all(item["format"] == "pcm" for item in audio)
    assert all(item["sample_rate"] == SAMPLE_RATE for item in audio)
    assert base64.b64decode(audio[0]["b64"])


def test_the_sources_event_reports_confidence_for_the_banner(client, ready_book):
    sources = dict(ask(client, ready_book))["sources"]

    assert "confidence" in sources
    assert 0.0 <= sources["confidence"] <= 1.0


def test_cancelling_a_finished_turn_truncates_its_stored_history(client, ready_book):
    events = ask(client, ready_book)
    turn_id = events[0][1]["turn_id"]
    sessions = client.app.state.sessions
    full = sessions.history("s1", ready_book)[0].answer

    response = client.post(
        "/api/chat/turns/{0}/cancel".format(turn_id), json={"played_seq": 0}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["turn_id"] == turn_id
    assert body["cancelled"] is True
    assert body["played_seq"] == 0
    stored = sessions.history("s1", ready_book)[0].answer
    assert stored == body["answer"]
    assert full.startswith(stored)
    assert len(stored) < len(full)


def test_cancelling_with_nothing_played_removes_the_exchange(client, ready_book):
    events = ask(client, ready_book)
    turn_id = events[0][1]["turn_id"]

    response = client.post(
        "/api/chat/turns/{0}/cancel".format(turn_id), json={"played_seq": -1}
    )

    assert response.status_code == 200
    assert response.json()["answer"] == ""
    assert client.app.state.sessions.history("s1", ready_book) == []


def test_cancelling_without_a_body_assumes_nothing_was_played(client, ready_book):
    turn_id = ask(client, ready_book)[0][1]["turn_id"]

    response = client.post("/api/chat/turns/{0}/cancel".format(turn_id))

    assert response.status_code == 200
    assert response.json()["played_seq"] == -1


def test_cancelling_an_unknown_turn_is_a_404(client):
    response = client.post("/api/chat/turns/nope/cancel", json={"played_seq": 0})

    assert response.status_code == 404


def test_an_empty_question_is_rejected_before_the_stream_opens(client, ready_book):
    response = client.post(
        "/api/chat", json={"book_id": ready_book, "session_id": "s1", "question": "  "}
    )

    assert response.status_code == 422


def test_an_unknown_book_is_a_404(client):
    response = client.post(
        "/api/chat", json={"book_id": "nope", "session_id": "s1", "question": "why?"}
    )

    assert response.status_code == 404


def test_chat_without_a_key_reports_503(client_without_key, ready_book):
    response = client_without_key.post(
        "/api/chat",
        json={"book_id": ready_book, "session_id": "s1", "question": "why?"},
    )

    assert response.status_code == 503
