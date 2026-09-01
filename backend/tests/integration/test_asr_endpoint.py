"""Transcription against a book that was really ingested (SPEC §5.1, §14.9).

The vocabulary hardening is only meaningful end to end. A hand-written word list
proves that :mod:`app.speech.phonetics` can repair a token; it proves nothing
about whether the names the extractor *actually harvests from a PDF* are the
ones the decoder needs biasing towards, or whether they survive chunking,
enrichment and persistence on the way. So every test here transcribes against
the fixture book's own stored vocabulary, harvested at ingest from its prose.

The last test closes the loop the product is built on: a mistranscribed name is
repaired, and the repaired question is the one that reaches retrieval.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from _support import (  # noqa: F401 - fixtures are used by injection
    ask,
    book_pdf,
    ingested_book,
    payloads,
)

from app.main import create_app
from app.services.openai_client import FakeAIClient

CLIP = ("question.webm", b"fake-opus-bytes", "audio/webm")

#: What the decoder heard: two of the book's invented names, mangled the way an
#: ASR model mangles names it has never seen.
MISHEARD = "Who kept the lantern at Castrel Cove for the Halvarde family?"
CORRECTED = "Who kept the lantern at Kestrel Cove for the Halvard family?"


def transcribe(client, book_id=None, clip=CLIP):
    data = {"book_id": book_id} if book_id is not None else None
    return client.post("/api/asr", files={"audio": clip}, data=data)


def test_a_clip_is_transcribed(client, ingested_book):
    response = transcribe(client, ingested_book)

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == FakeAIClient().transcript
    assert body["duration_ms"] >= 0


def test_the_books_own_names_are_sent_as_a_decoding_hint(
    client, fake_ai, ingested_book
):
    """§14.9: the prompt carries names harvested from this book's prose."""
    transcribe(client, ingested_book)

    prompt = fake_ai.transcribe_prompts[0]
    assert prompt is not None
    stored = client.app.state.store.get_vocabulary(ingested_book)
    assert stored  # the extractor found something to bias towards
    assert "Kestrel" in stored and "Halvard" in stored
    # Frequency order survives into the hint, which is what makes truncating it
    # safe: the names most likely to be spoken are the ones that fit.
    positions = [prompt.index(term) for term in stored]
    assert positions == sorted(positions)
    # Names, not prose: the hint biases spelling, it does not feed the model
    # the book.
    assert "basalt" not in prompt and "harbour" not in prompt


def test_no_book_means_no_hint_and_no_repair(settings, ingested_book):
    """Without a book there is no vocabulary, so nothing is biased or repaired."""
    ai = FakeAIClient(transcript=MISHEARD)
    with TestClient(create_app(settings=settings, ai=ai)) as plain:
        response = transcribe(plain)

    assert response.status_code == 200
    assert response.json()["text"] == MISHEARD
    assert ai.transcribe_prompts == [None]


def test_a_mistranscribed_name_is_repaired_against_the_ingested_book(
    settings, ingested_book
):
    ai = FakeAIClient(transcript=MISHEARD)
    with TestClient(create_app(settings=settings, ai=ai)) as client:
        response = transcribe(client, ingested_book)

    assert response.status_code == 200
    assert response.json()["text"] == CORRECTED


def test_ordinary_words_are_never_rewritten_into_names(settings, ingested_book):
    """The dangerous failure is the invisible one: a plausible wrong question."""
    heard = "Tell me about the harbour and the ferry timetable, please."
    ai = FakeAIClient(transcript=heard)
    with TestClient(create_app(settings=settings, ai=ai)) as client:
        response = transcribe(client, ingested_book)

    assert response.json()["text"] == heard


def test_an_unknown_book_degrades_to_plain_transcription(client, fake_ai, ingested_book):
    """A stale book id must cost the reader their retrieval quality, not their question."""
    response = transcribe(client, "no-such-book")

    assert response.status_code == 200
    assert fake_ai.transcribe_prompts == [None]


def test_the_repaired_question_is_the_one_that_reaches_retrieval(
    settings, ingested_book
):
    """The voice round trip: microphone → ASR → repair → chat → sources.

    What this pins is the wiring and the consequence, not the win: the question
    the retriever sees is the repaired one, and a repaired name changes what the
    book gives back. *How much better* the repaired retrieval is belongs to the
    eval harness's ASR-corruption arm (§14.10), which has a golden set to
    measure against; a fake client's retrieval order is not evidence of quality
    and must not be asserted as if it were.
    """
    ai = FakeAIClient(transcript=MISHEARD)
    with TestClient(create_app(settings=settings, ai=ai)) as client:
        transcript = transcribe(client, ingested_book).json()["text"]
        # Separate sessions: a second question in the same one would be
        # rewritten against the first, and this test is not about that.
        repaired = ask(client, ingested_book, transcript, session_id="repaired")
        as_heard = ask(client, ingested_book, MISHEARD, session_id="as-heard")

    assert transcript == CORRECTED
    assert payloads(repaired, "question")[0]["text"] == CORRECTED

    def retrieved(events):
        return [source["chunk_id"] for source in payloads(events, "sources")[0]["sources"]]

    assert retrieved(repaired) != retrieved(as_heard)
    chapters = {
        source["chapter_idx"] for source in payloads(repaired, "sources")[0]["sources"]
    }
    assert 1 in chapters  # "Chapter 1. Kestrel Cove", the chapter it names


def test_an_empty_recording_is_rejected(client, ingested_book):
    response = transcribe(client, ingested_book, clip=("q.webm", b"", "audio/webm"))

    assert response.status_code == 422


def test_a_decoder_that_repeats_the_prompt_is_not_believed(settings, ingested_book):
    """A hinted decoder that hears nothing sometimes returns the hint itself.

    The user records a question and what comes back is the list of names the
    server sent — which would then be answered as though they had asked it.
    """

    class EchoingAIClient(FakeAIClient):
        def transcribe(self, audio, filename, prompt=None):
            super().transcribe(audio, filename, prompt)
            # Echoes only while it is being hinted, exactly as the real one does.
            return prompt if prompt else "Who kept the lantern at Kestrel Cove?"

    ai = EchoingAIClient()
    with TestClient(create_app(settings=settings, ai=ai)) as client:
        response = transcribe(client, ingested_book)

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Who kept the lantern at Kestrel Cove?"
    assert not body["text"].lower().startswith("proper nouns")
    # It was asked twice: once with the hint, then again without it.
    assert ai.transcribe_prompts[0] and ai.transcribe_prompts[1] is None
