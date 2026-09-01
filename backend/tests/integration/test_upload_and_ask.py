"""The product, end to end and offline: upload a book, then talk to it.

Every test here drives the HTTP surface of SPEC §6 with a real PDF and
:class:`~app.services.openai_client.FakeAIClient`. Nothing is stubbed below the
API boundary — the PDF is parsed, the structure detected, the chunks cut,
enriched, summarized and embedded, and the answer is streamed back through the
same SSE writer a browser would read. What is asserted is therefore the
*contract*: which events arrive, in what order, carrying what shape. The fake
client's prose is not evidence of anything and is never treated as such.

The cancellation test is the exception that justifies the machinery in
:mod:`_support`: it needs a stream that is still running, which means a real
socket, which means uvicorn rather than ``TestClient``.
"""

from __future__ import annotations

import base64

import httpx
from fastapi.testclient import TestClient

from _support import (  # noqa: F401 - fixtures are used by injection
    PacedAIClient,
    QuotingAIClient,
    RewritingAIClient,
    STAGE_ORDER,
    ask,
    book_pdf,
    ingest,
    ingested_book,
    iter_sse,
    names,
    payloads,
    poll_status,
    serve,
    upload,
)

from app.ingest.chunker import KIND_BOOK_SUMMARY, KIND_CHAPTER_SUMMARY
from app.main import create_app
from app.rag.citations import split_answer_sentences
from app.services.openai_client import FakeAIClient
from app.speech.pcm import SAMPLE_RATE, duration_seconds

QUESTION = "Who kept the lantern at Kestrel Cove?"


# ------------------------------------------------------------------ ingestion


def test_health_reports_that_the_app_runs_without_a_key(client):
    body = client.get("/api/health").json()

    assert body == {"ok": True, "openai_configured": False}


def test_ingestion_walks_the_stages_and_ends_ready(settings, book_pdf):
    """The library card polls ``/status``; this is what it sees.

    A paced client is used only so the stages last long enough to be observed —
    the assertions are about their order and monotonic progress, never timing.
    """
    ai = PacedAIClient(call_delay=0.01)
    with TestClient(create_app(settings=settings, ai=ai)) as client:
        response = upload(client, book_pdf)
        assert response.status_code == 202
        assert response.json()["status"] == "parsing"

        observed, final = poll_status(client, response.json()["book_id"])

    assert final == {"status": "ready", "progress": 1.0, "error": None}

    stages = []
    for status, _ in observed:
        if status not in stages:
            stages.append(status)
    # Two stages are skippable (SPEC §14.4/§14.5), so what a poller sees is a
    # subsequence of the pipeline's order — never a reordering of it.
    assert stages == [stage for stage in STAGE_ORDER if stage in stages]
    assert len(stages) >= 4, stages
    assert stages[0] == "parsing" and stages[-1] == "ready"

    progress = [value for _, value in observed]
    assert progress == sorted(progress)
    assert progress[-1] == 1.0


def test_re_uploading_the_same_book_returns_the_existing_one(
    client, book_pdf, fake_ai, settings
):
    """Same bytes, same book: idempotent by sha256, and no second ingestion."""
    first = upload(client, book_pdf)
    assert first.status_code == 202
    book_id = first.json()["book_id"]
    poll_status(client, book_id)
    embed_calls = len(fake_ai.embed_calls)

    again = upload(client, book_pdf, filename="a-different-name.pdf")

    assert again.status_code == 200
    assert again.json() == {"book_id": book_id, "status": "ready"}
    assert len(fake_ai.embed_calls) == embed_calls
    assert [book["book_id"] for book in client.get("/api/books").json()["books"]] == [
        book_id
    ]
    # The second upload was still spooled to disk to be hashed; nothing of it
    # may survive that check.
    assert [path.name for path in settings.books_dir.iterdir()] == [
        "{0}.pdf".format(book_id)
    ]


def test_the_library_lists_the_ready_book(client, ingested_book):
    books = client.get("/api/books").json()["books"]

    assert len(books) == 1
    card = books[0]
    assert card["book_id"] == ingested_book
    assert card["title"] == "A Field Guide to the Meridian Archipelago"
    assert card["status"] == "ready"
    assert card["progress"] == 1.0
    assert card["pages"] == 4
    assert card["created_at"]


# ------------------------------------------------------------------------ toc


def test_the_toc_matches_the_books_own_structure(client, ingested_book):
    body = client.get("/api/books/{0}/toc".format(ingested_book)).json()

    assert body["title"] == "A Field Guide to the Meridian Archipelago"
    chapters = body["chapters"]
    assert [chapter["idx"] for chapter in chapters] == [1, 2, 3, 4]
    assert [chapter["title"] for chapter in chapters] == [
        "Chapter 1. Kestrel Cove",
        "Chapter 2. Thornbury Reach",
        "Chapter 3. Sable Light",
        "Chapter 4. Wexford Passage",
    ]
    starts = [chapter["page_start"] for chapter in chapters]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    assert starts[0] == 1
    for chapter in chapters:
        assert chapter["page_start"] <= chapter["page_end"]
    # The chapters tile the book: the last one runs to the final page.
    assert chapters[-1]["page_end"] == client.get("/api/books").json()["books"][0]["pages"]

    # SPEC §14.5: this book was summarized, so every entry carries its synopsis
    # — the same prose the summary node was indexed with, not a re-derivation.
    store = client.app.state.store
    nodes = {
        node.chapter_idx: node.text.strip()
        for node in store.get_summaries(ingested_book, kind=KIND_CHAPTER_SUMMARY)
    }
    assert {chapter["idx"] for chapter in chapters} == set(nodes)
    for chapter in chapters:
        assert chapter["synopsis"] == nodes[chapter["idx"]]
    # The book-level node belongs to no chapter (§14.5 stores it under index 0)
    # and must not surface as one.
    book_synopsis = store.get_summaries(ingested_book, kind=KIND_BOOK_SUMMARY)[0].text
    assert book_synopsis.strip() not in [chapter["synopsis"] for chapter in chapters]


class SummarylessAIClient(FakeAIClient):
    """A fake that declines to summarize; everything else it does unchanged.

    Summarizing is skippable by design (SPEC §14.5), so "no synopsis" is a state
    the shipped app reaches — an LLM failure, a chapter too thin to summarize —
    and the TOC has to say so by *omitting* the key rather than by carrying a
    null the UI would have to special-case.
    """

    def _reply(self, messages, purpose):
        if purpose == "summary":
            return ""
        return super()._reply(messages, purpose)


def test_a_chapter_with_no_summary_has_no_synopsis_field(settings, book_pdf):
    with TestClient(create_app(settings=settings, ai=SummarylessAIClient())) as client:
        book_id = ingest(client, book_pdf)
        chapters = client.get("/api/books/{0}/toc".format(book_id)).json()["chapters"]
        assert client.app.state.store.get_summaries(book_id) == []

    assert chapters
    for chapter in chapters:
        assert "synopsis" not in chapter, chapter
        # The rest of the entry is untouched — including the page keys, which
        # are nullable and must not be dropped along with the missing synopsis.
        assert {"idx", "title", "page_start", "page_end"} <= set(chapter)


def test_each_chapter_gets_a_retrievable_summary_node(client, ingested_book):
    """SPEC §14.5: synopses are indexed alongside the leaf chunks, not beside."""
    store = client.app.state.store

    chapter_nodes = store.get_summaries(ingested_book, kind=KIND_CHAPTER_SUMMARY)
    book_nodes = store.get_summaries(ingested_book, kind=KIND_BOOK_SUMMARY)

    assert [node.chapter_idx for node in chapter_nodes] == [1, 2, 3, 4]
    assert all(node.text.strip() for node in chapter_nodes)
    assert len(book_nodes) == 1
    index = store.load_index(ingested_book)
    summaries = [chunk for chunk in index.chunks if chunk.is_summary]
    assert len(summaries) == 5
    assert index.matrix.shape[0] == len(index.chunks)


# ---------------------------------------------------------------- the answer


def test_the_answer_stream_follows_the_documented_event_order(client, ingested_book):
    """SPEC §5.2/§14.7/§14.8: status → question → sources → delta+/audio+ →
    citations → done.

    Deltas and audio interleave — a segment is spoken while the next sentence is
    still being written, which is the whole latency argument — so what is fixed
    is the *ordering* between kinds, not a grouping of them.
    """
    events = ask(client, ingested_book, QUESTION)
    order = names(events)

    assert set(order) == {
        "status",
        "question",
        "sources",
        "delta",
        "audio",
        "citations",
        "done",
    }
    assert order[:3] == ["status", "question", "sources"]
    assert order[-2:] == ["citations", "done"]
    assert order.index("sources") < order.index("delta") < order.index("audio")
    assert [data["stage"] for data in payloads(events, "status")] == [
        "retrieving",
        "generating",
        "speaking",
    ]

    turn_ids = {data["turn_id"] for _, data in events}
    assert len(turn_ids) == 1 and turn_ids.pop()

    assert payloads(events, "question")[0]["text"] == QUESTION
    done = payloads(events, "done")[0]
    assert done["answer"] == "".join(
        data["text"] for data in payloads(events, "delta")
    ).strip()
    assert set(done["timings"]) == {
        "rewrite_ms",
        "retrieve_ms",
        "first_token_ms",
        "first_audio_ms",
        "total_ms",
    }
    assert done["timings"]["first_audio_ms"] > 0


def test_the_sources_event_names_real_chunks_with_a_confidence(client, ingested_book):
    sources = payloads(ask(client, ingested_book, QUESTION), "sources")[0]

    assert 0.0 < sources["confidence"] <= 1.0
    known = {chunk.id for chunk in client.app.state.store.load_index(ingested_book).chunks}
    assert sources["sources"]
    for source in sources["sources"]:
        assert source["chunk_id"] in known
        assert source["chapter_title"]
        assert len(source["preview"]) <= 200
        assert len(source["pages"]) == 2


def test_audio_segments_are_ordered_pcm_the_client_can_schedule(client, ingested_book):
    """SPEC §14.8: raw 24 kHz s16le mono, in ``seq`` order, one per sentence."""
    events = ask(client, ingested_book, QUESTION)
    audio = payloads(events, "audio")

    assert audio
    assert [segment["seq"] for segment in audio] == list(range(len(audio)))
    for segment in audio:
        assert segment["format"] == "pcm"
        assert segment["sample_rate"] == SAMPLE_RATE == 24000
        pcm = base64.b64decode(segment["b64"])
        assert pcm and len(pcm) % 2 == 0  # whole 16-bit samples
        assert duration_seconds(pcm) > 0.0
    # Distinct sentences, distinct audio: a duplicated or dropped segment would
    # be inaudible in a test that only counted them.
    assert len({segment["b64"] for segment in audio}) == len(audio)


def test_an_answer_that_quotes_nothing_is_shipped_uncited(client, ingested_book):
    """§14.7's honest default: nothing above the floor means no citation.

    The stock fake's prose shares no wording with the book and its embeddings
    are seeded per text, so no sentence can clear 0.3 against any chunk. That is
    the ungrounded case, and it has to produce an empty map — not a
    best-available guess — while the answer itself still ships.
    """
    events = ask(client, ingested_book, QUESTION)

    assert payloads(events, "citations")[0]["map"] == []
    assert payloads(events, "done")[0]["answer"]


def test_a_quoted_answer_is_attributed_to_the_chunk_it_came_from(settings, book_pdf):
    """§14.7 with a map that is not empty — the stock fake cannot produce one.

    Its embeddings are seeded from ``sha1(text)``, so a generated sentence has
    no cosine relationship with anything and nothing clears the 0.3 floor; the
    map is correctly empty, and every assertion about its *contents* is
    therefore vacuous. :class:`QuotingAIClient` embeds lexically and answers by
    quoting the excerpt it was given, which is the case attribution exists for.
    """
    ai = QuotingAIClient()
    with TestClient(create_app(settings=settings, ai=ai)) as client:
        book_id = ingest(client, book_pdf)
        events = ask(client, book_id, QUESTION)
        material = {
            chunk.id: " ".join((chunk.text + " " + chunk.parent_text).split())
            for chunk in client.app.state.store.load_index(book_id).chunks
        }

    answer = payloads(events, "done")[0]["answer"]
    sentences = split_answer_sentences(answer)
    retrieved = {
        source["chunk_id"] for source in payloads(events, "sources")[0]["sources"]
    }
    citations = payloads(events, "citations")[0]["map"]

    assert citations, "an answer copied out of the book must be attributable"
    assert [entry["sentence_idx"] for entry in citations] == sorted(
        entry["sentence_idx"] for entry in citations
    )
    for entry in citations:
        assert 0 <= entry["sentence_idx"] < len(sentences)
        assert 1 <= len(entry["chunk_ids"]) <= 4
        assert set(entry["chunk_ids"]) <= retrieved
        quoted = sentences[entry["sentence_idx"]]
        assert any(quoted in material[chunk_id] for chunk_id in entry["chunk_ids"]), (
            quoted,
            entry["chunk_ids"],
        )


def test_a_follow_up_question_is_rewritten_against_the_history(settings, ingested_book):
    """SPEC §5.2 step 1: the second question is resolved before it is retrieved.

    The stock fake echoes the prompt's last line, which for the rewrite template
    is the original question — indistinguishable from "no rewrite happened". So
    this app gets a fake that answers with a different standalone question, and
    the ``question`` event has to carry *that*.
    """
    ai = RewritingAIClient()
    with TestClient(create_app(settings=settings, ai=ai)) as client:
        first = ask(client, ingested_book, QUESTION)
        after_first = list(ai.rewrite_prompts)
        second = ask(client, ingested_book, "And what about its ferry?")
        history = client.app.state.sessions.history("s1", ingested_book)

    # An opening question has no history, so no rewrite call is made at all.
    assert after_first == []
    assert payloads(first, "status")[0]["stage"] == "retrieving"
    assert payloads(first, "question")[0]["text"] == QUESTION

    assert payloads(second, "status")[0]["stage"] == "rewriting"
    assert "retrieving" in [data["stage"] for data in payloads(second, "status")]
    assert payloads(second, "question")[0]["text"] == RewritingAIClient.REWRITTEN

    assert len(ai.rewrite_prompts) == 1
    prompt = ai.rewrite_prompts[0]
    assert QUESTION in prompt
    assert payloads(first, "done")[0]["answer"] in prompt
    assert [turn.question for turn in history] == [QUESTION, "And what about its ferry?"]


def test_deleting_a_book_removes_the_row_and_the_file(client, ingested_book, settings):
    pdf = settings.books_dir / "{0}.pdf".format(ingested_book)
    assert pdf.exists()

    assert client.delete("/api/books/{0}".format(ingested_book)).status_code == 204

    assert not pdf.exists()
    assert client.get("/api/books").json()["books"] == []
    assert client.get("/api/books/{0}/status".format(ingested_book)).status_code == 404
    assert (
        client.post(
            "/api/chat",
            json={"book_id": ingested_book, "session_id": "s1", "question": QUESTION},
        ).status_code
        == 404
    )


# ------------------------------------------------------------- cancellation


def test_cancelling_mid_stream_stops_the_events_and_truncates_the_history(
    settings, ingested_book, serve
):
    """SPEC §14.8, over a real socket, while the answer is still being spoken.

    ``TestClient`` cannot express this: starlette runs the whole ASGI call
    before it returns a response, so "mid-stream" does not exist there. Here the
    stream is consumed event by event and the cancel is posted from a second
    connection the moment the first audio segment lands — which is what a
    browser does when the reader interrupts.
    """
    # The first sentence synthesizes fast and every later one slowly, so the
    # cancel (posted the instant the first audio event lands) deterministically
    # arrives while the tail is still in flight — even on a loaded machine.
    # A uniform delay cannot guarantee that: with TTS concurrency 2 the first
    # two sentences finish together, and the release loop can emit both before
    # the cancel round-trip completes.
    import itertools
    import time as _time

    class StaggeredTTSClient(PacedAIClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._tts_order = itertools.count()

        def tts(self, text):
            _time.sleep(0.1 if next(self._tts_order) == 0 else 2.0)
            return FakeAIClient.tts(self, text)

    ai = StaggeredTTSClient()
    app = create_app(settings=settings, ai=ai)
    base = serve(app)
    seen = []
    cancel = None

    with httpx.Client(base_url=base, timeout=30.0) as reader, httpx.Client(
        base_url=base, timeout=30.0
    ) as controller:
        with reader.stream(
            "POST",
            "/api/chat",
            json={
                "book_id": ingested_book,
                "session_id": "s1",
                "question": QUESTION,
            },
        ) as response:
            assert response.status_code == 200
            for name, data in iter_sse(response.iter_lines()):
                seen.append((name, data))
                if name == "audio" and cancel is None:
                    cancel = controller.post(
                        "/api/chat/turns/{0}/cancel".format(data["turn_id"]),
                        json={"played_seq": 0},
                    )

    assert cancel is not None and cancel.status_code == 200
    body = cancel.json()
    assert body["cancelled"] is True
    assert body["played_seq"] == 0
    assert body["answer"]

    order = names(seen)
    # The terminal events never arrive: the turn went silent instead of
    # finishing, which is the whole point of cancelling one.
    assert "done" not in order
    assert "citations" not in order
    assert "error" not in order

    audio = payloads(seen, "audio")
    assert [segment["seq"] for segment in audio] == list(range(len(audio)))
    # Speech stopped early: fewer segments reached the wire than the answer
    # has sentences. (Exactly how many were already synthesized when the
    # cancel landed varies with machine load, so no synthesis count is
    # asserted — only that the tail never played.)
    from app.rag.citations import split_answer_sentences

    written = "".join(data["text"] for data in payloads(seen, "delta")).strip()
    assert 0 < len(audio) < len(split_answer_sentences(written))

    spoken = "".join(data["text"] for data in payloads(seen, "delta")).strip()
    assert spoken.startswith(body["answer"])
    assert len(body["answer"]) < len(spoken)

    # The stored transcript is what was heard, not what was written: a follow-up
    # question is resolved against the sentence that actually reached the ear.
    history = app.state.sessions.history("s1", ingested_book)
    assert [turn.answer for turn in history] == [body["answer"]]


def test_the_toc_carries_each_chapters_opening_prose(client, ingested_book):
    """The reader shows the book's own words on a chapter page, not a blurb."""
    toc = client.get("/api/books/{0}/toc".format(ingested_book)).json()
    chapters = toc["chapters"]
    assert chapters, "the fixture book has chapters"
    with_opening = [c for c in chapters if c.get("opening")]
    assert with_opening, "at least one chapter carries its opening text"
    for chapter in with_opening:
        assert chapter["opening"] == chapter["opening"].strip()
        assert "\n" not in chapter["opening"]
        assert len(chapter["opening"]) <= 900
