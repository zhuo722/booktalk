"""Shared machinery for the integration suite.

Three things live here that no unit test needs.

**Fakes that add one property each.** The stock
:class:`~app.services.openai_client.FakeAIClient` is deliberately featureless —
instant, unquotable and semantically blind — and each of those is exactly what
some part of the v2 contract needs in order to be visible at all.
:class:`PacedAIClient` takes measurable time, so ingestion has stages to poll
and an answer has a middle to interrupt. :class:`RewritingAIClient` returns a
different question from the one it was asked, so a rewrite can be told apart
from no rewrite. :class:`QuotingAIClient` embeds lexically and answers by
quoting the book, so citations have something to attribute. None of them changes
what the pipeline does; each removes one reason the pipeline's behaviour would
be indistinguishable from doing nothing.

**A real HTTP server.** ``TestClient`` cannot express a mid-stream anything:
starlette runs the whole ASGI call inside ``portal.call`` and hands back a
response whose body is already complete, so an SSE stream arrives as one string
and "cancel while it is running" has no meaning. :func:`serve` starts uvicorn on
an ephemeral loopback port instead — no egress, no fixture files, about fifty
milliseconds — and the cancellation test consumes the stream event by event over
a real socket, which is the only way to prove that nothing escapes after a
cancel.

**The fixture book.** A four-chapter PDF whose prose plants invented proper
nouns mid-sentence, because that is what the §14.9 vocabulary extractor reads
and what the ASR repair pass has to have to repair anything.

These are plain functions and fixtures in a module rather than a
``conftest.py``: ``tests/conftest.py`` is imported under the top-level name
``conftest`` and the unit suite imports from it by that name, which a second
``conftest.py`` under this directory would shadow.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import threading
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pytest
import uvicorn
from conftest import FixtureChapter

from app.index.bm25 import tokenize
from app.index.store import STATUS_FAILED, STATUS_READY
from app.services.openai_client import FakeAIClient

# ------------------------------------------------------------------- the book

#: Invented names, planted mid-sentence so :mod:`app.ingest.vocab` counts them
#: as proper nouns, and phonetically distinct from one another so the repair
#: pass in :mod:`app.speech.phonetics` has exactly one candidate per token.
CHAPTER_TITLES = (
    "Chapter 1. Kestrel Cove",
    "Chapter 2. Thornbury Reach",
    "Chapter 3. Sable Light",
    "Chapter 4. Wexford Passage",
)
CHAPTER_NAMES = (
    ("Kestrel", "Halvard"),
    ("Thornbury", "Sable"),
    ("Sable", "Halvard"),
    ("Wexford", "Kestrel"),
)
BOOK_TITLE = "A Field Guide to the Meridian Archipelago"

#: One sentence pattern per paragraph, so no two paragraphs share a line: a line
#: that repeats across pages is stripped as a running header, which would take
#: the planted names with it.
OPENINGS = (
    "The harbour at {a} was cut from basalt in 1873, and the {b} family kept "
    "its lantern burning for four generations.",
    "Charts of {a} still mark the shallow causeway that the {b} surveyors laid "
    "across the tidal flats one winter.",
    "A keeper walked out from {a} every evening at dusk while the {b} ledger "
    "recorded each wick that was trimmed.",
    "Fishing crews working out of {a} paid their quay dues to the {b} agent, "
    "who weighed the catch on a brass balance.",
)

#: Filler that deliberately shares no stem with the planted names: a lowercase
#: "meridian" scattered through the prose would disqualify "Meridian" from the
#: vocabulary, which is exactly the guard :mod:`app.ingest.vocab` is meant to
#: apply — and exactly the wrong thing to trip over here.
FILLER = (
    "basalt ferry orchard granite tide keeper market salt chart cove beacon "
    "fisher reef sail storm quarry bridge windmill cistern almanac ledger "
    "cooper tanner vineyard cliff gull anchor rope compass causeway lighthouse "
    "sextant barometer estuary shingle lantern rudder"
).split()


def filler_prose(seed: str, words: int = 55) -> str:
    """Deterministic, non-repeating filler that ends in a full stop."""
    generator = random.Random(seed)
    return " ".join(generator.choice(FILLER) for _ in range(words)) + "."


def fixture_chapters() -> List[FixtureChapter]:
    """The fixture book as :func:`conftest.book_pdf_factory` wants it."""
    chapters: List[FixtureChapter] = []
    for index, (title, (first, second)) in enumerate(
        zip(CHAPTER_TITLES, CHAPTER_NAMES), start=1
    ):
        paragraphs = [
            "{0} {1}".format(
                opening.format(a=first, b=second),
                filler_prose("c{0}p{1}".format(index, position)),
            )
            for position, opening in enumerate(OPENINGS)
        ]
        chapters.append(FixtureChapter(title=title, paragraphs=paragraphs))
    return chapters


@pytest.fixture
def book_pdf(book_pdf_factory):
    """The fixture book as a real PDF with an embedded table of contents."""
    return book_pdf_factory(
        fixture_chapters(), name="meridian.pdf", title=BOOK_TITLE, toc=True
    )


# ------------------------------------------------------------- paced fake AI


class PacedAIClient(FakeAIClient):
    """A :class:`FakeAIClient` whose calls take measurable wall time.

    The answers are identical — only the timing changes, and only so that a
    stage or a half-played answer exists long enough to be observed. Tests using
    this assert on *ordering*, never on durations.
    """

    def __init__(
        self, call_delay: float = 0.0, tts_delay: float = 0.0, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.call_delay = call_delay
        self.tts_delay = tts_delay

    def embed(self, texts: List[str]) -> List[List[float]]:
        time.sleep(self.call_delay)
        return super().embed(texts)

    def chat(self, messages: List[Dict[str, str]], **kw: Any) -> str:
        time.sleep(self.call_delay)
        return super().chat(messages, **kw)

    def chat_stream(
        self, messages: List[Dict[str, str]], **kw: Any
    ) -> Iterator[str]:
        time.sleep(self.call_delay)
        return super().chat_stream(messages, **kw)

    def tts(self, text: str) -> bytes:
        time.sleep(self.tts_delay)
        return super().tts(text)


class RewritingAIClient(FakeAIClient):
    """A fake whose rewrite step actually rewrites.

    The stock fake echoes the last line of the prompt, which for the rewrite
    template *is* the original question — true to the "already standalone" case
    but useless for proving that the rewritten query is what reached retrieval.
    This one answers with a fixed standalone question instead, so the
    ``question`` event has something to be compared against.
    """

    REWRITTEN = "Who kept the lantern at Kestrel Cove?"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        #: Every rewrite prompt seen, so a test can check the history reached it.
        self.rewrite_prompts: List[str] = []

    def _reply(self, messages: List[Dict[str, str]], purpose: str) -> str:
        if purpose == "rewrite":
            self.rewrite_prompts.append(messages[-1].get("content", ""))
            return self.REWRITTEN
        return super()._reply(messages, purpose)


#: The first numbered excerpt of an answer prompt, and the prose under it.
FIRST_EXCERPT = re.compile(
    r"^\[1\][^\n]*\n(?P<body>.+?)(?=\n\n\[\d+\]|\n\nQuestion:|\Z)", re.M | re.S
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> List[str]:
    return [piece for piece in _SENTENCE_END.split(text) if piece.strip()]


class QuotingAIClient(FakeAIClient):
    """A fake that quotes the book back, and embeds lexically rather than by hash.

    Both changes serve one purpose: making a *non-empty* citation map (SPEC
    §14.7) reachable offline. Attribution matches each generated sentence
    against the retrieved chunks, and the stock fake makes that impossible by
    construction — its answers share no wording with the book, and its
    embeddings are seeded from ``sha1(text)`` so that no two texts are ever
    similar. Here embeddings are a hashed bag of words (still deterministic,
    still offline, but overlapping text now points in overlapping directions)
    and the answer is the first excerpt, quoted verbatim.

    Which is the honest case to test, not a convenient one: an answer that
    really was lifted out of one chunk has exactly one correct attribution, and
    a citation layer that cannot find it is not doing anything.
    """

    def embed(self, texts: List[str]) -> List[List[float]]:
        self.embed_calls.append(list(texts))
        vectors: List[List[float]] = []
        for text in texts:
            vector = [0.0] * self.dims
            for token in tokenize(text):
                digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:8]
                vector[int(digest, 16) % self.dims] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors

    #: Sentences quoted back. Two is enough to show the map is per-sentence, and
    #: short enough that the whole answer stays under the back-pressure ceiling.
    QUOTED_SENTENCES = 2

    def _reply(self, messages: List[Dict[str, str]], purpose: str) -> str:
        if purpose != "answer":
            return super()._reply(messages, purpose)
        match = FIRST_EXCERPT.search(messages[-1].get("content", ""))
        if match is None:  # no excerpts retrieved: refuse, like the real prompt asks
            return "The book does not cover that."
        body = " ".join(match.group("body").split())
        return " ".join(_sentences(body)[: self.QUOTED_SENTENCES]) or body


# ------------------------------------------------------------- HTTP plumbing


def iter_sse(lines: Iterable[str]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Decode ``event:``/``data:`` pairs from a stream of SSE lines."""
    name: Optional[str] = None
    for raw in lines:
        line = raw.rstrip("\r")
        if line.startswith("event: "):
            name = line[len("event: ") :]
        elif line.startswith("data: ") and name is not None:
            yield name, json.loads(line[len("data: ") :])
            name = None


def sse_events(body: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Decode a fully buffered SSE body."""
    return list(iter_sse(body.split("\n")))


def ask(client, book_id: str, question: str, session_id: str = "s1"):
    """POST one question and decode the whole answer stream."""
    response = client.post(
        "/api/chat",
        json={"book_id": book_id, "session_id": session_id, "question": question},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return sse_events(response.text)


def names(events: Sequence[Tuple[str, Dict[str, Any]]]) -> List[str]:
    return [name for name, _ in events]


def payloads(
    events: Sequence[Tuple[str, Dict[str, Any]]], event: str
) -> List[Dict[str, Any]]:
    return [data for name, data in events if name == event]


def upload(client, path, filename: Optional[str] = None, content_type: str = "application/pdf"):
    """POST a file to ``/api/books``."""
    with open(str(path), "rb") as handle:
        return client.post(
            "/api/books",
            files={"file": (filename or path.name, handle.read(), content_type)},
        )


def upload_bytes(client, payload: bytes, filename: str = "book.pdf"):
    """POST raw bytes to ``/api/books`` (for the malformed-upload paths)."""
    return client.post(
        "/api/books", files={"file": (filename, payload, "application/pdf")}
    )


#: The stages ingestion may report, in the order it may report them. Two of them
#: are skippable (SPEC §14.4/§14.5), so an observed run is a *subsequence*.
STAGE_ORDER = (
    "parsing",
    "structuring",
    "chunking",
    "contextualizing",
    "summarizing",
    "embedding",
    STATUS_READY,
)


def poll_status(
    client, book_id: str, timeout: float = 60.0, interval: float = 0.005
) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """Poll ``/status`` until the book settles; return what was seen and the end.

    The observation list keeps one entry per *change*, so it is the ingestion's
    trajectory rather than a sample count.
    """
    observed: List[Tuple[str, float]] = []
    deadline = time.monotonic() + timeout
    while True:
        response = client.get("/api/books/{0}/status".format(book_id))
        assert response.status_code == 200, response.text
        payload = response.json()
        point = (payload["status"], round(float(payload["progress"]), 6))
        if not observed or observed[-1] != point:
            observed.append(point)
        if payload["status"] in (STATUS_READY, STATUS_FAILED):
            return observed, payload
        if time.monotonic() > deadline:
            raise AssertionError(
                "book {0} never settled; saw {1}".format(book_id, observed)
            )
        time.sleep(interval)


def ingest(client, path, filename: Optional[str] = None) -> str:
    """Upload a PDF and wait for it to be ready; returns the book id."""
    response = upload(client, path, filename=filename)
    assert response.status_code == 202, response.text
    book_id = response.json()["book_id"]
    _, final = poll_status(client, book_id)
    assert final["status"] == STATUS_READY, final
    return book_id


@pytest.fixture
def ingested_book(client, book_pdf) -> str:
    """The fixture book, uploaded through the API and ingested to ready."""
    return ingest(client, book_pdf)


@pytest.fixture
def serve():
    """Run an app on a real loopback HTTP server; returns its base URL.

    Yielded as a factory rather than a fixture value because the caller decides
    which AI client the app gets — a stream that can be cancelled has to be slow
    enough to still be running when the cancel arrives.
    """
    running: List[Tuple[uvicorn.Server, threading.Thread]] = []

    def start(app) -> str:
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            # Explicit asyncio: uvicorn's "auto" loop installs uvloop's policy
            # process-wide, which would outlive this fixture and reach every
            # TestClient in the rest of the session.
            loop="asyncio",
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, name="uvicorn-test", daemon=True)
        thread.start()
        running.append((server, thread))
        deadline = time.monotonic() + 15.0
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("the test server died during startup")
            if time.monotonic() > deadline:
                raise RuntimeError("the test server did not start in time")
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        return "http://127.0.0.1:{0}".format(port)

    yield start

    for server, thread in running:
        server.should_exit = True
        thread.join(timeout=15.0)
