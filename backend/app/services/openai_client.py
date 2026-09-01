"""The single seam between BookTalk and OpenAI (SPEC §9).

Every network call the app makes goes through :class:`AIClient`. That makes the
whole pipeline testable offline: swap in :class:`FakeAIClient` and nothing else
in the codebase changes.

Callers may tag a completion with ``purpose="rewrite" | "rerank" | "answer" |
"toc_levels" | "context" | "summary"`` — the last three are the ingest-time
calls of SPEC §14.2/§14.4/§14.5. :class:`OpenAIClient` ignores the tag;
:class:`FakeAIClient` uses it to return a canned result of the right *shape* for
that call site.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol

from app.config import Settings
from app.speech.pcm import SAMPLE_RATE, SAMPLE_WIDTH

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
REQUEST_TIMEOUT_S = 60.0
#: SPEC §14.8: raw 24 kHz s16le mono, no container and no decode step.
TTS_RESPONSE_FORMAT = "pcm"


class AIClient(Protocol):
    """The complete set of model calls BookTalk needs."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts; returns one vector per input, in order."""

    def transcribe(
        self, audio: bytes, filename: str, prompt: Optional[str] = None
    ) -> str:
        """Transcribe recorded audio; returns the raw transcript text.

        ``prompt`` biases the decoder towards a vocabulary it would otherwise
        mangle — for BookTalk, the book's own proper nouns (SPEC §14.9).
        """

    def chat_stream(self, messages: List[Dict[str, str]], **kw: Any) -> Iterator[str]:
        """Stream a chat completion, yielding text deltas."""

    def chat(self, messages: List[Dict[str, str]], **kw: Any) -> str:
        """Run a non-streaming chat completion, returning the full text."""

    def tts(self, text: str) -> bytes:
        """Synthesize speech; returns raw PCM (24 kHz, s16le, mono)."""


class OpenAIClient:
    """:class:`AIClient` backed by the official ``openai`` SDK."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # imported lazily so the app boots without a key

        self._settings = settings
        # base_url is always passed explicitly: with None the SDK falls back to
        # the OPENAI_BASE_URL env var, and a blank `OPENAI_BASE_URL=` line in
        # .env (loaded by dotenv as "") then silently breaks every request.
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or "https://api.openai.com/v1",
            max_retries=MAX_RETRIES,
            timeout=REQUEST_TIMEOUT_S,
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(
            model=self._settings.embed_model, input=list(texts)
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

    def transcribe(
        self, audio: bytes, filename: str, prompt: Optional[str] = None
    ) -> str:
        """Transcribe, biasing the decoder with ``prompt`` when one is given.

        The ``prompt`` parameter is on ``audio.transcriptions.create`` in the
        installed SDK and is accepted by both the gpt-4o transcribe models and
        whisper-1, but it is advisory and the semantics have moved before. So
        it degrades silently: if a request carrying a prompt is rejected, the
        same request is retried without it rather than failing the turn — a
        transcript with a mangled name beats no transcript at all.
        """

        def call(model: str, hint: Optional[str]) -> str:
            kwargs: Dict[str, Any] = {"model": model, "file": (filename, audio)}
            if hint:
                kwargs["prompt"] = hint
            result = self._client.audio.transcriptions.create(**kwargs)
            return (getattr(result, "text", "") or "").strip()

        def attempt(model: str) -> str:
            try:
                return call(model, prompt)
            except _PROMPT_UNSUPPORTED_ERRORS:
                if not prompt:
                    raise
                logger.info("ASR rejected the vocabulary prompt; retrying without it")
                return call(model, None)

        try:
            return attempt(self._settings.asr_model)
        except _MODEL_FALLBACK_ERRORS:
            return attempt(self._settings.asr_fallback_model)

    def chat_stream(self, messages: List[Dict[str, str]], **kw: Any) -> Iterator[str]:
        params = self._chat_params(kw)
        stream = self._client.chat.completions.create(
            messages=messages, stream=True, **params
        )
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece

    def chat(self, messages: List[Dict[str, str]], **kw: Any) -> str:
        params = self._chat_params(kw)
        response = self._client.chat.completions.create(messages=messages, **params)
        return (response.choices[0].message.content or "").strip()

    def tts(self, text: str) -> bytes:
        """Synthesize one sentence as raw PCM (SPEC §14.8).

        ``pcm`` is headerless 24 kHz s16le mono, which is what the browser's
        audio worklet wants anyway: no container to parse, no codec to decode,
        and a byte count that converts directly to a playback duration.
        """

        def call(model: str) -> bytes:
            response = self._client.audio.speech.create(
                model=model,
                voice=self._settings.tts_voice,
                input=text,
                response_format=TTS_RESPONSE_FORMAT,
            )
            return response.content

        try:
            return call(self._settings.tts_model)
        except _MODEL_FALLBACK_ERRORS:
            return call(self._settings.tts_fallback_model)

    def _chat_params(self, kw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize caller kwargs into SDK parameters.

        ``None`` means "unset", not "send null": call sites pass an optional
        model override straight through, and ``model=None`` would otherwise
        survive ``setdefault`` and reach the API as a null model name.
        """
        params = {key: value for key, value in kw.items() if value is not None}
        params.pop("purpose", None)
        params.setdefault("model", self._settings.llm_model)
        return params


def _fallback_errors() -> tuple:
    """Errors that mean "this model is unavailable" — retry with the fallback."""
    try:
        from openai import BadRequestError, NotFoundError, PermissionDeniedError

        return (NotFoundError, BadRequestError, PermissionDeniedError)
    except Exception:  # pragma: no cover - only if the SDK layout changes
        return (Exception,)


_MODEL_FALLBACK_ERRORS = _fallback_errors()


def _prompt_unsupported_errors() -> tuple:
    """Errors that mean "this request cannot carry a prompt" (SPEC §14.9).

    ``TypeError`` covers an SDK whose ``transcriptions.create`` has no such
    keyword at all; ``BadRequestError`` covers a model that rejects it.
    """
    try:
        from openai import BadRequestError, UnprocessableEntityError

        return (TypeError, BadRequestError, UnprocessableEntityError)
    except Exception:  # pragma: no cover - only if the SDK layout changes
        return (TypeError,)


_PROMPT_UNSUPPORTED_ERRORS = _prompt_unsupported_errors()


#: One second of fake speech per this many characters of text — enough that the
#: back-pressure estimate in tests has something proportional to measure.
FAKE_PCM_CHARS_PER_SECOND = 100.0

FAKE_TRANSCRIPT = "What does the book say about this chapter?"


def fake_pcm(text: str) -> bytes:
    """Deterministic PCM for one sentence: same text, same bytes, every time.

    Not silence — the bytes are the sha1 digest of the text, tiled — so a test
    can tell one segment from another and assert ordering, and so a truncated
    or duplicated segment is visible rather than indistinguishable. Length is
    proportional to the text, which is what makes the pacing tests meaningful.
    """
    seconds = max(1, len(text)) / FAKE_PCM_CHARS_PER_SECOND
    samples = max(1, int(seconds * SAMPLE_RATE))
    pattern = hashlib.sha1(text.encode("utf-8")).digest()
    total = samples * SAMPLE_WIDTH
    repeats = total // len(pattern) + 1
    return (pattern * repeats)[:total]


class FakeAIClient:
    """Deterministic offline stand-in for :class:`AIClient` (SPEC §9).

    Embeddings are seeded from ``sha1(text)``: identical text always yields an
    identical vector, but *similar* texts get unrelated vectors — tests may only
    rely on exact-text determinism, never on semantic similarity.
    """

    def __init__(self, dims: int = 1536, transcript: str = FAKE_TRANSCRIPT) -> None:
        self.dims = dims
        self.transcript = transcript
        self.embed_calls: List[List[str]] = []
        self.chat_calls: List[List[Dict[str, str]]] = []
        self.tts_calls: List[str] = []
        self.transcribe_calls: List[str] = []
        #: The vocabulary hint each ``transcribe`` call carried (SPEC §14.9).
        self.transcribe_prompts: List[Optional[str]] = []

    def embed(self, texts: List[str]) -> List[List[float]]:
        import numpy as np

        self.embed_calls.append(list(texts))
        vectors: List[List[float]] = []
        for text in texts:
            seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
            vector = np.random.RandomState(seed).normal(size=self.dims)
            norm = float(np.linalg.norm(vector)) or 1.0
            vectors.append((vector / norm).astype("float32").tolist())
        return vectors

    def transcribe(
        self, audio: bytes, filename: str, prompt: Optional[str] = None
    ) -> str:
        self.transcribe_calls.append(filename)
        self.transcribe_prompts.append(prompt)
        return self.transcript

    def chat_stream(self, messages: List[Dict[str, str]], **kw: Any) -> Iterator[str]:
        self.chat_calls.append(list(messages))
        return iter(_word_deltas(self._reply(messages, kw.get("purpose", "answer"))))

    def chat(self, messages: List[Dict[str, str]], **kw: Any) -> str:
        self.chat_calls.append(list(messages))
        return self._reply(messages, kw.get("purpose", "answer"))

    def tts(self, text: str) -> bytes:
        self.tts_calls.append(text)
        return fake_pcm(text)

    def _reply(self, messages: List[Dict[str, str]], purpose: str) -> str:
        last_user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user = message.get("content", "")
                break
        if purpose == "rewrite":
            return _last_line(last_user)
        if purpose == "rerank":
            return "[0, 1, 2, 3, 4, 5, 6, 7]"
        if purpose == "toc_levels":
            return _fake_toc_levels(last_user)
        if purpose == "context":
            return _fake_context(last_user)
        if purpose == "summary":
            return _fake_summary(last_user)
        chapter = _first_chapter_mention(last_user)
        return (
            "According to {chapter}, the excerpts cover exactly what you asked about. "
            "This canned answer comes from the offline fake client, so it repeats the "
            "retrieved material rather than reasoning about it."
        ).format(chapter=chapter)


def _word_deltas(text: str, per_delta: int = 3) -> List[str]:
    """Chop a canned answer into stream-shaped deltas."""
    words = text.split(" ")
    return [
        " ".join(words[i : i + per_delta]) + (" " if i + per_delta < len(words) else "")
        for i in range(0, len(words), per_delta)
    ]


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


_TOC_ENTRY = re.compile(r"^\s*\d+\.\s+(?P<title>.+?)\s*$")
_TAGGED = re.compile(r"<(?P<tag>excerpt|chapter[^>]*)>(?P<body>.*?)</", re.DOTALL)


def _fake_toc_levels(text: str) -> str:
    """Strict-JSON levels for the ingest-time TOC call (SPEC §14.2).

    Levels come from the offline heading families, so the canned answer is both
    deterministic and *correct* for a numbered outline — a test can assert that
    "Chapter 3" outranks "3.1 Ferries" without any network.
    """
    import json

    from app.ingest.headings import family_levels

    titles = [
        match.group("title")
        for match in (_TOC_ENTRY.match(line) for line in text.splitlines())
        if match
    ]
    if not titles:
        return "[]"
    levels = family_levels(titles) or [1] * len(titles)
    return json.dumps(
        [
            {"level": level, "title": title}
            for title, level in zip(titles, levels)
        ],
        ensure_ascii=False,
    )


def _fake_context(text: str) -> str:
    """A deterministic situating sentence built from the excerpt itself."""
    excerpt = _tagged_body(text, "excerpt")
    opening = " ".join(excerpt.split()[:12])
    return "This excerpt sits in the chapter and covers {0}.".format(opening or "the book")


def _fake_summary(text: str) -> str:
    """A deterministic synopsis: the first words of what was handed over."""
    body = " ".join(text.split())
    head = body.split(":", 1)[-1].strip() if ":" in body[:80] else body
    return "Summary: {0}.".format(" ".join(head.split()[:24]) or "the book")


def _tagged_body(text: str, tag: str) -> str:
    """Content of an ``<excerpt>``/``<chapter>`` block in a prompt."""
    for match in _TAGGED.finditer(text):
        if match.group("tag").split(" ")[0] == tag:
            return match.group("body").strip()
    return ""


def _first_chapter_mention(text: str) -> str:
    """Find the first chapter label in a context pack, for the canned answer."""
    latin = re.search(r"Chapter\s+\d+", text)
    if latin:
        return latin.group(0)
    cjk = re.search(r"第[一二三四五六七八九十百零0-9]+章", text)
    if cjk:
        return cjk.group(0)
    return "the book"


def build_client(settings: Settings) -> Optional[AIClient]:
    """Return a live client, or ``None`` when no API key is configured."""
    if not settings.openai_configured:
        return None
    return OpenAIClient(settings)


def batched(items: List[Any], size: int) -> Iterable[List[Any]]:
    """Yield ``items`` in consecutive slices of at most ``size``."""
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]
