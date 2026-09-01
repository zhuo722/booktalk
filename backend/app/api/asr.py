"""Speech-to-text endpoint (SPEC §5.1, hardened by §14.9).

Transcription is where a voice RAG app quietly loses: the model gets ordinary
words right and the book's invented names wrong, and a corrupted name is a
retrieval miss the reader never sees. This endpoint therefore does two things v1
did not, both of them free and both of them only possible because the corpus is
*one book we have already read*:

1. **Bias the decoder.** When the request names a book, its most frequent proper
   nouns go into the transcription ``prompt``, which is the documented way to
   push spelling towards a known vocabulary.
2. **Repair what still came back wrong.** The transcript is run through a Double
   Metaphone pass over that same vocabulary, which rewrites a token only when
   exactly one name sounds like it (:mod:`app.speech.phonetics`).

``book_id`` is optional and unknown books are ignored rather than rejected: a
mistranscribed question is a much better outcome than no question at all.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from app.api import get_ai, get_settings, get_store
from app.models import AsrResponse
from app.speech.phonetics import (
    PROMPT_TERMS,
    asr_prompt,
    echoes_prompt,
    repair_transcript,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["asr"])

# MediaRecorder gives webm/opus on Chrome and Firefox, mp4/m4a on Safari; the
# rest are accepted because the transcription API takes them anyway.
ALLOWED_SUFFIXES = {
    ".webm",
    ".mp4",
    ".m4a",
    ".mp3",
    ".mpeg",
    ".mpga",
    ".wav",
    ".ogg",
    ".oga",
    ".flac",
}
MIN_TRANSCRIPT_CHARS = 2
#: Vocabulary loaded per request — enough to fill the prompt and give the repair
#: pass the names actually worth repairing.
VOCABULARY_LIMIT = 200
EMPTY_AUDIO = "No audio was received."
UNSUPPORTED = "Unsupported audio format: {0}"
NOTHING_HEARD = "No speech was detected in that recording. Try again."


@router.post("/asr", response_model=AsrResponse)
def transcribe_audio(
    request: Request,
    audio: UploadFile = File(...),
    book_id: Optional[str] = Form(default=None),
) -> AsrResponse:
    """Transcribe a recorded clip, biased towards one book's vocabulary.

    ``duration_ms`` reports how long the transcription itself took — the client
    knows the length of its own recording, and the server cannot measure audio
    duration without decoding the container.
    """
    ai = get_ai(request)
    settings = get_settings(request)

    filename = (audio.filename or "recording.webm").strip()
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=UNSUPPORTED.format(suffix)
        )

    payload = audio.file.read()
    if not payload:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=EMPTY_AUDIO)
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Recording is larger than the {0} MB limit.".format(
                settings.max_upload_mb
            ),
        )

    vocabulary = _vocabulary(request, book_id)
    prompt = asr_prompt(vocabulary, limit=PROMPT_TERMS)

    started = time.monotonic()
    try:
        text = ai.transcribe(payload, filename, prompt=prompt or None)
        # A hinted decoder that hears nothing sometimes returns the hint. The
        # vocabulary is worth having, so the retry drops the hint rather than
        # the question: without it the same audio transcribes normally, or
        # comes back empty and is reported as silence below.
        if prompt and echoes_prompt(text, prompt):
            logger.info("ASR echoed the vocabulary prompt; retrying without it")
            text = ai.transcribe(payload, filename)
            if echoes_prompt(text, prompt):
                text = ""
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a 502
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail="Transcription failed: {0}".format(exc),
        ) from exc
    duration_ms = int(round((time.monotonic() - started) * 1000))

    text = (text or "").strip()
    if len(text) < MIN_TRANSCRIPT_CHARS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=NOTHING_HEARD)
    if vocabulary:
        text = repair_transcript(text, vocabulary).text
    return AsrResponse(text=text, duration_ms=duration_ms)


def _vocabulary(request: Request, book_id: Optional[str]) -> List[str]:
    """The named book's proper nouns, or an empty list.

    Every failure here is silent by design: a missing book, an un-ingested one
    or a broken read all degrade to plain v1 transcription rather than costing
    the user their question.
    """
    identifier = (book_id or "").strip()
    if not identifier:
        return []
    try:
        return get_store(request).get_vocabulary(identifier, limit=VOCABULARY_LIMIT)
    except Exception:  # noqa: BLE001 - biasing is an enhancement, never a gate
        logger.warning("could not load vocabulary for book %s", identifier, exc_info=True)
        return []
