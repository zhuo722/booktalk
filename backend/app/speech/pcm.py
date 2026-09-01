"""Raw PCM audio: the one format the whole speech path speaks (SPEC §14.8).

v1 shipped mp3 segments and made the browser decode each one. Raw PCM removes
that step entirely: OpenAI's ``response_format="pcm"`` returns headerless
signed 16-bit little-endian mono samples at 24 kHz, which a WebAudio worklet can
push straight onto the timeline. No container, no codec, no ffmpeg on either
side — and, because the format is fixed and headerless, *byte length is
duration*, which is what the back-pressure estimate in
:mod:`app.speech.turns` runs on.

The numbers are not negotiable at the wire: an SSE ``audio`` event says
``{"format": "pcm", "sample_rate": 24000}`` so a client never has to guess, but
every producer in this codebase emits exactly these constants.
"""

from __future__ import annotations

#: OpenAI's ``pcm`` response format, fixed by the API.
SAMPLE_RATE = 24000
#: s16le — two bytes per sample.
SAMPLE_WIDTH = 2
CHANNELS = 1
AUDIO_FORMAT = "pcm"

BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS


def duration_seconds(pcm: bytes) -> float:
    """How long ``pcm`` plays for, in seconds.

    Exact rather than estimated: a headerless fixed-rate format has no framing
    overhead, so the byte count *is* the duration. A trailing odd byte (which
    should never happen) is ignored rather than counted as half a sample.
    """
    if not pcm:
        return 0.0
    return (len(pcm) // SAMPLE_WIDTH) / float(SAMPLE_RATE * CHANNELS)


def seconds_to_bytes(seconds: float) -> int:
    """Byte length of ``seconds`` of audio, rounded down to a whole sample."""
    samples = int(max(0.0, seconds) * SAMPLE_RATE * CHANNELS)
    return samples * SAMPLE_WIDTH
