"""Post-hoc sentence → chunk attribution (SPEC §14.7).

v1 asked the model to write "(Chapter 3)" inline. That is the wrong instrument
for a product that is *spoken*: the marker is read aloud, it costs tokens on the
critical path, and the model is guessing at attribution it was never shown.

So citations are computed after the fact instead. The finished answer is split
into sentences, the sentences are embedded in one batch, and each is matched
against the chunks that were actually retrieved — the same trick RAGFlow's
``insert_citations`` uses. Two signals, because neither is sufficient alone:

* **vector similarity (weight 0.9)** catches paraphrase, which is most of what
  a generated sentence is;
* **token overlap (weight 0.1)** catches the names, numbers and dates that
  embeddings smooth away — and, being capped at 0.1, can never manufacture a
  citation on function words alone.

The threshold walks *down* — 0.63, then ×0.8 per step, never below 0.3 — so a
confident match is required first and a weaker one is accepted only when
nothing better exists. A sentence with no match above the floor simply gets no
citation: "we do not know where this came from" is the honest answer, and the
UI can show the sentence unmarked.

This module is pure: one embedding call, no database, no HTTP, no globals. It
runs concurrently with the TTS tail, so it adds nothing to time-to-first-audio.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from app.index.bm25 import tokenize
from app.index.dense import normalize, to_matrix
from app.index.store import BookIndex
from app.rag.retriever import Retrieval
from app.services.openai_client import AIClient

logger = logging.getLogger(__name__)

TK_WEIGHT = 0.1
VT_WEIGHT = 0.9
START_THRESHOLD = 0.63
THRESHOLD_DECAY = 0.8
THRESHOLD_FLOOR = 0.3
MAX_SOURCES = 4


@dataclass
class CitationChunk:
    """One candidate source: the chunk's id, its text and its embedding."""

    chunk_id: str
    text: str
    embedding: Sequence[float]


def attribute_citations(
    ai: AIClient,
    answer: str,
    chunks: Sequence[CitationChunk],
    tkweight: float = TK_WEIGHT,
    vtweight: float = VT_WEIGHT,
    threshold: float = START_THRESHOLD,
    decay: float = THRESHOLD_DECAY,
    floor: float = THRESHOLD_FLOOR,
    max_sources: int = MAX_SOURCES,
) -> List[Dict[str, Any]]:
    """Map each sentence of ``answer`` onto the chunks that support it.

    Returns ``[{"sentence_idx": int, "chunk_ids": [str, ...]}]`` with at most
    ``max_sources`` ids per sentence, ordered by similarity. Sentences with no
    match above the floor are omitted entirely, so an answer grounded in
    nothing returns ``[]``.
    """
    spans = split_answer_sentence_spans(answer)
    sentences = [piece for piece, _, _ in spans]
    if not sentences or not chunks:
        return []

    vectors = _embed(ai, sentences)
    if vectors is None:
        return []
    chunk_matrix = normalize(_chunk_matrix(chunks))
    chunk_tokens = [frozenset(tokenize(chunk.text)) for chunk in chunks]

    citations: List[Dict[str, Any]] = []
    for position, sentence in enumerate(sentences):
        vector_sim = _vector_similarity(chunk_matrix, vectors[position])
        token_sim = _token_similarity(sentence, chunk_tokens)
        similarity = tkweight * token_sim + vtweight * vector_sim
        picks = _pick(similarity, threshold, decay, floor, max_sources)
        if picks:
            citations.append(
                {
                    "sentence_idx": position,
                    # Character offsets into the answer, so the client can
                    # highlight by slicing instead of re-splitting.
                    "span": [spans[position][1], spans[position][2]],
                    "chunk_ids": [chunks[index].chunk_id for index in picks],
                }
            )
    return citations


def citation_chunks(index: BookIndex, retrieval: Retrieval) -> List[CitationChunk]:
    """Adapt a retrieval result into citation candidates.

    Only the chunks that actually reached the model are candidates.
    ``retrieval.scored`` is the whole top-8 selection, but a section block can
    still be dropped by the context budget, and a chunk from a dropped block
    fails on both counts: the model never read its text, so a sentence cannot
    have come from it, and the client was never sent its id, so it has nothing
    to render a citation against — ``sources`` is the only id vocabulary the
    client has.

    The embeddings are read straight out of the book's in-memory matrix (they
    are already L2-normalized), so attribution never re-embeds the book.
    """
    rows = index.matrix.shape[0] if index.matrix.size else 0
    shown = {source.chunk_id for source in retrieval.sources}
    return [
        CitationChunk(
            chunk_id=item.chunk.id,
            text=item.chunk.text,
            embedding=index.matrix[item.row],
        )
        for item in retrieval.scored
        if 0 <= item.row < rows and item.chunk.id in shown
    ]


# ----------------------------------------------------------------- similarity


def _embed(ai: AIClient, sentences: Sequence[str]) -> Optional[np.ndarray]:
    """One batched embedding call for the whole answer."""
    try:
        raw = ai.embed(list(sentences))
        if len(raw) != len(sentences):
            logger.warning(
                "embedding returned %d vectors for %d sentences",
                len(raw),
                len(sentences),
            )
            return None
        return normalize(to_matrix([list(vector) for vector in raw]))
    except Exception:  # noqa: BLE001 - the answer is already delivered
        logger.warning("citation embedding failed; answer stays uncited", exc_info=True)
        return None


def _chunk_matrix(chunks: Sequence[CitationChunk]) -> np.ndarray:
    """Stack chunk embeddings, zero-filling any that are missing or ragged.

    A chunk with no usable vector still competes on token overlap; it just
    never wins on cosine.
    """
    widths = [len(chunk.embedding) for chunk in chunks]
    dims = max(widths) if widths else 0
    rows = [
        list(chunk.embedding) if len(chunk.embedding) == dims else [0.0] * dims
        for chunk in chunks
    ]
    return to_matrix(rows, dims=dims)


def _vector_similarity(chunk_matrix: np.ndarray, sentence_vector: np.ndarray) -> np.ndarray:
    """Cosine of one sentence against every chunk (both sides normalized)."""
    rows = chunk_matrix.shape[0]
    if chunk_matrix.size == 0 or chunk_matrix.shape[1] != sentence_vector.shape[0]:
        return np.zeros((rows,), dtype=np.float32)
    return np.clip(chunk_matrix @ sentence_vector, 0.0, 1.0)


def _token_similarity(sentence: str, chunk_tokens: Sequence[Set[str]]) -> np.ndarray:
    """Fraction of the sentence's distinct tokens each chunk contains."""
    tokens = set(tokenize(sentence))
    if not tokens:
        return np.zeros((len(chunk_tokens),), dtype=np.float32)
    return np.asarray(
        [len(tokens & set(candidate)) / len(tokens) for candidate in chunk_tokens],
        dtype=np.float32,
    )


def _pick(
    similarity: np.ndarray,
    threshold: float,
    decay: float,
    floor: float,
    max_sources: int,
) -> List[int]:
    """Descending-threshold selection: strict first, lenient only if needed."""
    if similarity.size == 0 or max_sources <= 0:
        return []
    current = float(threshold)
    while True:
        matches = [index for index in range(similarity.size) if similarity[index] > current]
        if matches:
            matches.sort(key=lambda index: (-float(similarity[index]), index))
            return matches[:max_sources]
        following = current * float(decay)
        if following < floor or following >= current:
            return []
        current = following


# -------------------------------------------------------------- sentence split

# Deliberately *not* the streaming splitter from app.speech.sentences: that one
# optimizes for time-to-first-audio and will happily emit a fragment at a comma.
# Citations want display units — whole sentences a reader can see a source
# attached to — so this splitter only ever cuts at a real sentence end.
_ENDERS = ".!?。！？…"
_CJK_ENDERS = "。！？…"
_CLOSERS = "\"'”’)）]】》〉」』"
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿　-〿]")
_TRAILING_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)$")
_ABBREVIATIONS = frozenset(
    {
        "al", "approx", "ca", "cf", "dr", "e.g", "eds", "etc", "fig", "i.e",
        "jr", "mr", "mrs", "ms", "prof", "sr", "st", "vs",
    }
)


def split_answer_sentence_spans(answer: str) -> List[Tuple[str, int, int]]:
    """Split a finished answer into ``(sentence, start, end)`` triples.

    Offsets index into ``answer`` exactly as given (not stripped), so the
    client can highlight sentences by slicing rather than re-splitting — a
    second splitter on the client side is a second opinion about where
    sentences are, and the citation map is only meaningful under this one.
    """
    text = answer or ""
    if not text.strip():
        return []
    spans: List[Tuple[str, int, int]] = []

    def push(raw_start: int, raw_end: int) -> None:
        piece_start, piece_end = raw_start, raw_end
        while piece_start < piece_end and text[piece_start].isspace():
            piece_start += 1
        while piece_end > piece_start and text[piece_end - 1].isspace():
            piece_end -= 1
        if piece_end > piece_start:
            spans.append((text[piece_start:piece_end], piece_start, piece_end))

    start = 0
    cursor = 0
    length = len(text)
    while cursor < length:
        char = text[cursor]
        if char not in _ENDERS:
            cursor += 1
            continue
        end = cursor + 1
        while end < length and text[end] in _ENDERS:
            end += 1
        while end < length and text[end] in _CLOSERS:
            end += 1
        if _is_boundary(text, char, cursor, end, length):
            push(start, end)
            start = end
        cursor = max(end, cursor + 1)
    push(start, length)
    return spans


def split_answer_sentences(answer: str) -> List[str]:
    """The sentence texts alone; see :func:`split_answer_sentence_spans`."""
    return [piece for piece, _, _ in split_answer_sentence_spans(answer)]


def _is_boundary(text: str, char: str, cursor: int, end: int, length: int) -> bool:
    """Whether the run of enders at ``cursor`` really closes a sentence."""
    if char in _CJK_ENDERS:
        return True  # CJK punctuation is self-delimiting
    if _is_abbreviation(text[:cursor]):
        return False  # "e.g." and "Dr." keep going
    if end >= length:
        return True
    following = text[end]
    # A latin full stop needs whitespace (or CJK) after it, so "3.5" and
    # "example.com" stay in one piece.
    return following.isspace() or bool(_CJK.match(following))


def _is_abbreviation(prefix: str) -> bool:
    match = _TRAILING_WORD.search(prefix)
    if not match:
        return False
    return match.group(1).lower().rstrip(".") in _ABBREVIATIONS
