"""Sentence segmentation for streaming TTS (SPEC §5.2 step 4, revised by §14.8).

Two problems, one module.

**Where does a sentence end?** Answered by a vendored port of LiveKit Agents'
rule-based tokenizer (see :func:`split_sentences` below for the attribution):
a cascade of regexes that mark every full stop which is *not* a sentence end —
``Dr.``, ``e.g.``, ``U.S.``, ``3.5``, ``example.com`` — leaving the unmarked
ones as boundaries. Stdlib ``re`` only: no NLTK, no ``nltk.download()`` at
runtime, no corpora to ship. Three guards are bolted on top of it:

* **pipecat's non-whitespace lookahead** — on seeing ``.?!;…``, wait for the
  next non-whitespace character before committing. Their example is exactly the
  case that matters here: ``"$29."`` is not a sentence, ``"$29. Next"`` is.
* **stream2sentence's 12-character trailing guard** — never look for a boundary
  inside the last ``context_size`` characters of the buffer, because the token
  that would disambiguate them has not arrived yet.
* **never-split-numbers** — a full stop with digits on both sides is a decimal
  point or a thousands separator, never a sentence end.

**When is it worth speaking?** Answered by two-phase emission, which is where
the latency win lives. Time-to-first-audio is dominated by how many LLM tokens
we wait for before the first synthesis call, so the *first* fragment settles for
any soft delimiter (``,;:.!?…``) once it is ≥10 characters — "In chapter seven,"
is a perfectly good thing to start speaking — and force-emits at the next space
after ~30 words if the model writes a very long opening clause
(stream2sentence's ``minimum_first_fragment_length`` /
``force_first_fragment_after_words``). Every chunk *after* the first requires a
true sentence boundary and ≥20 characters (LiveKit's ``min_sentence_len``), so
the fragmenting costs nothing in prosody past the opening.

CJK punctuation is self-delimiting and needs no lookahead rule; the CJK enders
are simply always boundaries.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# --------------------------------------------------------------------- tuning

#: LiveKit ``min_sentence_len``: chunks after the first are merged up to this.
MIN_SENTENCE_CHARS = 20
#: stream2sentence ``minimum_first_fragment_length``.
FIRST_FRAGMENT_MIN_CHARS = 10
#: stream2sentence ``force_first_fragment_after_words``.
FORCE_FIRST_AFTER_WORDS = 30
#: stream2sentence ``context_size``: never tokenize the last N characters.
TRAILING_GUARD_CHARS = 12

ASCII_ENDERS = ".!?"
CJK_ENDERS = "。！？…"
SENTENCE_ENDERS = ASCII_ENDERS + CJK_ENDERS
#: Soft delimiters the *first* fragment may stop at (SPEC §14.8).
SOFT_DELIMITERS = ",;:，、；：" + SENTENCE_ENDERS
CLOSERS = "\"'”’)）]】》〉」』"

# Latin abbreviations whose full stop is not the end of a sentence — v1's list,
# kept because it covers what a non-fiction book actually contains ("approx.",
# "fig. 4", "cf. chapter two") and the ported LiveKit rules do not.
ABBREVIATIONS = frozenset(
    {
        "al", "approx", "ca", "cf", "dr", "e.g", "eds", "etc", "fig", "i.e",
        "jr", "mr", "mrs", "ms", "prof", "sr", "st", "vs",
    }
)

# Same-length sentinels, so every offset in the analysed copy still addresses
# the original string. U+E000/U+E001 are private-use: they carry no meaning of
# their own and are scrubbed out of the input before analysis.
_PRD = "\ue000"  # a full stop that is NOT a boundary
_STP = "\ue001"  # a full stop that IS a boundary, immune to later rules

_BOUNDARY_CHARS = frozenset(SENTENCE_ENDERS + _STP)
_SOFT_CHARS = frozenset(SOFT_DELIMITERS + _STP)


# ------------------------------------------------------- vendored LiveKit port
#
# Ported from LiveKit Agents' rule-based sentence tokenizer, Apache-2.0:
#   https://github.com/livekit/agents/blob/main/livekit-agents/livekit/agents/tokenize/_basic_sent.py
# The rule cascade below is theirs (which is in turn the widely-copied
# StackOverflow answer https://stackoverflow.com/a/31505798). Two deliberate
# changes:
#   * their ``<prd>``/``<stop>`` markers are multi-character, which shifts every
#     offset after them; ours are single characters, so the analysed copy stays
#     index-aligned with the caller's string and the emitted text is a *slice*
#     of the input rather than a reconstruction;
#   * they swap ``.”`` to ``”.`` so the boundary lands after a closing quote —
#     a rewrite we cannot afford for the same reason, so the boundary is
#     extended past closers when it is computed instead (see :func:`_cuts`).

_ALPHABET = r"([A-Za-z])"
_PREFIXES = r"\b(Mr|Mrs|Ms|Dr|Prof|St|Capt|Cpt|Lt|Sgt|Rev|Hon|Gen|Col)\."
_SUFFIXES = r"(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = (
    r"(Mr|Mrs|Ms|Dr|Prof|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s"
    r"|However\s|That\s|This\s|Wherever)"
)
_ACRONYM = r"([A-Z]\.[A-Z]\.(?:[A-Z]\.)?)"
_WEBSITES = r"\.(com|net|org|io|gov|edu|me)\b"
_MULTIPLE_DOTS = r"\.{2,}"


def _protect(text: str) -> str:
    """Mark every full stop that does not end a sentence.

    Returns a string of the same length as ``text``: protected stops become
    :data:`_PRD`, forced stops :data:`_STP`, everything else is untouched. Rule
    order matters and follows LiveKit's — a decision made by an earlier rule is
    immune to later ones, because the sentinel no longer matches ``\\.``.
    """
    marked = text.replace(_PRD, ".").replace(_STP, ".")

    def hide(match: "re.Match") -> str:
        return match.group(0).replace(".", _PRD)

    # An ellipsis *does* end a sentence: protect every dot of the run but the
    # last, which becomes the boundary.
    marked = re.sub(
        _MULTIPLE_DOTS,
        lambda m: _PRD * (len(m.group(0)) - 1) + _STP,
        marked,
    )
    marked = re.sub(_PREFIXES, hide, marked)
    marked = re.sub(_WEBSITES, lambda m: _PRD + m.group(1), marked)
    # never-split-numbers: digits on both sides of the stop, with or without
    # the space ("3.5", "1,200. 300" — merging those two is always safer than
    # speaking half a figure).
    marked = re.sub(r"([0-9])\.(?=\s*[0-9])", lambda m: m.group(1) + _PRD, marked)
    marked = marked.replace("Ph.D.", "Ph" + _PRD + "D" + _PRD)
    # An acronym followed by a sentence starter really is a sentence end.
    marked = re.sub(
        _ACRONYM + r"\s" + _STARTERS,
        lambda m: _acronym_stop(m.group(1)) + m.group(0)[len(m.group(1)) :],
        marked,
    )
    marked = re.sub(_ALPHABET + r"\." + _ALPHABET + r"\." + _ALPHABET + r"\.", hide, marked)
    marked = re.sub(_ALPHABET + r"\." + _ALPHABET + r"\.", hide, marked)
    marked = re.sub(
        r"\s" + _SUFFIXES + r"\.\s" + _STARTERS,
        lambda m: m.group(0).replace(".", _STP, 1),
        marked,
    )
    marked = re.sub(r"\s" + _SUFFIXES + r"\.", hide, marked)
    # A single letter before a stop is an initial ("J. Vell"), not a sentence.
    marked = re.sub(r"\s" + _ALPHABET + r"\.", hide, marked)
    marked = _hide_abbreviations(marked)
    return marked


def _acronym_stop(acronym: str) -> str:
    """``U.S.`` → its inner dots protected, its last one a forced boundary."""
    head, _, _ = acronym.rpartition(".")
    return head.replace(".", _PRD) + _STP


_ABBREVIATION_RE = re.compile(
    r"\b(?:{0})\.".format(
        "|".join(sorted((re.escape(word) for word in ABBREVIATIONS), key=len, reverse=True))
    ),
    re.IGNORECASE,
)


def _hide_abbreviations(marked: str) -> str:
    """Protect the stop after any abbreviation in :data:`ABBREVIATIONS`."""
    return _ABBREVIATION_RE.sub(lambda m: m.group(0)[:-1] + _PRD, marked)


def _cuts(text: str, chars) -> List[int]:
    """Offsets just past every boundary of the requested class, in order.

    A run of enders (``!?``) counts once, and the cut is pushed past any closing
    quotes or brackets so ``she said."`` is never split from its quotation mark.
    """
    marked = _protect(text)
    ends: List[int] = []
    index = 0
    length = len(marked)
    while index < length:
        if marked[index] not in chars:
            index += 1
            continue
        end = index + 1
        while end < length and marked[end] in _BOUNDARY_CHARS:
            end += 1
        while end < length and text[end] in CLOSERS:
            end += 1
        ends.append(end)
        index = end
    return ends


def split_sentences(
    text: str, min_sentence_len: int = MIN_SENTENCE_CHARS
) -> List[Tuple[str, int, int]]:
    """Split a finished string into ``(sentence, start, end)`` triples.

    LiveKit's signature and semantics, including the merging of anything
    shorter than ``min_sentence_len`` into its successor. ``start``/``end`` index
    the *input* string, so ``text[start:end]`` is always the sentence plus the
    whitespace around it.
    """
    if not text:
        return []
    sentences: List[Tuple[str, int, int]] = []
    start = 0
    for end in _cuts(text, _BOUNDARY_CHARS):
        piece = text[start:end].strip()
        if len(piece) < min_sentence_len:
            continue
        sentences.append((piece, start, end))
        start = end
    tail = text[start:].strip()
    if tail:
        sentences.append((tail, start, len(text)))
    return sentences


def split_text(text: str, min_sentence_len: int = MIN_SENTENCE_CHARS) -> List[str]:
    """:func:`split_sentences` without the offsets."""
    return [sentence for sentence, _, _ in split_sentences(text, min_sentence_len)]


# ------------------------------------------------------ streaming, two-phase


class SentenceSplitter:
    """Stateful two-phase splitter over a stream of LLM deltas.

    Feed it fragments; it returns only the chunks it is willing to hand to TTS.
    The first chunk of a turn is deliberately eager (soft delimiter, ≥10
    characters, forced after ~30 words) and every later one conservative (true
    sentence boundary, ≥20 characters). :meth:`flush` empties the buffer at the
    end of the answer, where nothing more is coming and every guard is moot.
    """

    def __init__(
        self,
        min_chars: int = MIN_SENTENCE_CHARS,
        first_min_chars: int = FIRST_FRAGMENT_MIN_CHARS,
        force_after_words: int = FORCE_FIRST_AFTER_WORDS,
        guard: int = TRAILING_GUARD_CHARS,
    ) -> None:
        self.min_chars = min_chars
        self.first_min_chars = first_min_chars
        self.force_after_words = force_after_words
        self.guard = guard
        self._buffer = ""
        self._emitted = 0

    def feed(self, text: str) -> List[str]:
        """Add a fragment; return every chunk that is now safe to speak."""
        if not text:
            return []
        self._buffer += text
        return self._drain(final=False)

    def flush(self) -> List[str]:
        """Emit whatever is left, then reset. Idempotent: a second call is []."""
        chunks = self._drain(final=True)
        self._buffer = ""
        self._emitted = 0
        return chunks

    @property
    def buffered(self) -> str:
        """Text held back so far (unterminated, too short, or inside the guard)."""
        return self._buffer

    def _drain(self, final: bool) -> List[str]:
        chunks: List[str] = []
        while self._buffer.strip():
            cut = self._cut(final)
            if cut is None:
                break
            piece = self._buffer[:cut].strip()
            self._buffer = self._buffer[cut:]
            if piece:
                chunks.append(piece)
                self._emitted += 1
        return chunks

    def _cut(self, final: bool) -> Optional[int]:
        """Where to cut the buffer next, or ``None`` to wait for more text."""
        buffer = self._buffer
        first = self._emitted == 0
        minimum = self.first_min_chars if first else self.min_chars
        # The trailing guard: characters this close to the end of the buffer
        # have not been disambiguated by what follows them yet.
        limit = len(buffer) if final else len(buffer) - self.guard

        for end in _cuts(buffer, _SOFT_CHARS if first else _BOUNDARY_CHARS):
            if end > limit:
                break
            # pipecat: commit only once a non-whitespace character follows.
            if not final and not buffer[end:].strip():
                break
            if len(buffer[:end].strip()) >= minimum:
                return end

        if first and not final:
            forced = _force_cut(buffer, self.force_after_words)
            if forced is not None and forced <= limit:
                return forced
        return len(buffer) if final else None


def _force_cut(text: str, words: int) -> Optional[int]:
    """Offset of the space after word ``words``, or ``None`` if it isn't there.

    The escape hatch for an opening clause with no punctuation in it: without
    this, a model that writes forty words before its first comma would hold the
    whole turn's audio hostage. Whitespace-delimited by nature, so it never
    fires on CJK — which does not need it, having a delimiter every few
    characters.
    """
    if words <= 0:
        return None
    count = 0
    for match in re.finditer(r"\S+(\s+)", text):
        count += 1
        if count >= words:
            return match.end()
    return None
