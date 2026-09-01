"""Repairing mistranscribed names against the book's vocabulary (SPEC §14.9).

ASR gets ordinary words right and invented names wrong: "Kestrel Cove" comes
back as "Castrel Cove", "Meridian" as "Meridien". Those are precisely the tokens
retrieval leans on — a corrupted proper noun is a BM25 miss and an embedding
that points somewhere else — which is why named-entity corruption accounts for
the overwhelming majority of ASR-induced RAG failures (arXiv 2608.22872), and
why phonetic repair against a closed vocabulary was the paper's only mitigation
that actually worked.

Double Metaphone is the right instrument because it encodes how a word *sounds*
and returns two codes for the genuinely ambiguous cases ("Thorne" → ``0RN`` and
``TRN``), so a transcript that heard the second pronunciation still matches.

Everything else here is about not making things worse. A wrong repair is more
damaging than the miss it replaces, because it is invisible: the reader sees a
plausible sentence that quietly retrieves the wrong chapter. So a token is
rewritten only when every one of these holds:

* it is at least four characters, latin, and not a common English word;
* it is not already in the vocabulary (nothing to fix);
* its phonetic code is at least three characters — two-character codes like
  ``FL`` collide across half the language ("fell", "veil", "foul", "Vell");
* it is textually similar as well as phonetically equal (ratio ≥ 0.6), which
  rejects same-sounding words of a completely different shape;
* and exactly **one** vocabulary term survives all of that. Two candidates mean
  we do not know, and the original wins by default.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

MIN_TOKEN_CHARS = 4
#: Shorter Double Metaphone codes match far too much to be evidence.
MIN_CODE_CHARS = 3
#: Textual agreement required on top of the phonetic match.
MIN_SIMILARITY = 0.6
#: Terms offered to the ASR decoder as a spelling hint.
PROMPT_TERMS = 60
#: Whisper truncates its prompt at 224 tokens; ~4 characters per token.
MAX_PROMPT_CHARS = 896
PROMPT_TEMPLATE = "Proper nouns from the book being discussed: {terms}."

_WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

# The most frequent English words, which must never be replaced by a name that
# happens to rhyme with them. Deliberately short: the other guards do the work,
# and this list only has to cover words a listener would notice being mangled.
COMMON_WORDS = frozenset(
    """
    about above after again against all also always another any are around away
    back because been before being below between both bring came cannot come
    could does done down during each early even ever every first found from
    gave give goes going gone good great half hand hard have having help here
    high hold home into just keep kept kind knew know large last late later
    least left less life like little live long look made make many more most
    much must name near need never next night none note only open other over
    part past people place point right said same seem seen sent several shall
    should since small some soon still such take taken tell than that their
    them then there these they thing think this those though three through time
    together told took toward turn under until upon used very want water well
    went were what when where whether which while whole whose will with within
    without word work would year years your
    """.split()
)


@dataclass
class Repair:
    """The result of a repair pass: the new text and what changed."""

    text: str
    replacements: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.replacements)


def double_metaphone(word: str) -> Tuple[str, str]:
    """``(primary, secondary)`` phonetic codes; ``("", "")`` when unavailable.

    Imported lazily so the module (and anything that merely wants
    :func:`asr_prompt`) works with the dependency absent.
    """
    encoder = _encoder()
    if encoder is None:
        return ("", "")
    try:
        primary, secondary = encoder(word)
    except Exception:  # noqa: BLE001 - a bad token must not fail a transcript
        return ("", "")
    return (primary or "", secondary or "")


_ENCODER: Optional[Callable[[str], Tuple[str, str]]] = None
_ENCODER_LOADED = False


def _encoder() -> Optional[Callable[[str], Tuple[str, str]]]:
    global _ENCODER, _ENCODER_LOADED
    if not _ENCODER_LOADED:
        _ENCODER_LOADED = True
        try:
            from metaphone import doublemetaphone

            _ENCODER = doublemetaphone
        except Exception:  # noqa: BLE001 - optional; repair simply no-ops
            logger.info("metaphone is not installed; ASR repair is disabled")
            _ENCODER = None
    return _ENCODER


def echoes_prompt(text: str, prompt: str) -> bool:
    """True when a transcript is the decoding hint coming back at us.

    A hinted decoder that hears no speech sometimes emits its own prompt
    instead of nothing — the user records a question and what returns is the
    list of names the server sent. The signal has to be the template's own
    opening words: a ratio of shared tokens looked more thorough and was
    wrong, because a book's vocabulary contains most of the ordinary words a
    short question is made of, and it rejected real questions.
    """
    said = (text or "").strip().lower()
    if not said or not (prompt or "").strip():
        return False
    lead = PROMPT_TEMPLATE.split("{")[0].strip().lower()
    # Enough of the opener to be unambiguous, short enough to survive the
    # decoder dropping a word or changing its punctuation.
    return bool(lead) and said.startswith(lead[:24])


def asr_prompt(
    vocabulary: Sequence[str],
    limit: int = PROMPT_TERMS,
    max_chars: int = MAX_PROMPT_CHARS,
) -> str:
    """A decoding hint listing the book's names, or ``""`` when there are none.

    The transcription ``prompt`` is advisory: it biases spelling towards words
    the decoder has been told to expect. Terms come in frequency order, so a
    truncated list still carries the names most likely to be spoken.
    """
    terms: List[str] = []
    length = 0
    for term in vocabulary:
        term = (term or "").strip()
        if not term:
            continue
        addition = len(term) + 2
        if len(terms) >= limit or length + addition > max_chars:
            break
        terms.append(term)
        length += addition
    if not terms:
        return ""
    return PROMPT_TEMPLATE.format(terms=", ".join(terms))


def phonetic_index(vocabulary: Sequence[str]) -> Dict[str, Set[str]]:
    """Map every usable phonetic code to the vocabulary terms that produce it."""
    index: Dict[str, Set[str]] = {}
    for term in vocabulary:
        term = (term or "").strip()
        if len(term) < MIN_TOKEN_CHARS or _CJK.search(term):
            continue
        for code in double_metaphone(term):
            if len(code) >= MIN_CODE_CHARS:
                index.setdefault(code, set()).add(term)
    return index


def repair_transcript(
    text: str,
    vocabulary: Sequence[str],
    min_similarity: float = MIN_SIMILARITY,
) -> Repair:
    """Rewrite mistranscribed names in ``text`` using the book's vocabulary.

    Word-for-word and in place: only whole tokens are considered, punctuation
    and spacing are untouched, and a token with anything less than one
    unambiguous match is left exactly as the model heard it.
    """
    original = text or ""
    if not original.strip() or not vocabulary:
        return Repair(text=original)
    index = phonetic_index(vocabulary)
    if not index:
        return Repair(text=original)
    known = {term.lower() for term in vocabulary}
    replacements: List[Tuple[str, str]] = []

    def substitute(match: "re.Match") -> str:
        token = match.group(0)
        candidate = _repair_token(token, index, known, min_similarity)
        if candidate is None:
            return token
        replacements.append((token, candidate))
        return candidate

    repaired = _WORD.sub(substitute, original)
    if replacements:
        logger.info(
            "ASR repair rewrote %d token(s): %s",
            len(replacements),
            ", ".join("{0}→{1}".format(a, b) for a, b in replacements),
        )
    return Repair(text=repaired, replacements=replacements)


def _repair_token(
    token: str,
    index: Dict[str, Set[str]],
    known: Set[str],
    min_similarity: float,
) -> Optional[str]:
    """The one vocabulary term this token should have been, or ``None``."""
    lowered = token.lower()
    if len(token) < MIN_TOKEN_CHARS or _CJK.search(token):
        return None
    if lowered in known or lowered in COMMON_WORDS:
        return None

    candidates: Set[str] = set()
    for code in double_metaphone(token):
        if len(code) >= MIN_CODE_CHARS:
            candidates |= index.get(code, set())
    plausible = sorted(
        term
        for term in candidates
        if term.lower() != lowered and _similar(lowered, term.lower(), min_similarity)
    )
    # Ambiguity is a reason to do nothing: two names that sound alike are
    # exactly the case where a wrong pick is both likely and invisible.
    return plausible[0] if len(plausible) == 1 else None


def _similar(left: str, right: str, threshold: float) -> bool:
    return SequenceMatcher(None, left, right).ratio() >= threshold
