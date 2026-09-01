"""The book's proper-noun vocabulary, harvested at ingest (SPEC §14.9).

Named-entity corruption is the dominant failure mode of voice RAG — arXiv
2608.22872 attributes 87–96 % of ASR-induced retrieval failures to it — and a
book hands us the fix for free: it is a closed corpus whose names we have
already read. Every island, founder and landmark in it can be listed once at
ingest and then used twice at query time: as a decoding hint on the ASR request,
and as the target set for the phonetic repair pass in
:mod:`app.speech.phonetics`.

The extractor is deliberately dumb, because the *signal* is strong. A proper
noun in running prose is capitalized where a common word is not, so the test is
distributional rather than lexical:

* it must appear capitalized **away from a sentence start** at least twice — one
  occurrence is a typo or a title-cased heading, and a sentence-initial capital
  says nothing at all;
* it must almost never appear lowercase — "Harbour" the place survives,
  "However" the adverb does not, and neither does "harbour" the noun.

No POS tagger, no NER model, no language pack. CJK is skipped on purpose: it has
no capitalization to read, and Double Metaphone is an English phonetic
algorithm, so a CJK book simply gets an empty vocabulary and the untouched v1
behaviour.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

#: Occurrences away from a sentence start needed to call something a name.
MIN_MID_SENTENCE = 2
#: Longest tolerated ratio of lowercase to capitalized uses.
MAX_LOWERCASE_RATIO = 0.2
MIN_TERM_CHARS = 3
MAX_TERM_CHARS = 32
#: Cap on the stored vocabulary. Beyond a couple of hundred names the ASR
#: prompt is full anyway, and the long tail is where false repairs come from.
MAX_TERMS = 200

_TOKEN = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_SENTENCE_ENDERS = ".!?…。！？\n\r"
#: Characters that may sit between a sentence end and its first word.
_LEADING = " \t \"'“”‘’([{«"

# Capitalized words that are structure, not names. They pass the distributional
# test in any book (always capitalized, often mid-sentence) and are useless or
# harmful as repair targets.
STRUCTURE_WORDS = frozenset(
    {
        "appendix", "book", "chapter", "conclusion", "contents", "epilogue",
        "figure", "foreword", "glossary", "index", "introduction", "note",
        "notes", "page", "part", "preface", "prologue", "section", "table",
        "volume",
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "sunday",
    }
)


class _Term:
    """Running counts for one candidate, keyed by its lowercase form."""

    __slots__ = ("mid", "start", "lower", "forms")

    def __init__(self) -> None:
        self.mid = 0
        self.start = 0
        self.lower = 0
        self.forms: Counter = Counter()

    @property
    def capitalized(self) -> int:
        return self.mid + self.start


def extract_vocabulary(
    texts: Iterable[str],
    max_terms: int = MAX_TERMS,
    min_mid_sentence: int = MIN_MID_SENTENCE,
) -> List[Tuple[str, int]]:
    """Rank the book's proper nouns, most frequent first.

    Returns ``(canonical_form, occurrences)`` pairs, where the canonical form is
    the capitalization the book itself uses most often — that is the spelling
    the reader sees on screen and the one the ASR prompt should suggest.
    """
    counts: Dict[str, _Term] = {}
    for text in texts:
        _count(text or "", counts)

    ranked = [
        (entry.forms.most_common(1)[0][0], entry.capitalized)
        for key, entry in counts.items()
        if _qualifies(key, entry, min_mid_sentence)
    ]
    # Frequency first; alphabetical within a tie so the list is stable across
    # runs (it is stored, prompted with, and diffed in tests).
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:max_terms]


def _count(text: str, counts: Dict[str, _Term]) -> None:
    """Tally one chunk's tokens into ``counts``."""
    for match in _TOKEN.finditer(text):
        token = match.group(0)
        if not (MIN_TERM_CHARS <= len(token) <= MAX_TERM_CHARS):
            continue
        if _CJK.search(token):
            continue
        entry = counts.get(token.lower())
        if entry is None:
            entry = counts[token.lower()] = _Term()
        if not token[0].isupper():
            entry.lower += 1
            continue
        entry.forms[token] += 1
        if _starts_sentence(text, match.start()):
            entry.start += 1
        else:
            entry.mid += 1


def _starts_sentence(text: str, position: int) -> bool:
    """Whether the token at ``position`` opens a sentence.

    A capital there is grammar, not evidence — which is the whole reason this
    function exists.
    """
    index = position - 1
    while index >= 0 and text[index] in _LEADING:
        index -= 1
    if index < 0:
        return True
    return text[index] in _SENTENCE_ENDERS


def _qualifies(key: str, entry: _Term, min_mid_sentence: int) -> bool:
    if key in STRUCTURE_WORDS:
        return False
    if entry.mid < min_mid_sentence:
        return False
    return entry.lower <= MAX_LOWERCASE_RATIO * entry.capitalized


def vocabulary_of(chunks: Sequence, max_terms: int = MAX_TERMS) -> List[Tuple[str, int]]:
    """:func:`extract_vocabulary` over the ``text`` of a chunk sequence."""
    return extract_vocabulary(
        (getattr(chunk, "text", "") for chunk in chunks), max_terms=max_terms
    )
