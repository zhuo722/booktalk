"""Heading-family voting (SPEC §14.2, tier 2.5).

Ported from RAGFlow's ``bullets_category``: instead of asking "does this one line
look like a heading?", score whole *families* of numbering conventions across the
entire book and keep the argmax. A book numbers its chapters exactly one way, so
the family that fires most often is the book's own convention — and every line
matching it is a heading, however plain its typography.

Each family is an ordered tuple of patterns, shallowest level first. Matching
walks the patterns from the deepest, so ``1.2.3 Ferries`` is recognized as a
third-level heading rather than as a first-level ``1.``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Tuple

MIN_VOTES = 3
MAX_HEADING_CHARS = 120
MAX_HEADING_WORDS = 20

_ROMAN = r"[IVXLCivxlc]+"
_NUMBER_WORDS = (
    "ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|THIRTEEN|"
    "FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|NINETEEN|TWENTY|THIRTY|FORTY|FIFTY"
)
_ENGLISH_ORDINAL = r"(?:\d{{1,3}}|{roman}|{words})".format(
    roman=_ROMAN, words=_NUMBER_WORDS
)
# ○ is U+25CB WHITE CIRCLE rather than the ideographic zero, and Chinese
# typesetting uses it as one — 西遊記 prints its tenth chapter as 第一○回.
# Leaving it out loses every chapter whose number contains a zero.
_CJK_NUMBER = r"[〇零○一二三四五六七八九十百千0-9]"


def _compile(patterns: Sequence[str]) -> Tuple[Pattern, ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


@dataclass(frozen=True)
class HeadingFamily:
    """One numbering convention, as an ordered ladder of level patterns."""

    name: str
    patterns: Tuple[Pattern, ...]

    def level_of(self, text: str) -> Optional[int]:
        """1-based level of ``text`` in this family, or ``None``.

        Deepest pattern first: ``2.1`` must not be read as the ``2.`` of level 1.
        """
        for depth in range(len(self.patterns) - 1, -1, -1):
            if self.patterns[depth].match(text):
                return depth + 1
        return None


FAMILIES: Tuple[HeadingFamily, ...] = (
    HeadingFamily(
        name="english",
        patterns=_compile(
            (
                r"^PART\s+{0}\b".format(_ENGLISH_ORDINAL),
                r"^(?:CHAPTER|BOOK)\s+{0}\b".format(_ENGLISH_ORDINAL),
                r"^SECTION\s+{0}\b".format(_ENGLISH_ORDINAL),
            )
        ),
    ),
    HeadingFamily(
        name="cjk",
        patterns=_compile(
            (
                r"^第\s*{0}+\s*[编編部篇]".format(_CJK_NUMBER),
                r"^第\s*{0}+\s*[章回]".format(_CJK_NUMBER),
                r"^第\s*{0}+\s*[节節]".format(_CJK_NUMBER),
                r"^第\s*{0}+\s*条".format(_CJK_NUMBER),
            )
        ),
    ),
    HeadingFamily(
        name="numeric",
        patterns=_compile(
            (
                r"^\d{1,2}[.、]?[ 　]+\S",
                r"^\d{1,2}\.\d{1,2}[.、]?[ 　]+\S",
                r"^\d{1,2}\.\d{1,2}\.\d{1,2}[.、]?[ 　]+\S",
            )
        ),
    ),
    HeadingFamily(
        name="markdown",
        patterns=_compile((r"^#[ \t]+\S", r"^##[ \t]+\S", r"^###[ \t]+\S")),
    ),
)

_SENTENCE_TAIL = re.compile(r"[,;:，；：]$")
_CJK_RANGE = re.compile(r"[㐀-䶿一-鿿　-〿＀-￯]")


@dataclass
class Vote:
    """The winning family and the tally that elected it."""

    family: HeadingFamily
    counts: Dict[str, int]

    @property
    def name(self) -> str:
        return self.family.name

    @property
    def matches(self) -> int:
        return self.counts.get(self.family.name, 0)

    def level_of(self, text: str) -> Optional[int]:
        return self.family.level_of(text) if is_heading_like(text) else None


def is_heading_like(text: str) -> bool:
    """Cheap shape test applied before any family pattern is tried.

    Headings are short and do not trail off into a clause. Screening on shape
    first keeps a body line that happens to open with "12 " from voting.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > MAX_HEADING_CHARS:
        return False
    if _SENTENCE_TAIL.search(stripped):
        return False
    if _CJK_RANGE.search(stripped):
        return True
    return len(stripped.split()) <= MAX_HEADING_WORDS


def any_family_level(text: str) -> Optional[int]:
    """Level of ``text`` in *any* family, without holding an election.

    Used where a single line has to be judged on its own — the repeated-line
    detector needs to know "is this a numbered heading?" before there is a
    winning family to ask.
    """
    if not is_heading_like(text):
        return None
    stripped = text.strip()
    for family in FAMILIES:
        level = family.level_of(stripped)
        if level is not None:
            return level
    return None


def tally(lines: Iterable[str]) -> Dict[str, int]:
    """Count, per family, how many lines match any of its level patterns."""
    counts: Dict[str, int] = {family.name: 0 for family in FAMILIES}
    for raw in lines:
        if not is_heading_like(raw):
            continue
        text = raw.strip()
        for family in FAMILIES:
            if family.level_of(text) is not None:
                counts[family.name] += 1
    return counts


def vote(lines: Iterable[str], min_votes: int = MIN_VOTES) -> Optional[Vote]:
    """Elect the book's heading convention, or ``None`` when none is used.

    Ties break on the declaration order of :data:`FAMILIES`, which puts explicit
    "Chapter"/"第N章" wording ahead of bare numbering — the safer reading when
    the evidence is balanced.
    """
    counts = tally(lines)
    best = max(FAMILIES, key=lambda family: counts[family.name])
    if counts[best.name] < min_votes:
        return None
    return Vote(family=best, counts=counts)


def normalize_levels(levels: Sequence[int]) -> List[int]:
    """Compact levels to 1..n, preserving order.

    A book with no PARTs numbers its chapters at family level 2; its chapters are
    still the top of *its* hierarchy, and the chunker's breadcrumb depends on
    that being level 1.
    """
    distinct = sorted({level for level in levels})
    ranking = {level: position + 1 for position, level in enumerate(distinct)}
    return [ranking[level] for level in levels]


def family_levels(titles: Sequence[str]) -> List[int]:
    """Level per title from the winning family; empty when no family wins.

    Used to give a printed contents page a hierarchy without an LLM call. Titles
    the family does not recognize inherit one level below their predecessor, so
    an unnumbered "Preface" under "PART ONE" still lands below it.
    """
    winner = vote(titles, min_votes=max(2, min(MIN_VOTES, len(titles))))
    if winner is None:
        return []
    raw: List[Optional[int]] = [winner.level_of(title) for title in titles]
    if not any(level is not None for level in raw):
        return []
    filled: List[int] = []
    for level in raw:
        if level is not None:
            filled.append(level)
        elif filled:
            filled.append(filled[-1] + 1)
        else:
            filled.append(1)
    return normalize_levels(filled)


__all__ = [
    "FAMILIES",
    "HeadingFamily",
    "MIN_VOTES",
    "Vote",
    "any_family_level",
    "family_levels",
    "is_heading_like",
    "normalize_levels",
    "tally",
    "vote",
]
