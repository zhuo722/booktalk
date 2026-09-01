"""Sentence segmentation for streaming TTS (SPEC §14.8).

Two contracts, tested separately because they answer different questions.
:func:`split_sentences` is the vendored LiveKit tokenizer — *where does a
sentence end* — and is table-driven over the cases that actually break naive
splitters. :class:`SentenceSplitter` is the two-phase streaming policy — *when
is it worth speaking* — and is tested by feeding text in small fragments, the
way LLM deltas arrive.
"""

from __future__ import annotations

from typing import List

import pytest

from app.speech.sentences import (
    FIRST_FRAGMENT_MIN_CHARS,
    MIN_SENTENCE_CHARS,
    TRAILING_GUARD_CHARS,
    SentenceSplitter,
    split_sentences,
    split_text,
)


def _stream(text: str, size: int = 7, **kwargs) -> List[str]:
    """Push text through the splitter in delta-sized fragments."""
    splitter = SentenceSplitter(**kwargs)
    emitted: List[str] = []
    for start in range(0, len(text), size):
        emitted.extend(splitter.feed(text[start : start + size]))
    emitted.extend(splitter.flush())
    return emitted


# --------------------------------------------------- where does a sentence end

BOUNDARY_CASES = [
    (
        "latin",
        "The harbour was built in 1873 by the Vell family. It still guards the "
        "northern approach. Nobody knows who lit the first lantern.",
        [
            "The harbour was built in 1873 by the Vell family.",
            "It still guards the northern approach.",
            "Nobody knows who lit the first lantern.",
        ],
    ),
    (
        "cjk-needs-no-whitespace",
        "灯塔建于一八七三年，至今仍在守望北方航道。渡轮每天清晨从旧石码头出发。",
        [
            "灯塔建于一八七三年，至今仍在守望北方航道。",
            "渡轮每天清晨从旧石码头出发。",
        ],
    ),
    (
        "decimal-is-not-a-boundary",
        "The channel is 3.5 metres deep at low tide.",
        ["The channel is 3.5 metres deep at low tide."],
    ),
    (
        "thousands-separator",
        "The bill came to $29.99 for the whole crossing.",
        ["The bill came to $29.99 for the whole crossing."],
    ),
    (
        "numbers-are-never-split-apart",
        "The fleet numbered 1,200. 300 of them were fishing boats.",
        ["The fleet numbered 1,200. 300 of them were fishing boats."],
    ),
    (
        "pipecat-lookahead-dollar-29",
        "The bill came to $29. Next we walked to the lighthouse.",
        ["The bill came to $29.", "Next we walked to the lighthouse."],
    ),
    (
        "latin-abbreviations",
        "It is shallow, i.e. wade-able, approx. two metres at low tide.",
        ["It is shallow, i.e. wade-able, approx. two metres at low tide."],
    ),
    (
        "titles-and-initials",
        "Dr. Vell wrote the almanac. J. Marsh edited it.",
        ["Dr. Vell wrote the almanac.", "J. Marsh edited it."],
    ),
    (
        "domain-name",
        "The archive lives at meridian.org and nowhere else.",
        ["The archive lives at meridian.org and nowhere else."],
    ),
    (
        "ellipsis-ends-a-sentence",
        "The keeper hesitated... Then he spoke.",
        ["The keeper hesitated...", "Then he spoke."],
    ),
    (
        "closing-quote-stays-attached",
        '"The tide will not wait," she said. Then she cast off.',
        ['"The tide will not wait," she said.', "Then she cast off."],
    ),
]


@pytest.mark.parametrize(
    "text,expected",
    [pytest.param(text, expected, id=name) for name, text, expected in BOUNDARY_CASES],
)
def test_sentence_boundaries(text, expected):
    assert split_text(text, min_sentence_len=1) == expected


def test_split_sentences_returns_offsets_into_the_input():
    text = "The ferry sails at dawn. It returns at dusk."

    spans = split_sentences(text, min_sentence_len=1)

    assert [sentence for sentence, _, _ in spans] == [
        "The ferry sails at dawn.",
        "It returns at dusk.",
    ]
    for sentence, start, end in spans:
        assert text[start:end].strip() == sentence


def test_short_sentences_merge_up_to_the_minimum():
    """One-word sentences are not worth a TTS round trip on their own."""
    assert split_text("Yes. No. Maybe.") == ["Yes. No. Maybe."]
    assert split_text("Yes. No. Maybe.", min_sentence_len=1) == ["Yes.", "No.", "Maybe."]


def test_empty_input_produces_no_sentences():
    assert split_sentences("") == []
    assert split_text("   ") == []


# ------------------------------------------------- when is it worth speaking


def test_first_fragment_emits_at_a_soft_delimiter():
    """The latency win: chunk one need not be a whole sentence."""
    text = (
        "In chapter seven, the keeper describes the harbour lights and the way "
        "they were lit each evening."
    )

    emitted = _stream(text)

    assert emitted[0] == "In chapter seven,"
    assert len(emitted[0]) >= FIRST_FRAGMENT_MIN_CHARS
    # Everything after the opening fragment is a whole sentence again.
    assert emitted[1].endswith(".")


def test_later_chunks_require_a_real_sentence_boundary():
    text = (
        "In chapter seven, the keeper, who was born on the island, describes "
        "the harbour lights in detail."
    )

    emitted = _stream(text)

    assert emitted[0] == "In chapter seven,"
    # The commas after the first fragment are ignored: phase two only cuts at
    # a sentence end, so the rest arrives in one piece.
    assert emitted[1:] == [
        "the keeper, who was born on the island, describes the harbour lights "
        "in detail."
    ]
    assert all(len(chunk) >= MIN_SENTENCE_CHARS for chunk in emitted[1:])


def test_cjk_first_fragment_stops_at_the_first_comma():
    text = "灯塔建于一八七三年，由维尔家族出资建造，至今仍在守望北方航道。渡轮每天清晨从旧石码头出发。"

    assert _stream(text) == [
        "灯塔建于一八七三年，",
        "由维尔家族出资建造，至今仍在守望北方航道。",
        "渡轮每天清晨从旧石码头出发。",
    ]


def test_first_fragment_is_forced_after_thirty_words():
    """An opening clause with no punctuation must not hold the audio hostage."""
    text = " ".join("word{0}".format(index) for index in range(60))

    emitted = _stream(text)

    assert len(emitted[0].split()) == 30
    assert emitted[0].startswith("word0 ")


def test_a_short_fragment_waits_for_ten_characters():
    splitter = SentenceSplitter(guard=0)

    assert splitter.feed("Yes, ") == []
    assert splitter.feed("the ferry runs, and") == ["Yes, the ferry runs,"]


def test_pipecat_lookahead_holds_a_boundary_until_something_follows():
    """"$29." is not a sentence until a non-whitespace character proves it."""
    splitter = SentenceSplitter(guard=0)

    assert splitter.feed("It cost $29.") == []
    assert splitter.feed(" ") == []
    assert splitter.feed("Next") == ["It cost $29."]


def test_trailing_guard_holds_the_last_characters_back():
    """A boundary inside the last 12 characters has not been disambiguated."""
    splitter = SentenceSplitter()

    assert splitter.feed("The ferry leaves at dawn.") == []
    assert splitter.buffered.endswith("dawn.")
    assert splitter.feed(" It returns at dusk.") == ["The ferry leaves at dawn."]
    assert TRAILING_GUARD_CHARS == 12


def test_flush_emits_the_unterminated_remainder_once():
    splitter = SentenceSplitter()
    splitter.feed("The last sentence has no terminator")

    assert splitter.flush() == ["The last sentence has no terminator"]
    assert splitter.flush() == []


def test_streaming_never_loses_or_duplicates_text():
    """Every character of the answer is spoken exactly once, in order."""
    text = (
        "The harbour was built in 1873, and rebuilt twice. Dr. Vell recorded "
        "3.5 metres of silt in the channel. Nobody has dredged it since."
    )

    emitted = _stream(text, size=3)

    assert "".join(emitted).replace(" ", "") == text.replace(" ", "")
