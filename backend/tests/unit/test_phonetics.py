"""Double Metaphone repair of mistranscribed names (SPEC §14.9).

Every vocabulary here is constructed for the case under test, because the whole
point of the layer is that the book supplies a small closed set of names — and
because the interesting assertions are the ones about what it refuses to do.
"""

from __future__ import annotations

import pytest

from app.speech.phonetics import (
    MAX_PROMPT_CHARS,
    Repair,
    asr_prompt,
    double_metaphone,
    phonetic_index,
    repair_transcript,
)

BOOK_NAMES = ["Kestrel", "Meridian", "Thornbury", "Halvard", "Sable"]


def repaired(text: str, vocabulary=BOOK_NAMES) -> str:
    return repair_transcript(text, vocabulary).text


# ------------------------------------------------------------------- repairs


@pytest.mark.parametrize(
    "heard,expected",
    [
        ("Castrel", "Kestrel"),
        ("Meridien", "Meridian"),
        ("Thornburry", "Thornbury"),
        ("Halvarde", "Halvard"),
    ],
)
def test_a_mistranscribed_name_is_repaired(heard, expected):
    assert repaired("Tell me about {0} Cove.".format(heard)) == (
        "Tell me about {0} Cove.".format(expected)
    )


def test_the_repair_reports_what_it_changed():
    result = repair_transcript("Who founded Castrel?", BOOK_NAMES)

    assert isinstance(result, Repair)
    assert result.changed is True
    assert result.replacements == [("Castrel", "Kestrel")]


def test_punctuation_and_spacing_survive_a_repair():
    assert repaired('"Castrel," she said — Meridien, too.') == (
        '"Kestrel," she said — Meridian, too.'
    )


def test_a_correctly_transcribed_name_is_left_alone():
    result = repair_transcript("Who founded Kestrel Cove?", BOOK_NAMES)

    assert result.changed is False


def test_a_lowercase_form_of_a_known_name_is_not_touched():
    """It is already the right word; only the casing is the model's choice."""
    assert repaired("who founded kestrel cove?") == "who founded kestrel cove?"


# ---------------------------------------------------------------- refusals


def test_a_common_english_word_is_never_replaced():
    """"Pepple" and "people" are homophones; only one of them is a name."""
    assert double_metaphone("people") == double_metaphone("Pepple")

    text = "The people came before the ferry left."
    assert repaired(text, ["Pepple", "Kestrel"]) == text


def test_two_similar_names_leave_the_original_alone():
    """Ambiguity is a reason to do nothing: a wrong pick is invisible."""
    vocabulary = ["Thornbury", "Thornberry"]

    assert repaired("They sailed past Thornbary.", vocabulary) == (
        "They sailed past Thornbary."
    )


def test_a_two_letter_code_repairs_nothing_at_all():
    """"Vell" and "fell" both encode to ``FL`` — far too coarse to act on."""
    assert double_metaphone("Vell")[0] == double_metaphone("fell")[0] == "FL"
    assert phonetic_index(["Vell"]) == {}

    assert repaired("The tide fell at dusk.", ["Vell"]) == "The tide fell at dusk."


def test_a_phonetic_match_of_a_different_shape_is_rejected():
    """Sounding alike is not enough; the spelling has to be close too.

    "supple" and "Sable" share a Double Metaphone code exactly, and differ far
    enough in writing (ratio 0.55) that the transcript is more likely right
    than the name is — which is what the similarity floor is for.
    """
    assert double_metaphone("supple") == double_metaphone("Sable")
    text = "The rope was supple enough to knot."

    assert repaired(text, ["Sable"]) == text
    # Drop the floor below their agreement and the same call does repair it,
    # so the refusal above is the guard and not a missing code match.
    assert repair_transcript(text, ["Sable"], min_similarity=0.5).replacements == [
        ("supple", "Sable")
    ]


def test_cjk_text_passes_through_untouched():
    text = "渡轮每天清晨从旧石码头出发。"

    assert repaired(text) == text


def test_an_empty_vocabulary_is_a_no_op():
    assert repaired("Anything at all here.", []) == "Anything at all here."
    assert repair_transcript("", BOOK_NAMES).text == ""


# ------------------------------------------------------------------ indexes


def test_the_index_maps_both_metaphone_codes():
    index = phonetic_index(["Thorne"])
    primary, secondary = double_metaphone("Thorne")

    assert primary != secondary
    assert "Thorne" in index.get(secondary, set())


# ------------------------------------------------------------- the ASR prompt


def test_the_prompt_lists_the_books_names_in_rank_order():
    prompt = asr_prompt(BOOK_NAMES)

    assert "Kestrel, Meridian, Thornbury" in prompt
    assert prompt.endswith(".")


def test_the_prompt_is_truncated_to_the_decoder_limit():
    prompt = asr_prompt(["Thornbury{0}".format(index) for index in range(500)])

    assert len(prompt) <= MAX_PROMPT_CHARS + len(asr_prompt(["x"]))
    assert "Thornbury0" in prompt


def test_the_prompt_honours_the_term_limit():
    prompt = asr_prompt(BOOK_NAMES, limit=2)

    assert prompt.count(",") == 1
    assert "Thornbury" not in prompt


def test_no_vocabulary_means_no_prompt():
    assert asr_prompt([]) == ""
    assert asr_prompt(["", "   "]) == ""
