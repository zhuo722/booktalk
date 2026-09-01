"""Reciprocal rank fusion arithmetic and ordering (SPEC §5.2 step 2)."""

from __future__ import annotations

import pytest

from app.index.fusion import DEFAULT_K, reciprocal_rank_fusion


def test_single_ranking_is_preserved():
    fused = reciprocal_rank_fusion([["a", "b", "c"]])

    assert [key for key, _ in fused] == ["a", "b", "c"]
    assert fused[0][1] == pytest.approx(1 / (DEFAULT_K + 1))


def test_scores_match_the_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]], k=60))

    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused["d"] == pytest.approx(1 / 63)


def test_agreement_between_lists_outranks_a_single_first_place():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]], k=60)

    assert [key for key, _ in fused] == ["a", "c", "b", "d"]


def test_small_k_sharpens_the_advantage_of_rank_one():
    sharp = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    flat = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))

    assert sharp["a"] / sharp["b"] > flat["a"] / flat["b"]


def test_ties_break_on_first_appearance():
    fused = reciprocal_rank_fusion([["x"], ["y"]])

    assert [key for key, _ in fused] == ["x", "y"]
    assert fused[0][1] == pytest.approx(fused[1][1])


def test_weights_scale_each_ranking():
    fused = dict(
        reciprocal_rank_fusion([["a"], ["b"]], k=60, weights=[2.0, 1.0])
    )

    assert fused["a"] == pytest.approx(2 / 61)
    assert fused["b"] == pytest.approx(1 / 61)


def test_k_arbitrates_between_a_first_place_and_broad_agreement():
    # "a" is rank 1 in one list only; "b" is rank 5 there but rank 1 in the
    # other. A small k makes rank one nearly decisive; a large k flattens the
    # curve until agreement across both lists wins. This is exactly the choice
    # SPEC §14.6 sweeps, and the reason 60 is not the default for one book.
    rankings = [["a", "x1", "x2", "x3", "b"], ["b"]]

    sharp = reciprocal_rank_fusion(rankings, k=1, weights=[1.0, 0.35])
    flat = reciprocal_rank_fusion(rankings, k=60, weights=[1.0, 0.35])

    assert sharp[0][0] == "a"
    assert flat[0][0] == "b"


def test_a_zero_weight_silences_a_ranking():
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]], k=20, weights=[1.0, 0.0]))

    assert fused["a"] == pytest.approx(1 / 21)
    assert fused["b"] == pytest.approx(0.0)


def test_the_default_k_is_tuned_for_a_single_book():
    # Cormack et al.'s 60 assumes million-document TREC corpora; a book is four
    # orders of magnitude smaller and its ranks are correspondingly sharper.
    assert DEFAULT_K == 20


def test_mismatched_weights_are_rejected():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_empty_input_returns_nothing():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_every_list_an_id_appears_in_contributes():
    two_lists = dict(reciprocal_rank_fusion([["a"], ["a"]]))
    three_lists = dict(reciprocal_rank_fusion([["a"], ["a"], ["a"]]))

    assert two_lists["a"] == pytest.approx(2 / (DEFAULT_K + 1))
    assert three_lists["a"] == pytest.approx(3 / (DEFAULT_K + 1))
