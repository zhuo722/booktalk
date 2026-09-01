"""Dense cosine search and the determinism contract of FakeAIClient."""

from __future__ import annotations

import numpy as np
import pytest

from app.index.dense import normalize, top_k, to_matrix
from app.index.store import decode_embedding, encode_embedding
from app.services.openai_client import FakeAIClient


def test_normalize_makes_every_row_a_unit_vector():
    matrix = normalize(to_matrix([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]]))

    assert np.allclose(np.linalg.norm(matrix[:2], axis=1), 1.0)
    assert np.allclose(matrix[0], [0.6, 0.8])
    assert np.allclose(matrix[2], [0.0, 0.0])  # a zero row stays zero


def test_top_k_orders_by_cosine():
    matrix = normalize(to_matrix([[1.0, 0.0], [0.9, 0.4], [0.0, 1.0]]))

    ranked = top_k(matrix, [1.0, 0.0], k=3)

    assert [row for row, _ in ranked] == [0, 1, 2]
    assert ranked[0][1] == pytest.approx(1.0)
    assert ranked[2][1] == pytest.approx(0.0, abs=1e-6)


def test_top_k_normalizes_the_query():
    matrix = normalize(to_matrix([[1.0, 0.0], [0.0, 1.0]]))

    long_query = top_k(matrix, [10.0, 0.0], k=1)
    unit_query = top_k(matrix, [1.0, 0.0], k=1)

    assert long_query == unit_query


def test_top_k_is_capped_and_deterministic():
    matrix = normalize(to_matrix([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))

    ranked = top_k(matrix, [1.0, 0.0], k=99)

    assert len(ranked) == 3
    assert [row for row, _ in ranked][:2] == [0, 1]  # ties break on row index


def test_empty_index_and_non_positive_k():
    empty = normalize(to_matrix([], dims=4))

    assert top_k(empty, [1.0, 0.0, 0.0, 0.0], k=5) == []
    assert top_k(normalize(to_matrix([[1.0, 0.0]])), [1.0, 0.0], k=0) == []


def test_dimension_mismatch_is_rejected():
    matrix = normalize(to_matrix([[1.0, 0.0]]))

    with pytest.raises(ValueError):
        top_k(matrix, [1.0, 0.0, 0.0], k=1)


def test_embeddings_survive_the_blob_round_trip():
    vector = [0.25, -0.5, 0.125]

    restored = decode_embedding(encode_embedding(vector))

    assert restored.dtype == np.float32
    assert np.allclose(restored, vector)
    assert decode_embedding(None).shape == (0,)


def test_fake_embeddings_are_deterministic_and_normalized():
    ai = FakeAIClient(dims=16)

    first, second = ai.embed(["the harbour"]), ai.embed(["the harbour"])

    assert first == second
    assert np.linalg.norm(first[0]) == pytest.approx(1.0, abs=1e-5)


def test_exact_text_is_retrieved_first_with_fake_embeddings():
    """The fake guarantees exact-text identity, not semantic similarity."""
    ai = FakeAIClient(dims=32)
    documents = ["chapter one", "chapter two", "chapter three"]
    matrix = normalize(to_matrix(ai.embed(documents)))

    ranked = top_k(matrix, ai.embed(["chapter two"])[0], k=3)

    assert ranked[0][0] == 1
    assert ranked[0][1] == pytest.approx(1.0, abs=1e-5)
