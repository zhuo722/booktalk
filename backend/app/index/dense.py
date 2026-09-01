"""Dense retrieval: cosine top-k over a normalized float32 matrix.

Rows are L2-normalized once at build time, so cosine similarity is a single
matrix-vector product. At book scale (tens of thousands of chunks) this is well
under a millisecond and needs no vector database.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

EPSILON = 1e-12


def to_matrix(vectors: Sequence[Sequence[float]], dims: int = 0) -> np.ndarray:
    """Stack embedding vectors into a ``(n, dims)`` float32 matrix."""
    if not vectors:
        return np.zeros((0, dims), dtype=np.float32)
    return np.asarray(vectors, dtype=np.float32)


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row; zero rows are left as zeros."""
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, EPSILON)


def top_k(
    matrix: np.ndarray, query: Sequence[float], k: int
) -> List[Tuple[int, float]]:
    """Return ``(row_index, cosine)`` for the ``k`` closest rows, best first.

    ``matrix`` must already be normalized (see :func:`normalize`); the query is
    normalized here. Ties break on the lower row index, so results are stable.
    """
    if k <= 0 or matrix.size == 0:
        return []
    vector = np.asarray(query, dtype=np.float32)
    if vector.ndim != 1 or vector.shape[0] != matrix.shape[1]:
        raise ValueError(
            "query has {0} dims, index has {1}".format(
                vector.shape[-1] if vector.ndim else 0, matrix.shape[1]
            )
        )
    norm = float(np.linalg.norm(vector))
    if norm > EPSILON:
        vector = vector / norm
    scores = matrix @ vector
    limit = min(k, scores.shape[0])
    # argpartition finds the top-k without a full sort, then we sort those k.
    candidates = np.argpartition(-scores, limit - 1)[:limit]
    ordered = sorted(candidates, key=lambda i: (-float(scores[i]), int(i)))
    return [(int(i), float(scores[i])) for i in ordered]
