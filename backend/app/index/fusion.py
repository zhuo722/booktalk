"""Reciprocal Rank Fusion (SPEC §5.2 step 2, revised by §14.6).

RRF merges rankings by position rather than score, so a BM25 score of 14.2 and a
cosine of 0.83 never have to be made commensurable:

    score(d) = Σ_lists w_list / (k + rank_list(d))     (rank is 1-based)

``k`` controls how quickly weight decays with rank. The familiar ``k = 60``
comes from Cormack et al., whose TREC corpora hold millions of documents: with
that much noise, a flat curve that rewards agreement across lists is the right
bet. One book is four orders of magnitude smaller, and its ranks are much more
informative, so the small-corpus guidance (k = 10–20) applies instead —
``DEFAULT_K = 20`` here, swept in eval over {10, 20, 60}.

``weights`` scales each list's contribution (Anthropic's own hybrid-search
cookbook uses an uneven split). The retriever passes them in leg order —
dense first, then BM25 — from settings, so a sweep needs no code change.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Sequence, Tuple

DEFAULT_K = 20


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]],
    k: int = DEFAULT_K,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[Hashable, float]]:
    """Fuse ranked id lists into one ranking of ``(id, score)``, best first.

    Ties are broken by the order in which ids were first seen, which keeps the
    output deterministic for equal scores.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")

    scores: Dict[Hashable, float] = {}
    first_seen: Dict[Hashable, int] = {}
    order = 0
    for list_index, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else float(weights[list_index])
        for rank, key in enumerate(ranking, start=1):
            if key not in first_seen:
                first_seen[key] = order
                order += 1
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
