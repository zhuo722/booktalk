"""Lexical retrieval: BM25 over a language-aware tokenizer.

Dense search alone misses rare proper nouns, numbers and exact names — exactly
the things readers ask a non-fiction book about — so every query also runs
through BM25 and the two rankings are fused.

The tokenizer splits a string into CJK and non-CJK runs: CJK runs go through
jieba (no whitespace to split on), latin runs through a regex. jieba is imported
lazily because loading its dictionary costs about a second.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi

CJK_RUN = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")
LATIN_TOKEN = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[^\W\d_]+", re.UNICODE)
_MIN_CJK_TOKEN = 1
# Placeholder token for documents that tokenize to nothing; BM25Okapi divides
# by the average document length and so cannot take an empty document.
_EMPTY_DOC_TOKEN = "§"

_JIEBA = None


def _jieba():
    """Import jieba on first CJK query/document (dictionary load is slow)."""
    global _JIEBA
    if _JIEBA is None:
        import jieba

        _JIEBA = jieba
    return _JIEBA


def has_cjk(text: str) -> bool:
    """True when the text contains any CJK ideograph."""
    return bool(CJK_RUN.search(text))


def tokenize(text: str) -> List[str]:
    """Lowercased tokens: jieba for CJK runs, regex words for everything else."""
    if not text:
        return []
    tokens: List[str] = []
    cursor = 0
    for match in CJK_RUN.finditer(text):
        if match.start() > cursor:
            tokens.extend(_latin_tokens(text[cursor : match.start()]))
        tokens.extend(_cjk_tokens(match.group(0)))
        cursor = match.end()
    if cursor < len(text):
        tokens.extend(_latin_tokens(text[cursor:]))
    return tokens


def _latin_tokens(text: str) -> List[str]:
    return [match.group(0).lower() for match in LATIN_TOKEN.finditer(text)]


def _cjk_tokens(text: str) -> List[str]:
    words = [word.strip() for word in _jieba().lcut(text)]
    return [word for word in words if len(word) >= _MIN_CJK_TOKEN]


class BM25Index:
    """BM25-Okapi over pre-tokenized documents, addressed by row index."""

    def __init__(self, documents: Sequence[str]) -> None:
        self.size = len(documents)
        self._tokenized: List[List[str]] = [tokenize(doc) for doc in documents]
        # BM25Okapi rejects an empty corpus, and a document with no tokens
        # would divide by zero on average length; both are padded here.
        corpus = [tokens or [_EMPTY_DOC_TOKEN] for tokens in self._tokenized] or [
            [_EMPTY_DOC_TOKEN]
        ]
        self._bm25: Optional[BM25Okapi] = BM25Okapi(corpus) if documents else None

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Return ``(row_index, score)`` for the ``k`` best documents."""
        if k <= 0 or self._bm25 is None or self.size == 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            range(self.size), key=lambda i: (-float(scores[i]), i)
        )[: min(k, self.size)]
        return [(int(i), float(scores[i])) for i in ranked if float(scores[i]) > 0.0]
