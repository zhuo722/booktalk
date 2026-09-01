"""Cross-encoder reranking of fused candidates (SPEC §14.6).

RRF fuses two *rankings*; it never looks at the query and a passage together.
A cross-encoder does, which is why a rerank pass over the fused head is the
cheapest remaining accuracy win. The model is FlashRank's
``ms-marco-TinyBERT-L-2-v2``: 5 MB of ONNX, ~7.6 ms for 20 passages on this
machine, no torch — 0.2 % of the first-audio latency budget.

Three properties this module exists to guarantee:

* **Never on the import path.** ``flashrank`` and ``onnxruntime`` are imported
  inside :func:`load_reranker`, so the app (and the test suite) boots without
  them installed.
* **Never a surprise download.** :class:`flashrank.Ranker` fetches the model
  from HuggingFace in its constructor. ``allow_download=False`` refuses to
  construct one unless the model is already unpacked in the cache directory,
  which is what the offline test suite and any air-gapped run rely on.
* **Never fatal.** Every failure path — missing dependency, missing model,
  no network, a broken ONNX session — returns ``None`` and leaves the caller
  with the fusion order it already had.

The retriever takes a reranker as a *parameter*; it never loads one itself.
Loading happens once, at application startup, through :func:`prewarm` (the
first inference on a cold session is ~10× the steady-state cost, and that is
latency we refuse to spend on a reader's first question).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ms-marco-TinyBERT-L-2-v2"
# The cross-encoder truncates at 512 tokens, which is why it is fed CHILD
# chunks (~256-320 tokens) and never parents (1600) — a truncated parent would
# be scored on its first third.
MAX_LENGTH = 512


class Reranker(Protocol):
    """Scores query/passage pairs. One score per passage, same order, 0..1."""

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        ...


class FlashRankReranker:
    """Adapter from FlashRank's request/response objects to :class:`Reranker`.

    FlashRank returns the passages *re-sorted* with a ``score`` key added, so
    the adapter round-trips positions through the passage ``id`` field to hand
    the caller scores back in the original order.
    """

    def __init__(self, ranker: Any, model_name: str = DEFAULT_MODEL) -> None:
        self._ranker = ranker
        self.model_name = model_name

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        if not passages:
            return []
        from flashrank import RerankRequest

        request = RerankRequest(
            query=query,
            passages=[
                {"id": position, "text": text}
                for position, text in enumerate(passages)
            ],
        )
        scores = [0.0] * len(passages)
        for item in self._ranker.rerank(request):
            position = int(item["id"])
            if 0 <= position < len(scores):
                scores[position] = float(item["score"])
        return scores


_CACHE: Dict[Tuple[str, str], Optional[Reranker]] = {}
_LOCK = threading.Lock()


def load_reranker(
    model_name: str = DEFAULT_MODEL,
    cache_dir: Optional[Union[str, Path]] = None,
    allow_download: bool = True,
    max_length: int = MAX_LENGTH,
) -> Optional[Reranker]:
    """Build a reranker, or return ``None`` when one cannot be had.

    Results (including failures) are memoized per ``(model, cache_dir)`` so a
    machine with no model does not re-attempt a download on every query.
    """
    directory = Path(cache_dir) if cache_dir is not None else Path("/tmp")
    key = (model_name, str(directory))
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]

    reranker = _build(model_name, directory, allow_download, max_length)
    with _LOCK:
        _CACHE[key] = reranker
    return reranker


def _build(
    model_name: str, directory: Path, allow_download: bool, max_length: int
) -> Optional[Reranker]:
    if not (directory / model_name).is_dir() and not allow_download:
        logger.info(
            "reranker %s not cached in %s and downloads are disabled; skipping",
            model_name,
            directory,
        )
        return None
    try:
        from flashrank import Ranker
    except Exception:  # noqa: BLE001 - optional dependency
        logger.info("flashrank not installed; retrieval will use fusion order")
        return None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        ranker = Ranker(
            model_name=model_name,
            cache_dir=str(directory),
            max_length=max_length,
            log_level="WARNING",
        )
    except Exception:  # noqa: BLE001 - offline, disk, or a bad ONNX session
        logger.warning(
            "could not load reranker %s; retrieval will use fusion order",
            model_name,
            exc_info=True,
        )
        return None
    return FlashRankReranker(ranker, model_name=model_name)


def get_reranker(settings: Any, allow_download: bool = True) -> Optional[Reranker]:
    """The reranker configured by ``settings``, or ``None`` when disabled.

    ``settings`` is duck-typed rather than imported so this module stays usable
    from the eval harness with a plain namespace.
    """
    if not getattr(settings, "flashrank", False):
        return None
    return load_reranker(
        model_name=getattr(settings, "flashrank_model", DEFAULT_MODEL),
        cache_dir=getattr(settings, "flashrank_cache_dir", None),
        allow_download=allow_download,
    )


def prewarm(settings: Any) -> Optional[Reranker]:
    """Load the model at startup so no reader pays for the cold session.

    Safe to call from a background thread: the first call may download 5 MB.
    """
    reranker = get_reranker(settings)
    if reranker is None:
        return None
    try:
        reranker.score("warm up the onnx session", ["a short passage of prose"])
    except Exception:  # noqa: BLE001 - a warm-up failure must not block boot
        logger.warning("reranker warm-up failed", exc_info=True)
        return None
    logger.info("reranker ready: %s", getattr(reranker, "model_name", "?"))
    return reranker


def reset_cache() -> None:
    """Forget memoized load attempts (tests, and re-configuration in eval)."""
    with _LOCK:
        _CACHE.clear()
