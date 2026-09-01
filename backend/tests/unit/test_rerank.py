"""Loading and adapting the cross-encoder reranker (SPEC §14.6).

The contract these tests pin down is "never fatal, never a surprise download":
every path through :mod:`app.index.rerank` must either hand back a working
reranker or ``None``, and none of them may reach the network. Every test here
passes ``allow_download=False`` or a disabled setting, so the suite stays
offline even on a machine that has never fetched the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.config import Settings
from app.index import rerank


@pytest.fixture(autouse=True)
def clear_reranker_cache():
    """Memoized load attempts must not leak between tests."""
    rerank.reset_cache()
    yield
    rerank.reset_cache()


class StubRanker:
    """Mimics ``flashrank.Ranker``: returns passages re-sorted, scores added."""

    def __init__(self, scores: List[float]) -> None:
        self.scores = scores
        self.requests: List[Any] = []

    def rerank(self, request: Any) -> List[Dict[str, Any]]:
        self.requests.append(request)
        passages = [dict(passage) for passage in request.passages]
        for passage, score in zip(passages, self.scores):
            passage["score"] = score
        passages.sort(key=lambda passage: passage["score"], reverse=True)
        return passages


# ---------------------------------------------------------------- the adapter


def test_adapter_returns_scores_in_the_input_order():
    # FlashRank hands back a *re-sorted* list; the adapter must undo that.
    ranker = rerank.FlashRankReranker(StubRanker([0.1, 0.9, 0.5]))

    scores = ranker.score("question", ["first", "second", "third"])

    assert scores == pytest.approx([0.1, 0.9, 0.5])


def test_adapter_sends_the_query_and_every_passage():
    stub = StubRanker([0.2, 0.4])
    ranker = rerank.FlashRankReranker(stub)

    ranker.score("who founded the lighthouse", ["alpha", "bravo"])

    request = stub.requests[0]
    assert request.query == "who founded the lighthouse"
    assert [passage["text"] for passage in request.passages] == ["alpha", "bravo"]


def test_adapter_short_circuits_on_no_passages():
    stub = StubRanker([])

    assert rerank.FlashRankReranker(stub).score("question", []) == []
    assert stub.requests == []


# ----------------------------------------------------------------- the loader


def test_an_uncached_model_is_skipped_when_downloads_are_disabled(tmp_path: Path):
    assert rerank.load_reranker(cache_dir=tmp_path, allow_download=False) is None


def test_load_attempts_are_memoized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    attempts = []

    def counted(model_name, directory, allow_download, max_length):
        attempts.append(model_name)
        return None

    monkeypatch.setattr(rerank, "_build", counted)

    rerank.load_reranker(cache_dir=tmp_path, allow_download=False)
    rerank.load_reranker(cache_dir=tmp_path, allow_download=False)

    # A machine with no model must not re-attempt a download per query.
    assert attempts == [rerank.DEFAULT_MODEL]


def test_a_broken_model_directory_degrades_to_none(tmp_path: Path):
    # The directory exists (so the download guard passes) but holds no ONNX
    # file: constructing the Ranker raises, and that must not propagate.
    (tmp_path / rerank.DEFAULT_MODEL).mkdir()

    assert rerank.load_reranker(cache_dir=tmp_path, allow_download=False) is None


def test_a_missing_flashrank_install_degrades_to_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "flashrank":
            raise ImportError("no flashrank here")
        return real_import(name, *args, **kwargs)

    (tmp_path / rerank.DEFAULT_MODEL).mkdir()
    monkeypatch.setattr(builtins, "__import__", refuse)

    assert rerank.load_reranker(cache_dir=tmp_path, allow_download=False) is None


# --------------------------------------------------------------- settings glue


def test_settings_can_turn_the_reranker_off(tmp_path: Path):
    settings = Settings(data_dir=tmp_path, flashrank=False)

    assert rerank.get_reranker(settings) is None
    assert rerank.prewarm(settings) is None


def test_settings_point_the_cache_at_the_data_directory(tmp_path: Path):
    settings = Settings(data_dir=tmp_path)

    assert settings.flashrank is True
    assert settings.flashrank_cache_dir == tmp_path / "flashrank"
    assert rerank.get_reranker(settings, allow_download=False) is None


def test_prewarm_survives_a_model_that_scores_badly(monkeypatch: pytest.MonkeyPatch):
    class Exploding:
        model_name = "boom"

        def score(self, query, passages):
            raise RuntimeError("onnx session died")

    monkeypatch.setattr(rerank, "get_reranker", lambda settings: Exploding())

    assert rerank.prewarm(Settings()) is None


def test_prewarm_runs_one_inference_to_warm_the_session(monkeypatch: pytest.MonkeyPatch):
    class Counting:
        model_name = "tiny"

        def __init__(self):
            self.calls = 0

        def score(self, query, passages):
            self.calls += 1
            return [0.5] * len(passages)

    warm = Counting()
    monkeypatch.setattr(rerank, "get_reranker", lambda settings: warm)

    assert rerank.prewarm(Settings()) is warm
    assert warm.calls == 1
