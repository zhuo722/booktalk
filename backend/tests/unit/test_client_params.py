"""The AIClient seam: kwarg normalization and the ingest-time canned replies."""

from __future__ import annotations

import json

from app.config import Settings
from app.services.openai_client import OpenAIClient


def client() -> OpenAIClient:
    """A live client object; constructing one makes no network call."""
    return OpenAIClient(Settings(openai_api_key="test-key-not-used"))


def test_an_optional_model_override_of_none_falls_back_to_the_default():
    """``model=None`` means "unset" — it must never reach the API as a null."""
    params = client()._chat_params({"purpose": "context", "temperature": 0, "model": None})

    assert params == {"temperature": 0, "model": "gpt-4o-mini"}


def test_an_explicit_model_override_wins():
    params = client()._chat_params({"purpose": "summary", "model": "gpt-4o"})

    assert params["model"] == "gpt-4o"
    assert "purpose" not in params


def test_the_fake_client_answers_the_ingest_call_sites(fake_ai):
    levels = fake_ai.chat(
        [{"role": "user", "content": "Entries:\n1. PART ONE\n2. Chapter 1\n3. Chapter 2"}],
        purpose="toc_levels",
    )
    context = fake_ai.chat(
        [{"role": "user", "content": "<excerpt>\nthe ferry leaves at dawn\n</excerpt>"}],
        purpose="context",
    )
    summary = fake_ai.chat(
        [{"role": "user", "content": "Chapter: Isle of Basalt\n\nthe ferry leaves at dawn"}],
        purpose="summary",
    )

    assert json.loads(levels) == [
        {"level": 1, "title": "PART ONE"},
        {"level": 2, "title": "Chapter 1"},
        {"level": 2, "title": "Chapter 2"},
    ]
    assert context == "This excerpt sits in the chapter and covers the ferry leaves at dawn."
    assert summary.startswith("Summary: ")


def test_the_fake_client_is_deterministic(fake_ai):
    messages = [{"role": "user", "content": "<excerpt>\nthe quarry closed\n</excerpt>"}]

    assert fake_ai.chat(messages, purpose="context") == fake_ai.chat(
        messages, purpose="context"
    )


def test_a_blank_base_url_env_var_never_reaches_the_sdk(monkeypatch):
    """A blank ``OPENAI_BASE_URL=`` line in .env loads as "" — with base_url
    left to the SDK's env fallback that empty string became the base URL and
    every request failed with a connection error. The client must therefore
    always pass an explicit base_url."""
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    c = OpenAIClient(Settings(openai_api_key="test-key-not-used"))
    assert str(c._client.base_url) == "https://api.openai.com/v1/"


def test_a_configured_base_url_is_used(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    c = OpenAIClient(
        Settings(openai_api_key="k", openai_base_url="https://gw.example.com/v1")
    )
    assert str(c._client.base_url).startswith("https://gw.example.com/v1")
