"""Public-demo guard rails and the separate chat provider (DeepSeek)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from fastapi.testclient import TestClient

from app import demo
from app.config import DEEPSEEK_BASE_URL, Settings, load_settings
from app.main import create_app
from app.services.openai_client import OpenAIClient


def _no_env_file(tmp_path: Path) -> Path:
    return tmp_path / "missing.env"


def test_a_deepseek_key_routes_chat_to_deepseek(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")

    settings = load_settings(_no_env_file(tmp_path))

    assert settings.chat_api_key == "sk-deepseek"
    assert settings.chat_base_url == DEEPSEEK_BASE_URL
    assert settings.llm_model == "deepseek-chat"
    client = OpenAIClient(settings)
    assert str(client._chat_client.base_url).startswith(DEEPSEEK_BASE_URL)
    assert str(client._client.base_url).startswith("https://api.openai.com")


def test_without_a_chat_key_chat_shares_the_openai_client(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    settings = load_settings(_no_env_file(tmp_path))

    assert settings.chat_api_key == ""
    assert settings.llm_model == "gpt-4o-mini"
    client = OpenAIClient(settings)
    assert client._chat_client is client._client


def test_an_explicit_llm_model_wins_over_the_deepseek_default(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("BOOKTALK_LLM_MODEL", "deepseek-reasoner")

    assert load_settings(_no_env_file(tmp_path)).llm_model == "deepseek-reasoner"


def test_the_limiter_refuses_past_the_per_ip_limit_and_recovers():
    now = [0.0]
    limiter = demo.RateLimiter(clock=lambda: now[0])
    rule = demo.Rule("POST", demo.re.compile("^/x$"), 2, 60.0, 100, "things")

    assert limiter.check(rule, "a") is None
    assert limiter.check(rule, "a") is None
    assert "2 things per hour" in limiter.check(rule, "a")
    assert limiter.check(rule, "b") is None

    now[0] = 61.0
    assert limiter.check(rule, "a") is None


def test_the_limiter_enforces_the_daily_total_across_ips():
    limiter = demo.RateLimiter(clock=lambda: 0.0)
    rule = demo.Rule("POST", demo.re.compile("^/x$"), 10, 60.0, 2, "things")

    assert limiter.check(rule, "a") is None
    assert limiter.check(rule, "b") is None
    assert "today's limit" in limiter.check(rule, "c")


def test_demo_mode_protects_books_present_at_boot(settings: Settings, fake_ai, store):
    store.create_book(
        book_id="ab" * 16, title="Seed", filename="seed.pdf", sha256="x", status="ready"
    )
    demo_settings = dataclasses.replace(settings, demo=True)

    with TestClient(create_app(settings=demo_settings, ai=fake_ai)) as client:
        refused = client.delete("/api/books/" + "ab" * 16)
        missing = client.delete("/api/books/" + "cd" * 16)

    assert refused.status_code == 403
    assert "demo library" in refused.json()["detail"]
    assert missing.status_code == 404


def test_demo_mode_rate_limits_uploads(settings: Settings, fake_ai):
    demo_settings = dataclasses.replace(settings, demo=True)
    headers = {"x-forwarded-for": "203.0.113.9, 10.0.0.1"}

    with TestClient(create_app(settings=demo_settings, ai=fake_ai)) as client:
        codes = [
            client.post(
                "/api/books",
                files={"file": ("a.txt", b"nope", "text/plain")},
                headers=headers,
            ).status_code
            for _ in range(4)
        ]

    assert codes == [400, 400, 400, 429]
