"""Application settings, read from the process environment (see SPEC §8).

Deliberately dependency-light: a frozen dataclass populated from ``os.environ``
plus an optional ``.env`` file loaded by python-dotenv. No pydantic-settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


def _env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None else value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env_str(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


CONTEXTUAL_ENV = "BOOKTALK_CONTEXTUAL"


def contextual_enabled(default: bool = True) -> bool:
    """Whether per-chunk LLM contextual enrichment is on (SPEC §14.4).

    Read straight from the process environment rather than from a
    :class:`Settings` snapshot: ingestion runs on a background thread that is
    handed a store and a client, not the settings object, and this must not pull
    ``.env`` (and with it an API key) into a test process.
    """
    return _env_bool(CONTEXTUAL_ENV, default)


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the runtime configuration."""

    openai_api_key: str = ""
    openai_base_url: Optional[str] = None
    # Optional separate provider for chat completions only (answer, rewrite,
    # ingest enrichment) — e.g. DeepSeek, which has no embeddings, ASR or TTS.
    # Empty means chat goes to OpenAI like everything else.
    chat_api_key: str = ""
    chat_base_url: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    asr_model: str = "gpt-4o-mini-transcribe"
    asr_fallback_model: str = "whisper-1"
    tts_model: str = "gpt-4o-mini-tts"
    tts_fallback_model: str = "tts-1"
    tts_voice: str = "alloy"
    embed_model: str = "text-embedding-3-small"
    embed_dims: int = 1536
    max_upload_mb: int = 200
    data_dir: Path = BACKEND_DIR / "data"
    rerank: bool = False
    contextual: bool = True
    # Retrieval knobs (SPEC §14.6). k=20 and an even 0.5/0.5 split are the
    # starting point for the eval sweep, not a measured optimum: TREC's k=60
    # assumes million-document corpora, a single book is four orders smaller.
    rrf_k: int = 20
    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    flashrank: bool = True
    flashrank_model: str = "ms-marco-TinyBERT-L-2-v2"
    cors_origins: List[str] = field(default_factory=lambda: list(DEFAULT_CORS_ORIGINS))
    static_dir: Optional[Path] = None
    # Public-demo mode: per-IP rate limits, and the books present at boot
    # cannot be deleted by visitors (see app/demo.py).
    demo: bool = False

    @property
    def openai_configured(self) -> bool:
        """True when an API key is present, i.e. live OpenAI calls are possible."""
        return bool(self.openai_api_key)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def db_path(self) -> Path:
        return self.data_dir / "booktalk.db"

    @property
    def books_dir(self) -> Path:
        return self.data_dir / "books"

    @property
    def flashrank_cache_dir(self) -> Path:
        """Where the reranker's ONNX model is unpacked (5 MB, downloaded once)."""
        return self.data_dir / "flashrank"

    @property
    def rrf_weights(self) -> "tuple":
        """Per-leg RRF weights in the retriever's leg order: dense, then BM25."""
        return (self.dense_weight, self.bm25_weight)

    def ensure_dirs(self) -> None:
        """Create the data directories this configuration points at."""
        self.books_dir.mkdir(parents=True, exist_ok=True)


def load_settings(env_file: Optional[Path] = None) -> Settings:
    """Build a :class:`Settings` from the environment.

    ``backend/.env`` is loaded first when present, but never overrides variables
    that are already exported in the process environment.
    """
    dotenv_path = env_file if env_file is not None else BACKEND_DIR / ".env"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path, override=False)

    data_dir = Path(_env_str("BOOKTALK_DATA_DIR", "") or str(BACKEND_DIR / "data"))
    if not data_dir.is_absolute():
        data_dir = (BACKEND_DIR / data_dir).resolve()

    origins_raw = _env_str("BOOKTALK_CORS_ORIGINS")
    origins = (
        [o.strip() for o in origins_raw.split(",") if o.strip()]
        if origins_raw
        else list(DEFAULT_CORS_ORIGINS)
    )

    static_raw = _env_str("BOOKTALK_STATIC_DIR")
    static_dir = Path(static_raw).expanduser() if static_raw else None

    # BOOKTALK_CHAT_API_KEY / _BASE_URL are generic; DEEPSEEK_API_KEY is the
    # shorthand that also implies DeepSeek's endpoint and model.
    deepseek_key = _env_str("DEEPSEEK_API_KEY")
    chat_key = _env_str("BOOKTALK_CHAT_API_KEY") or deepseek_key
    chat_base_url = _env_str("BOOKTALK_CHAT_BASE_URL") or (
        DEEPSEEK_BASE_URL if chat_key and chat_key == deepseek_key else ""
    )
    default_llm = DEEPSEEK_MODEL if chat_base_url == DEEPSEEK_BASE_URL else "gpt-4o-mini"

    return Settings(
        openai_api_key=_env_str("OPENAI_API_KEY"),
        openai_base_url=_env_str("OPENAI_BASE_URL") or None,
        chat_api_key=chat_key,
        chat_base_url=chat_base_url or None,
        llm_model=_env_str("BOOKTALK_LLM_MODEL", default_llm),
        asr_model=_env_str("BOOKTALK_ASR_MODEL", "gpt-4o-mini-transcribe"),
        tts_model=_env_str("BOOKTALK_TTS_MODEL", "gpt-4o-mini-tts"),
        tts_voice=_env_str("BOOKTALK_TTS_VOICE", "alloy"),
        embed_model=_env_str("BOOKTALK_EMBED_MODEL", "text-embedding-3-small"),
        max_upload_mb=_env_int("BOOKTALK_MAX_UPLOAD_MB", 200),
        data_dir=data_dir,
        rerank=_env_bool("BOOKTALK_RERANK", False),
        contextual=contextual_enabled(),
        rrf_k=_env_int("BOOKTALK_RRF_K", 20),
        dense_weight=_env_float("BOOKTALK_DENSE_WEIGHT", 0.5),
        bm25_weight=_env_float("BOOKTALK_BM25_WEIGHT", 0.5),
        flashrank=_env_bool("BOOKTALK_FLASHRANK", True),
        flashrank_model=_env_str(
            "BOOKTALK_FLASHRANK_MODEL", "ms-marco-TinyBERT-L-2-v2"
        ),
        cors_origins=origins,
        static_dir=static_dir,
        demo=_env_bool("BOOKTALK_DEMO", False),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (used by the uvicorn entrypoint)."""
    return load_settings()
