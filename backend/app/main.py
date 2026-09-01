"""FastAPI application factory.

``create_app`` takes the two things tests want to control — the settings and the
AI client — so the whole HTTP surface can run offline against
:class:`FakeAIClient` with no monkeypatching.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import asr, books, chat
from app.config import BACKEND_DIR, Settings, load_settings
from app.db import init_db
from app.index import rerank
from app.index.store import Store
from app.models import HealthResponse
from app.services.openai_client import AIClient, build_client
from app.sessions import SessionStore
from app.speech.turns import TurnRegistry

logger = logging.getLogger(__name__)

DEFAULT_STATIC_DIR = BACKEND_DIR.parent / "frontend" / "dist"


def create_app(
    settings: Optional[Settings] = None,
    ai: Optional[AIClient] = None,
    store: Optional[Store] = None,
) -> FastAPI:
    """Build the application.

    ``ai=None`` means "decide from the configuration": a real client when
    ``OPENAI_API_KEY`` is set, otherwise nothing at all — and the endpoints that
    need a model answer 503 with instructions instead of failing obscurely.
    """
    settings = settings or load_settings()
    settings.ensure_dirs()
    init_db(settings.db_path)

    app = FastAPI(title="BookTalk", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.store = store or Store(settings.db_path)
    app.state.ai = ai if ai is not None else build_client(settings)
    app.state.sessions = SessionStore()
    app.state.turns = TurnRegistry()
    app.state.reranker = None
    _prewarm_reranker(app, settings)

    @app.get("/api/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        """Liveness plus whether model calls are possible at all."""
        return HealthResponse(ok=True, openai_configured=settings.openai_configured)

    app.include_router(books.router)
    app.include_router(asr.router)
    app.include_router(chat.router)
    _mount_static(app, settings)
    return app


def _prewarm_reranker(app: FastAPI, settings: Settings) -> None:
    """Load the cross-encoder off the request path (SPEC §14.6).

    On a background daemon thread, because the first load may fetch 5 MB and
    build an ONNX session, and no reader should pay for that on their first
    question. Gated on ``openai_configured``: an app with no API key makes no
    network calls at all, which is exactly the property the offline test suite
    depends on. Until (or unless) it lands, ``app.state.reranker`` stays
    ``None`` and retrieval runs on fusion order.
    """
    if not settings.flashrank or not settings.openai_configured:
        return

    def load() -> None:
        try:
            app.state.reranker = rerank.prewarm(settings)
        except Exception:  # noqa: BLE001 - a cold reranker must not block boot
            logger.warning("reranker pre-warm failed", exc_info=True)

    threading.Thread(target=load, name="rerank-prewarm", daemon=True).start()


def _mount_static(app: FastAPI, settings: Settings) -> None:
    """Serve a built frontend from ``/`` when one exists.

    Mounted last so the API routes always win; absent in development, where Vite
    serves the UI on its own port.
    """
    static_dir: Optional[Path] = settings.static_dir or DEFAULT_STATIC_DIR
    if static_dir is None or not static_dir.is_dir():
        return
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    logger.info("serving static frontend from %s", static_dir)


def __getattr__(name: str) -> Any:
    """Build ``app`` on first access, for ``uvicorn app.main:app``.

    Lazy so that importing :func:`create_app` (as the tests do) does not create
    a database in the default data directory as a side effect.
    """
    if name == "app":
        application = create_app()
        globals()["app"] = application
        return application
    raise AttributeError("module {0!r} has no attribute {1!r}".format(__name__, name))
