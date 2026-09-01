"""HTTP layer: routers and the shared request dependencies.

Everything the routers need hangs off ``app.state``, which :func:`create_app`
populates — including the :class:`AIClient`, so tests can swap in the fake by
passing one argument.
"""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from app.config import Settings
from app.index.store import Store
from app.services.openai_client import AIClient
from app.sessions import SessionStore
from app.speech.turns import TurnRegistry

NO_KEY_DETAIL = (
    "OpenAI is not configured. Add OPENAI_API_KEY to backend/.env "
    "(see .env.example) and restart the server."
)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> Store:
    return request.app.state.store


def get_sessions(request: Request) -> SessionStore:
    return request.app.state.sessions


def get_turns(request: Request) -> TurnRegistry:
    """The registry of in-flight answers, for stamping and cancelling turns."""
    return request.app.state.turns


def get_ai(request: Request) -> AIClient:
    """The configured AI client, or a 503 explaining how to configure one."""
    client = request.app.state.ai
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=NO_KEY_DETAIL
        )
    return client
