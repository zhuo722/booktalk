"""Conversational query rewriting (SPEC §5.2 step 1).

A follow-up like "and its harbour?" retrieves nothing on its own. With history
present we spend one small completion turning it into a standalone query; with
no history the function is the identity and costs nothing.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from app.rag.prompts import REWRITE_SYSTEM, REWRITE_USER_TEMPLATE
from app.services.openai_client import AIClient
from app.sessions import Turn

logger = logging.getLogger(__name__)

HISTORY_TURNS = 3
MAX_REWRITE_CHARS = 400


def rewrite_question(
    ai: AIClient,
    question: str,
    history: Sequence[Turn],
    model: Optional[str] = None,
) -> str:
    """Return a standalone version of ``question``.

    Falls back to the original question whenever there is no history, or the
    model returns something empty, oversized, or otherwise implausible — a bad
    rewrite is much worse than no rewrite.
    """
    question = question.strip()
    if not history or not question:
        return question

    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {
            "role": "user",
            "content": REWRITE_USER_TEMPLATE.format(
                history=format_history(history), question=question
            ),
        },
    ]
    try:
        rewritten = ai.chat(messages, purpose="rewrite", temperature=0, model=model)
    except Exception:  # noqa: BLE001 - retrieval still works with the original
        logger.warning("query rewrite failed; using the original question", exc_info=True)
        return question

    cleaned = _clean(rewritten)
    if not cleaned or len(cleaned) > MAX_REWRITE_CHARS:
        return question
    return cleaned


def format_history(history: Sequence[Turn], turns: int = HISTORY_TURNS) -> str:
    """Render the last ``turns`` exchanges for the rewrite prompt."""
    lines: List[str] = []
    for turn in list(history)[-turns:]:
        lines.append("User: {0}".format(turn.question.strip()))
        lines.append("Assistant: {0}".format(turn.answer.strip()))
    return "\n".join(lines)


def _clean(text: str) -> str:
    """Strip the wrapping the model sometimes adds around a rewritten query."""
    cleaned = (text or "").strip()
    if len(cleaned) >= 2 and cleaned[0] in "\"'“”" and cleaned[-1] in "\"'“”":
        cleaned = cleaned[1:-1].strip()
    return cleaned.splitlines()[0].strip() if cleaned else ""
