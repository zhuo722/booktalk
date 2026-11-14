"""Public-demo guard rails (``BOOKTALK_DEMO=1``).

A hosted demo spends the owner's API credit on every visitor's question, so in
demo mode the endpoints that cost money are rate limited per client IP and
capped globally per day, and the books present at boot (the seeded example
library) cannot be deleted. Everything is in-process memory: one container,
one worker, and a restart simply resets the counters.
"""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, FrozenSet, Optional, Pattern, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

HOUR = 3600.0
DAY = 86400.0


@dataclass(frozen=True)
class Rule:
    method: str
    path: Pattern[str]
    per_ip: int
    per_ip_window_s: float
    per_day: int
    what: str


RULES: Tuple[Rule, ...] = (
    Rule("POST", re.compile(r"^/api/books/?$"), 3, HOUR, 20, "uploads"),
    Rule("POST", re.compile(r"^/api/chat/?$"), 30, HOUR, 600, "questions"),
    Rule("POST", re.compile(r"^/api/asr/?$"), 40, HOUR, 800, "voice inputs"),
)

DELETE_BOOK = re.compile(r"^/api/books/([0-9a-f]+)/?$")
PROTECTED_DETAIL = "This example book is part of the demo library and can't be deleted."


class RateLimiter:
    """Sliding-window counters keyed by (rule, client) plus a daily total."""

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)

    def check(self, rule: Rule, client: str) -> Optional[str]:
        """Record a hit, or return why it is refused (and record nothing)."""
        now = self._clock()
        with self._lock:
            mine = self._hits[(rule.what, client)]
            total = self._hits[(rule.what, "*")]
            _expire(mine, now - rule.per_ip_window_s)
            _expire(total, now - DAY)
            if len(total) >= rule.per_day:
                return (
                    "The public demo has reached today's limit for {0}. "
                    "Please try again tomorrow, or run BookTalk locally."
                ).format(rule.what)
            if len(mine) >= rule.per_ip:
                wait_min = max(1, int((mine[0] + rule.per_ip_window_s - now) / 60) + 1)
                return (
                    "Demo limit: {0} {1} per hour. Please try again in about {2} min."
                ).format(rule.per_ip, rule.what, wait_min)
            mine.append(now)
            total.append(now)
            return None


def _expire(hits: Deque[float], cutoff: float) -> None:
    while hits and hits[0] <= cutoff:
        hits.popleft()


def client_ip(request: Request) -> str:
    """The visitor's address; behind the host's proxy that is the first XFF hop."""
    forwarded = request.headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    if first:
        return first
    return request.client.host if request.client else "unknown"


def install(app: FastAPI, protected: FrozenSet[str]) -> None:
    """Attach the demo middleware; ``protected`` are book ids nobody may delete."""
    limiter = RateLimiter()
    app.state.demo_protected = protected

    @app.middleware("http")
    async def demo_guard(request: Request, call_next):  # type: ignore[no-untyped-def]
        method = request.method.upper()
        path = request.url.path
        if method == "DELETE":
            match = DELETE_BOOK.match(path)
            if match and match.group(1) in protected:
                return JSONResponse({"detail": PROTECTED_DETAIL}, status_code=403)
        for rule in RULES:
            if method == rule.method and rule.path.match(path):
                refused = limiter.check(rule, client_ip(request))
                if refused:
                    return JSONResponse({"detail": refused}, status_code=429)
                break
        return await call_next(request)
