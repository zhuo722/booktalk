#!/usr/bin/env bash
#
# Run the whole app for development: the FastAPI backend on :8000 and the Vite
# dev server on :5173, in one terminal, stopped together by Ctrl-C.
#
# Vite proxies /api to the backend (see frontend/vite.config.ts), so the app is
# used at the frontend URL; the backend URL is printed for curl and /docs.
#
# Written for the bash that ships with macOS (3.2) — no `wait -n`, no
# associative arrays, no `${var,,}`. Do not "modernize" it: /bin/bash here is
# still 3.2, and the point of this script is that it runs on a clean machine.

set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
PYTHON="$BACKEND/.venv/bin/python"
VITE="$FRONTEND/node_modules/.bin/vite"
BACKEND_PORT=8000
FRONTEND_PORT=5173
BACKEND_URL="http://127.0.0.1:$BACKEND_PORT"
FRONTEND_URL="http://localhost:$FRONTEND_PORT"

backend_pid=""
frontend_pid=""

die() {
    printf 'dev.sh: %s\n' "$1" >&2
    exit 1
}

# Both children are killed by pid rather than by port: this script may not be
# the only thing on the machine, and a stray `lsof | kill` is how you lose an
# unrelated server. `exec` in each subshell below makes $! the real process, so
# the pid we hold is the one that owns the port.
cleanup() {
    status=$?
    trap - EXIT INT TERM
    for pid in $backend_pid $frontend_pid; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in $backend_pid $frontend_pid; do
        wait "$pid" 2>/dev/null || true
    done
    exit "$status"
}

[ -x "$PYTHON" ] || die "no backend virtualenv at $PYTHON
  create it with:  cd backend && python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'"
[ -x "$VITE" ] || die "frontend dependencies are not installed
  install them with:  cd frontend && npm install"

trap cleanup EXIT INT TERM

# --reload is the whole reason to run the backend this way; uvicorn's reloader
# forwards the signal to its worker, so killing this pid stops both.
( cd "$BACKEND" && exec "$PYTHON" -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$BACKEND_PORT" --reload ) &
backend_pid=$!

# --strictPort rather than Vite's default hunt for a free port: the port is
# printed below and baked into the backend's CORS origins, so a silent move to
# 5174 would produce a page that loads and then cannot talk to anything.
( cd "$FRONTEND" && exec "$VITE" --port "$FRONTEND_PORT" --strictPort ) &
frontend_pid=$!

printf '\n  BookTalk\n  app      %s\n  api      %s\n\n  Ctrl-C stops both.\n\n' \
    "$FRONTEND_URL" "$BACKEND_URL"

# bash 3.2 has no `wait -n`, so the two children are watched by polling: the
# moment either one exits, the trap takes the other down with it.
while kill -0 "$backend_pid" 2>/dev/null && kill -0 "$frontend_pid" 2>/dev/null; do
    sleep 1
done
