#!/bin/sh
# Seed an empty data directory with the pre-indexed example library, then serve.
set -e
DATA="${BOOKTALK_DATA_DIR:-/data}"
mkdir -p "$DATA/books"
if [ ! -f "$DATA/booktalk.db" ] && [ -f /app/seed/booktalk.db ]; then
  cp /app/seed/booktalk.db "$DATA/booktalk.db"
  cp /app/seed/books/*.pdf "$DATA/books/" 2>/dev/null || true
  echo "seeded $DATA with the example library"
fi
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-7860}" --proxy-headers --forwarded-allow-ips='*'
