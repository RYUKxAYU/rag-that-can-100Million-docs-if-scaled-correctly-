#!/usr/bin/env sh
set -eu

mkdir -p /app/data /app/logs /app/backups

exec uvicorn app.main:app \
    --host "${APP_HOST:-0.0.0.0}" \
    --port "${APP_PORT:-8000}" \
    --loop uvloop \
    --http httptools
