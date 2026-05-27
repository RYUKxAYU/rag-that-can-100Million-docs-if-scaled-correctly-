#!/usr/bin/env sh
set -eu

export APP_ENV="${APP_ENV:-production}"
export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${APP_PORT:-8000}"
export OFFLINE_MODE="${OFFLINE_MODE:-true}"
export LOCAL_DATA_PATH="${LOCAL_DATA_PATH:-/app/data}"
export BACKUP_PATH="${BACKUP_PATH:-/app/backups}"
export LOG_PATH="${LOG_PATH:-/app/logs}"

mkdir -p "$LOCAL_DATA_PATH" "$BACKUP_PATH" "$LOG_PATH"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU runtime detected:"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
  echo "Warning: NVIDIA runtime not detected. Host GPU support may be unavailable."
fi

exec /app/scripts/entrypoint.sh
