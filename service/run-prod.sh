#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

exec "$APP_DIR/.venv/bin/uvicorn" app.main:app \
  --host 0.0.0.0 \
  --port 8088 \
  --workers 2

