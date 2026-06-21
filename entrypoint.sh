#!/usr/bin/env sh
set -e

WORKERS=${WORKERS:-1}

exec gunicorn \
  measure_detector_v2.server:app \
  --workers "$WORKERS" \
  --worker-class uvicorn.workers.UvicornWorker \
  --preload \
  --max-requests 1000 \
  --max-requests-jitter 100 \
  --graceful-timeout 30 \
  --keep-alive 5 \
  --timeout 120 \
  --bind 0.0.0.0:8123
