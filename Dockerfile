# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

LABEL maintainer="Simon Waloschek <waloschek@pm.me>"

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/model.optimized.onnx \
    WORKERS=1 \
    ORT_INTRA_OP_NUM_THREADS=0 \
    ORT_INTER_OP_NUM_THREADS=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
COPY models/ models/
COPY entrypoint.sh entrypoint.sh

RUN pip install --upgrade pip && \
    pip install . && \
    chmod +x entrypoint.sh

EXPOSE 8123

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://127.0.0.1:8123/health || exit 1

CMD ["/app/entrypoint.sh"]
