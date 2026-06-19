# syntax=docker/dockerfile:1
# ─── NCDC Chatbot backend (FastAPI + uv) ──────────────────────────────────
# Build:  docker build -t ncdc-backend ./Backend
# Run:    docker run --env-file Backend/.env -p 8000:8000 ncdc-backend
FROM python:3.13-slim AS base

# uv: fast, reproducible installs straight from uv.lock.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    # Persist the FastEmbed ONNX model here so it survives restarts (mount a volume).
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

# libgomp1 is required at runtime by onnxruntime (pulled in by fastembed).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Install dependencies only — cached unless the lockfiles change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# 2) Copy the source and install the project itself.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Persisted at runtime via volumes: uploaded PDFs + the embedding model cache.
RUN mkdir -p /app/storage/documents /app/.cache/fastembed

EXPOSE 8000

# Production server: no --reload. WEB_CONCURRENCY controls worker count.
# Note: each worker runs startup (create_all/seed_admin are idempotent) and warms
# its own copy of the embedding model, so keep the count modest for memory.
ENV WEB_CONCURRENCY=2
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY}"]
