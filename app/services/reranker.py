"""Cross-encoder reranking to sharpen retrieval precision.

A dense bi-encoder is fast but coarse: it embeds query and passage independently.
A cross-encoder scores each (query, passage) pair jointly and orders results far
more accurately. We retrieve a wide candidate set with dense search, then rerank
and keep only the best few.

Runs on CPU via FastEmbed ONNX (~tens of ms for ~30 candidates), so it barely
moves the latency budget — which is dominated by LLM generation. The model loads
lazily and is reused; its call is serialised with a lock for thread safety.
"""
from __future__ import annotations

import logging
import threading
from functools import lru_cache

from app.config import settings

logger = logging.getLogger("ncdc.reranker")

_lock = threading.Lock()


@lru_cache(maxsize=1)
def _model():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    # Pass cache_dir explicitly so the model always resolves to our configured,
    # persistent path — not FastEmbed's OS-temp default (which Windows wipes).
    return TextCrossEncoder(
        model_name=settings.reranker_model,
        cache_dir=settings.fastembed_cache_path,
    )


def warmup() -> None:
    """Pre-load the reranker off the request path. Best-effort."""
    if settings.reranker_enabled:
        with _lock:
            list(_model().rerank("warmup", ["warmup passage"]))


def _passage_text(p: dict) -> str:
    return p.get("context") or p.get("text") or p.get("snippet") or ""


def rerank(query: str, passages: list[dict], top_k: int) -> list[dict]:
    """Return the top_k passages reordered by cross-encoder relevance.

    Falls back to the incoming (dense) order on any failure or when disabled.
    """
    if not passages:
        return []
    if not settings.reranker_enabled:
        return passages[:top_k]
    try:
        with _lock:
            scores = list(_model().rerank(query, [_passage_text(p) for p in passages]))
    except Exception:  # noqa: BLE001 - never let reranking break a query
        logger.exception("Rerank failed; falling back to dense order")
        return passages[:top_k]
    for p, s in zip(passages, scores, strict=False):
        p["rerank_score"] = float(s)
    ranked = sorted(passages, key=lambda p: p.get("rerank_score", float("-inf")), reverse=True)
    return ranked[:top_k]
