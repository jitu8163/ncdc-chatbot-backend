"""Cross-encoder reranking with BAAI/bge-reranker-v2-m3 (multilingual).

A cross-encoder scores each (query, passage) pair jointly, which is far more
precise than the bi-encoder vector similarity used for first-stage retrieval.
We over-fetch with hybrid search, then rerank down to the few passages actually
fed to the LLM — this is the single biggest lever on citation correctness.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model, device=settings.embed_device)


def rerank(query: str, passages: list[str]) -> list[float]:
    """Return a relevance score per passage (higher = more relevant)."""
    if not passages:
        return []
    scores = _model().predict(
        [(query, p) for p in passages],
        batch_size=settings.embed_batch_size,
        show_progress_bar=False,
    )
    return [float(s) for s in scores]


def warmup() -> None:
    """Load + JIT the cross-encoder so the first real query is fast."""
    if settings.use_reranker:
        rerank("warmup", ["warmup passage"])
