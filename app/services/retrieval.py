"""Two-stage retrieval: hybrid recall (child chunks) -> cross-encoder rerank
-> parent expansion.

We search and rerank the small *child* chunks (precise matching), then expand
each surviving hit to its larger *parent* block (richer context for the LLM) and
deduplicate so the LLM never sees the same parent twice. Results are cached in
Redis keyed by the query + category.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

from app.config import settings
from app.services import cache, embeddings, qdrant_service, reranker

logger = logging.getLogger("ncdc.retrieval")


@contextmanager
def _timed(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("  %-14s %6.0f ms", label, (time.perf_counter() - start) * 1000)


def _expand_to_parents(candidates: list[dict]) -> list[dict]:
    """Collapse child hits to unique parents, preserving best-child order/metadata."""
    seen: set[tuple[str, int | None]] = set()
    passages: list[dict] = []
    for c in candidates:
        key = (c.get("document_id"), c.get("parent_ordinal"))
        if key in seen:
            continue
        seen.add(key)
        passages.append(
            {
                "document_id": c.get("document_id"),
                "document_title": c.get("document_title", "Document"),
                "page": c.get("page"),
                "parent_ordinal": c.get("parent_ordinal"),
                "section": c.get("section"),
                "snippet": c.get("text", ""),            # precise child match
                "context": c.get("parent_text") or c.get("text", ""),  # parent for LLM
                "score": c.get("score"),
                "rerank_score": c.get("rerank_score"),
            }
        )
    return passages


def retrieve(query: str, category: str | None = None) -> list[dict]:
    """Return the top reranked, parent-expanded passages for a query."""
    cache_key_parts = (query, category or "")
    cached = cache.get_json("retr", *cache_key_parts)
    if cached is not None:
        logger.info("  retrieval cache HIT")
        return cached

    with _timed("embed dense"):
        dense = embeddings.embed_query(query)
    with _timed("embed sparse"):
        sparse = embeddings.sparse_embed_query(query) if settings.use_hybrid_search else None

    with _timed("qdrant search"):
        candidates = qdrant_service.search(
            query_text=query,
            dense_vector=dense,
            sparse_vector=sparse,
            limit=settings.retrieve_top_k,
            category=category,
        )
    if not candidates:
        return []

    if settings.use_reranker:
        try:
            with _timed(f"rerank ({len(candidates)})"):
                scores = reranker.rerank(query, [c["text"] for c in candidates])
            for cand, score in zip(candidates, scores, strict=True):
                cand["rerank_score"] = float(score)
            candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
            candidates = [
                c for c in candidates if c["rerank_score"] >= settings.min_rerank_score
            ] or candidates
        except Exception:  # noqa: BLE001 - reranker is best-effort, never fail the query
            logger.exception("Reranking failed; falling back to hybrid order")

    # Expand to unique parents, then keep the top-k distinct parent passages.
    passages = _expand_to_parents(candidates)[: settings.rerank_top_k]
    cache.set_json("retr", passages, settings.retrieval_cache_ttl, *cache_key_parts)
    return passages
