"""Single-stage dense retrieval (MVP).

We embed the query with the dense model and run a plain vector search in Qdrant,
then return the top chunks directly to the LLM. No BM25 hybrid arm, no reranker
and no parent expansion — kept deliberately simple for the POC. Results are cached
in Redis keyed by the query + category.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

from app.config import settings
from app.services import cache, embeddings, qdrant_service

logger = logging.getLogger("ncdc.retrieval")


@contextmanager
def _timed(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("  %-14s %6.0f ms", label, (time.perf_counter() - start) * 1000)


def _to_passages(candidates: list[dict]) -> list[dict]:
    """Shape Qdrant hits into the passage dicts the LLM + citation code expect."""
    passages: list[dict] = []
    for c in candidates:
        text = c.get("text", "")
        passages.append(
            {
                "document_id": c.get("document_id"),
                "document_title": c.get("document_title", "Document"),
                "page": c.get("page"),
                "section": c.get("section"),
                "snippet": text,
                "context": text,   # same chunk fed to the LLM (single-level chunking)
                "score": c.get("score"),
            }
        )
    return passages


def retrieve(query: str, category: str | None = None) -> list[dict]:
    """Return the top dense-search passages for a query."""
    cache_key_parts = (query, category or "")
    cached = cache.get_json("retr", *cache_key_parts)
    if cached is not None:
        logger.info("  retrieval cache HIT")
        return cached

    with _timed("embed dense"):
        dense = embeddings.embed_query(query)

    with _timed("qdrant search"):
        candidates = qdrant_service.search(
            dense_vector=dense,
            limit=settings.retrieve_top_k,
            category=category,
        )
    if not candidates:
        return []

    passages = _to_passages(candidates)[: settings.final_top_k]
    cache.set_json("retr", passages, settings.retrieval_cache_ttl, *cache_key_parts)
    return passages
