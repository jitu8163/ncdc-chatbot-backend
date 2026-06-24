"""Two-stage retrieval: dense vector search -> cross-encoder rerank.

We embed the query, pull a wide candidate set from Qdrant (retrieve_top_k),
de-duplicate it, then a cross-encoder reranks and keeps the best final_top_k
passages for the LLM. Results are cached in Redis keyed by the query + category.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
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


def _dedupe(passages: list[dict]) -> list[dict]:
    """Drop passages with identical text (duplicate uploads / overlapping chunks),
    keeping the first (highest dense score) occurrence."""
    seen: set[str] = set()
    out: list[dict] = []
    for p in passages:
        key = (p.get("snippet") or "").strip()
        if key and key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def retrieve_staged(query: str) -> Iterator[dict]:
    """Run retrieval as a generator, surfacing each stage as it happens.

    Yields ``{"stage": <name>}`` progress events ("embedding" -> "retrieving" ->
    "reranking") as the work proceeds, then exactly one terminal
    ``{"result": [...passages...]}``. The streaming chat endpoint relays the stage
    events to the UI so the user sees what the pipeline is doing; a cache hit skips
    straight to the result.
    """
    cache_key_parts = (query,)
    cached = cache.get_json("retr", *cache_key_parts)
    if cached is not None:
        logger.info("  retrieval cache HIT")
        yield {"result": cached}
        return

    yield {"stage": "embedding"}
    with _timed("embed dense"):
        dense = embeddings.embed_query(query)

    yield {"stage": "retrieving"}
    with _timed("qdrant search"):
        candidates = qdrant_service.search(
            dense_vector=dense,
            limit=settings.retrieve_top_k,
        )
    if not candidates:
        yield {"result": []}
        return

    passages = _dedupe(_to_passages(candidates))

    yield {"stage": "reranking"}
    with _timed("rerank"):
        passages = reranker.rerank(query, passages, settings.final_top_k)

    cache.set_json("retr", passages, settings.retrieval_cache_ttl, *cache_key_parts)
    yield {"result": passages}


def retrieve(query: str) -> list[dict]:
    """Return the best passages for a query (dense search + cross-encoder rerank)."""
    passages: list[dict] = []
    for event in retrieve_staged(query):
        if "result" in event:
            passages = event["result"]
    return passages
