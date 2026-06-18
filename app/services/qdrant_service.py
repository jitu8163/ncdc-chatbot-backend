"""Qdrant vector store with plain dense vector search (MVP — no BM25/hybrid).

Scale notes (1000-page docs × 100 uploads ≈ hundreds of thousands of points):
  * A single collection with payload indexes on `document_id`, `enabled`, `category`
    keeps filtered search fast.
  * Enable/disable and delete are O(filter) payload/point operations — we never
    re-embed to toggle a document's visibility.
  * Points are upserted in batches by the ingestion pipeline.
"""
from __future__ import annotations

import logging
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient, models

from app.config import settings
from app.services.chunking import Chunk

logger = logging.getLogger("ncdc.qdrant")

DENSE = "dense"


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=settings.qdrant_timeout,
    )


def ensure_collection() -> None:
    client = get_client()
    if client.collection_exists(settings.qdrant_collection):
        # If the embedding model (and thus vector size) changed, the stored vectors
        # are no longer comparable — drop and recreate so re-indexing rebuilds them.
        info = client.get_collection(settings.qdrant_collection)
        current = info.config.params.vectors[DENSE].size
        if current == settings.embedding_dim:
            return
        logger.warning(
            "Dense vector size changed (%s -> %s); recreating collection %r. "
            "All documents must be re-indexed.",
            current, settings.embedding_dim, settings.qdrant_collection,
        )
        client.delete_collection(settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            DENSE: models.VectorParams(
                size=settings.embedding_dim, distance=models.Distance.COSINE
            )
        },
        # Keep raw vectors on disk; HNSW graph + payload index stay in RAM.
        on_disk_payload=True,
    )
    for field in ("document_id", "category"):
        client.create_payload_index(
            settings.qdrant_collection, field, models.PayloadSchemaType.KEYWORD
        )
    client.create_payload_index(
        settings.qdrant_collection, "enabled", models.PayloadSchemaType.BOOL
    )


def _point_id(document_id: str, ordinal: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{ordinal}"))


def upsert_chunks(
    document_id: str,
    document_title: str,
    category: str | None,
    enabled: bool,
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
) -> None:
    points: list[models.PointStruct] = []
    for chunk, dense in zip(chunks, dense_vectors, strict=True):
        points.append(
            models.PointStruct(
                id=_point_id(document_id, chunk.ordinal),
                vector={DENSE: dense},
                payload={
                    "document_id": document_id,
                    "document_title": document_title,
                    "category": category,
                    "enabled": enabled,
                    "page": chunk.page,
                    "section": chunk.section,
                    "ordinal": chunk.ordinal,
                    "text": chunk.text,   # matched + fed to the LLM
                },
            )
        )
    get_client().upsert(settings.qdrant_collection, points=points, wait=True)


def delete_document(document_id: str) -> None:
    get_client().delete(
        settings.qdrant_collection,
        points_selector=models.FilterSelector(filter=_doc_filter(document_id)),
        wait=True,
    )


def set_document_enabled(document_id: str, enabled: bool) -> None:
    get_client().set_payload(
        settings.qdrant_collection,
        payload={"enabled": enabled},
        points=models.FilterSelector(filter=_doc_filter(document_id)),
        wait=True,
    )


def _doc_filter(document_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
    )


def _search_filter(category: str | None) -> models.Filter:
    must = [models.FieldCondition(key="enabled", match=models.MatchValue(value=True))]
    if category:
        must.append(
            models.FieldCondition(key="category", match=models.MatchValue(value=category))
        )
    return models.Filter(must=must)


def search(
    dense_vector: list[float],
    limit: int,
    category: str | None = None,
) -> list[dict]:
    """Plain dense vector search.

    Returns a list of payload dicts augmented with the retrieval `score`.
    """
    client = get_client()
    flt = _search_filter(category)

    result = client.query_points(
        collection_name=settings.qdrant_collection,
        query=dense_vector,
        using=DENSE,
        query_filter=flt,
        limit=limit,
        with_payload=True,
    )

    hits: list[dict] = []
    for point in result.points:
        payload = dict(point.payload or {})
        payload["score"] = point.score
        hits.append(payload)
    return hits
