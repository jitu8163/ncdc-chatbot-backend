"""Qdrant vector store with native hybrid search (dense + BM25 sparse, RRF fusion).

Scale notes (1000-page docs × 100 uploads ≈ hundreds of thousands of points):
  * A single collection with payload indexes on `document_id`, `enabled`, `category`
    keeps filtered search fast.
  * Enable/disable and delete are O(filter) payload/point operations — we never
    re-embed to toggle a document's visibility.
  * Points are upserted in batches by the ingestion pipeline.
"""
from __future__ import annotations

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient, models

from app.config import settings
from app.services.chunking import Chunk

DENSE = "dense"
SPARSE = "bm25"


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


def ensure_collection() -> None:
    client = get_client()
    if client.collection_exists(settings.qdrant_collection):
        return
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            DENSE: models.VectorParams(
                size=settings.embedding_dim, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
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
    sparse_vectors: list[tuple[list[int], list[float]]],
) -> None:
    points: list[models.PointStruct] = []
    for chunk, dense, (idx, val) in zip(chunks, dense_vectors, sparse_vectors, strict=True):
        points.append(
            models.PointStruct(
                id=_point_id(document_id, chunk.ordinal),
                vector={
                    DENSE: dense,
                    SPARSE: models.SparseVector(indices=idx, values=val),
                },
                payload={
                    "document_id": document_id,
                    "document_title": document_title,
                    "category": category,
                    "enabled": enabled,
                    "page": chunk.page,
                    "section": chunk.section,
                    "ordinal": chunk.ordinal,
                    "parent_ordinal": chunk.parent_ordinal,
                    "text": chunk.text,                 # child (matched) text
                    "parent_text": chunk.parent_text,   # parent context for the LLM
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
    query_text: str,
    dense_vector: list[float],
    sparse_vector: tuple[list[int], list[float]] | None,
    limit: int,
    category: str | None = None,
) -> list[dict]:
    """Hybrid (RRF) when a sparse vector is supplied, else dense-only.

    Returns a list of payload dicts augmented with the retrieval `score`.
    """
    client = get_client()
    flt = _search_filter(category)

    if sparse_vector is not None:
        idx, val = sparse_vector
        result = client.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector, using=DENSE, filter=flt, limit=limit
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=idx, values=val),
                    using=SPARSE,
                    filter=flt,
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
    else:
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
