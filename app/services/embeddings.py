"""Dense (BAAI/bge-m3) and sparse (BM25 via FastEmbed) embedding helpers.

* Dense — `BAAI/bge-m3` (1024-dim, multilingual) runs locally via
  sentence-transformers, so there is no per-embedding API cost and the model
  handles the multilingual NCDC corpus well.
* Sparse — BM25 (FastEmbed, ONNX) gives the hybrid search its lexical/exact-match
  arm, critical for medical terms, drug names and dosages that dense vectors blur.

Both models are loaded lazily on first use and cached process-wide. Query
embeddings are additionally cached in Redis (they are deterministic).
"""
from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.services import cache


@lru_cache(maxsize=1)
def _dense_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model, device=settings.embed_device)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed texts with bge-m3. Callers should pre-batch large lists."""
    if not texts:
        return []
    vectors = _dense_model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=settings.embed_batch_size,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    cached = cache.get_json("emb", settings.embedding_model, text)
    if cached is not None:
        return cached
    vector = embed_texts([text])[0]
    cache.set_json("emb", vector, settings.embedding_cache_ttl, settings.embedding_model, text)
    return vector


@lru_cache(maxsize=1)
def _bm25():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name="Qdrant/bm25")


def sparse_embed_documents(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    """Return (indices, values) pairs suitable for a Qdrant SparseVector."""
    out: list[tuple[list[int], list[float]]] = []
    for emb in _bm25().embed(texts):
        out.append((emb.indices.tolist(), emb.values.tolist()))
    return out


def sparse_embed_query(text: str) -> tuple[list[int], list[float]]:
    emb = next(iter(_bm25().query_embed(text)))
    return emb.indices.tolist(), emb.values.tolist()


def warmup() -> None:
    """Load + JIT the dense and sparse models so the first real query is fast."""
    embed_texts(["warmup"])
    if settings.use_hybrid_search:
        sparse_embed_query("warmup")
