"""Dense embedding helper via FastEmbed (ONNX, CPU-friendly).

A small multilingual model (default
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384-dim) runs
locally through FastEmbed's ONNX runtime. It is ~20× lighter and faster than a
large bi-encoder on CPU (~20 ms/query vs several seconds) while still covering
the multilingual NCDC corpus — the key lever for a low-cost, low-latency box.

The model is loaded lazily on first use and cached process-wide. Query embeddings
are additionally cached in Redis (they are deterministic).
"""
from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.services import cache


@lru_cache(maxsize=1)
def _dense_model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed documents/passages. Callers should pre-batch large lists."""
    if not texts:
        return []
    vectors = _dense_model().embed(texts, batch_size=settings.embed_batch_size)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    cached = cache.get_json("emb", settings.embedding_model, text)
    if cached is not None:
        return cached
    # query_embed applies any query-side preprocessing the model expects.
    vector = next(iter(_dense_model().query_embed(text))).tolist()
    cache.set_json("emb", vector, settings.embedding_cache_ttl, settings.embedding_model, text)
    return vector


def warmup() -> None:
    """Load + JIT the dense model so the first real query is fast."""
    embed_texts(["warmup"])
