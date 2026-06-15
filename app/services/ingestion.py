"""Document ingestion pipeline: extract -> chunk -> embed (batched) -> index.

Runs in a background thread (FastAPI BackgroundTasks). Status is persisted on
the Document row so the admin UI can poll progress, and so a server restart
leaves a clear `processing`/`failed` trail rather than a silent hang.

For very large corpora this same `process_document` function can be driven by a
Celery/RQ worker instead — it only needs a document id and a DB session factory.
"""
from __future__ import annotations

import logging
from datetime import datetime

from app.config import settings
from app.database import SessionLocal
from app.models import Document, DocumentStatus
from app.services import embeddings, qdrant_service
from app.services.chunking import chunk_document

logger = logging.getLogger(__name__)


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def process_document(document_id: str) -> None:
    """End-to-end indexing for one document. Safe to call from a worker thread."""
    db = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        if doc is None:
            logger.warning("process_document: %s not found", document_id)
            return

        doc.status = DocumentStatus.processing
        doc.error = None
        db.commit()

        qdrant_service.ensure_collection()

        # 1) Extract + chunk (streams pages internally).
        chunks, pages = chunk_document(doc.file_path)
        if not chunks:
            raise ValueError("No extractable text found in document (scanned PDF?).")

        # 2) Embed + index in batches to bound memory and API payload size.
        total = 0
        for batch in _batched(chunks, settings.embed_batch_size):
            texts = [c.text for c in batch]
            dense = embeddings.embed_texts(texts)
            sparse = (
                embeddings.sparse_embed_documents(texts)
                if settings.use_hybrid_search
                else [([], []) for _ in texts]
            )
            qdrant_service.upsert_chunks(
                document_id=doc.id,
                document_title=doc.title,
                category=doc.category,
                enabled=doc.enabled,
                chunks=batch,
                dense_vectors=dense,
                sparse_vectors=sparse,
            )
            total += len(batch)
            logger.info("Indexed %s/%s chunks for %s", total, len(chunks), doc.id)

        doc.page_count = pages
        doc.chunk_count = total
        doc.status = DocumentStatus.indexed if doc.enabled else DocumentStatus.disabled
        doc.indexed_at = datetime.utcnow()
        db.commit()
        logger.info("Document %s indexed (%s pages, %s chunks)", doc.id, pages, total)

    except Exception as exc:  # noqa: BLE001 - record failure for the admin UI
        logger.exception("Ingestion failed for %s", document_id)
        db.rollback()
        doc = db.get(Document, document_id)
        if doc is not None:
            doc.status = DocumentStatus.failed
            doc.error = str(exc)[:2000]
            db.commit()
    finally:
        db.close()
