"""Knowledge Base Management: upload, manage, version, enable/disable documents."""
from __future__ import annotations

import hashlib
import os
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.deps import get_current_user, require_admin
from app.models import AuditLog, Document, DocumentStatus, User
from app.schemas import DocumentOut, DocumentPage, DocumentUpdate
from app.services import cache, ingestion, pdf_processor, qdrant_service

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES = {"application/pdf"}


def _audit(db: Session, user_id: str | None, action: str, target: str, detail: str | None = None):
    db.add(AuditLog(user_id=user_id, action=action, target=target, detail=detail))


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    replaces_id: str | None = Form(None),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Only PDF documents are supported.")

    title = title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="A title is required for every document.")

    os.makedirs(settings.upload_dir, exist_ok=True)
    doc_id = uuid.uuid4().hex
    safe_name = f"{doc_id}.pdf"
    dest = os.path.join(settings.upload_dir, safe_name)

    # Stream to disk in chunks; enforce size limit without loading file in memory.
    hasher = hashlib.sha256()
    size = 0
    max_bytes = settings.max_upload_mb * 1024 * 1024
    with open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                out.close()
                os.remove(dest)
                raise HTTPException(status_code=413, detail="File exceeds maximum upload size.")
            hasher.update(chunk)
            out.write(chunk)

    doc = Document(
        id=doc_id,
        title=title,
        original_filename=file.filename or safe_name,
        file_path=dest,
        checksum=hasher.hexdigest(),
        file_size=size,
        status=DocumentStatus.pending,
        uploaded_by=admin.id,
        replaces_id=replaces_id,
    )

    # If replacing an older document, bump version and disable the predecessor.
    if replaces_id:
        old = db.get(Document, replaces_id)
        if old:
            doc.version = old.version + 1
            old.enabled = False
            old.status = DocumentStatus.disabled
            qdrant_service.set_document_enabled(old.id, False)

    db.add(doc)
    _audit(db, admin.id, "document.upload", doc.id, doc.title)
    db.commit()
    db.refresh(doc)

    background.add_task(ingestion.process_document, doc.id)
    return doc


@router.get("", response_model=list[DocumentOut])
def list_documents(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
    status_filter: DocumentStatus | None = None,
):
    q = db.query(Document)
    if status_filter:
        q = q.filter(Document.status == status_filter)
    return q.order_by(Document.created_at.desc()).all()


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, db: Session = Depends(get_db), _: User = Depends(require_admin)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.patch("/{document_id}", response_model=DocumentOut)
def update_document(
    document_id: str,
    payload: DocumentUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if payload.title is not None:
        doc.title = payload.title
    if payload.enabled is not None and payload.enabled != doc.enabled:
        doc.enabled = payload.enabled
        # Toggle visibility in Qdrant without re-embedding.
        qdrant_service.set_document_enabled(doc.id, payload.enabled)
        if doc.status in (DocumentStatus.indexed, DocumentStatus.disabled):
            doc.status = DocumentStatus.indexed if payload.enabled else DocumentStatus.disabled

    _audit(db, admin.id, "document.update", doc.id, doc.title)
    db.commit()
    db.refresh(doc)
    return doc


@router.post("/{document_id}/reindex", response_model=DocumentOut)
def reindex_document(
    document_id: str,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    qdrant_service.delete_document(doc.id)
    doc.status = DocumentStatus.pending
    doc.chunk_count = 0
    doc.progress = 0.0
    _audit(db, admin.id, "document.reindex", doc.id, doc.title)
    db.commit()
    background.add_task(ingestion.process_document, doc.id)
    db.refresh(doc)
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str, db: Session = Depends(get_db), admin: User = Depends(require_admin)
):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    qdrant_service.delete_document(doc.id)
    if doc.file_path and os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except OSError:
            pass
    _audit(db, admin.id, "document.delete", doc.id, doc.title)
    db.delete(doc)
    db.commit()


@router.get("/{document_id}/page/{page}", response_model=DocumentPage)
def get_page_text(
    document_id: str,
    page: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Full extracted text of one page, for the in-app source viewer (any signed-in
    user). The citation snippet is a substring of this text, so the frontend can
    highlight the relevant lines within the complete page. Cached per page."""
    doc = db.get(Document, document_id)
    if not doc or not doc.file_path or not os.path.exists(doc.file_path):
        raise HTTPException(status_code=404, detail="Document not found")

    cached = cache.get_json("page", document_id, str(page))
    if cached is not None:
        return DocumentPage(**cached)

    text, section = pdf_processor.page_text(doc.file_path, page)
    result = DocumentPage(
        document_id=document_id,
        document_title=doc.title,
        page=page,
        section=section,
        text=text,
    )
    cache.set_json("page", result.model_dump(), settings.retrieval_cache_ttl, document_id, str(page))
    return result


@router.get("/{document_id}/view")
def view_document(document_id: str, db: Session = Depends(get_db)):
    """Serve the source PDF inline so citation hyperlinks (#page=N) work in-browser.

    Public (no auth) so citation links open directly; tighten if documents are sensitive.
    """
    doc = db.get(Document, document_id)
    if not doc or not os.path.exists(doc.file_path):
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(
        doc.file_path,
        media_type="application/pdf",
        filename=doc.original_filename,
        content_disposition_type="inline",
    )
