"""Relational data model (MySQL/SQLite via SQLAlchemy 2.0)."""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


class UserRole(str, enum.Enum):
    admin = "admin"
    user = "user"


class DocumentStatus(str, enum.Enum):
    pending = "pending"        # uploaded, not yet processed
    processing = "processing"  # extraction / embedding in progress
    indexed = "indexed"        # searchable
    failed = "failed"
    disabled = "disabled"      # kept but excluded from retrieval


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.user)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(512), index=True)
    original_filename: Mapped[str] = mapped_column(String(512))
    category: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    # If this document replaces an older one, point back to it for version history.
    replaces_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("documents.id"), nullable=True
    )

    file_path: Mapped[str] = mapped_column(String(1024))
    checksum: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    # Ingestion completion percentage (0–100), updated live during processing so
    # the admin UI can show upload/indexing progress.
    progress: Mapped[float] = mapped_column(Float, default=0.0)

    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus), default=DocumentStatus.pending, index=True
    )
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    uploaded_by: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("users.id"), nullable=True
    )
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    # JSON-encoded list of citations + follow-up suggestions for assistant turns.
    citations_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    followups_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    answered: Mapped[bool] = mapped_column(Boolean, default=True)  # False when no info found
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    message_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("chat_messages.id"), index=True
    )
    rating: Mapped[int] = mapped_column(Integer)  # +1 thumbs up, -1 thumbs down
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class QueryLog(Base):
    """One row per question — powers the analytics dashboard."""

    __tablename__ = "query_logs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    message_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    question: Mapped[str] = mapped_column(Text)
    normalized_question: Mapped[str] = mapped_column(String(512), index=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    answered: Mapped[bool] = mapped_column(Boolean, default=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    # comma-separated document ids cited in the answer (most-accessed-docs stat)
    cited_document_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    action: Mapped[str] = mapped_column(String(128), index=True)
    target: Mapped[str | None] = mapped_column(String(255), nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
