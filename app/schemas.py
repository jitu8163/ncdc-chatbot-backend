"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models import DocumentStatus, UserRole


# ─── Auth ──────────────────────────────────────────────────────────────
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str | None = None
    role: UserRole = UserRole.user


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    email: str  # plain str on output (avoids rejecting reserved TLDs like .local)
    full_name: str | None
    role: UserRole
    is_active: bool


# ─── Documents ─────────────────────────────────────────────────────────
class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    original_filename: str
    version: int
    status: DocumentStatus
    enabled: bool
    page_count: int
    chunk_count: int
    progress: float
    file_size: int
    error: str | None
    created_at: datetime
    indexed_at: datetime | None


class DocumentUpdate(BaseModel):
    title: str | None = None
    enabled: bool | None = None


class DocumentPage(BaseModel):
    """Full extracted text of a single page, for the in-app source viewer."""
    document_id: str
    document_title: str
    page: int | None
    section: str | None
    text: str


# ─── Chat ──────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None
    language: str | None = None  # optional hint; otherwise auto-detected by the LLM


class Citation(BaseModel):
    document_id: str
    document_title: str
    page: int | None = None
    section: str | None = None
    snippet: str | None = None
    url: str | None = None  # clickable deep link to the source page


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    title: str | None = None
    answer: str
    answered: bool
    citations: list[Citation] = []
    latency_ms: int = 0


class ChatSessionSummary(BaseModel):
    """One conversation in a user's history list."""
    id: str
    title: str | None
    created_at: str | None


class FeedbackIn(BaseModel):
    message_id: str
    rating: int = Field(ge=-1, le=1)
    comment: str | None = None


class FeedbackEntry(BaseModel):
    """One feedback record paired with the question and answer it refers to."""
    id: str
    rating: int
    question: str | None
    answer: str
    comment: str | None
    created_at: str | None


# ─── Analytics ─────────────────────────────────────────────────────────
class FAQItem(BaseModel):
    question: str
    count: int


class DocUsageItem(BaseModel):
    document_id: str
    title: str
    count: int


class DailyUsageItem(BaseModel):
    date: str
    count: int


class AnalyticsSummary(BaseModel):
    total_questions: int
    answered_questions: int
    unanswered_questions: int
    avg_latency_ms: float
    daily_usage: list[DailyUsageItem]
    frequently_asked: list[FAQItem]
    most_accessed_documents: list[DocUsageItem]
    feedback_positive: int
    feedback_negative: int
