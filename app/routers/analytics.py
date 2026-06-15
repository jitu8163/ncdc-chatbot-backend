"""Analytics dashboard + audit log endpoints (admin only)."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends

from app.database import get_db
from app.deps import require_admin
from app.models import AuditLog, Document, Feedback, QueryLog, User
from app.schemas import (
    AnalyticsSummary,
    DailyUsageItem,
    DocUsageItem,
    FAQItem,
)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummary)
def summary(days: int = 30, db: Session = Depends(get_db), _: User = Depends(require_admin)):
    since = datetime.utcnow() - timedelta(days=days)
    logs = db.query(QueryLog).filter(QueryLog.created_at >= since).all()

    total = len(logs)
    answered = sum(1 for q in logs if q.answered)
    avg_latency = (sum(q.latency_ms for q in logs) / total) if total else 0.0

    # Daily usage.
    per_day: Counter[str] = Counter()
    faq: Counter[str] = Counter()
    doc_hits: Counter[str] = Counter()
    for q in logs:
        if q.created_at:
            per_day[q.created_at.strftime("%Y-%m-%d")] += 1
        faq[q.normalized_question] += 1
        if q.cited_document_ids:
            for did in q.cited_document_ids.split(","):
                if did:
                    doc_hits[did] += 1

    daily_usage = [DailyUsageItem(date=d, count=c) for d, c in sorted(per_day.items())]
    frequently_asked = [FAQItem(question=q, count=c) for q, c in faq.most_common(10)]

    # Resolve document titles for the most-accessed list.
    most_accessed: list[DocUsageItem] = []
    if doc_hits:
        titles = {
            d.id: d.title
            for d in db.query(Document).filter(Document.id.in_(list(doc_hits))).all()
        }
        for did, count in doc_hits.most_common(10):
            most_accessed.append(
                DocUsageItem(document_id=did, title=titles.get(did, "(deleted)"), count=count)
            )

    pos = (
        db.query(func.count(Feedback.id))
        .filter(Feedback.rating > 0, Feedback.created_at >= since)
        .scalar()
        or 0
    )
    neg = (
        db.query(func.count(Feedback.id))
        .filter(Feedback.rating < 0, Feedback.created_at >= since)
        .scalar()
        or 0
    )

    return AnalyticsSummary(
        total_questions=total,
        answered_questions=answered,
        unanswered_questions=total - answered,
        avg_latency_ms=round(avg_latency, 1),
        daily_usage=daily_usage,
        frequently_asked=frequently_asked,
        most_accessed_documents=most_accessed,
        feedback_positive=int(pos),
        feedback_negative=int(neg),
    )


@router.get("/audit-logs")
def audit_logs(
    limit: int = 100, db: Session = Depends(get_db), _: User = Depends(require_admin)
):
    rows = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "user_id": r.user_id,
            "action": r.action,
            "target": r.target,
            "detail": r.detail,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
