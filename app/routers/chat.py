"""Chat / AI Question-Answering with citations, follow-ups and feedback.

Public endpoints (citizens + healthcare workers) — no login required, in line
with the SOW exclusion of a user-registration system. Conversation state is
keyed by an anonymous `session_id`.
"""
from __future__ import annotations

import json
import re
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.graph import run_chat
from app.models import ChatMessage, ChatSession, Feedback, QueryLog
from app.schemas import ChatRequest, ChatResponse, Citation, FeedbackIn
from app.services import cache

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())[:512]


@router.post("", response_model=ChatResponse)
def ask(payload: ChatRequest, request: Request, db: Session = Depends(get_db)):
    started = time.perf_counter()

    # 0) Rate limit per client IP (Redis-backed, no-op if Redis is down).
    client_ip = request.client.host if request.client else "anonymous"
    if cache.rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please slow down and try again shortly.",
        )

    # 1) Resolve / create the session.
    session = db.get(ChatSession, payload.session_id) if payload.session_id else None
    if session is None:
        session = ChatSession(language=payload.language)
        db.add(session)
        db.flush()

    history = [{"role": m.role, "content": m.content} for m in session.messages]

    # 2) Persist the user turn.
    user_msg = ChatMessage(session_id=session.id, role="user", content=payload.question)
    db.add(user_msg)

    # 3) Run the LangGraph pipeline (classify -> rewrite -> retrieve -> generate
    #    -> citations -> format).
    final = run_chat(
        question=payload.question,
        history=history,
        language=payload.language,
    )
    citations = [Citation(**c) for c in final.get("citations", [])]

    # 4) Persist the assistant turn.
    assistant_msg = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=final["answer"],
        answered=final["answered"],
        citations_json=json.dumps([c.model_dump() for c in citations]),
        followups_json=json.dumps(final.get("followups", [])),
    )
    db.add(assistant_msg)
    db.flush()

    latency_ms = int((time.perf_counter() - started) * 1000)

    # 5) Analytics log.
    db.add(
        QueryLog(
            session_id=session.id,
            message_id=assistant_msg.id,
            question=payload.question,
            normalized_question=_normalize(payload.question),
            language=payload.language,
            answered=final["answered"],
            latency_ms=latency_ms,
            cited_document_ids=",".join({c.document_id for c in citations}) or None,
        )
    )
    db.commit()

    return ChatResponse(
        session_id=session.id,
        message_id=assistant_msg.id,
        answer=final["answer"],
        answered=final["answered"],
        citations=citations,
        followups=final.get("followups", []),
        latency_ms=latency_ms,
    )


@router.post("/sessions", response_model=dict)
def new_session(db: Session = Depends(get_db)):
    session = ChatSession()
    db.add(session)
    db.commit()
    return {"session_id": session.id}


@router.get("/sessions/{session_id}", response_model=dict)
def get_session(session_id: str, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = []
    for m in session.messages:
        messages.append(
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "answered": m.answered,
                "citations": json.loads(m.citations_json) if m.citations_json else [],
                "followups": json.loads(m.followups_json) if m.followups_json else [],
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
        )
    return {"session_id": session.id, "messages": messages}


@router.post("/feedback", status_code=status.HTTP_201_CREATED)
def submit_feedback(payload: FeedbackIn, db: Session = Depends(get_db)):
    msg = db.get(ChatMessage, payload.message_id)
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")
    db.add(Feedback(message_id=payload.message_id, rating=payload.rating, comment=payload.comment))
    db.commit()
    return {"status": "ok"}
