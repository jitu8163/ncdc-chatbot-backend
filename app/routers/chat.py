"""Chat / AI Question-Answering with citations and feedback.

All endpoints require a logged-in user (citizen or healthcare worker). Only
admins can upload knowledge-base content; any authenticated user can chat.
Conversation state is keyed by `session_id` and tied to the signed-in user.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.deps import get_current_user
from app.graph import build_citations, run_chat
from app.models import ChatMessage, ChatSession, Feedback, QueryLog, User
from app.schemas import ChatRequest, ChatResponse, ChatSessionSummary, Citation, FeedbackIn
from app.services import cache, error_log, llm, retrieval

logger = logging.getLogger("ncdc.chat")

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())[:512]


def _owned_session(db: Session, session_id: str | None, user: User) -> ChatSession | None:
    """Fetch a session the caller is allowed to use, or None for 'start a new one'.

    Returns None when no id is given. Raises 404 if the id is unknown or belongs
    to another user (we 404 rather than 403 so we don't reveal that the id exists).
    A legacy session with no owner is claimed by the current user.
    """
    if not session_id:
        return None
    session = db.get(ChatSession, session_id)
    if session is None or (session.user_id is not None and session.user_id != user.id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    if session.user_id is None:
        session.user_id = user.id
    return session


def _ndjson(event: dict) -> str:
    """Serialise one stream event as a single newline-delimited JSON line."""
    return json.dumps(event, ensure_ascii=False) + "\n"


def _word_chunks(text: str) -> Iterator[str]:
    """Split text into small chunks so canned replies also stream word-by-word."""
    for token in re.findall(r"\S+\s*", text):
        yield token


def _pace_words(text_iter: Iterator[str], delay: float) -> Iterator[str]:
    """Re-chunk a stream of arbitrary text deltas into word-sized pieces and
    sleep `delay` seconds between each, so the answer is visibly typed out in the
    UI instead of arriving at the model's full speed.

    Buffers across deltas so words split across model tokens stay intact. Runs
    inside Starlette's threadpool (sync generator), so the sleep does not block
    the event loop.
    """
    buf = ""
    for piece in text_iter:
        buf += piece
        # Emit every complete "word + trailing whitespace" unit; keep the tail.
        while True:
            m = re.match(r"\S+\s+", buf)
            if not m:
                break
            buf = buf[m.end():]
            if delay:
                time.sleep(delay)
            yield m.group(0)
    if buf:
        if delay:
            time.sleep(delay)
        yield buf


@router.post("", response_model=ChatResponse)
def ask(
    payload: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    started = time.perf_counter()

    # 0) Rate limit per client IP (Redis-backed, no-op if Redis is down).
    client_ip = request.client.host if request.client else "anonymous"
    if cache.rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please slow down and try again shortly.",
        )

    # 1) Resolve / create the session (enforcing per-user ownership).
    session = _owned_session(db, payload.session_id, user)
    is_new = session is None
    if session is None:
        session = ChatSession(language=payload.language, user_id=user.id)
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
    if final.get("error"):
        error_log.log_error(
            payload.question,
            "LLM generation error",
            session_id=session.id,
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
    )
    db.add(assistant_msg)
    db.flush()

    # Name a brand-new conversation from its first message.
    if is_new and not session.title:
        session.title = llm.generate_title(payload.question)

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
        title=session.title,
        answer=final["answer"],
        answered=final["answered"],
        citations=citations,
        latency_ms=latency_ms,
    )


@router.post("/stream")
def ask_stream(
    payload: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Same pipeline as `ask`, but streams the answer to the client as it is
    generated (newline-delimited JSON events: meta -> delta* -> done).

    The conversation is always persisted server-side once generation finishes,
    even if the client navigates away mid-stream, so it still appears in history.
    """
    started = time.perf_counter()

    client_ip = request.client.host if request.client else "anonymous"
    if cache.rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please slow down and try again shortly.",
        )

    session = _owned_session(db, payload.session_id, user)
    is_new = session is None
    if session is None:
        session = ChatSession(language=payload.language, user_id=user.id)
        db.add(session)
        db.flush()
    session_id = session.id
    # Build history before recording the current turn (the question is passed separately).
    history = [{"role": m.role, "content": m.content} for m in session.messages]
    db.add(ChatMessage(session_id=session_id, role="user", content=payload.question))

    delay = settings.stream_word_delay

    def event_stream() -> Iterator[str]:
        # Tell the client its session id immediately so a brand-new chat is anchored.
        yield _ndjson({"type": "meta", "session_id": session_id})

        passages: list[dict] = []
        streamed_any = False
        logged_error = False
        try:
            if llm.classify_question(payload.question) == "chitchat":
                answer = llm.CHITCHAT_REPLY
                for piece in _pace_words(_word_chunks(answer), delay):
                    yield _ndjson({"type": "delta", "text": piece})
                    streamed_any = True
                final = {"answer": answer, "answered": True, "sources_used": []}
            else:
                search_query = llm.rewrite_query(payload.question, history)
                passages = retrieval.retrieve(search_query)
                # A short meta follow-up ("are you sure?", "why?") about the prior
                # answer is answered from the conversation, not by grounding strictly
                # in freshly-retrieved (often irrelevant) chunks.
                followup = bool(history) and llm.is_followup(payload.question)
                generate = llm.stream_followup if followup else llm.stream_answer
                logger.info(
                    "chat user=%s session=%s followup=%s query=%r rewritten=%r retrieved=%d",
                    user.id, session_id, followup, payload.question, search_query, len(passages),
                )
                final_holder: list[dict] = []

                def text_deltas() -> Iterator[str]:
                    for event in generate(
                        question=payload.question,
                        passages=passages,
                        history=history,
                        language=payload.language,
                    ):
                        if event["type"] == "delta":
                            yield event["text"]
                        else:
                            final_holder.append(event)

                for piece in _pace_words(text_deltas(), delay):
                    yield _ndjson({"type": "delta", "text": piece})
                    streamed_any = True

                final = final_holder[0] if final_holder else {
                    "answer": llm.NO_INFO_MESSAGE,
                    "answered": False,
                    "sources_used": [],
                }
        except Exception as exc:  # noqa: BLE001 - never leak a 500 mid-stream
            error_log.log_error(
                payload.question,
                f"{type(exc).__name__}: {exc}",
                session_id=session_id,
                language=payload.language,
            )
            logged_error = True
            final = {
                "answer": llm.SERVICE_BUSY_MESSAGE,
                "answered": False,
                "sources_used": [],
                "error": True,
            }
            passages = []

        # The LLM layer may catch its own provider error and report it via the
        # final event rather than raising — record that here (once).
        if final.get("error") and not logged_error:
            error_log.log_error(
                payload.question,
                "Response generation error during streaming",
                session_id=session_id,
                language=payload.language,
            )

        # Some results never produce deltas — no passages retrieved, an LLM/stream
        # error, or a "no information found" reply. Stream the final answer now so
        # the client shows the message instead of an empty bubble.
        if not streamed_any and final["answer"]:
            for piece in _pace_words(_word_chunks(final["answer"]), delay):
                yield _ndjson({"type": "delta", "text": piece})

        citations = [Citation(**c) for c in build_citations(passages, final["sources_used"])]

        assistant_msg = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=final["answer"],
            answered=final["answered"],
            citations_json=json.dumps([c.model_dump() for c in citations]),
        )
        db.add(assistant_msg)
        db.flush()

        # Name a brand-new conversation from its first message (cosmetic; the
        # answer has already fully streamed, so this only gates the done event).
        if is_new and not session.title:
            session.title = llm.generate_title(payload.question)

        latency_ms = int((time.perf_counter() - started) * 1000)
        db.add(
            QueryLog(
                session_id=session_id,
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
        logger.info(
            "chat done session=%s answered=%s latency=%dms",
            session_id, final["answered"], latency_ms,
        )

        yield _ndjson(
            {
                "type": "done",
                "message_id": assistant_msg.id,
                "session_id": session_id,
                "title": session.title,
                "answer": final["answer"],
                "answered": final["answered"],
                "citations": [c.model_dump() for c in citations],
                "latency_ms": latency_ms,
            }
        )

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.post("/sessions", response_model=dict)
def new_session(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    session = ChatSession(user_id=user.id)
    db.add(session)
    db.commit()
    return {"session_id": session.id}


@router.get("/sessions", response_model=list[ChatSessionSummary])
def list_sessions(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """List the signed-in user's conversations (most recent first)."""
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )
    return [
        ChatSessionSummary(
            id=s.id,
            title=s.title,
            created_at=s.created_at.isoformat() if s.created_at else None,
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}", response_model=dict)
def get_session(
    session_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    session = _owned_session(db, session_id, user)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.commit()  # persist owner-claim of any legacy session
    messages = []
    for m in session.messages:
        messages.append(
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "answered": m.answered,
                "citations": json.loads(m.citations_json) if m.citations_json else [],
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
        )
    return {"session_id": session.id, "messages": messages}


@router.post("/feedback", status_code=status.HTTP_201_CREATED)
def submit_feedback(
    payload: FeedbackIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    msg = db.get(ChatMessage, payload.message_id)
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")
    # Only allow rating messages in the caller's own conversations.
    session = db.get(ChatSession, msg.session_id)
    if session is None or (session.user_id is not None and session.user_id != user.id):
        raise HTTPException(status_code=404, detail="Message not found")
    db.add(Feedback(message_id=payload.message_id, rating=payload.rating, comment=payload.comment))
    db.commit()
    return {"status": "ok"}
