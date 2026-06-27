"""Context-aware conversation routing core (DIRECT / MEMORY / RAG).

This is the single "brain" shared by BOTH chat entrypoints — the non-streaming
LangGraph (`app.graph.pipeline`) and the streaming endpoint
(`app.routers.chat.ask_stream`) — so they can never drift apart again.

Responsibilities:
  * ConversationState — the small, persisted memory of a session (active topic,
    last RAG query + retrieved context, last answer, rolling summary).
  * The Context Analyzer — classify each turn into DIRECT / MEMORY / RAG, keep the
    active topic stable across follow-ups, and rewrite the turn into a standalone,
    topic-resolved search query.
  * MEMORY answering — try to answer a follow-up from the persisted context; report
    when that's insufficient so the caller falls through to a fresh RAG lookup.
  * Retrieval guardrails — flag weak (low-relevance) retrievals so callers avoid
    fabricating answers.
  * State updates — fold each turn's outcome back into ConversationState.

The heavy lifting (LLM calls) lives in `app.services.llm`; this module adds the
deterministic fast-paths, fallbacks, state management and logging around them.
"""
from __future__ import annotations

import json
import logging
from typing import TypedDict

from app.config import settings
from app.services import llm

logger = logging.getLogger("ncdc.conversation")

# Re-export the route constants so callers use one vocabulary.
DIRECT, MEMORY, RAG = llm.DIRECT, llm.MEMORY, llm.RAG


class ConversationState(TypedDict, total=False):
    active_topic: str            # the subject under discussion (sticky across follow-ups)
    last_rag_query: str          # the last standalone query we retrieved on
    last_rag_context: list[dict]  # trimmed passages from the last RAG turn (for MEMORY)
    last_answer: str             # the assistant's last answer text
    conversation_summary: str    # short rolling gist of the conversation


def new_state() -> ConversationState:
    return {
        "active_topic": "",
        "last_rag_query": "",
        "last_rag_context": [],
        "last_answer": "",
        "conversation_summary": "",
    }


def load_state(raw: str | None) -> ConversationState:
    """Parse the persisted ConversationState JSON (ChatSession.state_json). Tolerant
    of missing/garbled values — always returns a complete state dict."""
    state = new_state()
    if not raw:
        return state
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return state
    if isinstance(data, dict):
        for key in state:
            if key in data and data[key] is not None:
                state[key] = data[key]  # type: ignore[literal-required]
    if not isinstance(state.get("last_rag_context"), list):
        state["last_rag_context"] = []
    return state


def dump_state(state: ConversationState) -> str:
    return json.dumps(state, ensure_ascii=False)


# ─── Context Analyzer ────────────────────────────────────────────────────────
def analyze(question: str, state: ConversationState, history: list[dict] | None) -> dict:
    """Classify the turn and produce routing metadata.

    Returns ``{"route", "reason", "rewritten_query", "active_topic"}`` where route
    is one of DIRECT / MEMORY / RAG.

    Deterministic fast-paths settle the easy cases with no LLM call (obvious
    smalltalk -> DIRECT; a first turn that isn't smalltalk is a RAG lookup).
    Otherwise one small/fast LLM call routes + rewrites; the reliable local
    follow-up regex is the safety net, and we fall back to it on any error.
    """
    q = (question or "").strip()
    topic = state.get("active_topic", "")

    # 1) Obvious smalltalk -> DIRECT, no LLM call. Topic is preserved.
    subtype = llm.classify_smalltalk(q)
    if subtype:
        return {
            "route": DIRECT,
            "reason": f"smalltalk:{subtype}",
            "rewritten_query": q,
            "active_topic": topic,
            "subtype": subtype,
        }

    # 2) First real turn of a session — nothing to follow up on, so it's a lookup.
    if not history:
        return {
            "route": RAG,
            "reason": "first-turn knowledge lookup",
            "rewritten_query": q,
            "active_topic": q[:80],  # provisional; refined after the LLM/retrieval
        }

    regex_followup = llm.is_followup(q)
    try:
        out = llm.analyze_context_llm(q, history, active_topic=topic)
        route = out["route"]
        # Trust the deterministic follow-up signal over a RAG misroute: the regex
        # only fires on clearly back-referential phrasing, and such turns must keep
        # the active topic rather than start a new lookup that drops it.
        if regex_followup and route == RAG:
            route = MEMORY
            out["reason"] = (out.get("reason") or "") + " | regex-followup override"
        # MEMORY/DIRECT must never silently change the topic.
        if route in (MEMORY, DIRECT) and topic:
            out["active_topic"] = topic
        out["route"] = route
        return out
    except Exception:  # noqa: BLE001 - provider slow/unavailable: deterministic fallback
        logger.warning("Context analyzer LLM unavailable; using regex/topic fallback")
        if regex_followup:
            return {
                "route": MEMORY,
                "reason": "regex follow-up (analyzer fallback)",
                "rewritten_query": _fallback_rewrite(q, topic),
                "active_topic": topic,
            }
        return {
            "route": RAG,
            "reason": "knowledge lookup (analyzer fallback)",
            "rewritten_query": q,
            "active_topic": q[:80],
        }


def _fallback_rewrite(question: str, topic: str) -> str:
    """Cheap standalone-query rewrite when the LLM analyzer is unavailable: prepend
    the active topic to a short referential follow-up so retrieval has something to
    bite on (e.g. topic 'dengue' + 'how many deaths?' -> 'dengue how many deaths?')."""
    q = (question or "").strip()
    if topic and len(q.split()) <= 6 and topic.lower() not in q.lower():
        return f"{topic} {q}".strip()
    return q


# ─── MEMORY answering ────────────────────────────────────────────────────────
def answer_from_memory(
    question: str,
    state: ConversationState,
    history: list[dict] | None,
    language: str | None,
) -> dict:
    """Attempt to answer a MEMORY follow-up from the persisted context + history.

    Returns the same shape as ``llm.answer_from_context`` —
    ``{answer, answered, sources_used, needs_retrieval}``. When needs_retrieval is
    True the caller should run RAG with the rewritten query instead.
    """
    return llm.answer_from_context(
        question=question,
        passages=state.get("last_rag_context") or [],
        history=history,
        language=language,
    )


# ─── Retrieval guardrails ────────────────────────────────────────────────────
def best_relevance(passages: list[dict] | None) -> float | None:
    """Highest reranker score among passages (None if unscored/empty)."""
    scores = [p["rerank_score"] for p in (passages or []) if "rerank_score" in p]
    return max(scores) if scores else None


def is_weak_retrieval(passages: list[dict] | None) -> bool:
    """True when retrieval is empty or its best passage scores below the configured
    relevance floor — a signal to avoid answering (no fabrication) and instead say
    the documents don't cover it / ask to rephrase."""
    if not passages:
        return True
    best = best_relevance(passages)
    if best is None:  # reranker disabled or scores missing — don't second-guess
        return False
    return best < settings.retrieval_min_score


# ─── State updates ───────────────────────────────────────────────────────────
def _trim_context(passages: list[dict] | None) -> list[dict]:
    """Keep only the few top passages, and only the fields MEMORY answering needs,
    so the persisted session state stays small."""
    trimmed: list[dict] = []
    for p in (passages or [])[: settings.memory_context_passages]:
        trimmed.append(
            {
                "document_id": p.get("document_id"),
                "document_title": p.get("document_title", "Document"),
                "page": p.get("page"),
                "section": p.get("section"),
                "text": (p.get("context") or p.get("text") or p.get("snippet") or "")[:1500],
                "rerank_score": p.get("rerank_score"),
            }
        )
    return trimmed


def _summarize(state: ConversationState, topic: str, question: str, answer: str) -> str:
    """Maintain a tiny rolling gist without an extra LLM call: the active topic plus
    a trimmed last exchange. Enough to anchor the analyzer; cheap to store."""
    gist = f"Topic: {topic}. " if topic else ""
    gist += f"Last asked: {question.strip()[:120]}"
    return gist[:400]


def update_after_rag(
    state: ConversationState,
    *,
    active_topic: str,
    rewritten_query: str,
    passages: list[dict] | None,
    question: str,
    answer: str,
    answered: bool,
) -> ConversationState:
    """Fold a RAG turn into the state: refresh the active topic and remember the
    query + retrieved context so the next follow-up can be answered from memory."""
    topic = (active_topic or state.get("active_topic") or rewritten_query or question)[:120]
    state["active_topic"] = topic
    state["last_rag_query"] = rewritten_query
    # Only overwrite remembered context when this turn actually retrieved something,
    # so a weak/empty retrieval doesn't wipe usable prior context.
    if passages:
        state["last_rag_context"] = _trim_context(passages)
    state["last_answer"] = answer
    state["conversation_summary"] = _summarize(state, topic, question, answer)
    return state


def update_after_simple(
    state: ConversationState,
    *,
    question: str,
    answer: str,
    active_topic: str | None = None,
) -> ConversationState:
    """Fold a DIRECT or MEMORY turn into the state. The active topic and remembered
    context are preserved (a follow-up must not reset the topic); only the last
    answer / summary are refreshed."""
    if active_topic:
        state["active_topic"] = active_topic[:120]
    state["last_answer"] = answer
    state["conversation_summary"] = _summarize(
        state, state.get("active_topic", ""), question, answer
    )
    return state


def log_turn(
    *,
    session_id: str,
    route: str,
    reason: str,
    active_topic: str,
    rewritten_query: str,
    passages: list[dict] | None = None,
    sources_used: list[int] | None = None,
    answered: bool | None = None,
) -> None:
    """Single structured observability line per routing decision / outcome."""
    best = best_relevance(passages)
    scores = [round(p.get("rerank_score"), 2) for p in (passages or []) if p.get("rerank_score") is not None]
    docs = [f"{p.get('document_title')}#p{p.get('page')}" for p in (passages or [])]
    logger.info(
        "route=%s topic=%r reason=%r rewritten=%r retrieved=%d best_score=%s scores=%s "
        "answered=%s sources=%s docs=%s session=%s",
        route, active_topic, reason, rewritten_query,
        len(passages or []), None if best is None else round(best, 2), scores,
        answered, sources_used, docs, session_id,
    )
