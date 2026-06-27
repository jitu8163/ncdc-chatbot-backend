"""The chat LangGraph — context-aware DIRECT / MEMORY / RAG routing.

Flow:

    START
      -> analyze                       (Context Analyzer: route + active topic + rewrite)
         -> DIRECT -> direct -> END    (greeting / smalltalk; no retrieval, no LLM)
         -> MEMORY -> memory           (answer from previously retrieved context)
              -> answered  -> citations -> END
              -> needs_retrieval -> rewrite -> ...        (fall through to RAG)
         -> RAG    -> rewrite          (Query Rewriter: promote the standalone query)
              -> retrieve              (dense vector search + cross-encoder rerank)
              -> generate              (guardrailed Context Builder + LLM)
              -> citations             (Citation Engine: doc/page/section/hyperlink)
              -> END

The Context Analyzer, MEMORY answering, retrieval guardrails and conversation-state
updates live in `app.services.conversation`, shared with the streaming endpoint so
the two chat paths route identically. ConversationState is threaded in/out via
`conv_state` and folded back in `run_chat`.
"""
from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.graph.state import ChatState
from app.services import cache, conversation, llm, retrieval

logger = logging.getLogger("ncdc.graph")

DIRECT, MEMORY, RAG = conversation.DIRECT, conversation.MEMORY, conversation.RAG


def _timed(node: Callable[[ChatState], ChatState]) -> Callable[[ChatState], ChatState]:
    """Wrap a graph node to log how long it takes (ms)."""
    name = node.__name__.removesuffix("_node")

    @functools.wraps(node)
    def wrapper(state: ChatState) -> ChatState:
        start = time.perf_counter()
        try:
            return node(state)
        finally:
            logger.info("step %-9s %6.0f ms", name, (time.perf_counter() - start) * 1000)

    return wrapper


# ─── Nodes ─────────────────────────────────────────────────────────────────
def analyze_node(state: ChatState) -> ChatState:
    """Context Analyzer: classify the turn, keep/refresh the active topic and rewrite
    the query — all in one fast call (with deterministic fast-paths/fallbacks)."""
    info = conversation.analyze(
        state["question"], state.get("conv_state") or {}, state.get("history")
    )
    return {
        "route": info["route"],
        "reason": info.get("reason", ""),
        "active_topic": info.get("active_topic", ""),
        "rewritten_query": info.get("rewritten_query", state["question"]),
        "subtype": info.get("subtype", ""),
    }


def direct_node(state: ChatState) -> ChatState:
    """DIRECT: greeting / acknowledgement / thanks / farewell — a canned, intent-
    appropriate reply with no retrieval and no LLM call."""
    returning = bool(state.get("history"))
    answer = llm.smalltalk_reply(state.get("subtype") or "", returning=returning)
    return {
        "answer": answer,
        "answered": True,
        "passages": [],
        "sources_used": [],
        "citations": [],
    }


def memory_node(state: ChatState) -> ChatState:
    """MEMORY: try to answer the follow-up from the previously retrieved context +
    conversation. If that material is insufficient, signal needs_retrieval so the
    graph falls through to a fresh RAG lookup with the rewritten query."""
    result = conversation.answer_from_memory(
        question=state["question"],
        state=state.get("conv_state") or {},
        history=state.get("history"),
        language=state.get("language"),
    )
    if result["needs_retrieval"]:
        return {"needs_retrieval": True}
    # Answered from memory: cite against the remembered context so citations resolve.
    return {
        "needs_retrieval": False,
        "answer": result["answer"],
        "answered": result["answered"],
        "sources_used": result["sources_used"],
        "passages": (state.get("conv_state") or {}).get("last_rag_context") or [],
    }


def rewrite_node(state: ChatState) -> ChatState:
    """Query Rewriter: promote the analyzer's standalone, topic-resolved query into
    the field retrieval uses. (The rewrite itself happened in the analyzer call.)"""
    return {"search_query": state.get("rewritten_query") or state["question"]}


def retrieve_node(state: ChatState) -> ChatState:
    passages = retrieval.retrieve(state["search_query"])
    return {"passages": passages, "weak_retrieval": conversation.is_weak_retrieval(passages)}


def _answer_signature(state: ChatState) -> tuple[str, ...]:
    sig = [state["search_query"], state.get("language") or ""]
    for p in state["passages"]:
        sig.append(f"{p.get('document_id')}:{p.get('page')}")
    return tuple(sig)


def generate_node(state: ChatState) -> ChatState:
    # Guardrail: with no passages or only weak (low-relevance) matches, don't ask the
    # LLM to answer — say the documents don't cover it rather than risk fabrication.
    if not state.get("passages") or state.get("weak_retrieval"):
        logger.info("guardrail: weak/empty retrieval -> no-info (query=%r)", state["search_query"])
        return {"answer": llm.NO_INFO_MESSAGE, "answered": False, "sources_used": []}

    cached = cache.get_json("ans", *_answer_signature(state))
    if cached is not None:
        return cached
    result = llm.generate_answer(
        question=state.get("rewritten_query") or state["question"],
        passages=state["passages"],
        history=state.get("history"),
        language=state.get("language"),
    )
    out: ChatState = {
        "answer": result["answer"],
        "answered": result["answered"],
        "sources_used": result["sources_used"],
    }
    if result.get("error"):
        out["error"] = True
    # Don't cache failed generations — we want a retry to re-attempt.
    if not result.get("error"):
        cache.set_json("ans", out, settings.answer_cache_ttl, *_answer_signature(state))
    return out


def _citation_url(document_id: str, page: int | None) -> str:
    base = f"{settings.public_base_url.rstrip('/')}/api/documents/{document_id}/view"
    return f"{base}#page={page}" if page else base


def build_citations(passages: list[dict], sources_used: list[int]) -> list[dict]:
    """Map the LLM's cited source numbers to deduplicated citation records.

    Shared by the graph's citations_node and the streaming chat endpoint.
    """
    citations: list[dict] = []
    seen: set[tuple[str, int | None]] = set()
    for idx in sources_used or []:
        if 1 <= idx <= len(passages):
            p = passages[idx - 1]
            key = (p["document_id"], p.get("page"))
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                {
                    "document_id": p["document_id"],
                    "document_title": p.get("document_title", "Document"),
                    "page": p.get("page"),
                    "section": p.get("section"),
                    "snippet": (p.get("snippet") or p.get("context") or p.get("text") or "")[:300],
                    "url": _citation_url(p["document_id"], p.get("page")),
                }
            )
    return citations


def citations_node(state: ChatState) -> ChatState:
    return {"citations": build_citations(state.get("passages", []), state.get("sources_used", []))}


# ─── Routing ─────────────────────────────────────────────────────────────────
def _route_from_analyze(state: ChatState) -> str:
    return state.get("route", RAG)


def _route_from_memory(state: ChatState) -> str:
    return "retrieve" if state.get("needs_retrieval") else "answered"


# ─── Graph assembly ──────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _graph():
    g = StateGraph(ChatState)
    g.add_node("analyze", _timed(analyze_node))
    g.add_node("direct", _timed(direct_node))
    g.add_node("memory", _timed(memory_node))
    g.add_node("rewrite", _timed(rewrite_node))
    g.add_node("retrieve", _timed(retrieve_node))
    g.add_node("generate", _timed(generate_node))
    g.add_node("citations", _timed(citations_node))

    g.add_edge(START, "analyze")
    g.add_conditional_edges(
        "analyze",
        _route_from_analyze,
        {DIRECT: "direct", MEMORY: "memory", RAG: "rewrite"},
    )
    g.add_edge("direct", END)
    # MEMORY: answer from context, or fall through to RAG retrieval.
    g.add_conditional_edges(
        "memory", _route_from_memory, {"retrieve": "rewrite", "answered": "citations"}
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "citations")
    g.add_edge("citations", END)
    return g.compile()


def run_chat(
    question: str,
    history: list[dict] | None = None,
    language: str | None = None,
    conv_state: dict | None = None,
) -> ChatState:
    """Execute the chat graph and return the final state, including the updated
    ConversationState under ``conv_state`` for the caller to persist."""
    state_in = conv_state if conv_state is not None else conversation.new_state()
    initial: ChatState = {
        "question": question,
        "history": history or [],
        "language": language,
        "conv_state": state_in,
    }
    start = time.perf_counter()
    result = _graph().invoke(initial)
    logger.info("chat pipeline total %6.0f ms", (time.perf_counter() - start) * 1000)

    # Fold this turn's outcome back into the conversation state for persistence.
    updated = _update_state(state_in, result)
    result["conv_state"] = updated

    conversation.log_turn(
        session_id="-",
        route=result.get("route", "?"),
        reason=result.get("reason", ""),
        active_topic=updated.get("active_topic", ""),
        rewritten_query=result.get("search_query") or result.get("rewritten_query", ""),
        passages=result.get("passages"),
        sources_used=result.get("sources_used"),
        answered=result.get("answered"),
    )
    return result


def _update_state(state_in: dict, result: ChatState) -> dict:
    """Apply conversation.update_* based on which route actually produced the answer."""
    route = result.get("route", RAG)
    question = result.get("question", "")
    answer = result.get("answer", "")
    # A RAG path ran whenever we ended up retrieving (pure RAG, or MEMORY that fell
    # through to retrieval). Detect it by the presence of a search_query/passages set
    # by the retrieval branch.
    ran_rag = bool(result.get("search_query")) and route != DIRECT and not (
        route == MEMORY and not result.get("needs_retrieval")
    )
    if ran_rag:
        return conversation.update_after_rag(
            state_in,
            active_topic=result.get("active_topic", ""),
            rewritten_query=result.get("search_query") or result.get("rewritten_query", ""),
            passages=result.get("passages"),
            question=question,
            answer=answer,
            answered=bool(result.get("answered")),
        )
    return conversation.update_after_simple(
        state_in,
        question=question,
        answer=answer,
        active_topic=result.get("active_topic") if route != DIRECT else None,
    )
