"""The chat LangGraph.

Flow (maps to the architecture's node list):

    START
      -> classify            (Question Classifier)
         -> chitchat -> END  (greeting / smalltalk short-circuit, no retrieval)
         -> rewrite          (Query Rewriter — resolve follow-ups to a standalone query)
            -> retrieve       (Dense vector search)
            -> generate       (Context Builder + LLM)
            -> citations      (Citation Engine: builder/page/section/hyperlink)
            -> format         (Response Formatter)
            -> END
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
from app.services import cache, llm, retrieval

logger = logging.getLogger("ncdc.graph")


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
def classify_node(state: ChatState) -> ChatState:
    return {"route": llm.classify_question(state["question"])}


def chitchat_node(state: ChatState) -> ChatState:
    return {
        "answer": llm.CHITCHAT_REPLY,
        "answered": True,
        "passages": [],
        "sources_used": [],
        "citations": [],
        "followups": [],
    }


def rewrite_node(state: ChatState) -> ChatState:
    return {"search_query": llm.rewrite_query(state["question"], state.get("history"))}


def retrieve_node(state: ChatState) -> ChatState:
    passages = retrieval.retrieve(state["search_query"], category=state.get("category"))
    return {"passages": passages}


def _answer_signature(state: ChatState) -> tuple[str, ...]:
    sig = [state["search_query"], state.get("language") or ""]
    for p in state["passages"]:
        sig.append(f"{p.get('document_id')}:{p.get('page')}")
    return tuple(sig)


def generate_node(state: ChatState) -> ChatState:
    cached = cache.get_json("ans", *_answer_signature(state))
    if cached is not None:
        return cached
    result = llm.generate_answer(
        question=state["question"],
        passages=state["passages"],
        history=state.get("history"),
        language=state.get("language"),
    )
    out: ChatState = {
        "answer": result["answer"],
        "answered": result["answered"],
        "sources_used": result["sources_used"],
        "followups": result["followups"],
    }
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
                    "snippet": (p.get("snippet") or p.get("context") or "")[:300],
                    "url": _citation_url(p["document_id"], p.get("page")),
                }
            )
    return citations


def citations_node(state: ChatState) -> ChatState:
    return {"citations": build_citations(state["passages"], state.get("sources_used", []))}


def format_node(state: ChatState) -> ChatState:
    # Final shaping: keep at most 3 follow-ups; drop them when no answer was found.
    followups = state.get("followups", [])[:3] if state.get("answered") else []
    return {"followups": followups}


def _route(state: ChatState) -> str:
    return state.get("route", "question")


# ─── Graph assembly ──────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _graph():
    g = StateGraph(ChatState)
    g.add_node("classify", _timed(classify_node))
    g.add_node("chitchat", _timed(chitchat_node))
    g.add_node("rewrite", _timed(rewrite_node))
    g.add_node("retrieve", _timed(retrieve_node))
    g.add_node("generate", _timed(generate_node))
    g.add_node("citations", _timed(citations_node))
    g.add_node("format", _timed(format_node))

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify", _route, {"chitchat": "chitchat", "question": "rewrite"}
    )
    g.add_edge("chitchat", END)
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "citations")
    g.add_edge("citations", "format")
    g.add_edge("format", END)
    return g.compile()


def run_chat(
    question: str,
    history: list[dict] | None = None,
    language: str | None = None,
    category: str | None = None,
) -> ChatState:
    """Execute the chat graph and return the final state."""
    initial: ChatState = {
        "question": question,
        "history": history or [],
        "language": language,
        "category": category,
    }
    start = time.perf_counter()
    result = _graph().invoke(initial)
    logger.info("chat pipeline total %6.0f ms", (time.perf_counter() - start) * 1000)
    return result
