"""Shared state passed between LangGraph nodes."""
from __future__ import annotations

from typing import TypedDict


class ChatState(TypedDict, total=False):
    # Inputs
    question: str
    language: str | None
    history: list[dict]            # [{role, content}, ...] prior turns
    conv_state: dict               # persisted ConversationState (active_topic, ...)

    # Working values (produced by the Context Analyzer)
    route: str                     # "DIRECT" | "MEMORY" | "RAG"
    reason: str                    # why the analyzer chose this route (observability)
    active_topic: str              # sticky subject under discussion
    rewritten_query: str           # standalone, topic-resolved search query
    search_query: str              # query actually used for retrieval (== rewritten_query)
    subtype: str                   # smalltalk subtype for DIRECT (greeting/ack/thanks/bye)
    needs_retrieval: bool          # MEMORY couldn't answer from context -> fall through to RAG

    passages: list[dict]           # retrieval results (or last_rag_context for MEMORY answers)
    weak_retrieval: bool           # best relevance below the guardrail floor

    # Outputs
    answer: str
    answered: bool
    sources_used: list[int]
    citations: list[dict]
    error: bool
