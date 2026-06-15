"""Shared state passed between LangGraph nodes."""
from __future__ import annotations

from typing import TypedDict


class ChatState(TypedDict, total=False):
    # Inputs
    question: str
    language: str | None
    category: str | None
    history: list[dict]            # [{role, content}, ...] prior turns

    # Working values
    route: str                     # "chitchat" | "question"
    search_query: str              # rewritten, standalone query
    passages: list[dict]           # reranked, parent-expanded retrieval results

    # Outputs
    answer: str
    answered: bool
    sources_used: list[int]
    citations: list[dict]
    followups: list[str]
