"""Grounded answer generation with gpt-4o-mini.

Enforces the SOW guardrails: answer strictly from supplied NCDC passages, never
diagnose / prescribe / give clinical recommendations, stay multilingual, and emit
the exact fallback string when the context does not contain the answer.
"""
from __future__ import annotations

import json
import logging

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

NO_INFO_MESSAGE = (
    "Relevant information could not be found in the available NCDC guideline documents."
)

_client: OpenAI | None = None


def _openai() -> OpenAI:
    global _client
    if _client is None:
        # base_url=None -> OpenAI; set it for an OpenAI-compatible provider (e.g. Groq).
        _client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
        )
    return _client


SYSTEM_PROMPT = f"""You are the NCDC Guideline Assistant. You help citizens and \
healthcare workers understand National Centre for Disease Control (NCDC) guideline \
documents.

STRICT RULES:
1. Answer ONLY using the numbered SOURCES provided in the user message. Never use \
outside or prior knowledge.
2. If the sources do not contain enough information to answer, you MUST set \
"answered" to false and set "answer" to exactly this string:
   "{NO_INFO_MESSAGE}"
3. You must NOT diagnose diseases, prescribe medicines, or give individual clinical \
recommendations. You may quote what a guideline states in general terms, but add no \
personal medical advice.
4. Reply in the SAME language as the user's question. Translate guideline content as \
needed while preserving meaning.
5. Cite the sources you used by their number. Only cite sources that actually support \
your answer.
6. Be concise, factual and faithful to the guideline wording.

Respond ONLY with a JSON object of this exact shape:
{{
  "answer": "<your grounded answer or the no-info string>",
  "answered": <true|false>,
  "sources_used": [<source numbers you relied on>],
  "followups": ["<up to 3 relevant follow-up questions in the user's language>"]
}}"""


def _build_context(passages: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        header = f"[{i}] {p.get('document_title', 'Document')}"
        if p.get("page"):
            header += f", page {p['page']}"
        if p.get("section"):
            header += f", section: {p['section']}"
        # Feed the larger parent block when available; fall back to the snippet.
        body = p.get("context") or p.get("text") or p.get("snippet") or ""
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def classify_question(question: str) -> str:
    """Classify a turn so the graph can skip retrieval for non-questions.

    Returns one of: "chitchat" (greeting/thanks/smalltalk) or "question".
    """
    try:
        resp = _openai().chat.completions.create(
            model=settings.openai_chat_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user's message for an NCDC guideline assistant. "
                        'Respond with JSON {"label": "chitchat" | "question"}. '
                        '"chitchat" = greetings, thanks, goodbyes or smalltalk with no '
                        'information need. "question" = anything that may need the '
                        "guideline documents to answer."
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        label = json.loads(resp.choices[0].message.content).get("label", "question")
        return "chitchat" if label == "chitchat" else "question"
    except Exception:  # noqa: BLE001 - on any failure, treat as a real question
        logger.exception("Question classification failed")
        return "question"


def rewrite_query(question: str, history: list[dict] | None) -> str:
    """Rewrite a possibly-elliptical follow-up into a standalone search query."""
    if not history:
        return question
    recent = [t for t in history if t.get("role") == "user"][-4:]
    if not recent:
        return question
    try:
        convo = "\n".join(f"{t['role']}: {t['content']}" for t in history[-6:])
        resp = _openai().chat.completions.create(
            model=settings.openai_chat_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's latest message into a single, standalone "
                        "search query that resolves pronouns and references using the "
                        "conversation. Keep the original language. Respond with JSON "
                        '{"query": "..."}.'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Conversation:\n{convo}\n\nLatest message: {question}",
                },
            ],
        )
        rewritten = json.loads(resp.choices[0].message.content).get("query", "").strip()
        return rewritten or question
    except Exception:  # noqa: BLE001
        logger.exception("Query rewrite failed; using original question")
        return question


CHITCHAT_REPLY = (
    "Hello! I'm the NCDC Guideline Assistant. Ask me anything about the NCDC "
    "guideline documents and I'll answer with citations to the source."
)


def generate_answer(
    question: str,
    passages: list[dict],
    history: list[dict] | None = None,
    language: str | None = None,
) -> dict:
    """Return {answer, answered, sources_used: list[int], followups: list[str]}."""
    if not passages:
        return {"answer": NO_INFO_MESSAGE, "answered": False, "sources_used": [], "followups": []}

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Carry recent session turns so follow-ups keep context (kept short on purpose).
    for turn in (history or [])[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    lang_hint = f"\n\n(Respond in: {language})" if language else ""
    messages.append(
        {
            "role": "user",
            "content": (
                f"SOURCES:\n{_build_context(passages)}\n\n"
                f"QUESTION: {question}{lang_hint}"
            ),
        }
    )

    try:
        resp = _openai().chat.completions.create(
            model=settings.openai_chat_model,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception:  # noqa: BLE001
        logger.exception("LLM generation failed")
        return {"answer": NO_INFO_MESSAGE, "answered": False, "sources_used": [], "followups": []}

    answer = (data.get("answer") or "").strip() or NO_INFO_MESSAGE
    answered = bool(data.get("answered", False)) and answer != NO_INFO_MESSAGE
    sources_used = [int(s) for s in data.get("sources_used", []) if isinstance(s, (int, float))]
    followups = [str(f) for f in data.get("followups", [])][:3]
    return {
        "answer": answer,
        "answered": answered,
        "sources_used": sources_used,
        "followups": followups,
    }
