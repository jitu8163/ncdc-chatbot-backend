"""Grounded answer generation with gpt-4o-mini.

Enforces the SOW guardrails: answer strictly from supplied NCDC passages, never
diagnose / prescribe / give clinical recommendations, stay multilingual, and emit
the exact fallback string when the context does not contain the answer.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator

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


# Greetings / thanks / smalltalk with no information need. Matched locally so we
# don't spend a network round-trip (and ~1s) classifying every turn. Deliberately
# conservative: anything that isn't an obvious pleasantry falls through to
# "question" and goes through retrieval, so we never skip docs for a real need.
_CHITCHAT_RE = re.compile(
    r"^\s*(?:"
    r"hi+|hey+|hello+|hiya|yo|sup|"
    r"good\s*(?:morning|afternoon|evening|day)|greetings|"
    r"thanks?(?:\s*you)?|thank\s*you|thx|ty|cheers|"
    r"bye+|goodbye|see\s*you|see\s*ya|cya|"
    r"ok(?:ay)?|cool|nice|great|awesome|got\s*it|"
    r"how\s*are\s*you|who\s*are\s*you|what\s*can\s*you\s*do"
    r")[\s!.?]*$",
    re.IGNORECASE,
)


def classify_question(question: str) -> str:
    """Classify a turn so the pipeline can skip retrieval for non-questions.

    Returns "chitchat" (greeting/thanks/smalltalk) or "question". Uses a cheap
    local pattern instead of an LLM call to keep latency down.
    """
    return "chitchat" if _CHITCHAT_RE.match(question or "") else "question"


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


_JSON_UNESCAPE = {
    '"': '"', "\\": "\\", "/": "/",
    "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t",
}


class _AnswerFieldDecoder:
    """Incrementally decode the JSON string value of the top-level "answer" key
    from a streamed chat completion.

    The model still returns a single JSON object (answer + answered +
    sources_used + followups), but we want to forward just the answer prose to
    the client as it arrives. This walks the streamed characters, finds the
    answer value and JSON-unescapes it on the fly; the remaining metadata is
    parsed from the complete buffer once the stream ends.
    """

    def __init__(self) -> None:
        self._raw = ""
        self._pos = 0
        self._in_value = False
        self._done = False
        self._escape = False
        self._uni = ""  # collects "u" + 4 hex digits of a \uXXXX escape

    def feed(self, delta: str) -> str:
        if self._done:
            return ""
        self._raw += delta
        out: list[str] = []
        while self._pos < len(self._raw):
            if not self._in_value:
                m = re.search(r'"answer"\s*:\s*"', self._raw[self._pos:])
                if not m:
                    break  # value start not in buffer yet; wait for more
                self._pos += m.end()
                self._in_value = True
                continue
            ch = self._raw[self._pos]
            self._pos += 1
            if self._uni:
                self._uni += ch
                if len(self._uni) == 5:
                    try:
                        out.append(chr(int(self._uni[1:], 16)))
                    except ValueError:
                        pass
                    self._uni = ""
                    self._escape = False
                continue
            if self._escape:
                if ch == "u":
                    self._uni = "u"
                else:
                    out.append(_JSON_UNESCAPE.get(ch, ch))
                    self._escape = False
                continue
            if ch == "\\":
                self._escape = True
                continue
            if ch == '"':  # closing quote -> answer value complete
                self._done = True
                break
            out.append(ch)
        return "".join(out)


def stream_answer(
    question: str,
    passages: list[dict],
    history: list[dict] | None = None,
    language: str | None = None,
) -> Iterator[dict]:
    """Stream a grounded answer.

    Yields:
      {"type": "delta", "text": <answer chunk>}   — zero or more, in order
      {"type": "final", "answer", "answered", "sources_used", "followups"}  — exactly once

    The final event carries the canonical values (parsed from the complete JSON)
    used for persistence + citations.
    """
    if not passages:
        yield {
            "type": "final",
            "answer": NO_INFO_MESSAGE,
            "answered": False,
            "sources_used": [],
            "followups": [],
        }
        return

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
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

    decoder = _AnswerFieldDecoder()
    raw_parts: list[str] = []
    shown_parts: list[str] = []
    try:
        stream = _openai().chat.completions.create(
            model=settings.openai_chat_model,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content or ""
            if not piece:
                continue
            raw_parts.append(piece)
            text = decoder.feed(piece)
            if text:
                shown_parts.append(text)
                yield {"type": "delta", "text": text}
    except Exception:  # noqa: BLE001
        logger.exception("LLM streaming generation failed")
        yield {
            "type": "final",
            "answer": NO_INFO_MESSAGE,
            "answered": False,
            "sources_used": [],
            "followups": [],
        }
        return

    streamed = "".join(shown_parts).strip()
    try:
        data = json.loads("".join(raw_parts))
    except Exception:  # noqa: BLE001 - fall back to the text we already streamed
        logger.warning("Could not parse streamed answer JSON; using extracted text")
        data = {}

    answer = (data.get("answer") or "").strip() or streamed or NO_INFO_MESSAGE
    answered = bool(data.get("answered", False)) and answer != NO_INFO_MESSAGE
    sources_used = [int(s) for s in data.get("sources_used", []) if isinstance(s, (int, float))]
    followups = [str(f) for f in data.get("followups", [])][:3]
    yield {
        "type": "final",
        "answer": answer,
        "answered": answered,
        "sources_used": sources_used,
        "followups": followups,
    }
