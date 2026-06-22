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

# Shown when the language model itself can't be reached (rate limit, timeout,
# provider error) — distinct from NO_INFO_MESSAGE so we don't blame the documents
# for an infrastructure problem.
SERVICE_BUSY_MESSAGE = (
    "The assistant is temporarily unavailable (the language model is rate-limited or "
    "unreachable). Please try again in a little while."
)

_client: OpenAI | None = None


def _openai() -> OpenAI:
    global _client
    if _client is None:
        # base_url=None -> OpenAI; set it for an OpenAI-compatible provider (e.g. Groq).
        _client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            # Cap every call so a stalled provider can't exceed the latency budget
            # (the SDK default is 600s). Disable the automatic retries that would
            # otherwise multiply that wait on a transient error.
            timeout=settings.llm_request_timeout,
            max_retries=0,
        )
    return _client


SYSTEM_PROMPT = f"""You are the NCDC Guideline Assistant. You help citizens and \
healthcare workers understand National Centre for Disease Control (NCDC) guideline \
documents.

RULES:
1. Ground factual claims about the guidelines in the numbered SOURCES provided in \
the user message — do not invent guideline facts. Within the sources, you SHOULD \
synthesize, summarize and paraphrase across passages; you do not need a word-for-word \
match. You may ALSO use the CONVERSATION so far (the earlier questions and your own \
earlier answers) as context.
2. FOLLOW-UPS: When the user's latest message is a follow-up about your previous \
answer — e.g. "are you sure?", "why?", "how?", "can you explain more?", "what do you \
mean?", "is that correct?" — do NOT treat it as a brand-new question and do NOT reply \
with a greeting or restate your role. Instead, use the CONVERSATION to confirm, \
justify, clarify or expand on what you already told the user. If you answer such a \
follow-up from the conversation rather than from the numbered sources, set \
"sources_used" to [] and "answered" to true. Never claim the information is \
unavailable when the conversation already contains it.
3. Be helpful: if the sources or the conversation contain information that answers or \
partially answers the question, give the most useful grounded answer you can and note \
any limits. Only when NEITHER the sources NOR the conversation address the question at \
all should you set "answered" to false and set "answer" to exactly this string:
   "{NO_INFO_MESSAGE}"
4. You must NOT diagnose diseases, prescribe medicines, or give individual clinical \
recommendations. You may explain what a guideline states in general terms, but add no \
personal medical advice.
5. Reply in the SAME language as the user's question. Translate guideline content as \
needed while preserving meaning.
6. Cite the sources you used by their number. Only cite sources that actually support \
your answer.
7. Be concise, factual and faithful to the guideline wording.

Respond ONLY with a JSON object of this exact shape:
{{
  "answer": "<your grounded answer or the no-info string>",
  "answered": <true|false>,
  "sources_used": [<source numbers you relied on, or [] if answered from the conversation>]
}}"""


FOLLOWUP_SYSTEM_PROMPT = """You are the NCDC Guideline Assistant, continuing an \
ongoing conversation. The user's latest message is a FOLLOW-UP about your previous \
answer — for example confirming it ("are you sure?"), asking why or how, or asking \
you to explain or expand.

RULES:
1. Use the CONVERSATION above as your primary context: confirm, justify, clarify or \
expand on what you already told the user. Any SOURCES provided are supplementary.
2. Do NOT greet the user, restate your role, or send a generic message. Do NOT say \
the information is unavailable when the conversation already contains it.
3. Stay faithful to what the guidelines (as reflected in the conversation and any \
sources) actually say — do not invent new facts, and give no personal medical advice \
(no diagnosis or prescriptions).
4. Reply in the SAME language as the user.

Respond ONLY with a JSON object of this exact shape:
{
  "answer": "<your answer>",
  "answered": true,
  "sources_used": [<source numbers you relied on, or [] if answered from the conversation>]
}"""


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


# Short "meta" follow-ups that refer to the PREVIOUS answer rather than introducing
# a new topic. These have little/no retrievable content of their own, so they must
# be answered from the conversation — not by running fresh retrieval and grounding
# strictly in whatever (often irrelevant) chunks come back.
_FOLLOWUP_RE = re.compile(
    r"^\s*(?:"
    r"(?:are|r)\s*(?:you|u)\s*(?:sure|certain|serious)|(?:you|u)\s*sure|sure|"
    r"really|seriously|"
    r"why(?:\s*(?:not|so|is\s*(?:that|this)))?|"
    r"how(?:\s*(?:so|come|is\s*(?:that|this)))?|"
    r"what\s*(?:do|does|du)\s*(?:you|u|that|this|it)\s*mean|"
    r"is\s*(?:that|this|it)\s*(?:correct|right|true|sure|so)|"
    r"(?:can|could|will)\s*(?:you|u)\s*(?:please\s*)?(?:explain|elaborate|clarify)"
    r"(?:\s*(?:more|that|it|this))?|"
    r"(?:please\s*)?(?:explain|elaborate|clarify)(?:\s*(?:more|that|it|this))?|"
    r"(?:tell|give)\s*me\s*more|more\s*details?|"
    r"go\s*on|continue|and|prove\s*it|says?\s*who"
    r")[\s!.?]*$",
    re.IGNORECASE,
)


def is_followup(question: str) -> bool:
    """True for a short meta follow-up about the previous answer (e.g. "are you
    sure?", "why?", "explain more"). Caller should also confirm history exists."""
    return bool(_FOLLOWUP_RE.match(question or ""))


def rewrite_query(question: str, history: list[dict] | None) -> str:
    """Rewrite a possibly-elliptical follow-up into a standalone search query."""
    if not history:
        return question
    recent = [t for t in history if t.get("role") == "user"][-4:]
    if not recent:
        return question
    try:
        window = settings.chat_history_window
        convo = "\n".join(f"{t['role']}: {t['content']}" for t in history[-window:])
        # Tight per-call timeout: this runs *before* streaming starts, so if the
        # rewrite is slow we fall back to the original question rather than make
        # the user wait. (Handled by the except below on timeout.)
        resp = _openai().with_options(timeout=settings.rewrite_timeout).chat.completions.create(
            model=settings.rewrite_model or settings.openai_chat_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's latest message into a single, standalone "
                        "search query that resolves pronouns and references using the "
                        "conversation. If the latest message is a confirmation or "
                        "meta follow-up (e.g. 'are you sure?', 'why?', 'how?', "
                        "'explain more', 'what do you mean?'), build the query around "
                        "the SPECIFIC TOPIC of the assistant's most recent answer so "
                        "the right documents are retrieved — never output a generic "
                        "query like 'are you sure'. Keep the original language. "
                        'Respond with JSON {"query": "..."}.'
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


def generate_title(question: str) -> str:
    """Summarise a conversation's first message into a short, specific title.

    Used once per conversation. Runs on the small/fast rewrite model with a tight
    timeout and falls back to a trimmed version of the question on any failure, so
    it can never block or break the chat flow.
    """
    fallback = re.sub(r"\s+", " ", (question or "New conversation").strip())[:60]
    try:
        resp = _openai().with_options(timeout=settings.rewrite_timeout).chat.completions.create(
            model=settings.rewrite_model or settings.openai_chat_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a short, specific conversation title (3-6 words, "
                        "Title Case, no surrounding quotes, no trailing punctuation) "
                        "that summarises the user's message. Keep the original "
                        'language. Respond with JSON {"title": "..."}.'
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        title = (json.loads(resp.choices[0].message.content).get("title") or "").strip()
        return title[:60] or fallback
    except Exception:  # noqa: BLE001
        logger.exception("Title generation failed; using fallback")
        return fallback


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
    """Return {answer, answered, sources_used: list[int]}."""
    if not passages:
        return {"answer": NO_INFO_MESSAGE, "answered": False, "sources_used": []}

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Carry recent session turns so follow-ups keep context (bounded by the window).
    for turn in (history or [])[-settings.chat_history_window:]:
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
        return {
            "answer": SERVICE_BUSY_MESSAGE,
            "answered": False,
            "sources_used": [],
            "error": True,
        }

    answer = (data.get("answer") or "").strip() or NO_INFO_MESSAGE
    answered = bool(data.get("answered", False)) and answer != NO_INFO_MESSAGE
    sources_used = [int(s) for s in data.get("sources_used", []) if isinstance(s, (int, float))]
    return {
        "answer": answer,
        "answered": answered,
        "sources_used": sources_used,
    }


def _loads_json_object(raw: str) -> dict | None:
    """Parse a JSON object from a model response, tolerating stray wrapping.

    Since the streaming call no longer enforces JSON mode at the API level, the
    model could (rarely) wrap the object in markdown fences or add a stray word.
    Try a strict parse first, then fall back to the outermost {...} slice.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:  # noqa: BLE001
                return None
    return None


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


def _history_messages(history: list[dict] | None) -> list[dict]:
    return [
        {"role": t["role"], "content": t["content"]}
        for t in (history or [])[-settings.chat_history_window:]
    ]


def stream_answer(
    question: str,
    passages: list[dict],
    history: list[dict] | None = None,
    language: str | None = None,
) -> Iterator[dict]:
    """Stream a grounded answer for a (possibly first) question.

    Yields zero or more {"type":"delta",...} then exactly one {"type":"final",...}.
    """
    # Give up immediately only when there's nothing to work with at all: no
    # retrieved passages AND no prior conversation to fall back on.
    if not passages and not history:
        yield {
            "type": "final",
            "answer": NO_INFO_MESSAGE,
            "answered": False,
            "sources_used": [],
        }
        return

    lang_hint = f"\n\n(Respond in: {language})" if language else ""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_history_messages(history),
        {
            "role": "user",
            "content": f"SOURCES:\n{_build_context(passages)}\n\nQUESTION: {question}{lang_hint}",
        },
    ]
    yield from _stream_json_answer(messages)


def stream_followup(
    question: str,
    passages: list[dict],
    history: list[dict] | None = None,
    language: str | None = None,
) -> Iterator[dict]:
    """Stream an answer to a conversational follow-up about the previous answer
    ("are you sure?", "why?", "explain more"). Answers primarily from the
    CONVERSATION (retrieval is supplementary) and never short-circuits to the
    no-info message, so the assistant confirms/justifies/expands instead of
    deflecting.
    """
    lang_hint = f"\n\n(Respond in: {language})" if language else ""
    sources = _build_context(passages)
    sources_block = f"SOURCES (optional, may help):\n{sources}\n\n" if sources.strip() else ""
    messages = [
        {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
        *_history_messages(history),
        {"role": "user", "content": f"{sources_block}FOLLOW-UP: {question}{lang_hint}"},
    ]
    yield from _stream_json_answer(messages)


def _stream_json_answer(messages: list[dict]) -> Iterator[dict]:
    """Stream one grounded-answer completion from `messages`: zero or more delta
    events, then exactly one final event {answer, answered, sources_used[, error]}.
    Shared by stream_answer and stream_followup.
    """
    decoder = _AnswerFieldDecoder()
    raw_parts: list[str] = []
    shown_parts: list[str] = []
    try:
        # NOTE: we deliberately do NOT pass response_format={"type": "json_object"}
        # here. Groq buffers the *entire* response into a single chunk when JSON
        # mode is on, which defeats streaming (the answer arrives all at once).
        # The SYSTEM_PROMPT already instructs the model to emit a JSON object, so
        # we still get well-formed JSON — just streamed token-by-token, which the
        # decoder below turns into live answer deltas.
        stream = _openai().chat.completions.create(
            model=settings.openai_chat_model,
            messages=messages,
            temperature=0.1,
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
            "answer": SERVICE_BUSY_MESSAGE,
            "answered": False,
            "sources_used": [],
            "error": True,
        }
        return

    streamed = "".join(shown_parts).strip()
    data = _loads_json_object("".join(raw_parts))
    if data is None:  # fall back to the text we already streamed
        logger.warning("Could not parse streamed answer JSON; using extracted text")
        data = {}

    answer = (data.get("answer") or "").strip() or streamed or NO_INFO_MESSAGE
    answered = bool(data.get("answered", False)) and answer != NO_INFO_MESSAGE
    sources_used = [int(s) for s in data.get("sources_used", []) if isinstance(s, (int, float))]
    yield {
        "type": "final",
        "answer": answer,
        "answered": answered,
        "sources_used": sources_used,
    }
