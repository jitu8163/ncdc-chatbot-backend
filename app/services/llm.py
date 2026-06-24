"""Grounded answer generation with gpt-4o-mini.

Enforces the SOW guardrails: answer strictly from supplied NCDC passages, never
diagnose / prescribe / give clinical recommendations, stay multilingual, and emit
the exact fallback string when the context does not contain the answer.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from app.config import settings

logger = logging.getLogger(__name__)

# Transient provider failures worth one quick retry before giving up (Groq's free
# tier 429-rate-limits and occasionally times out under bursty testing).
_RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)

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


# Smalltalk with no information need, split by INTENT so each gets an appropriate
# reply instead of the same canned greeting. Matched locally (no LLM round-trip)
# and deliberately conservative: anything that isn't an obvious pleasantry falls
# through to the LLM router, so we never skip docs for a real need.
_SMALLTALK_RES: list[tuple[str, re.Pattern]] = [
    # Acknowledgements: "ok", "okay", "got it", "cool", "makes sense" — a brief
    # nod, NOT a reason to re-introduce the assistant.
    ("ack", re.compile(
        r"^\s*(?:ok(?:ay)?|kk?|cool|nice|great|awesome|good|fine|alright|"
        r"got\s*it|understood|makes\s*sense|sounds\s*good|noted|perfect|"
        r"wonderful|excellent|right|i\s*see|gotcha|thumbs\s*up|👍)[\s!.?]*$",
        re.IGNORECASE)),
    # Thanks.
    ("thanks", re.compile(
        r"^\s*(?:thanks?(?:\s*(?:you|a\s*lot|so\s*much))?|thank\s*(?:you|u)|thx|ty|"
        r"much\s*appreciated|appreciate\s*it|cheers)[\s!.?]*$",
        re.IGNORECASE)),
    # Farewells.
    ("bye", re.compile(
        r"^\s*(?:bye+|goodbye|see\s*(?:you|ya)|cya|take\s*care|"
        r"good\s*night|gn|that'?s\s*all)[\s!.?]*$",
        re.IGNORECASE)),
    # Greetings / who-are-you — the only case that warrants the full intro.
    ("greeting", re.compile(
        r"^\s*(?:hi+|hey+|hello+|hiya|yo|sup|namaste|namaskar|"
        r"good\s*(?:morning|afternoon|evening|day)|greetings|"
        r"how\s*are\s*you|who\s*(?:are\s*you|r\s*u)|what\s*(?:can\s*you\s*do|are\s*you)|"
        r"what\s*can\s*you\s*help(?:\s*with)?)[\s!.?]*$",
        re.IGNORECASE)),
]


def classify_smalltalk(question: str) -> str | None:
    """Return the smalltalk INTENT of a trivial non-question — one of "greeting",
    "ack", "thanks", "bye" — or None when the turn needs real routing
    (follow-up / knowledge lookup). Cheap local match, no LLM call."""
    q = question or ""
    for label, pattern in _SMALLTALK_RES:
        if pattern.match(q):
            return label
    return None


def classify_question(question: str) -> str:
    """Coarse route for the non-streaming graph pipeline: "chitchat" for any
    smalltalk, else "question". The streaming endpoint uses the finer
    `classify_smalltalk` + `route_query` instead."""
    return "chitchat" if classify_smalltalk(question) else "question"


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


# Referential follow-up *openers* that may carry trailing context, e.g.
# "explain that for children", "tell me more about the dosage", "what about
# pregnant women", "why is that the case for infants". Unlike _FOLLOWUP_RE these
# are not end-anchored: they point back at the previous answer, so extra words
# after the trigger don't make them a brand-new question. Kept deliberately to
# explicitly back-referential phrasings so genuine standalone questions ("why do
# measles spread?") still go through normal retrieval.
_FOLLOWUP_PREFIX_RE = re.compile(
    r"^\s*(?:"
    # confirmations, possibly trailing: "are you sure about this", "you sure?"
    r"(?:are|r)\s*(?:you|u)\s*(?:really\s*)?(?:sure|certain|serious|positive)\b|"
    r"(?:you|u)\s*(?:sure|certain)\b|"
    # "is that correct ...", "is this accurate for adults", "was it right"
    r"(?:is|are|was|were|isn'?t)\s*(?:that|this|it|these|those)\s*"
    r"(?:correct|right|true|accurate|sure|so|real)\b|"
    # explain / elaborate / clarify / expand / rephrase / summarise
    r"(?:can|could|would|will)?\s*(?:you|u)?\s*(?:please\s*)?"
    r"(?:explain|elaborate|clarify|expand|rephrase|summari[sz]e)\b|"
    # "tell me more", "give me more details"
    r"(?:tell|give)\s*(?:me\s*)?(?:some\s*)?more\b|"
    # "what about X", "how about X"
    r"(?:what|how)\s*about\b|"
    # "what do you mean ..."
    r"what\s*(?:do|does|did)\s*(?:you|that|this|it)\s*mean\b|"
    # referential why/how: "why is that ...", "how does this work"
    r"(?:why|how)\s*(?:is|are|was|were|does|do|did|come|so)?\s*"
    r"(?:that|this|it|these|those|they)\b|"
    # challenges to the prior answer
    r"says?\s*who\b|prove\s*it\b|according\s*to\s*(?:what|whom|who)\b"
    r")",
    re.IGNORECASE,
)


def is_followup(question: str) -> bool:
    """True for a meta follow-up about the previous answer — either a bare phrase
    ("are you sure?", "why?", "explain more") or a back-referential opener that
    carries extra context ("explain that for children", "what about infants?").
    Caller should also confirm history exists."""
    q = question or ""
    return bool(_FOLLOWUP_RE.match(q) or _FOLLOWUP_PREFIX_RE.match(q))


_VALID_ROUTES = {"smalltalk", "followup", "knowledge"}

_ROUTER_SYSTEM_PROMPT = (
    "You are the query router for the NCDC Guideline Assistant — a RAG chatbot that "
    "answers questions about National Centre for Disease Control (NCDC) guideline "
    "documents. Decide how the user's LATEST message should be handled, using the "
    "conversation for context, and return STRICT JSON.\n\n"
    "Choose exactly one route:\n"
    "- \"knowledge\": a NEW question whose answer must be looked up in the guideline "
    "documents (the vector database). This is the default for any genuine "
    "information need on a fresh topic.\n"
    "- \"followup\": a message about the assistant's PREVIOUS answer — it asks to "
    "confirm, justify, clarify, challenge, translate or expand what was just said, "
    "rather than introducing a new topic. It is answered from the CONVERSATION (the "
    "last reference), NOT by a fresh document lookup. Examples: 'are you sure?', "
    "'are you sure about this?', 'why?', 'how so?', 'is that correct?', 'really?', "
    "'explain more', 'what do you mean?', 'tell me more', 'what about children?', "
    "'and for adults?', 'says who?', 'can you simplify that?'.\n"
    "- \"smalltalk\": a greeting, thanks, acknowledgement or farewell with no "
    "information need ('hi', 'ok', 'thanks', 'great', 'bye').\n\n"
    "Also produce \"query\": a single standalone search query for the knowledge "
    "base, with pronouns/references resolved from the conversation. For a followup, "
    "build it around the SPECIFIC TOPIC of the assistant's most recent answer — "
    "never a generic query like 'are you sure'. For smalltalk, repeat the message. "
    "Keep the original language.\n\n"
    'Respond ONLY with JSON: {"route": "knowledge|followup|smalltalk", "query": "..."}.'
)


def route_query(question: str, history: list[dict] | None) -> dict:
    """Route a turn and produce a standalone search query.

    Returns ``{"route": "smalltalk"|"followup"|"knowledge", "query": str}``.

    A fast local pass settles the easy cases with no LLM call (obvious smalltalk;
    a first turn that isn't smalltalk is always a knowledge lookup). Otherwise one
    small/fast LLM call — the same rewrite call we already make — both classifies
    the turn AND rewrites the query, so there is no extra round-trip. The reliable
    local follow-up regex is OR'd in as a safety net, and we fall back to it on any
    error. The tight `rewrite_timeout` keeps this from delaying time-to-first-token.
    """
    q = (question or "").strip()

    smalltalk = classify_smalltalk(q)
    if smalltalk:
        return {"route": "smalltalk", "subtype": smalltalk, "query": q}

    regex_followup = bool(history) and is_followup(q)
    if not history:
        # Nothing to follow up on — a real message on a fresh session is a lookup.
        return {"route": "knowledge", "query": q}

    window = settings.chat_history_window
    convo = "\n".join(f"{t['role']}: {t['content']}" for t in history[-window:])
    try:
        resp = _openai().with_options(timeout=settings.rewrite_timeout).chat.completions.create(
            model=settings.rewrite_model or settings.openai_chat_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Conversation:\n{convo}\n\nLatest message: {question}",
                },
            ],
        )
        data = json.loads(resp.choices[0].message.content) or {}
        route = str(data.get("route", "")).strip().lower()
        query = (data.get("query") or "").strip() or q
        if route not in _VALID_ROUTES:
            route = "followup" if regex_followup else "knowledge"
        # Trust the deterministic follow-up signal over a "knowledge" misroute:
        # the regex only fires on clearly back-referential phrasing.
        if regex_followup and route == "knowledge":
            route = "followup"
        return {"route": route, "query": query}
    except Exception:  # noqa: BLE001
        logger.exception("Query routing failed; falling back to regex signal")
        return {"route": "followup" if regex_followup else "knowledge", "query": q}


def rewrite_query(question: str, history: list[dict] | None) -> str:
    """Rewrite a possibly-elliptical follow-up into a standalone search query.

    Thin wrapper over `route_query` for the non-streaming graph pipeline, which
    only needs the rewritten query.
    """
    return route_query(question, history)["query"]


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


# Intent-specific smalltalk replies. Only a genuine greeting gets the full
# self-introduction; acknowledgements ("ok"), thanks and farewells get a short,
# natural reply so the assistant doesn't re-introduce itself every turn.
GREETING_REPLY = (
    "Hello! I'm the NCDC Guideline Assistant. Ask me anything about the NCDC "
    "guideline documents and I'll answer with citations to the source."
)
ACK_REPLY = "Sure — let me know if there's anything else you'd like to ask about the NCDC guidelines."
THANKS_REPLY = "You're welcome! Feel free to ask if you have any more questions about the NCDC guidelines."
BYE_REPLY = "Goodbye! Come back anytime you have questions about the NCDC guidelines."

_SMALLTALK_REPLIES = {
    "greeting": GREETING_REPLY,
    "ack": ACK_REPLY,
    "thanks": THANKS_REPLY,
    "bye": BYE_REPLY,
}

# Backwards-compatible alias for the non-streaming graph's chitchat node.
CHITCHAT_REPLY = GREETING_REPLY


def smalltalk_reply(subtype: str | None) -> str:
    """Pick the reply for a smalltalk intent ("greeting"/"ack"/"thanks"/"bye").
    Falls back to the acknowledgement reply for an unknown/None subtype."""
    return _SMALLTALK_REPLIES.get(subtype or "", ACK_REPLY)


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
    passages: list[dict] | None = None,
    history: list[dict] | None = None,
    language: str | None = None,
) -> Iterator[dict]:
    """Answer a conversational follow-up about the previous answer ("are you
    sure?", "why?", "explain more") from the CONVERSATION, and never short-circuit
    to the no-info message — the assistant confirms/justifies/expands instead of
    deflecting.

    Unlike `stream_answer` this uses a single non-streaming JSON-mode completion:
    follow-up answers are short and the small chat model emits far more reliable
    JSON when it isn't also being asked to stream. The caller (chat router) paces
    the returned answer out word-by-word, so the UI still types it in. Yields
    exactly one {"type": "final", ...} event.
    """
    lang_hint = f"\n\n(Respond in: {language})" if language else ""
    sources = _build_context(passages or [])
    sources_block = f"SOURCES (optional, may help):\n{sources}\n\n" if sources.strip() else ""
    messages = [
        {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
        *_history_messages(history),
        {"role": "user", "content": f"{sources_block}FOLLOW-UP: {question}{lang_hint}"},
    ]
    try:
        resp = _create_chat_completion(messages, json_mode=True)
        data = _loads_json_object(resp.choices[0].message.content or "") or {}
    except Exception:  # noqa: BLE001
        logger.exception("Follow-up generation failed")
        yield {
            "type": "final",
            "answer": SERVICE_BUSY_MESSAGE,
            "answered": False,
            "sources_used": [],
            "error": True,
        }
        return

    answer = (data.get("answer") or "").strip()
    # A follow-up should never deflect with the no-info string — it's about the
    # conversation we already have. If the model emitted it (or nothing), fall back
    # to a graceful re-engagement rather than blaming the documents.
    if not answer or answer == NO_INFO_MESSAGE:
        answer = (
            "Let me clarify based on what I shared above — could you tell me which "
            "part you'd like me to confirm or expand on?"
        )
        answered = False
    else:
        answered = bool(data.get("answered", True))
    sources_used = [int(s) for s in data.get("sources_used", []) if isinstance(s, (int, float))]
    yield {
        "type": "final",
        "answer": answer,
        "answered": answered,
        "sources_used": sources_used,
    }


def _create_chat_completion(messages: list[dict], *, json_mode: bool):
    """Make a non-streaming chat completion, retrying once on a transient provider
    error. Returns the SDK response. Used where reliable parsing matters more than
    token-by-token streaming (e.g. short follow-up answers)."""
    last_exc: Exception | None = None
    kwargs: dict = {"model": settings.openai_chat_model, "messages": messages, "temperature": 0.1}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(2):
        try:
            return _openai().chat.completions.create(**kwargs)
        except _RETRYABLE_ERRORS as exc:
            last_exc = exc
            if attempt == 0:
                logger.warning("LLM call transient error (%s); retrying once", type(exc).__name__)
                time.sleep(0.8)
    raise last_exc  # type: ignore[misc]


def _create_chat_stream(messages: list[dict]):
    """Open a streaming chat completion, retrying once on a transient provider
    error (rate limit / timeout / connection / 5xx).

    The retry happens before any token is consumed, so it can't duplicate output.
    A short backoff gives a rate-limit window a moment to reset; the per-call
    timeout still bounds the total wait.
    """
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            return _openai().chat.completions.create(
                model=settings.openai_chat_model,
                messages=messages,
                temperature=0.1,
                stream=True,
            )
        except _RETRYABLE_ERRORS as exc:
            last_exc = exc
            if attempt == 0:
                logger.warning("LLM stream transient error (%s); retrying once", type(exc).__name__)
                time.sleep(0.8)
    raise last_exc  # type: ignore[misc]


def _stream_json_answer(messages: list[dict]) -> Iterator[dict]:
    """Stream one grounded-answer completion from `messages`: zero or more delta
    events, then exactly one final event {answer, answered, sources_used[, error]}.
    Used by stream_answer (the grounded-answer path).
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
        stream = _create_chat_stream(messages)
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
