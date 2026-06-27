# Context-Aware Hybrid RAG — Refactor Notes

This document describes the refactor of the NCDC chatbot's query routing and
conversational follow-up handling from a "send (almost) everything to the vector
DB" design into a **context-aware DIRECT / MEMORY / RAG** workflow.

The existing architecture — Qdrant vector store, FastEmbed embeddings, two-stage
retrieval (dense + cross-encoder rerank), FastAPI endpoints, SQLAlchemy schema,
streaming NDJSON protocol and the React frontend — is **preserved**. The changes
are additive and centred on a new routing "brain".

---

## 1. Architecture

```
                            ┌─────────────────────────┐
        User Query  ─────►  │   Context Analyzer      │   (1 fast LLM call +
   (+ history, active topic,│   - route: DIRECT/MEMORY│    deterministic
    last retrieved context) │           /RAG          │    fast-paths/fallbacks)
                            │   - active_topic (sticky)│
                            │   - rewritten_query     │
                            └───────────┬─────────────┘
                                        │ route
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
        ┌──────────┐            ┌──────────────┐           ┌──────────────┐
        │  DIRECT  │            │    MEMORY    │           │     RAG      │
        │ canned   │            │ answer from  │           │ rewrite →    │
        │ reply,   │            │ last context │           │ retrieve →   │
        │ no LLM   │            │ + history    │           │ rerank →     │
        └────┬─────┘            └──────┬───────┘           │ guardrail →  │
             │                         │                   │ generate     │
             │             answered ◄──┤                   └──────┬───────┘
             │                         │ needs_retrieval          │
             │                         └─────────►  (RAG path) ────┤
             │                                                     │
             ▼                                                     ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │ Citations  →  Conversation-state update  →  Final Response        │
        └─────────────────────────────────────────────────────────────────┘
```

### LangGraph nodes (`app/graph/pipeline.py`)

| Node        | Role                                                                 |
|-------------|----------------------------------------------------------------------|
| `analyze`   | **Context Analyzer** — route + active topic + standalone rewrite      |
| `direct`    | DIRECT reply (greeting/thanks/ack/bye), no retrieval, no LLM          |
| `memory`    | MEMORY — answer from remembered context, or signal `needs_retrieval`  |
| `rewrite`   | **Query Rewriter** — promote the standalone query into the search field |
| `retrieve`  | Dense vector search + cross-encoder rerank (existing pipeline)        |
| `generate`  | Guardrailed answer generation (existing grounded LLM call)            |
| `citations` | Citation Engine (existing, unchanged)                                 |

Conditional edges: `analyze → {direct | memory | rewrite}`, and
`memory → {rewrite (retrieve) | citations (answered)}`.

### Why both chat paths share one brain

The non-streaming `POST /api/chat` runs the LangGraph; the streaming
`POST /api/chat/stream` (used by the frontend) builds its own loop to emit token
deltas. Previously they routed **differently**, which is how the follow-up bug
survived on the streaming path. Both now call the same
`app/services/conversation.py` functions (`analyze`, `answer_from_memory`,
`is_weak_retrieval`, `update_after_*`), so they cannot drift apart.

---

## 2. Modified / new files

| File | Change |
|------|--------|
| `app/services/conversation.py` | **New.** ConversationState, Context Analyzer (`analyze`), MEMORY answering, retrieval guardrails, state updates, structured `log_turn`. |
| `app/services/llm.py` | Added `analyze_context_llm` (enriched router: route + active_topic + rewritten_query + reason) and `answer_from_context` (MEMORY answer with `needs_retrieval`). Broadened greeting regex ("hi there"). Existing functions kept for back-compat. |
| `app/graph/state.py` | Extended `ChatState` with `route`, `reason`, `active_topic`, `rewritten_query`, `needs_retrieval`, `weak_retrieval`, `conv_state`. |
| `app/graph/pipeline.py` | Rebuilt graph into `analyze → DIRECT/MEMORY/RAG` with conditional edges + guardrail; `run_chat` now threads ConversationState in/out. |
| `app/routers/chat.py` | Both endpoints load/save ConversationState and route via `conversation.analyze`; streaming path implements MEMORY→RAG fallback + observability. |
| `app/models.py` | Added nullable `ChatSession.state_json` (persisted ConversationState). |
| `app/main.py` | Idempotent `_ensure_schema()` adds `state_json` to pre-existing tables (no migration tool needed). |
| `app/config.py` | Added `memory_context_passages` and `retrieval_min_score` (guardrail floor). |

No endpoints, request/response schemas, the Qdrant collection, the embedding /
rerank models, or the frontend contract were changed.

---

## 3. Routing logic

`conversation.analyze(question, state, history)` returns
`{route, reason, rewritten_query, active_topic}`:

1. **Deterministic fast-paths (no LLM):** obvious smalltalk → `DIRECT`; a first
   turn that isn't smalltalk → `RAG` (nothing to follow up on yet).
2. **Otherwise one fast LLM call** (`analyze_context_llm`, on the small/fast
   rewrite model, tight timeout) classifies the turn, keeps/refreshes the active
   topic and rewrites a standalone query — all in one round-trip.
3. **Safety nets:** the deterministic follow-up regex overrides a `RAG` misroute
   to `MEMORY` for clearly back-referential phrasing; on any LLM error we fall
   back to the regex signal (follow-up → MEMORY, else RAG).

- **DIRECT** — greetings, thanks, acknowledgements, farewells. Bypasses
  retrieval and the LLM (canned, intent-appropriate reply).
- **MEMORY** — depends on the conversation: meta questions about the last answer
  ("are you sure?", "why?", "explain that") *and* elliptical follow-ups for more
  detail on the active topic ("how many deaths?", "when was that reported?").
  Answered from the previously retrieved context + history first; if that's
  insufficient, it **falls through to RAG** with the rewritten query.
- **RAG** — new informational questions / new topics: rewrite → retrieve →
  rerank → guardrail → generate.

---

## 4. Conversation state management

`ConversationState` (persisted as JSON in `ChatSession.state_json`):

```json
{
  "active_topic": "dengue",
  "last_rag_query": "How many dengue deaths were reported?",
  "last_rag_context": [ { "document_id": "...", "page": 3, "text": "...", "rerank_score": 6.1 } ],
  "last_answer": "Dengue is a viral disease ...",
  "conversation_summary": "Topic: dengue. Last asked: how many deaths?"
}
```

- Loaded per request (`load_state`), threaded through the graph / streaming loop,
  and saved back (`dump_state`) on the session row inside the same DB transaction
  that records the turn — so it survives across requests and server restarts.
- `last_rag_context` is trimmed to the top `memory_context_passages` passages
  (and the fields MEMORY needs) to keep the row and the prompt small.
- Updates are route-aware: `update_after_rag` refreshes topic + remembered
  context after a retrieval turn; `update_after_simple` preserves topic/context
  for DIRECT and MEMORY turns.

---

## 5. Active topic tracking

- The active topic is **sticky**: MEMORY and DIRECT turns never overwrite it
  (enforced both in the analyzer prompt and defensively in `analyze`), so
  "Are you sure?" or "How many deaths?" keep the topic as *dengue*.
- It is refreshed only on a genuine new lookup (`update_after_rag`).
- Worked example:
  - `"Tell me about dengue"` → RAG → `active_topic = "dengue"`.
  - `"Are you sure?"` → MEMORY → topic unchanged (`dengue`).
  - `"How many deaths?"` → MEMORY → topic unchanged; rewritten to
    `"How many dengue deaths were reported?"`.

---

## 6. Query rewriting

- Performed inside the analyzer call (no extra round-trip): the model resolves
  pronouns/ellipsis against the active topic and conversation to emit one
  standalone search query.
- `"How many deaths?"` + topic *dengue* → `"How many dengue deaths were reported?"`
- `"When was that published?"` + topic *dengue* → `"When was the dengue report published?"`
- The graph's `rewrite` node simply promotes this into the retrieval field.
- **Fallback** (analyzer LLM unavailable): a short referential follow-up is
  prefixed with the active topic (`dengue how many deaths?`) so retrieval still
  has context.

---

## 7. Retrieval guardrails (no fabrication)

- After rerank, `is_weak_retrieval(passages)` flags empty results or a best
  cross-encoder score below `retrieval_min_score` (default `-8.0`).
- On a weak/empty retrieval the `generate` step is **skipped** — the user gets
  "Relevant information could not be found …" instead of a fabricated answer
  (also saves an LLM call).
- Verified separation: real in-corpus queries score **+5.6 … +8.0**; an
  irrelevant input ("hi there") scored **−8.85**. Raise the floor (e.g. `-2.0`)
  to be stricter.

---

## 8. Observability

One structured line per turn (`ncdc.conversation`), e.g.:

```
route=RAG topic='dengue' reason='first-turn knowledge lookup'
rewritten='How many dengue deaths were reported?' retrieved=6
best_score=6.10 scores=[6.1, 5.8, ...] answered=True sources=[1,3]
docs=['Dengue-Guidelines#p3', ...] session=<id>
```

Covers route, active topic, rewritten query, retrieval scores and the final
source documents used — the fields needed to debug routing decisions.

---

## 9. Example conversations (new behaviour)

**A. The original failure, now fixed**

```
User:  Tell me about dengue.
Bot:   [RAG answer about dengue]                 route=RAG     topic=dengue
User:  Are you sure?
Bot:   [confirms/justifies from memory]          route=MEMORY  topic=dengue
User:  How many deaths?
Bot:   Dengue deaths reported: …                 route=MEMORY→RAG
       (rewritten: "How many dengue deaths were reported?")
```
Previously the last turn became a standalone `"How many deaths?"` lookup and
returned *"No documents found"*. Now the topic is carried and the query rewritten.

**B. DIRECT bypass**

```
User:  Hi there            → DIRECT (no retrieval, no LLM)  → greeting reply
User:  Thanks!             → DIRECT                          → "You're welcome…"
```

**C. MEMORY answered from context (no new retrieval)**

```
User:  What is leptospirosis?      → RAG     (stores context as memory)
User:  Can you explain that more?  → MEMORY  (answered from remembered context;
                                              citations resolve to stored passages)
```

**D. Guardrail (no fabrication)**

```
User:  <off-topic / not in the corpus>  → RAG → weak retrieval (best_score < floor)
Bot:   Relevant information could not be found in the available NCDC guideline documents.
```

---

## Notes / limitations

- When the analyzer LLM is *fully unavailable* (e.g. provider outage), routing
  degrades to the deterministic regex + topic-prefix fallback; a brand-new
  elliptical opener may not be topic-resolved until the LLM is reachable again.
- `conversation_summary` is a cheap rolling gist (no extra LLM call); swap in a
  periodic summarization call if richer long-context memory is needed.
- The DIRECT/MEMORY/RAG split and the rewriter still depend on the LLM provider;
  see `docs`/`.env` for the active provider and its rate limits.
```
