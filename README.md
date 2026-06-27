# NCDC Guideline Chatbot — Backend

A web-based, **RAG (Retrieval-Augmented Generation)** chatbot that answers questions
**strictly** from uploaded NCDC guideline PDFs, with **verifiable citations** (document,
page, section) and **clickable source links** — per the project SOW.

- **Framework:** FastAPI (Python 3.11–3.13)
- **Orchestration:** LangGraph (classify → rewrite → retrieve → generate → cite → format)
- **Answer model:** Google `gemini-2.5-flash` (via its OpenAI-compatible API)
- **Embeddings:** `BAAI/bge-m3` (dense, 1024-dim, local) + BM25 (sparse, FastEmbed)
- **Vector DB:** Qdrant (native hybrid search, RRF fusion)
- **Reranker:** `BAAI/bge-reranker-v2-m3` multilingual cross-encoder (sentence-transformers)
- **Relational DB:** PostgreSQL (SQLite supported for dev)
- **Cache / rate limiting:** Redis

## Why hybrid search + reranking (design decision)

The SOW demands answers grounded *only* in the documents, with correct citations.
That is a **high-precision retrieval** problem, so the pipeline is multi-stage:

1. **Hybrid recall** — dense vectors (`bge-m3`, semantic) **+** BM25 sparse vectors (exact
   terms: drug names, dosages, disease names, section numbers), fused with Reciprocal Rank
   Fusion inside Qdrant. Pulls ~`RETRIEVE_TOP_K` (40) **child** candidates.
2. **Cross-encoder rerank** — `bge-reranker-v2-m3` re-scores each (question, passage) pair.
   This is the biggest lever on citation correctness.
3. **Parent expansion** — surviving children are expanded to their larger **parent** blocks
   and deduplicated, so the LLM sees `RERANK_TOP_K` (8) coherent context passages while
   citations stay pinned to the precise child page/section.

Both retrieval arms are toggleable (`USE_HYBRID_SEARCH`, `USE_RERANKER`).

## Orchestration (LangGraph)

The chat turn runs as a compiled LangGraph (`app/graph/`):

```
START → classify ─(chitchat)→ END
              └──(question)→ rewrite → retrieve → generate → citations → format → END
```

- **classify** — Question Classifier: greetings/smalltalk short-circuit to a canned reply,
  skipping retrieval and the LLM call.
- **rewrite** — Query Rewriter: resolves follow-up pronouns/references against the session
  history into a standalone search query (big win for multi-turn retrieval).
- **retrieve** — hybrid search + rerank + parent expansion (cached in Redis).
- **generate / citations / format** — grounded answer, citation mapping, response shaping.

## Caching & rate limiting (Redis)

Redis is **best-effort** — if it is down, every call is a cache miss and the request still
succeeds. It backs: query-embedding cache, retrieval-result cache, grounded-answer cache,
and a per-IP fixed-window rate limiter on `POST /api/chat` (`RATE_LIMIT_PER_MINUTE`).

## Scaling design (1000-page docs × 100 uploads)

- **Streaming extraction** — pages are read one at a time (PyMuPDF), never the whole PDF
  in memory.
- **Parent-child chunking** — small child chunks are embedded/searched; their larger parent
  block is fed to the LLM. Both stay within one page → exact citations.
- **Batched embedding + indexing** — chunks are embedded/upserted in batches
  (`EMBED_BATCH_SIZE`), so a 1000-page document indexes incrementally.
- **Background ingestion** — uploads return immediately; indexing runs in the background
  with status tracked on the document (`pending → processing → indexed/failed`).
- **Filtered vector search** — Qdrant payload indexes on `document_id`, `enabled`,
  `category`. Enable/disable and delete are metadata/point ops — **no re-embedding**.
- For heavier throughput, swap FastAPI `BackgroundTasks` for Celery/RQ driving the same
  `ingestion.process_document(document_id)`.

## Quick start

```bash
# 1. Start Qdrant + PostgreSQL + Redis (or use your own)
docker compose up -d

# 2. Configure
cp .env.example .env          # then edit: GEMINI_API_KEY, SECRET_KEY, DB creds, admin pwd

# 3. Install deps (uv) and run
uv sync
uv run python main.py         # http://localhost:8000  | docs: /docs
```

> First run creates the DB tables, seeds the admin from `FIRST_ADMIN_*`, and ensures the
> Qdrant collection. The `bge-m3`, `bge-reranker-v2-m3` and BM25 models download on first
> use (cached locally) — the first ingest/query is slow while they warm up. Set
> `EMBED_DEVICE=cuda` if a GPU is available.

### Dev without PostgreSQL
Set `DATABASE_URL=sqlite:///./ncdc_chatbot.db` in `.env`. Redis is optional everywhere —
without it, caching and rate limiting are simply disabled.

## API overview

| Area | Endpoint | Auth |
|------|----------|------|
| Login | `POST /api/auth/login` | public |
| Current user | `GET /api/auth/me` | bearer |
| Manage users | `POST/GET /api/auth/users` | admin |
| Upload document | `POST /api/documents` (multipart) | admin |
| List / get / update / delete | `/api/documents…` | admin |
| Enable/disable / re-categorize | `PATCH /api/documents/{id}` | admin |
| Re-index | `POST /api/documents/{id}/reindex` | admin |
| View source PDF (citation link) | `GET /api/documents/{id}/view#page=N` | public |
| **Ask a question** | `POST /api/chat` | public |
| Session history | `GET /api/chat/sessions/{id}` | public |
| Feedback (👍/👎) | `POST /api/chat/feedback` | public |
| Analytics dashboard | `GET /api/analytics/summary` | admin |
| Audit logs | `GET /api/analytics/audit-logs` | admin |

### Ask example
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the recommended management for a category III dog bite?"}'
```
Response includes `answer`, `answered`, `citations[]` (document, page, section, snippet,
clickable `url`), and `followups[]`. When the docs lack the answer, the bot returns the
mandated string: *"Relevant information could not be found in the available NCDC guideline
documents."*

## Project layout
```
app/
  main.py            FastAPI app, startup (tables, admin seed, Qdrant)
  config.py          settings (.env)
  database.py        SQLAlchemy engine/session
  models.py          users, documents, sessions, messages, feedback, query_logs, audit
  schemas.py         Pydantic request/response models
  security.py        JWT + bcrypt
  deps.py            auth dependencies (current user / admin guard)
  routers/           auth, documents, chat, analytics
  graph/
    state.py         LangGraph shared state (TypedDict)
    pipeline.py      compiled graph: classify→rewrite→retrieve→generate→cite→format
  services/
    pdf_processor.py extraction (streaming, page + section detection)
    chunking.py      structure-aware parent-child chunks
    embeddings.py    dense (bge-m3) + sparse (BM25)
    qdrant_service.py hybrid search, upsert, enable/disable, delete
    reranker.py      bge-reranker-v2-m3 cross-encoder
    retrieval.py     hybrid recall -> rerank -> parent expansion (cached)
    llm.py           grounded Gemini answer + classifier + query rewriter
    cache.py         Redis cache + rate limiting (best-effort)
    ingestion.py     extract -> chunk -> embed -> index (background)
```

## SOW module coverage
- **2.1 UI** — backend APIs for chat, sessions, follow-ups, feedback, copy. UI is the
  separate React app in [`../Frontend`](../Frontend).
- **2.2 Knowledge Base** — upload/replace/enable-disable/categorize/version CRUD.
- **2.3 Processing & Indexing** — PDF extract, chunk, embed, re-index.
- **2.4 AI QA** — grounded answers, multilingual, follow-ups, session context, guardrails.
- **2.5 Citations** — document name, page, section, clickable hyperlink.
- **2.6 Analytics** — totals, daily usage, FAQ, most-accessed docs, feedback summary.
- **2.7 Administration** — document management, audit logs, settings via `.env`.

## Notes / next steps
- Frontend is the separate React + Vite app in [`../Frontend`](../Frontend); these APIs
  are CORS-enabled. This backend is API-only and no longer serves any static UI.
- Add Alembic migrations before production (tables are auto-created for dev convenience).
- Scanned/image PDFs need OCR (not in scope) — ingestion fails clearly if no text found.
