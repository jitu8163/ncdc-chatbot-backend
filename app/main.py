"""FastAPI application entrypoint for the NCDC Guideline Chatbot backend."""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from app.config import settings
from app.database import Base, SessionLocal, engine
from app.models import User, UserRole
from app.routers import analytics, auth, chat, documents
from app.security import hash_password
from app.services import embeddings, ocr, qdrant_service, reranker

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ncdc")

# Route uvicorn's own loggers through the root handler so its access/error lines get
# the same timestamped format instead of the bare "INFO: ..." default.
for _name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
    _lg = logging.getLogger(_name)
    _lg.handlers.clear()
    _lg.propagate = True


def _ensure_schema() -> None:
    """Apply the few additive columns that `create_all` can't add to pre-existing
    tables. Idempotent and best-effort: checks the live schema first and only issues
    an ADD COLUMN when the column is missing, so it's safe to run on every startup
    against SQLite (dev) or PostgreSQL (prod) without a migration tool."""
    # (table, column, DDL type) — append future additive columns here.
    additions = [("chat_sessions", "state_json", "TEXT")]
    inspector = inspect(engine)
    for table, column, ddl_type in additions:
        try:
            existing = {c["name"] for c in inspector.get_columns(table)}
        except Exception:  # noqa: BLE001 - table not created yet / inspect failed
            continue
        if column in existing:
            continue
        try:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))
            logger.info("Schema: added %s.%s", table, column)
        except Exception:  # noqa: BLE001 - another worker may have added it first
            logger.exception("Schema: could not add %s.%s (continuing)", table, column)


def _seed_first_admin() -> None:
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            db.add(
                User(
                    email=settings.first_admin_email,
                    full_name="Administrator",
                    role=UserRole.admin,
                    hashed_password=hash_password(settings.first_admin_password),
                )
            )
            db.commit()
            logger.info("Seeded first admin: %s", settings.first_admin_email)
    finally:
        db.close()


def _warmup_models() -> None:
    """Pre-load the local embedding model off the request path so the first user
    query doesn't pay the (slow) cold-load cost. Runs in a background thread;
    failures are non-fatal (the model loads lazily on first use anyway)."""
    try:
        embeddings.warmup()
        logger.info("Embedding model warmed up")
    except Exception:  # noqa: BLE001
        logger.exception("Model warmup failed (will load lazily on first query)")
    try:
        ocr.warmup()
        logger.info("OCR model warmed up")
    except Exception:  # noqa: BLE001
        logger.exception("OCR warmup failed (will load lazily on first scanned page)")
    try:
        reranker.warmup()
        logger.info("Reranker model warmed up")
    except Exception:  # noqa: BLE001
        logger.exception("Reranker warmup failed (will load lazily on first query)")


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    _ensure_schema()
    _seed_first_admin()
    try:
        qdrant_service.ensure_collection()
    except Exception:  # noqa: BLE001 - don't block startup if Qdrant is briefly down
        logger.exception("Could not ensure Qdrant collection on startup")
    # Warm models in the background so startup stays fast but the first query is warm.
    threading.Thread(target=_warmup_models, name="model-warmup", daemon=True).start()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="RAG chatbot over NCDC guideline documents (Groq LLM + Qdrant).",
    lifespan=lifespan,
)

# CORS — origins come from settings.cors_origins (env: CORS_ORIGINS). Defaults to
# "*" for local dev; set to the deployed frontend domain(s) in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(analytics.router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "app": settings.app_name, "environment": settings.environment}
