"""Application configuration loaded from environment / .env file."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # App
    app_name: str = "NCDC Guideline Chatbot"
    environment: str = "development"
    log_level: str = "INFO"
    public_base_url: str = "http://localhost:8000"

    # Security
    secret_key: str = "change-me"
    access_token_expire_minutes: int = 480
    first_admin_email: str = "admin@ncdc.local"
    first_admin_password: str = "change-me-now"

    # Relational DB (PostgreSQL in prod; sqlite for dev convenience)
    database_url: str = "sqlite:///./ncdc_chatbot.db"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "ncdc_documents"
    # Per-operation timeout (seconds). The default qdrant-client REST timeout is
    # short; a remote (e.g. Qdrant Cloud) cluster with high round-trip latency can
    # make `wait=True` upserts during ingestion exceed it. Keep this generous so
    # indexing writes don't time out — searches still return quickly.
    qdrant_timeout: float = 60.0

    # Redis (cache + rate limiting)
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    embedding_cache_ttl: int = 604800   # 7 days — query embeddings are stable
    retrieval_cache_ttl: int = 3600     # 1 hour — re-index invalidates via version bump
    answer_cache_ttl: int = 3600        # 1 hour — cached grounded answers
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 30     # /api/chat requests per client IP per minute

    # Answer LLM (OpenAI-compatible Chat Completions API).
    # Default points at Groq for sub-second generation; set OPENAI_API_KEY to your
    # Groq key. Leave openai_base_url empty to use OpenAI (gpt-4o-mini) instead.
    openai_api_key: str = ""
    openai_base_url: str | None = "https://api.groq.com/openai/v1"
    openai_chat_model: str = "llama-3.1-8b-instant"
    # Follow-up query rewrite runs *before* streaming starts, so it uses a small,
    # fast model (the task is trivial) to keep time-to-first-token low. Falls back
    # to openai_chat_model if left blank.
    rewrite_model: str = "llama-3.1-8b-instant"
    # Hard timeouts so a stalled provider can never blow the latency budget.
    # llm_request_timeout caps any single LLM call (the SDK default is 600s);
    # rewrite_timeout is a tighter cap on the pre-stream rewrite, which falls back
    # to the original question if the model is slow.
    llm_request_timeout: float = 12.0
    rewrite_timeout: float = 1.5

    # Multi-turn memory: how many of the most recent conversation messages (user +
    # assistant turns combined) are loaded as context for follow-up rewriting and
    # answer generation. ~16 ≈ the last 8 exchanges — enough for "Why?"/"Are you
    # sure?" follow-ups without bloating the prompt or latency.
    chat_history_window: int = 16

    # Paced streaming: artificial delay (seconds) inserted between word chunks sent
    # to the client so the answer is visibly typed out rather than appearing at the
    # model's full speed. ~0.04s ≈ a steady "medium" reveal. Set to 0 to disable.
    stream_word_delay: float = 0.04

    # Embeddings — small multilingual ONNX bi-encoder via FastEmbed (CPU-friendly).
    # 384-dim. Changing this model requires re-indexing the corpus.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    embed_device: str = "cpu"           # unused by the ONNX path; kept for compatibility

    # Retrieval — dense vector search, then a cross-encoder reranker sharpens the
    # final ordering. retrieve_top_k is the wide candidate set fed to the reranker;
    # final_top_k is what survives and goes to the LLM.
    retrieve_top_k: int = 30   # candidates pulled from Qdrant (reranker input)
    final_top_k: int = 6       # passages kept and sent to the LLM

    # Cross-encoder reranking. Scores each (query, passage) pair jointly — far more
    # accurate than raw vector distance. Runs on CPU via FastEmbed ONNX in ~tens of
    # ms for ~30 candidates, negligible next to LLM generation. Set
    # reranker_enabled=false to fall back to plain dense order.
    reranker_enabled: bool = True
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # PDF extraction
    # Tables are pulled out as markdown (PyMuPDF find_tables) so the row/column
    # structure survives chunking.
    extract_tables: bool = True

    # OCR fallback for scanned / image-only pages (RapidOCR — ONNX on CPU, so no
    # system Tesseract binary is required). When a page's text layer has fewer than
    # ocr_min_chars characters, the page is rendered at ocr_dpi and OCR'd. OCR is
    # ~5-10x slower than text extraction, so it only runs on pages that need it.
    ocr_enabled: bool = True
    ocr_dpi: int = 200          # render resolution; higher = better accuracy, slower
    ocr_min_chars: int = 80     # text-layer length below which a page is OCR'd

    # Ingestion / chunking (MVP: single-level fixed-size chunks)
    # Each chunk is embedded + searched and fed to the LLM as-is. Chunks never cross
    # a page boundary, so every chunk maps to exactly one page → exact citations.
    chunk_tokens: int = 500
    chunk_overlap_tokens: int = 60
    embed_batch_size: int = 64
    upload_dir: str = "./storage/documents"
    max_upload_mb: int = 200


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
