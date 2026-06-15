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

    # Redis (cache + rate limiting)
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    embedding_cache_ttl: int = 604800   # 7 days — query embeddings are stable
    retrieval_cache_ttl: int = 3600     # 1 hour — re-index invalidates via version bump
    answer_cache_ttl: int = 3600        # 1 hour — cached grounded answers
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 30     # /api/chat requests per client IP per minute

    # Answer LLM (OpenAI-compatible Chat Completions API).
    # Leave openai_base_url empty for OpenAI itself; set it to an OpenAI-compatible
    # endpoint (e.g. Groq: https://api.groq.com/openai/v1) to use another provider.
    openai_api_key: str = ""
    openai_base_url: str | None = None
    openai_chat_model: str = "gpt-4o-mini"

    # Embeddings — BAAI/bge-m3 dense (local, multilingual) + BM25 sparse
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embed_device: str = "cpu"           # "cuda" if a GPU is available

    # Retrieval
    use_hybrid_search: bool = True
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    retrieve_top_k: int = 40
    rerank_top_k: int = 8
    min_rerank_score: float = 0.0

    # Ingestion / parent-child chunking
    # Children are embedded + searched (precise match); the larger parent block
    # is fed to the LLM (richer context). Both stay within one page → exact citations.
    parent_chunk_tokens: int = 1200
    child_chunk_tokens: int = 350
    child_overlap_tokens: int = 60
    embed_batch_size: int = 64
    upload_dir: str = "./storage/documents"
    max_upload_mb: int = 200


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
