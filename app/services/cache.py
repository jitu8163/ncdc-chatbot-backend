"""Redis-backed caching and rate limiting (best-effort).

Maps to the architecture's Redis usage: embedding_cache, retrieval_cache,
prompt_cache (answers) and rate_limit_cache. Every helper degrades gracefully —
if Redis is unreachable the call is treated as a cache miss and the request
still succeeds, so Redis is an optimisation, never a hard dependency.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from functools import lru_cache
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _redis():
    if not settings.cache_enabled:
        return None
    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        return client
    except Exception:  # noqa: BLE001 - cache is optional
        logger.warning("Redis unavailable; caching/rate-limiting disabled.")
        return None


def _key(namespace: str, *parts: str) -> str:
    raw = "|".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"ncdc:{namespace}:{digest}"


def get_json(namespace: str, *parts: str) -> Any | None:
    client = _redis()
    if client is None:
        return None
    try:
        raw = client.get(_key(namespace, *parts))
        return json.loads(raw) if raw else None
    except Exception:  # noqa: BLE001
        return None


def set_json(namespace: str, value: Any, ttl: int, *parts: str) -> None:
    client = _redis()
    if client is None:
        return
    try:
        client.set(_key(namespace, *parts), json.dumps(value), ex=ttl)
    except Exception:  # noqa: BLE001
        pass


def rate_limited(client_id: str) -> bool:
    """Fixed-window per-minute limiter. Returns True if the caller is over quota."""
    if not settings.rate_limit_enabled:
        return False
    client = _redis()
    if client is None:
        return False
    try:
        window = int(time.time() // 60)
        key = f"ncdc:rate:{client_id}:{window}"
        count = client.incr(key)
        if count == 1:
            client.expire(key, 60)
        return count > settings.rate_limit_per_minute
    except Exception:  # noqa: BLE001
        return False
