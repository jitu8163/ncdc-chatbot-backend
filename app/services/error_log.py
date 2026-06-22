"""Persistent JSON log of failed / errored chat responses.

Every question that fails to produce an answer (LLM/provider error, retrieval
crash, "failed to fetch" style infrastructure failure) is appended as one entry
to a JSON file on disk so it can be reviewed later. Kept deliberately simple and
self-contained — no DB table, just an append-to-a-JSON-array file guarded by a
lock so concurrent requests don't corrupt it.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime

from app.config import settings

logger = logging.getLogger("ncdc.errorlog")

_lock = threading.Lock()


def _path() -> str:
    return os.path.join(settings.upload_dir, "..", "error_logs.json")


def log_error(
    question: str,
    error: str,
    *,
    session_id: str | None = None,
    language: str | None = None,
) -> None:
    """Append one failed-question record to the JSON error log. Never raises."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "question": question,
        "error": error,
        "session_id": session_id,
        "language": language,
    }
    try:
        with _lock:
            path = os.path.abspath(_path())
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data: list[dict] = []
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as fh:
                        data = json.load(fh)
                        if not isinstance(data, list):
                            data = []
                except (json.JSONDecodeError, OSError):
                    data = []
            data.append(entry)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception:  # noqa: BLE001 - logging must never break a request
        logger.exception("Failed to write error log entry")
