"""OCR fallback for scanned / image-only PDF pages.

Uses RapidOCR (ONNX on CPU) so no system Tesseract binary has to be installed —
the recognition models ship inside the package. The engine is loaded lazily and
reused across pages; its `__call__` is serialised with a lock because the
ingestion pipeline can process several documents (and therefore pages) at once.
"""
from __future__ import annotations

import logging
import threading
from functools import lru_cache

from app.config import settings

logger = logging.getLogger("ncdc.ocr")

# RapidOCR's underlying ONNX session is not guaranteed thread-safe; serialise calls.
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _engine():
    # Imported lazily so the (slow) model load only happens when OCR is first used.
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def warmup() -> None:
    """Pre-load the OCR models off the request/ingestion path. Best-effort."""
    if settings.ocr_enabled:
        _engine()


def image_to_text(png_bytes: bytes) -> str:
    """Return OCR text for a PNG-encoded page image; '' on failure or no text."""
    if not settings.ocr_enabled:
        return ""
    try:
        with _lock:
            result, _ = _engine()(png_bytes)
    except Exception:  # noqa: BLE001 - OCR is best-effort; never break ingestion
        logger.exception("OCR failed for a page image")
        return ""
    if not result:
        return ""
    # RapidOCR returns rows of [box, text, score] roughly in reading order.
    return "\n".join(line[1] for line in result if len(line) >= 2 and line[1]).strip()
