"""PDF text + structure extraction using PyMuPDF.

Designed to stream page-by-page so a 1000-page document never needs to be
fully materialised in memory. For each page we return the text plus a best-effort
"section" label (nearest preceding heading) used for citation references.

Three extraction layers run per page:
  1. Plain text from the text layer (fast, the common case).
  2. Tables detected by PyMuPDF are emitted as markdown so the row/column
     structure survives chunking (find_tables / to_markdown).
  3. OCR fallback (RapidOCR) for scanned / image-only pages whose text layer is
     empty or sparse — the page is rendered to an image and recognised so a
     purely scanned document still indexes its text. Toggle via settings.ocr_*.

Running headers/footers that repeat across most pages are stripped so they
don't pollute every chunk.
"""
from __future__ import annotations

import logging
import re
import statistics
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import fitz  # PyMuPDF

from app.config import settings
from app.services import ocr

logger = logging.getLogger("ncdc.pdf")


@dataclass
class PageContent:
    page_number: int          # 1-based, matches what users see in a PDF viewer
    text: str
    section: str | None       # nearest heading at/above this page


def _looks_like_heading(text: str) -> bool:
    text = text.strip()
    if not (3 <= len(text) <= 90):
        return False
    if text.endswith((".", ",", ";", ":")):
        return False
    # Heading-ish: title case / all caps / numbered section (e.g. "2.4 Treatment")
    words = text.split()
    if len(words) > 12:
        return False
    return True


def _tables(page: fitz.Page) -> tuple[list[str], list[fitz.Rect]]:
    """Return (markdown tables, their bounding rects) for a page."""
    if not settings.extract_tables:
        return [], []
    markdowns: list[str] = []
    rects: list[fitz.Rect] = []
    try:
        for table in page.find_tables().tables:
            md = (table.to_markdown() or "").strip()
            if md:
                markdowns.append(md)
                rects.append(fitz.Rect(table.bbox))
    except Exception:  # noqa: BLE001 - table detection is best-effort
        logger.exception("Table extraction failed for page %s", page.number)
    return markdowns, rects


def _in_any(rect: fitz.Rect, rects: list[fitz.Rect]) -> bool:
    if not rects:
        return False
    cx, cy = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
    return any(r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1 for r in rects)


def _norm(line: str) -> str:
    # Normalise for boilerplate matching: drop digits (page numbers vary) + ws.
    return re.sub(r"\s+", " ", re.sub(r"\d+", "", line)).strip().lower()


def _boilerplate(doc: fitz.Document) -> set[str]:
    """Normalised lines that repeat on most pages (running headers/footers)."""
    if doc.page_count < 4:
        return set()
    counts: dict[str, int] = {}
    sample = min(doc.page_count, 15)
    for index in range(sample):
        seen: set[str] = set()
        for line in doc.load_page(index).get_text("text").splitlines():
            key = _norm(line)
            if 6 <= len(key) <= 80:
                seen.add(key)
        for key in seen:
            counts[key] = counts.get(key, 0) + 1
    threshold = max(3, int(sample * 0.6))
    return {key for key, n in counts.items() if n >= threshold}


def _ocr_page(page: fitz.Page) -> str:
    """Render a page to an image and OCR it; '' on any failure."""
    try:
        pix = page.get_pixmap(dpi=settings.ocr_dpi)
        return ocr.image_to_text(pix.tobytes("png"))
    except Exception:  # noqa: BLE001 - never let a bad page break ingestion
        logger.exception("OCR rendering failed for page %s", page.number)
        return ""


def iter_pages(file_path: str) -> Iterator[PageContent]:
    """Yield page content lazily; carries the current section across pages."""
    doc = fitz.open(file_path)
    current_section: str | None = None
    try:
        body_size = _estimate_body_font_size(doc)
        boilerplate = _boilerplate(doc)

        for index in range(doc.page_count):
            page = doc.load_page(index)
            data = page.get_text("dict")
            table_md, table_rects = _tables(page)
            plain_lines: list[str] = []

            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    line_text = "".join(s.get("text", "") for s in spans).strip()
                    if not line_text or _norm(line_text) in boilerplate:
                        continue
                    # Skip text that lives inside a detected table (emitted as md below).
                    if _in_any(fitz.Rect(line.get("bbox")), table_rects):
                        continue
                    plain_lines.append(line_text)

                    max_size = max((s.get("size", 0.0) for s in spans), default=0.0)
                    is_bold = any(int(s.get("flags", 0)) & 16 for s in spans)
                    if (max_size >= body_size * 1.15 or is_bold) and _looks_like_heading(
                        line_text
                    ):
                        current_section = line_text

            parts = plain_lines + table_md
            text = "\n".join(parts).strip()

            # Scanned / image-only page: the text layer is empty or sparse. Render
            # the page and OCR it so scanned PDFs still index. Only runs on pages
            # that need it (OCR is far slower than text extraction).
            if settings.ocr_enabled and len(text) < settings.ocr_min_chars:
                ocr_text = _ocr_page(page)
                if len(ocr_text) > len(text):
                    logger.info(
                        "OCR recovered %s chars on page %s", len(ocr_text), index + 1
                    )
                    text = ocr_text

            yield PageContent(page_number=index + 1, text=text, section=current_section)
    finally:
        doc.close()


def _estimate_body_font_size(doc: fitz.Document) -> float:
    sizes: list[float] = []
    sample = min(doc.page_count, 10)
    for index in range(sample):
        data = doc.load_page(index).get_text("dict")
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        sizes.append(float(span.get("size", 0.0)))
    return statistics.median(sizes) if sizes else 11.0


def page_count(file_path: str) -> int:
    doc = fitz.open(file_path)
    try:
        return doc.page_count
    finally:
        doc.close()
