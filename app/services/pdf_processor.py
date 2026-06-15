"""PDF text + structure extraction using PyMuPDF.

Designed to stream page-by-page so a 1000-page document never needs to be
fully materialised in memory. For each page we return the text plus a best-effort
"section" label (nearest preceding heading) used for citation references.
"""
from __future__ import annotations

import statistics
from collections.abc import Iterator
from dataclasses import dataclass

import fitz  # PyMuPDF


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


def iter_pages(file_path: str) -> Iterator[PageContent]:
    """Yield page content lazily; carries the current section across pages."""
    doc = fitz.open(file_path)
    current_section: str | None = None
    try:
        # Establish a body-text font size baseline from a sample of pages.
        body_size = _estimate_body_font_size(doc)

        for index in range(doc.page_count):
            page = doc.load_page(index)
            data = page.get_text("dict")
            plain_lines: list[str] = []

            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    line_text = "".join(s.get("text", "") for s in spans).strip()
                    if not line_text:
                        continue
                    plain_lines.append(line_text)

                    max_size = max((s.get("size", 0.0) for s in spans), default=0.0)
                    is_bold = any(int(s.get("flags", 0)) & 16 for s in spans)
                    if (max_size >= body_size * 1.15 or is_bold) and _looks_like_heading(
                        line_text
                    ):
                        current_section = line_text

            text = "\n".join(plain_lines).strip()
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
