"""Simple single-level chunking (MVP).

Each page's text is split into fixed-size, slightly-overlapping token windows
(``chunk_tokens`` with ``chunk_overlap_tokens`` overlap). The same chunk is both
embedded/searched and fed to the LLM — no parent/child distinction.

We never cross page boundaries, so every chunk maps to exactly one page number —
keeping citations (page + section) exact, which the SOW requires.
"""
from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from app.config import settings
from app.services.pdf_processor import iter_pages

# cl100k_base is a reasonable, model-agnostic tokenizer for sizing chunks.
_enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str             # embedded + searched + fed to the LLM
    page: int
    section: str | None
    ordinal: int          # position within the document (unique)


def _split_tokens(text: str, max_tokens: int, overlap: int) -> list[str]:
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    step = max(1, max_tokens - overlap)
    out: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + max_tokens]
        if not window:
            break
        out.append(_enc.decode(window))
        if start + max_tokens >= len(tokens):
            break
    return out


def chunk_document(file_path: str) -> tuple[list[Chunk], int]:
    """Return (chunks, page_count). Streams pages to keep memory bounded."""
    chunks: list[Chunk] = []
    ordinal = 0
    pages = 0

    for page in iter_pages(file_path):
        pages = page.page_number
        if not page.text.strip():
            continue

        for window in _split_tokens(
            page.text, settings.chunk_tokens, settings.chunk_overlap_tokens
        ):
            window = window.strip()
            if not window:
                continue
            chunks.append(
                Chunk(
                    text=window,
                    page=page.page_number,
                    section=page.section,
                    ordinal=ordinal,
                )
            )
            ordinal += 1

    return chunks, pages
