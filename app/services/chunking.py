"""Structure-aware parent-child chunking.

Two granularities are produced per page:

* **Parent** blocks — larger, section-coherent spans (up to ``parent_chunk_tokens``).
  These are what we feed to the LLM, so the answer sees enough surrounding context.
* **Child** chunks — small windows (``child_chunk_tokens``) split from each parent.
  These are what we embed and search, because small passages match a query far
  more precisely than a whole page.

We never cross page boundaries, so every chunk still maps to exactly one page
number — keeping citations (page + section) exact, which the SOW requires. At
retrieval time we match children, then expand to their parent for generation.
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
    text: str             # child text — embedded + searched + reranked
    parent_text: str      # larger parent block — fed to the LLM as context
    page: int
    section: str | None
    ordinal: int          # child position within the document (unique)
    parent_ordinal: int   # parent position within the document (groups children)


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
    """Return (child_chunks, page_count). Streams pages to keep memory bounded."""
    chunks: list[Chunk] = []
    child_ordinal = 0
    parent_ordinal = 0
    pages = 0

    for page in iter_pages(file_path):
        pages = page.page_number
        if not page.text.strip():
            continue

        # 1) Split the page into parent blocks (no overlap between parents).
        for parent_text in _split_tokens(page.text, settings.parent_chunk_tokens, 0):
            parent_text = parent_text.strip()
            if not parent_text:
                continue

            # 2) Split each parent into overlapping child windows.
            for child in _split_tokens(
                parent_text, settings.child_chunk_tokens, settings.child_overlap_tokens
            ):
                child = child.strip()
                if not child:
                    continue
                chunks.append(
                    Chunk(
                        text=child,
                        parent_text=parent_text,
                        page=page.page_number,
                        section=page.section,
                        ordinal=child_ordinal,
                        parent_ordinal=parent_ordinal,
                    )
                )
                child_ordinal += 1
            parent_ordinal += 1

    return chunks, pages
