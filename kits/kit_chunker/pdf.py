from __future__ import annotations

from typing import List, Tuple

from kit_common.models import Chunk
from kit_common.utils import make_id
from .tokenizers import TokenEstimator, get_token_estimator
from .splitters import split_text


def extract_text_from_pdf(path: str) -> list[tuple[int, str]]:
    from pypdf import PdfReader  # local import to keep optional

    reader = PdfReader(path)
    pages: list[tuple[int, str]] = []
    for i, p in enumerate(reader.pages, start=1):
        txt = p.extract_text() or ""
        pages.append((i, txt))
    return pages


def split_pdf(
    path: str,
    *,
    max_tokens: int = 512,
    overlap: int = 64,
    token_estimator: TokenEstimator | None = None,
    source: str | None = None,
    doc_id: str | None = None,
) -> list[Chunk]:
    est = token_estimator or get_token_estimator()
    pages = extract_text_from_pdf(path)
    chunks: list[Chunk] = []
    for page_num, text in pages:
        page_chunks = split_text(
            text,
            max_tokens=max_tokens,
            overlap=overlap,
            strategy="token",
            token_estimator=est,
            doc_id=doc_id,
            source=source,
        )
        for ch in page_chunks:
            ch.page = page_num
            # Recompute id to include page
            start, end = ch.start or 0, ch.end or 0
            ch.id = make_id(str(doc_id or source or ""), str(page_num), str(start), str(end), ch.text[:32])
            ch.metadata["source"] = source
        chunks.extend(page_chunks)
    return chunks
