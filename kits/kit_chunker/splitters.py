from __future__ import annotations

import re
from typing import Iterable

from kit_common.models import Chunk
from kit_common.utils import make_id
from .tokenizers import TokenEstimator, get_token_estimator


def _find_split_boundary(text: str, start: int, end: int) -> int:
    # try to snap to nearest whitespace or period before end within 50 chars
    window_start = max(start, end - 50)
    segment = text[window_start:end]
    match = re.search(r"[\s\.]\S*$", segment)
    if match:
        return window_start + match.start() + 1
    return end


def split_text(
    text: str,
    *,
    max_tokens: int = 512,
    overlap: int = 64,
    strategy: str = "token",
    token_estimator: TokenEstimator | None = None,
    doc_id: str | None = None,
    source: str | None = None,
) -> list[Chunk]:
    est = token_estimator or get_token_estimator()
    chunks: list[Chunk] = []

    if not text:
        return []

    if strategy not in {"token", "paragraph"}:
        strategy = "token"

    if strategy == "paragraph":
        paragraphs = [p for p in re.split(r"\n\n+", text) if p.strip()]
        cur_offset = 0
        for p in paragraphs:
            p_start = text.find(p, cur_offset)
            p_end = p_start + len(p)
            cur_offset = p_end
            # pack paragraphs greedily within max_tokens
            buffer = p
            buf_start = p_start
            while buffer:
                # grow until next paragraph would exceed; here we split paragraph itself if too big
                lo, hi = 0, len(buffer)
                best = 0
                while lo <= hi:
                    mid = (lo + hi) // 2
                    part = buffer[:mid]
                    if est.count(part) <= max_tokens:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                cut = _find_split_boundary(buffer, 0, best) if best < len(buffer) else best
                part = buffer[:cut]
                start = buf_start
                end = buf_start + len(part)
                cid = make_id(str(doc_id or source or ""), str(None), str(start), str(end), part[:32])
                chunks.append(
                    Chunk(
                        id=cid,
                        doc_id=doc_id,
                        text=part,
                        start=start,
                        end=end,
                        tokens=est.count(part),
                        metadata={"source": source},
                    )
                )
                # prepare next slice within paragraph, apply token overlap where applicable
                # compute next start by backing off overlap tokens within current part
                if cut >= len(buffer):
                    buffer = ""
                else:
                    # back off by overlap tokens
                    next_start = cut
                    while next_start > 0 and est.count(buffer[next_start:cut]) < overlap:
                        # move to previous space to keep words intact
                        prev_space = buffer.rfind(" ", 0, next_start)
                        next_start = prev_space if prev_space != -1 else next_start - 1
                    buf_start += next_start
                    buffer = buffer[next_start:]
        return chunks

    # token strategy
    n = len(text)
    start = 0
    while start < n:
        # binary search max end such that token count <= max_tokens
        lo, hi = start, n
        best_end = start
        while lo <= hi:
            mid = (lo + hi) // 2
            part = text[start:mid]
            if est.count(part) <= max_tokens:
                best_end = mid
                lo = mid + 1
            else:
                hi = mid - 1
        end = _find_split_boundary(text, start, best_end) if best_end < n else best_end
        part = text[start:end]
        if not part:
            # avoid infinite loop: advance by one char
            end = min(n, start + 1)
            part = text[start:end]
        cid = make_id(str(doc_id or source or ""), str(None), str(start), str(end), part[:32])
        chunks.append(
            Chunk(
                id=cid,
                doc_id=doc_id,
                text=part,
                start=start,
                end=end,
                tokens=est.count(part),
                metadata={"source": source},
            )
        )
        if end >= n:
            break
        # compute next start to include overlap tokens
        next_start = end
        while next_start > start and est.count(text[next_start:end]) < overlap:
            prev_space = text.rfind(" ", start, next_start)
            next_start = prev_space if prev_space != -1 else next_start - 1
        start = next_start
    return chunks


def split_markdown(
    md: str,
    *,
    max_tokens: int = 512,
    overlap: int = 64,
    header_regex: str = r"^#{1,6}\s+.+$",
    token_estimator: TokenEstimator | None = None,
    doc_id: str | None = None,
    source: str | None = None,
) -> list[Chunk]:
    est = token_estimator or get_token_estimator()
    lines = md.splitlines()
    header_re = re.compile(header_regex)
    sections: list[tuple[str, int]] = []  # (text, start_offset)
    cur: list[str] = []
    cur_start = 0
    offset = 0
    for i, line in enumerate(lines):
        line_len = len(line) + 1  # +\n
        if header_re.match(line):
            if cur:
                section_text = "\n".join(cur)
                sections.append((section_text, cur_start))
                cur = []
            cur = [line]
            cur_start = offset
        else:
            if not cur:
                cur_start = offset
            cur.append(line)
        offset += line_len
    if cur:
        sections.append(("\n".join(cur), cur_start))

    chunks: list[Chunk] = []
    for sec_text, sec_start in sections:
        # reuse split_text token strategy for the section
        sub_chunks = split_text(
            sec_text,
            max_tokens=max_tokens,
            overlap=overlap,
            strategy="token",
            token_estimator=est,
            doc_id=doc_id,
            source=source,
        )
        # adjust offsets to original markdown string
        for ch in sub_chunks:
            ch.start = (ch.start or 0) + sec_start
            ch.end = (ch.end or 0) + sec_start
            # recompute id to include absolute offsets
            ch.id = make_id(str(doc_id or source or ""), str(None), str(ch.start), str(ch.end), ch.text[:32])
        chunks.extend(sub_chunks)
    return chunks

