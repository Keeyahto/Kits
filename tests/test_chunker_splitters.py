from __future__ import annotations

import types

import pytest

from kit_chunker.splitters import split_text, split_markdown
from kit_chunker.tokenizers import get_token_estimator


def test_split_text_token_strategy_respects_limits():
    text = " ".join(["word" + str(i) for i in range(1000)])
    est = get_token_estimator(name="fallback")
    chunks = split_text(text, max_tokens=50, overlap=10, token_estimator=est)
    assert all(ch.text.strip() for ch in chunks)
    assert all(ch.tokens is not None and ch.tokens <= 50 for ch in chunks)
    # ensure overlap leads to more than one chunk
    assert len(chunks) > 1


def test_split_markdown_headers():
    md = """# Title\n\nPara1 line.\n\n## Sub\ntext under sub.\n"""
    est = get_token_estimator(name="fallback")
    chunks = split_markdown(md, max_tokens=20, overlap=5, token_estimator=est)
    # first chunk should start near beginning
    assert chunks[0].start == 0
    assert any("Sub" in c.text for c in chunks)

