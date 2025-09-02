from __future__ import annotations

from kit_chunker import pdf as pdf_mod


def test_split_pdf_monkeypatch(monkeypatch):
    # simulate two pages of text
    monkeypatch.setattr(pdf_mod, "extract_text_from_pdf", lambda p: [(1, "hello world"), (2, "bye world")])
    chunks = pdf_mod.split_pdf("dummy.pdf", max_tokens=10, overlap=2, source="sample.pdf", doc_id="doc1")
    assert all(c.page in (1, 2) for c in chunks)
    assert any(c.page == 2 for c in chunks)

