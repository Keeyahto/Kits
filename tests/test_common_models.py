from __future__ import annotations

from kit_common.models import Document, Chunk, Embedding, SearchResult, QARequest, QAResponse


def test_models_dump_and_validation():
    doc = Document(id="d1", text="hello", source="s")
    ch = Chunk(id="c1", doc_id=doc.id, text="hello", start=0, end=5)
    emb = Embedding(vector=[0.0, 1.0], model="m", dim=2)
    sr = SearchResult(id="x", score=0.5, payload={"text": "t"})
    req = QARequest(question="q", top_k=3)
    resp = QAResponse(answer="a", sources=[sr])

    for m in (doc, ch, emb, sr, req, resp):
        d = m.model_dump()
        assert isinstance(d, dict)

