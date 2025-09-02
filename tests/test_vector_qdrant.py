from __future__ import annotations

import os

import pytest

from kit_vector import QdrantBackend, CollectionParams
from kit_common.errors import ExternalServiceError


def test_qdrant_backend_init_without_client(monkeypatch):
    # if qdrant-client import fails, constructor should raise
    import kit_vector.qdrant_backend as qb

    old = qb.QdrantClient
    qb.QdrantClient = None  # type: ignore
    with pytest.raises(ExternalServiceError):
        QdrantBackend(url="http://localhost:6333")
    qb.QdrantClient = old  # restore


@pytest.mark.skipif(not os.getenv("QDRANT_URL"), reason="no qdrant configured")
def test_qdrant_e2e_small(monkeypatch):
    url = os.environ["QDRANT_URL"]
    api_key = os.getenv("QDRANT_API_KEY")
    backend = QdrantBackend(url=url, api_key=api_key)
    params = CollectionParams(name="kits_test", vector_size=4, distance="cosine")
    backend.recreate(params)
    vectors = [[0.0, 0.1, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0]]
    payloads = [{"text": "a", "source": None, "page": None, "tenant": None} for _ in vectors]
    ids = [f"id{i}" for i in range(len(vectors))]
    n = backend.upsert(params.name, vectors, payloads, ids)
    assert n == 3
    res = backend.search(params.name, vectors[1], k=2)
    assert len(res) >= 1
    assert res[0].id in ids
