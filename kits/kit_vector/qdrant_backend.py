from __future__ import annotations

from typing import Any

from kit_common.errors import ExternalServiceError
from kit_common.logging import get_logger
from kit_common.models import SearchResult
from .models import CollectionParams

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
except Exception:  # pragma: no cover - used in runtime, mocked in tests
    QdrantClient = None  # type: ignore
    Distance = None  # type: ignore
    VectorParams = None  # type: ignore
    PointStruct = None  # type: ignore
    Filter = None  # type: ignore


class QdrantBackend:
    def __init__(self, url: str, api_key: str | None = None, timeout_s: float = 10.0):
        if QdrantClient is None:
            raise ExternalServiceError("qdrant-client is not available")
        self._client = QdrantClient(url=url, api_key=api_key, timeout=timeout_s)
        self._log = get_logger(__name__)

    def ensure_collection(self, params: CollectionParams) -> None:
        try:
            dist = Distance.COSINE if params.distance == "cosine" else Distance.DOT
            if not self._client.collection_exists(collection_name=params.name):
                self._client.create_collection(
                    collection_name=params.name,
                    vectors_config=VectorParams(size=params.vector_size, distance=dist),
                )
        except Exception as e:  # noqa: BLE001
            raise ExternalServiceError(f"ensure_collection failed: {e}") from e

    def upsert(
        self, name: str, vectors: list[list[float]], payloads: list[dict], ids: list[str] | None = None
    ) -> int:
        try:
            points: list[PointStruct] = []
            for idx, vec in enumerate(vectors):
                pid = ids[idx] if ids else None
                payload = payloads[idx] if idx < len(payloads) else {}
                points.append(PointStruct(id=pid, vector=vec, payload=payload))
            res = self._client.upsert(collection_name=name, points=points)
            self._log.info("upsert", extra={"count": len(points)})
            # acknowledge count inserted
            return len(points)
        except Exception as e:  # noqa: BLE001
            raise ExternalServiceError(f"upsert failed: {e}") from e

    def search(self, name: str, query: list[float], *, k: int = 5, filter: dict | None = None) -> list[SearchResult]:
        try:
            qfilter = Filter(**filter) if filter else None  # type: ignore[arg-type]
            hits = self._client.search(collection_name=name, query_vector=query, limit=k, query_filter=qfilter)
            results: list[SearchResult] = []
            for h in hits:
                results.append(SearchResult(id=str(h.id), score=float(h.score), payload=h.payload or {}))
            return results
        except Exception as e:  # noqa: BLE001
            raise ExternalServiceError(f"search failed: {e}") from e

    def recreate(self, params: CollectionParams) -> None:
        try:
            if self._client.collection_exists(collection_name=params.name):
                self._client.delete_collection(collection_name=params.name)
            self.ensure_collection(params)
        except Exception as e:  # noqa: BLE001
            raise ExternalServiceError(f"recreate failed: {e}") from e

