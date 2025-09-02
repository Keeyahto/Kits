from __future__ import annotations

from typing import Protocol

from kit_common.models import SearchResult
from .models import CollectionParams


class VectorBackend(Protocol):
    def ensure_collection(self, params: CollectionParams) -> None: ...

    def upsert(
        self, name: str, vectors: list[list[float]], payloads: list[dict], ids: list[str] | None = None
    ) -> int: ...

    def search(self, name: str, query: list[float], *, k: int = 5, filter: dict | None = None) -> list[SearchResult]: ...

    def recreate(self, params: CollectionParams) -> None: ...

