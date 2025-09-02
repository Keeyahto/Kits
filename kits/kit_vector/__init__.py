from __future__ import annotations

from typing import Optional

from kit_common.config import Settings, load_settings
from .models import CollectionParams
from .base import VectorBackend
from .qdrant_backend import QdrantBackend


def get_default_backend(settings: Optional[Settings] = None) -> VectorBackend:
    st = settings or load_settings()
    if not st.qdrant_url:
        raise ValueError("Qdrant URL is not configured")
    return QdrantBackend(url=st.qdrant_url, api_key=st.qdrant_api_key)

__all__ = ["CollectionParams", "VectorBackend", "QdrantBackend", "get_default_backend"]

