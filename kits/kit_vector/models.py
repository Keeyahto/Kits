from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from kit_common.models import SearchResult as _SearchResult


class CollectionParams(BaseModel):
    name: str
    vector_size: int
    distance: Literal["cosine", "dot"]


# re-export for type reference convenience if needed by users
SearchResult = _SearchResult

