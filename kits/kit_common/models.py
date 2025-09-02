from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    text: str
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    doc_id: str | None = None
    text: str
    start: int | None = None
    end: int | None = None
    page: int | None = None
    tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Embedding(BaseModel):
    vector: list[float]
    model: str
    dim: int


class SearchResult(BaseModel):
    id: str
    score: float
    payload: dict[str, Any]


class QARequest(BaseModel):
    question: str
    top_k: int = 5
    tenant: str | None = None


class QAResponse(BaseModel):
    answer: str
    sources: list[SearchResult] = Field(default_factory=list)

