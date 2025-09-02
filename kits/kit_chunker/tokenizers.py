from __future__ import annotations

from math import ceil
from typing import Protocol


class TokenEstimator(Protocol):
    def count(self, text: str) -> int: ...


class _TiktokenEstimator:
    def __init__(self):
        import tiktoken  # local import

        try:
            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover - safe fallback
            self.enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self.enc.encode(text))


class _CharEstimator:
    def count(self, text: str) -> int:
        return int(ceil(len(text) / 4))


def get_token_estimator(name: str = "tiktoken") -> TokenEstimator:
    if name == "tiktoken":
        try:
            return _TiktokenEstimator()
        except Exception:  # pragma: no cover - fallback when tiktoken missing
            return _CharEstimator()
    return _CharEstimator()

