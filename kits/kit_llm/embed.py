from __future__ import annotations

from typing import List

from kit_common.config import Settings, load_settings
from .client import get_default_client


def embed_texts(
    texts: list[str],
    *,
    model: str | None = None,
    batch_size: int = 64,
    normalize: bool = True,
    timeout_s: float = 60.0,
) -> List[List[float]]:
    st: Settings = load_settings()
    client = get_default_client(st)
    return client.embed_texts(
        texts, model=model, batch_size=batch_size, normalize=normalize, timeout_s=timeout_s
    )

