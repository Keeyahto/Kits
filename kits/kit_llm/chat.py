from __future__ import annotations

from typing import Any

from kit_common.config import Settings, load_settings
from .client import get_default_client


def chat(
    messages: list[dict[str, str]],
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    timeout_s: float = 60.0,
    tools: list[dict] | None = None,
) -> dict:
    st: Settings = load_settings()
    client = get_default_client(st)
    msgs = list(messages)
    if system is not None:
        msgs = [{"role": "system", "content": system}] + msgs
    return client.chat(
        msgs, model=model, temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s, tools=tools
    )

