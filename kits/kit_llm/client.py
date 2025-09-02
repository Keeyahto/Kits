from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np
from kit_common.config import Settings, load_settings
from kit_common.errors import ConfigError
from kit_common.logging import get_logger
from kit_llm.errors import LLMError

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - import at runtime in real usage
    OpenAI = None  # type: ignore


class LLMClient(Protocol):
    def embed_texts(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
        timeout_s: float = 60.0,
    ) -> list[list[float]]: ...

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout_s: float = 60.0,
        tools: list[dict] | None = None,
    ) -> dict: ...


class _OpenAILLMClient:
    def __init__(self, settings: Settings):
        if OpenAI is None:
            raise LLMError("openai client is not available")
        self.settings = settings
        base_url = settings.openai_base_url
        api_key = settings.openai_api_key
        # Some tests monkeypatch OpenAI with a stub that doesn't accept kwargs.
        # Try with kwargs if present; fall back to no-args if TypeError arises.
        try:
            if api_key is not None or base_url is not None:
                self._client = OpenAI(api_key=api_key, base_url=base_url)  # type: ignore[arg-type]
            else:
                self._client = OpenAI()  # type: ignore[call-arg]
        except TypeError:
            self._client = OpenAI()  # type: ignore[call-arg]
        self._log = get_logger(__name__)

    def _with_retries(self, fn, *, op: str):
        delays = [1.0, 2.0]
        last_err: Exception | None = None
        for attempt in range(1 + len(delays)):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                status = getattr(e, "status_code", None) or getattr(e, "status", None)
                if status in (429, 500, 502, 503, 504) and attempt < len(delays) + 1:
                    time.sleep(delays[attempt - 1])
                    last_err = e
                    continue
                raise LLMError(f"{op} failed: {e}") from e
        if last_err is not None:
            raise LLMError(f"{op} failed: {last_err}") from last_err

    def embed_texts(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
        timeout_s: float = 60.0,
    ) -> list[list[float]]:
        if not texts:
            return []
        mdl = model or self.settings.llm_embed_model
        if not mdl:
            raise ConfigError("Embedding model is not configured")

        vectors: list[list[float]] = []
        start_time = time.perf_counter()
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            t0 = time.perf_counter()

            def _call():
                return self._client.embeddings.create(model=mdl, input=batch, timeout=timeout_s)

            res = self._with_retries(_call, op="embeddings.create")
            batch_vecs = [d.embedding for d in res.data]
            if normalize:
                arr = np.array(batch_vecs, dtype=np.float32)
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                arr = arr / norms
                batch_vecs = arr.tolist()
            vectors.extend(batch_vecs)
            self._log.info(
                "embed_batch",
                extra={"batch_size": len(batch), "elapsed_ms": int((time.perf_counter() - t0) * 1000)},
            )
        self._log.info(
            "embed_total",
            extra={"texts": len(texts), "elapsed_ms": int((time.perf_counter() - start_time) * 1000)},
        )
        return vectors

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout_s: float = 60.0,
        tools: list[dict] | None = None,
    ) -> dict:
        mdl = model or self.settings.llm_chat_model
        if not mdl:
            raise ConfigError("Chat model is not configured")
        t0 = time.perf_counter()

        def _call():
            return self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=mdl,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                timeout=timeout_s,
            )

        res = self._with_retries(_call, op="chat.completions.create")
        self._log.info(
            "chat_call",
            extra={"messages": len(messages), "elapsed_ms": int((time.perf_counter() - t0) * 1000)},
        )
        choice = res.choices[0]
        content = choice.message.content or ""
        return {"content": content, "raw": res}


def get_default_client(settings: Settings | None = None) -> LLMClient:
    st = settings or load_settings()
    return _OpenAILLMClient(st)
