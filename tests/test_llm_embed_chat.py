from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

import kit_llm.client as client_mod
from kit_llm.client import get_default_client
from kit_llm.embed import embed_texts
from kit_llm.chat import chat as chat_func
from kit_common.config import Settings
from kit_common.errors import ConfigError
from kit_llm.errors import LLMError


class _FakeEmbeddings:
    def __init__(self, dim: int):
        self.dim = dim

    def create(self, model: str, input: list[str], timeout: float):  # noqa: ARG002
        class D:
            def __init__(self, v):
                self.embedding = v

        vecs = [[float(i % 7) for i in range(self.dim)] for _ in input]
        return types.SimpleNamespace(data=[D(v) for v in vecs])


class _FakeChat:
    def __init__(self, should_fail: bool = False, status_code: int | None = None):
        self.should_fail = should_fail
        self.status_code = status_code

    def completions(self):
        return self

    def create(self, **kwargs):  # noqa: D401, ANN001
        if self.should_fail:
            e = Exception("rate limit")
            setattr(e, "status_code", self.status_code)
            raise e
        msg = types.SimpleNamespace(content="ok")
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(choices=[choice])


class _FakeOpenAI:
    def __init__(self, *args, **kwargs):  # noqa: D401, ANN001
        self.embeddings = _FakeEmbeddings(dim=1536)
        self.chat = types.SimpleNamespace(completions=_FakeChat())


def _install_fake_openai(monkeypatch):
    monkeypatch.setattr(client_mod, "OpenAI", _FakeOpenAI)


def test_embed_texts_basic(monkeypatch):
    _install_fake_openai(monkeypatch)
    # model must be configured
    st = Settings(llm_embed_model="text-embedding-3-small")
    monkeypatch.setenv("LLM_EMBED_MODEL", st.llm_embed_model)

    vecs = embed_texts(["a", "b", "c"], model=st.llm_embed_model, batch_size=2)
    assert len(vecs) == 3
    assert all(len(v) == 1536 for v in vecs)
    # normalized vectors
    norms = [np.linalg.norm(v) for v in vecs]
    assert all(abs(n - 1) < 1e-5 for n in norms)


def test_chat_injects_system_and_handles_error(monkeypatch):
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("LLM_CHAT_MODEL", "gpt-4o-mini")
    resp = chat_func([
        {"role": "user", "content": "hi"},
    ], system="sys")
    assert "content" in resp and isinstance(resp["raw"], object)

    # force retries then error
    class _RetryOpenAI(_FakeOpenAI):
        def __init__(self):  # noqa: D401
            self.embeddings = _FakeEmbeddings(dim=1536)
            self.chat = types.SimpleNamespace(completions=_FakeChat(should_fail=True, status_code=429))

    monkeypatch.setattr(client_mod, "OpenAI", _RetryOpenAI)
    with pytest.raises(LLMError):
        chat_func([{ "role": "user", "content": "hi" }])


def test_embed_requires_model(monkeypatch):
    _install_fake_openai(monkeypatch)
    monkeypatch.delenv("LLM_EMBED_MODEL", raising=False)
    with pytest.raises(ConfigError):
        embed_texts(["x"])  # no model configured

