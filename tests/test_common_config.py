from __future__ import annotations

import os
from pathlib import Path

from kit_common import load_settings, Settings


def test_load_settings_env_and_file(tmp_path: Path, monkeypatch):
    envfile = tmp_path / ".env"
    envfile.write_text("OPENAI_API_KEY=from_file\nLOG_LEVEL=DEBUG\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    st = load_settings(str(envfile))
    assert isinstance(st, Settings)
    assert st.openai_api_key == "from_file"
    assert st.log_level == "DEBUG"

    # env overrides
    monkeypatch.setenv("OPENAI_API_KEY", "from_env")
    st2 = load_settings(str(envfile))
    assert st2.openai_api_key == "from_env"


def test_settings_empty_ok(monkeypatch):
    for k in [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "LLM_CHAT_MODEL",
        "LLM_EMBED_MODEL",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "LOG_LEVEL",
    ]:
        monkeypatch.delenv(k, raising=False)
    st = load_settings()
    assert st.openai_api_key is None
    assert st.log_level == "INFO"

