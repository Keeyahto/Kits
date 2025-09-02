from __future__ import annotations

from typing import Any

import os
from pydantic import BaseModel
from dotenv import load_dotenv as _load_dotenv


class Settings(BaseModel):
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    llm_chat_model: str | None = None
    llm_embed_model: str | None = None
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    log_level: str = "INFO"


def _read_env() -> dict[str, str | None]:
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_base_url": os.getenv("OPENAI_BASE_URL"),
        "llm_chat_model": os.getenv("LLM_CHAT_MODEL"),
        "llm_embed_model": os.getenv("LLM_EMBED_MODEL"),
        "qdrant_url": os.getenv("QDRANT_URL"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
    }


def load_settings(env_file: str | None = None) -> Settings:
    if env_file is not None:
        _load_dotenv(env_file)
    else:
        # Load default .env if present
        _load_dotenv()

    values = _read_env()
    return Settings(**values)

