from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover - optional
    orjson = None  # type: ignore


class JsonLikeFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Include common extras if present
        for key in ("extra", "event", "context"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        if orjson is not None:
            return orjson.dumps(payload).decode("utf-8")
        return json.dumps(payload, ensure_ascii=False)


_configured_names: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    # Respect LOG_LEVEL from environment
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    if name not in _configured_names:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLikeFormatter())
        handler.setLevel(level)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.propagate = False
        _configured_names.add(name)
    return logger

