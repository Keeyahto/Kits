from __future__ import annotations

import json
import logging

from kit_common.logging import get_logger


def test_get_logger_json_and_level(monkeypatch, capsys):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    log = get_logger("test")
    assert log.level == logging.WARNING
    log.warning("hello")
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["level"] == "WARNING"
    assert data["msg"] == "hello"

