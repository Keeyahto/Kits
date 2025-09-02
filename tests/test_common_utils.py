from __future__ import annotations

from kit_common.utils import normalize_text, make_id


def test_normalize_text_spaces_newlines_bom():
    raw = "\ufeffHello\r\nWorld\t !  !  \rEnd"
    norm = normalize_text(raw)
    assert "\r" not in norm
    assert "\t" not in norm
    assert "  " not in norm
    assert norm.count("\n") == 2
    assert norm.startswith("Hello")


def test_make_id_stable():
    a = make_id("x", "y", "z")
    b = make_id("x", "y", "z")
    assert a == b

