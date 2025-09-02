from __future__ import annotations

import hashlib
import re


_BOM = "\ufeff"


def normalize_text(text: str) -> str:
    # Remove UTF-8 BOM if present
    if text.startswith(_BOM):
        text = text.lstrip(_BOM)
    # Normalize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace tabs with single space
    text = text.replace("\t", " ")
    # Collapse multiple spaces inside lines, preserve newlines
    text = re.sub(r"[ ]{2,}", " ", text)
    # Strip spaces around line breaks
    text = re.sub(r" *\n *", "\n", text)
    return text


def make_id(*parts: str) -> str:
    joined = "\x1f".join(parts)
    h = hashlib.sha1()
    h.update(joined.encode("utf-8"))
    return h.hexdigest()

