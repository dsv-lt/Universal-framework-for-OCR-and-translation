"""Demo intermediate layer for `_example_demo` (ROT13 substitution cipher)."""

from __future__ import annotations

import codecs
from typing import Any


def transform(raw_text: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Decode ROT13 cipher text into plain text for the translation stage."""
    _ = context or {}
    decoded = codecs.encode(raw_text, "rot_13")
    notes = ["Applied ROT13 intermediate cipher transform"] if decoded != raw_text else []
    return {
        "text": decoded,
        "original_script": raw_text,
        "notes": notes,
        "metadata": {"cipher": "rot13"},
    }
