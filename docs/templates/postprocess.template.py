"""Template for the optional Language Pack intermediate postprocessor.

Place this file as `packs/<pack_id>/postprocess.py` and set:
    postprocess:
      enabled: true
      module_path: "postprocess.py"

If a language does NOT need deterministic decoding/normalization before
translation (e.g., most conlangs), omit this file and keep `enabled: false`.
"""

from __future__ import annotations

from typing import Any


def transform(raw_text: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Transform raw OCR text before passing it to the translation stage.

    Args:
        raw_text: Raw text returned by the OCR/VLM recognizer.
        context: Optional metadata dict (pack_id, region_index, confidence, etc.).

    Returns:
        Dictionary matching `PostprocessResult`:
          - text (str): Text sent to the `[Source] -> EN` translator.
          - original_script (str): Preserved representation in original script.
          - notes (list[str]): Human-readable decoding notes (e.g., decoded numerals).
          - metadata (dict): Any structured side information.
    """
    _ = context or {}

    # Example: simple normalization (replace with cipher / numeral decoder as needed)
    cleaned = " ".join(raw_text.split())
    notes: list[str] = []

    return {
        "text": cleaned,
        "original_script": raw_text,
        "notes": notes,
        "metadata": {},
    }
