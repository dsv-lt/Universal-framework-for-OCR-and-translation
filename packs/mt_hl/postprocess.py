"""Intermediate layer for Mushoku Tensei Human Language ([HL] <-> Basq + numerals)."""

from __future__ import annotations

import re
from typing import Any

VOWELS = "aeiou"
CONSONANT_PAIRS = [
    ("b", "z"),
    ("c", "y"),
    ("d", "x"),
    ("f", "w"),
    ("g", "v"),
    ("h", "t"),
    ("j", "s"),
    ("k", "r"),
    ("l", "q"),
    ("m", "p"),
    ("n", "n"),
]

CIPHER_MAP: dict[str, str] = {}
for _v in VOWELS:
    CIPHER_MAP[_v] = _v
    CIPHER_MAP[_v.upper()] = _v.upper()
for _a, _b in CONSONANT_PAIRS:
    CIPHER_MAP[_a] = _b
    CIPHER_MAP[_b] = _a
    CIPHER_MAP[_a.upper()] = _b.upper()
    CIPHER_MAP[_b.upper()] = _a.upper()

# [HL] numeral symbols: k=1(I), e=5(V), r=10(X), s=50(L), z=100(C), w=500(D), g=1000(M)
NUMERAL_VALUES_HL: dict[str, tuple[int, str]] = {
    "k": (1, "I"),
    "e": (5, "V"),
    "r": (10, "X"),
    "s": (50, "L"),
    "z": (100, "C"),
    "w": (500, "D"),
    "g": (1000, "M"),
}

_COMMON_NON_NUMERAL_BASQ = {"ere", "ek", "er", "be", "ke"}


def swap_consonants(text: str) -> str:
    """Apply symmetric consonant substitution ([HL] <-> Basq); vowels stay unchanged."""
    return "".join(CIPHER_MAP.get(ch, ch) for ch in text)


def _decode_hl_numeral(hl_seq: str) -> tuple[int, str] | None:
    if not hl_seq or any(ch.lower() not in NUMERAL_VALUES_HL for ch in hl_seq):
        return None
    vals = [NUMERAL_VALUES_HL[ch.lower()][0] for ch in hl_seq]
    roms = [NUMERAL_VALUES_HL[ch.lower()][1] for ch in hl_seq]
    total = 0
    for idx, val in enumerate(vals):
        if idx + 1 < len(vals) and val < vals[idx + 1]:
            total -= val
        else:
            total += val
    return total, "".join(roms)


def transform(raw_text: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize recognized Basque text, compute [HL] script, and flag numeral candidates."""
    _ = context or {}
    cleaned = raw_text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?|\n?```$", "", cleaned).strip()
    cleaned = "\n".join(" ".join(line.split()) for line in cleaned.splitlines() if line.strip())

    hl_script = swap_consonants(cleaned)
    notes: list[str] = []
    numeral_hints: list[dict[str, Any]] = []

    for tok_basq in re.findall(r"\b[A-Za-z]{2,8}\b", cleaned):
        if tok_basq.lower() in _COMMON_NON_NUMERAL_BASQ:
            continue
        tok_hl = swap_consonants(tok_basq)
        # Flag repeated-letter numerals (e.g. Rrr -> kkk=3) or all-uppercase Roman sequences
        is_repeated = len(set(tok_hl.lower())) == 1 and len(tok_hl) >= 2
        is_all_upper_numeral = tok_basq.isupper() and len(tok_basq) >= 2
        if is_repeated or is_all_upper_numeral:
            decoded = _decode_hl_numeral(tok_hl)
            if decoded is not None:
                val, roman = decoded
                notes.append(f"Numeral candidate '{tok_basq}' ([HL] '{tok_hl}') = {val} ({roman})")
                numeral_hints.append({"basq": tok_basq, "hl": tok_hl, "value": val, "roman": roman})

    return {
        "text": cleaned,
        "original_script": hl_script,
        "notes": notes,
        "metadata": {"hl_script": hl_script, "numeral_candidates": numeral_hints},
    }
