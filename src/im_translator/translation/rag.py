"""Glossary RAG lookup and grammar-guided prompt builder."""

from __future__ import annotations

import re

from im_translator.core.pack_loader import LoadedPack
from im_translator.core.schemas import GlossaryEntry

_TOKEN_RE = re.compile(r"[^\W\d_][\w'-]*", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [tok.strip("'-_").lower() for tok in _TOKEN_RE.findall(text) if tok.strip("'-_")]


class GlossaryRAG:
    """Lightweight lexical RAG over a Language Pack's glossary.json and grammar.md."""

    def __init__(
        self,
        glossary: list[GlossaryEntry] | None = None,
        grammar_text: str = "",
        source_label: str = "Source",
    ) -> None:
        self.glossary = glossary or []
        self.grammar_text = grammar_text.strip()
        self.source_label = source_label

    @classmethod
    def from_pack(cls, pack: LoadedPack) -> "GlossaryRAG":
        label = pack.manifest.intermediate_lang_code or pack.manifest.name
        return cls(
            glossary=pack.glossary,
            grammar_text=pack.grammar_text,
            source_label=label,
        )

    def retrieve(self, text: str, top_k: int = 12, min_stem_len: int = 4) -> list[GlossaryEntry]:
        """Find relevant glossary entries using exact, variant, and prefix/stem matches."""
        if not text.strip() or not self.glossary:
            return []

        tokens = _tokenize(text)
        token_set = set(tokens)
        lowered_text = text.lower()
        scored: list[tuple[int, int, GlossaryEntry]] = []

        for idx, entry in enumerate(self.glossary):
            term_l = entry.term.strip().lower()
            variants_l = [v.strip().lower() for v in entry.variants if v.strip()]
            score = 0

            # 1. Exact term match (single token or multi-word phrase)
            if term_l in token_set or (" " in term_l and term_l in lowered_text):
                score += 100

            # 2. Exact variant match
            if any(v in token_set or (" " in v and v in lowered_text) for v in variants_l):
                score += 90

            # 3. Stem/prefix overlap (helpful for agglutinative suffixes)
            if score == 0:
                candidates = [term_l] + variants_l
                for cand in candidates:
                    if len(cand) < min_stem_len:
                        continue
                    stem = cand[:min_stem_len]
                    for tok in tokens:
                        if len(tok) >= min_stem_len and (
                            tok.startswith(cand) or cand.startswith(tok) or tok.startswith(stem)
                        ):
                            score = max(score, 40 + min(len(cand), len(tok)))

            if score > 0:
                if entry.is_proper_noun:
                    score += 5
                scored.append((score, -idx, entry))

        scored.sort(reverse=True)
        return [item[2] for item in scored[:top_k]]

    def build_to_en_messages(self, source_text: str, top_k: int = 12) -> tuple[list[dict[str, str]], list[GlossaryEntry]]:
        """Build system + user messages for `[Source] -> English` translation."""
        hits = self.retrieve(source_text, top_k=top_k)

        system_parts = [
            f"You are an expert translator converting text from {self.source_label} into natural, accurate English.",
            "Translate ONLY what is present in the source text. Do not hallucinate missing content.",
            "Preserve crossed-out spans (~~...~~), unknown markers ([?]), and proper names faithfully.",
            "Output only the English translation without extra commentary.",
        ]

        if self.grammar_text:
            system_parts.append(f"## Grammar & Style Reference\n{self.grammar_text}")

        if hits:
            glossary_lines = ["## Relevant Glossary Entries"]
            for entry in hits:
                flags = ["PROPER_NOUN — DO NOT TRANSLATE/ALTER"] if entry.is_proper_noun else []
                if entry.pos:
                    flags.append(entry.pos)
                meta = f" ({'; '.join(flags)})" if flags else ""
                note = f" — {entry.notes}" if entry.notes else ""
                glossary_lines.append(f"- `{entry.term}`{meta} => {entry.en}{note}")
            system_parts.append("\n".join(glossary_lines))

        messages = [
            {"role": "system", "content": "\n\n".join(system_parts)},
            {"role": "user", "content": f"Translate to English:\n\n{source_text}"},
        ]
        return messages, hits
