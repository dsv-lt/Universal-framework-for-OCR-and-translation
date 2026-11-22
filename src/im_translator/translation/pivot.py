"""Two-step pivot translator: [Source] -> [EN] -> [Target]."""

from __future__ import annotations

from dataclasses import dataclass

from im_translator.core.pack_loader import LoadedPack
from im_translator.core.schemas import GlossaryEntry
from im_translator.translation.client import OpenAICompatibleClient
from im_translator.translation.rag import GlossaryRAG

_LANG_NAMES = {
    "en": "English",
    "ru": "Russian",
    "ja": "Japanese",
    "jp": "Japanese",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "zh": "Chinese",
    "ko": "Korean",
}


@dataclass
class TranslationOutput:
    en_text: str
    target_text: str
    target_lang: str
    glossary_hits: list[GlossaryEntry]


class PivotTranslator:
    """Translate text via English pivot using GlossaryRAG + OpenAICompatibleClient."""

    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    def translate(
        self,
        source_text: str,
        pack: LoadedPack,
        target_lang: str = "ru",
    ) -> TranslationOutput:
        """Execute `[Source] -> EN` and optionally `EN -> [target_lang]`."""
        cleaned = source_text.strip()
        if not cleaned:
            return TranslationOutput(
                en_text="",
                target_text="",
                target_lang=target_lang,
                glossary_hits=[],
            )

        rag = GlossaryRAG.from_pack(pack)
        norm_target = target_lang.strip().lower()

        if pack.manifest.translation.pivot_via_en or norm_target == "en":
            en_messages, hits = rag.build_to_en_messages(cleaned)
            en_text = self.client.chat_completion(en_messages)
        else:
            # Direct translation path when pivot_via_en is explicitly disabled
            hits = rag.retrieve(cleaned)
            lang_label = _LANG_NAMES.get(norm_target, target_lang)
            en_messages, _ = rag.build_to_en_messages(cleaned)
            en_messages[0]["content"] = en_messages[0]["content"].replace("English", lang_label)
            en_messages[1]["content"] = f"Translate to {lang_label}:\n\n{cleaned}"
            direct_text = self.client.chat_completion(en_messages)
            return TranslationOutput(
                en_text="",
                target_text=direct_text,
                target_lang=target_lang,
                glossary_hits=hits,
            )

        if norm_target in ("en", "eng", "english"):
            return TranslationOutput(
                en_text=en_text,
                target_text=en_text,
                target_lang=target_lang,
                glossary_hits=hits,
            )

        target_label = _LANG_NAMES.get(norm_target, target_lang)
        pivot_messages = [
            {
                "role": "system",
                "content": (
                    f"You are an accurate translator translating from English to {target_label}. "
                    "Preserve proper nouns, formatting, ~~strikethroughs~~, and [?] markers. "
                    f"Output only the {target_label} translation."
                ),
            },
            {
                "role": "user",
                "content": f"Translate the following English text into {target_label}:\n\n{en_text}",
            },
        ]
        target_text = self.client.chat_completion(pivot_messages)
        return TranslationOutput(
            en_text=en_text,
            target_text=target_text,
            target_lang=target_lang,
            glossary_hits=hits,
        )
