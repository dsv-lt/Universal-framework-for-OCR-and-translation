"""Tests for GlossaryRAG, PivotTranslator, and OpenAI-compatible VLM/LLM clients."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np

from im_translator.core.pack_loader import load_pack
from im_translator.core.schemas import GlossaryEntry
from im_translator.recognition.vlm_llamacpp import VLMLlamaCppRecognizer
from im_translator.translation.client import OpenAICompatibleClient
from im_translator.translation.pivot import PivotTranslator
from im_translator.translation.rag import GlossaryRAG

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_PACK = REPO_ROOT / "packs" / "_example_demo"


def test_glossary_rag_exact_variant_and_stem_matching() -> None:
    entries = [
        GlossaryEntry(
            term="Rudeus",
            en="Rudeus",
            pos="proper_noun",
            variants=["Rudeusen"],
            is_proper_noun=True,
            notes="Keep exact spelling.",
        ),
        GlossaryEntry(
            term="ur",
            en="water",
            pos="noun",
            variants=["urak"],
            is_proper_noun=False,
        ),
        GlossaryEntry(
            term="kalma",
            en="lamp",
            pos="noun",
            variants=[],
            is_proper_noun=False,
        ),
    ]
    rag = GlossaryRAG(glossary=entries, grammar_text="SOV word order.", source_label="TestLang")
    # Exact variant ("Rudeusen") + stem match ("kalmar" starts with "kalma")
    hits = rag.retrieve("Rudeusen kalmar")
    hit_terms = {h.term for h in hits}
    assert "Rudeus" in hit_terms
    assert "kalma" in hit_terms
    assert "ur" not in hit_terms

    messages, msg_hits = rag.build_to_en_messages("Rudeusen kalmar")
    assert len(msg_hits) == 2
    sys_prompt = messages[0]["content"]
    assert "SOV word order." in sys_prompt
    assert "PROPER_NOUN" in sys_prompt
    assert "Rudeus" in sys_prompt


def test_pivot_translator_two_step_and_en_shortcut() -> None:
    pack = load_pack(DEMO_PACK)
    recorded_prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "mock-llm"}]})
        body = json.loads(request.content.decode("utf-8"))
        user_msg = body["messages"][1]["content"]
        recorded_prompts.append(user_msg)
        if "Translate to English" in user_msg:
            content = "Welcome to the forest of Aethelgard."
        else:
            content = "Добро пожаловать в лес Этельгарда."
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    transport = httpx.MockTransport(handler)
    client = OpenAICompatibleClient(base_url="http://mock.local/v1", transport=transport)
    assert client.health_check()["ok"] is True

    translator = PivotTranslator(client)
    res_ru = translator.translate("silva Aethelgard", pack=pack, target_lang="ru")
    assert res_ru.en_text == "Welcome to the forest of Aethelgard."
    assert res_ru.target_text == "Добро пожаловать в лес Этельгарда."
    assert len(recorded_prompts) == 2
    assert {g.term for g in res_ru.glossary_hits} == {"silva", "Aethelgard"}

    # When target_lang == "en", second hop must be skipped
    recorded_prompts.clear()
    res_en = translator.translate("silva Aethelgard", pack=pack, target_lang="en")
    assert res_en.en_text == res_en.target_text
    assert len(recorded_prompts) == 1


def test_vlm_llamacpp_recognizer_with_mock_transport() -> None:
    pack = load_pack(DEMO_PACK)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        parts = body["messages"][0]["content"]
        assert any(p.get("type") == "image_url" for p in parts)
        return httpx.Response(200, json={"choices": [{"message": {"content": "uryyb"}}]})

    rec = VLMLlamaCppRecognizer(
        base_url="http://mock.local/v1",
        transport=httpx.MockTransport(handler),
    )
    crop = np.zeros((32, 96, 3), dtype=np.uint8)
    out = rec.recognize([crop], pack)
    assert len(out) == 1
    assert out[0].text == "uryyb"
