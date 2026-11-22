"""Translate 15 real MT_im_HL1 OCR results + 3 Basque sanity sentences via llama-server (Gemma-3-12B)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from im_translator.core.pack_loader import load_pack
from im_translator.translation.client import OpenAICompatibleClient
from im_translator.translation.pivot import PivotTranslator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = PROJECT_ROOT / "packs" / "mt_hl"
OCR_JSON = PROJECT_ROOT / "outputs" / "mt_hl_e2e_ocr.json"
OUT_JSON = PROJECT_ROOT / "outputs" / "mt_hl_e2e_translated.json"

SANITY_SENTENCES = [
    "Etxe hau zaharra da.",
    "Magia beltza eta argia zerutik jaisten da.",
    "Rudeus Greyrat eta Sylphiette Asura erresumara joan dira.",
]


def main() -> None:
    pack = load_pack(PACK_DIR)
    ocr_data = json.loads(OCR_JSON.read_text(encoding="utf-8"))

    client = OpenAICompatibleClient(
        base_url="http://127.0.0.1:8081/v1",
        api_key="not-needed",
        model="gemma-3-12b-it-Q4_K_M",
        timeout=180.0,
    )
    translator = PivotTranslator(client)

    sanity_results: list[dict[str, Any]] = []
    for sent in SANITY_SENTENCES:
        t0 = time.perf_counter()
        out = translator.translate(sent, pack, target_lang="ru")
        sanity_results.append(
            {
                "source_basq": sent,
                "en_pivot": out.en_text,
                "ru_target": out.target_text,
                "glossary_hits": [h.model_dump(mode="json") for h in out.glossary_hits],
                "seconds": round(time.perf_counter() - t0, 2),
            }
        )
        print(f"[SANITY] {sent} -> EN: {out.en_text} | RU: {out.target_text}", flush=True)

    frame_results: list[dict[str, Any]] = []
    for frame in ocr_data.get("frames", []):
        src = frame.get("postprocessed_basq", "").strip()
        if not src:
            frame_results.append({**frame, "en_pivot": "", "ru_target": "", "glossary_hits": []})
            continue
        t0 = time.perf_counter()
        out = translator.translate(src, pack, target_lang="ru")
        dt = round(time.perf_counter() - t0, 2)
        frame_results.append(
            {
                "index": frame["index"],
                "file": frame["file"],
                "detected_total": frame["detected_total"],
                "ocr_lines_count": frame["ocr_lines_count"],
                "postprocessed_basq": src,
                "hl_script": frame["hl_script"],
                "notes": frame["notes"],
                "en_pivot": out.en_text,
                "ru_target": out.target_text,
                "glossary_hits": [h.model_dump(mode="json") for h in out.glossary_hits],
                "seconds": dt,
            }
        )
        print(
            f"[{frame['index']}/{len(ocr_data.get('frames', []))}] {frame['file']} ({dt}s) | "
            f"EN: {out.en_text[:80]} | RU: {out.target_text[:80]}",
            flush=True,
        )

    OUT_JSON.write_text(
        json.dumps(
            {
                "llm_model": "gemma-3-12b-it-Q4_K_M.gguf",
                "endpoint": "http://127.0.0.1:8081/v1",
                "sanity_checks": sanity_results,
                "frames": frame_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved translated E2E results to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
