"""Run real DBNet++ -> dewarp -> Qwen2.5-VL-3B (+mt_hl LoRA) -> postprocess OCR on 15 MT_im_HL1 frames."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "hf_cache"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "models" / "torch_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from PIL import Image

from im_translator.core.pack_loader import load_pack
from im_translator.core.paths import normalize_user_path
from im_translator.core.schemas import DetectionConfig, DetectionGranularity, DetectorType
from im_translator.detection.dewarp import dewarp_polygon, group_and_sort_regions
from im_translator.detection.registry import get_detector
from im_translator.recognition.vlm_peft import VLMPeftRecognizer

SRC_DIR = normalize_user_path(os.environ.get("MT_HL_IMAGES_DIR", r"D:\Personal\download\MT_im_HL1"))
MODELS_DIR = PROJECT_ROOT / "models"
PACK_DIR = PROJECT_ROOT / "packs" / "mt_hl"
BENCH_JSON = PROJECT_ROOT / "outputs" / "mt_hl_det_benchmark_248.json"
OUT_JSON = PROJECT_ROOT / "outputs" / "mt_hl_e2e_ocr.json"

MUST_INCLUDE = [
    "Снимок экрана 2026-09-26 021556.png",
    "Снимок экрана 2026-09-23 204343.png",
    "Снимок экрана 2026-09-26 142154.png",
    "Снимок экрана 2026-09-26 233909.png",
]


def _pick_15_frames() -> list[Path]:
    selected: list[str] = [name for name in MUST_INCLUDE if (SRC_DIR / name).exists()]
    if BENCH_JSON.exists():
        data = json.loads(BENCH_JSON.read_text(encoding="utf-8"))
        cands = [
            r
            for r in data.get("per_image", [])
            if r["file"] not in selected
            and 3 <= r["dbnet"]["count"] <= 25
            and (r["dbnet"]["mean_conf"] or 0.0) >= 0.85
        ]
        cands.sort(key=lambda r: float(r["dbnet"]["mean_conf"] or 0.0), reverse=True)
        # Pick 11 diverse frames spread evenly across timestamp order among high-conf candidates
        top_pool = sorted(cands[:60], key=lambda r: r["file"])
        step = max(1, len(top_pool) // max(1, 15 - len(selected)))
        for r in top_pool[::step]:
            if len(selected) >= 15:
                break
            selected.append(r["file"])
    for p in sorted(SRC_DIR.glob("*.png")):
        if len(selected) >= 15:
            break
        if p.name not in selected:
            selected.append(p.name)
    return [SRC_DIR / name for name in selected[:15]]


def main() -> None:
    pack = load_pack(PACK_DIR)
    detector = get_detector(DetectorType.DBNET, MODELS_DIR, device="cpu")
    recognizer = VLMPeftRecognizer(MODELS_DIR, device="auto")

    frames = _pick_15_frames()
    print(f"Selected {len(frames)} representative MT_im_HL1 frames.", flush=True)

    det_cfg = DetectionConfig(detector=DetectorType.DBNET, granularity=DetectionGranularity.LINE)
    outputs: list[dict[str, Any]] = []
    t_total = time.perf_counter()

    for idx, img_path in enumerate(frames, 1):
        with Image.open(img_path) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)

        raw_regions = detector.detect(rgb, det_cfg)
        sorted_regions = group_and_sort_regions(
            raw_regions,
            direction=pack.manifest.script.direction,
            granularity=DetectionGranularity.LINE,
        )

        # Keep up to top 12 highest-confidence lines per frame preserving reading order
        if len(sorted_regions) > 12:
            top_ids = {
                id(r)
                for r in sorted(sorted_regions, key=lambda x: x.confidence, reverse=True)[:12]
            }
            sorted_regions = [r for r in sorted_regions if id(r) in top_ids]

        crops = []
        for reg in sorted_regions:
            crop = dewarp_polygon(rgb, reg.polygon, padding=6)
            ch, cw = crop.shape[:2]
            if ch < 32 or cw < 32:
                scale = max(32.0 / max(ch, 1), 32.0 / max(cw, 1))
                pil_tmp = Image.fromarray(crop).resize(
                    (max(32, int(round(cw * scale))), max(32, int(round(ch * scale)))),
                    Image.Resampling.BICUBIC,
                )
                crop = np.asarray(pil_tmp, dtype=np.uint8)
            crops.append(crop)
        recs = recognizer.recognize(crops, pack) if crops else []

        line_entries: list[dict[str, Any]] = []
        raw_lines: list[str] = []
        for reg, rec in zip(sorted_regions, recs):
            txt = rec.text.strip()
            line_entries.append(
                {
                    "region_id": reg.id,
                    "confidence": reg.confidence,
                    "raw_basq": txt,
                }
            )
            if txt:
                raw_lines.append(txt)

        joined_raw = "\n".join(raw_lines)
        post = pack.apply_postprocess(joined_raw)

        outputs.append(
            {
                "index": idx,
                "file": img_path.name,
                "detected_total": len(raw_regions),
                "ocr_lines_count": len(sorted_regions),
                "lines": line_entries,
                "raw_ocr_basq": joined_raw,
                "postprocessed_basq": post.text,
                "hl_script": post.original_script,
                "notes": post.notes,
            }
        )
        print(
            f"[{idx}/{len(frames)}] {img_path.name}: det={len(raw_regions)} -> ocr={len(line_entries)} lines | "
            f"preview={joined_raw.replace(chr(10), ' / ')[:90]}",
            flush=True,
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "elapsed_seconds": round(time.perf_counter() - t_total, 2),
                "detector": "dbnet",
                "recognizer": "Qwen/Qwen2.5-VL-3B-Instruct+mt_hl_vlm_lora",
                "frames": outputs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(outputs)} E2E OCR results to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
