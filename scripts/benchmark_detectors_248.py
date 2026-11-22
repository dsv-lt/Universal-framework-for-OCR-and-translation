"""Benchmark real ONNX CRAFT & DBNet++ detectors across all 248 frames in MT_im_HL1."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from PIL import Image

from im_translator.core.paths import normalize_user_path
from im_translator.core.schemas import DetectionConfig, DetectorType
from im_translator.detection.registry import get_detector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = normalize_user_path(os.environ.get("MT_HL_IMAGES_DIR", r"D:\Personal\download\MT_im_HL1"))
MODELS_DIR = PROJECT_ROOT / "models"
OUT_JSON = PROJECT_ROOT / "outputs" / "mt_hl_det_benchmark_248.json"

_DETECTORS: dict[str, Any] = {}


def _init_worker() -> None:
    for dt in (DetectorType.DBNET, DetectorType.CRAFT):
        _DETECTORS[dt.value] = get_detector(dt, MODELS_DIR, device="cpu")


def _process_one(img_path_str: str) -> dict[str, Any]:
    path = Path(img_path_str)
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    img_area = max(1.0, float(h * w))

    entry: dict[str, Any] = {"file": path.name, "width": w, "height": h}
    for dt in (DetectorType.DBNET, DetectorType.CRAFT):
        det = _DETECTORS[dt.value]
        t0 = time.perf_counter()
        regions = det.detect(rgb, DetectionConfig(detector=dt))
        dt_ms = (time.perf_counter() - t0) * 1000.0
        confs = [float(r.confidence) for r in regions]
        full_frame_hits = 0
        for r in regions:
            xs = [p[0] for p in r.polygon]
            ys = [p[1] for p in r.polygon]
            bw = max(xs) - min(xs)
            bh = max(ys) - min(ys)
            if (bw * bh) / img_area > 0.70:
                full_frame_hits += 1
        entry[dt.value] = {
            "count": len(regions),
            "ms": round(dt_ms, 2),
            "mean_conf": round(sum(confs) / len(confs), 4) if confs else None,
            "full_frame_hits": full_frame_hits,
        }
    return entry


def main() -> None:
    files = sorted(SRC_DIR.glob("*.png"))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(files)} PNG files in {SRC_DIR}", flush=True)

    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    with ProcessPoolExecutor(max_workers=4, initializer=_init_worker) as pool:
        futs = {pool.submit(_process_one, str(p)): p.name for p in files}
        for done_idx, fut in enumerate(as_completed(futs), 1):
            fname = futs[fut]
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                errors.append({"file": fname, "error": str(exc)})
            if done_idx % 50 == 0 or done_idx == len(files):
                print(f"Processed {done_idx}/{len(files)} images ({time.perf_counter() - t0:.1f}s)", flush=True)

    wall_s = time.perf_counter() - t0
    results.sort(key=lambda x: x["file"])

    summary: dict[str, Any] = {
        "source_dir": str(SRC_DIR),
        "total_images": len(files),
        "processed": len(results),
        "errors": errors,
        "wall_seconds": round(wall_s, 2),
        "detectors": {},
    }
    for dt_name in ("dbnet", "craft"):
        counts = [r[dt_name]["count"] for r in results]
        nonzero = sum(1 for c in counts if c > 0)
        confs = [r[dt_name]["mean_conf"] for r in results if r[dt_name]["mean_conf"] is not None]
        ms_list = [r[dt_name]["ms"] for r in results]
        ff_hits = sum(r[dt_name]["full_frame_hits"] for r in results)
        summary["detectors"][dt_name] = {
            "images_with_detections": nonzero,
            "zero_detection_images": len(results) - nonzero,
            "total_regions": int(sum(counts)),
            "mean_regions_per_image": round(sum(counts) / max(1, len(counts)), 2),
            "min_regions": min(counts) if counts else 0,
            "max_regions": max(counts) if counts else 0,
            "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
            "mean_ms_per_image": round(sum(ms_list) / max(1, len(ms_list)), 2),
            "full_frame_border_hits": ff_hits,
        }

    OUT_JSON.write_text(
        json.dumps({"summary": summary, "per_image": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
