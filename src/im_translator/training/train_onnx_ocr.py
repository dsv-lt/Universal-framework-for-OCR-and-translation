"""ONNX OCR (CTC) training plan builder and environment validator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from im_translator.core.pack_loader import load_pack
from im_translator.training.schemas import TrainOnnxOCRConfig


def check_train_onnx_dependencies() -> dict[str, bool]:
    pkgs = ("torch", "onnxruntime", "datasets", "jiwer")
    return {pkg: importlib.util.find_spec(pkg) is not None for pkg in pkgs}


def prepare_onnx_ocr_training(
    config: TrainOnnxOCRConfig,
    packs_dir: Path,
) -> dict[str, Any]:
    """Validate dataset & pack for ONNX OCR training and build a reproducible run plan."""
    pack = load_pack(packs_dir / config.pack_id)
    dataset_path = Path(config.dataset_path).resolve()
    deps = check_train_onnx_dependencies()
    missing = [name for name, ok in deps.items() if not ok]

    dataset_exists = dataset_path.is_file()
    sample_count = 0
    if dataset_exists:
        with dataset_path.open("r", encoding="utf-8") as fh:
            sample_count = sum(1 for line in fh if line.strip())

    return {
        "stage": "train_onnx_ocr",
        "pack_id": pack.pack_id,
        "charset_size": len(pack.manifest.script.charset),
        "dataset_path": str(dataset_path),
        "dataset_exists": dataset_exists,
        "sample_count": sample_count,
        "output_dir": str(Path(config.output_dir).resolve()),
        "dependencies": deps,
        "missing_dependencies": missing,
        "ready": dataset_exists and sample_count > 0 and len(missing) == 0,
        "install_hint": "pip install 'im-translator[onnx,vlm,train]'" if missing else "",
        "config": config.model_dump(mode="json"),
    }
