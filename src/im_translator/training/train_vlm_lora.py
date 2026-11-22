"""VLM QLoRA (Qwen2.5-VL with finetune_vision_layers=True) training planner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from im_translator.core.pack_loader import load_pack
from im_translator.training.schemas import TrainVLMLoRAConfig


def check_vlm_train_dependencies() -> dict[str, bool]:
    pkgs = ("torch", "transformers", "peft", "accelerate", "datasets", "jiwer")
    return {pkg: importlib.util.find_spec(pkg) is not None for pkg in pkgs}


def prepare_vlm_lora_training(
    config: TrainVLMLoRAConfig,
    packs_dir: Path,
) -> dict[str, Any]:
    """Validate dataset & pack for VLM LoRA fine-tuning and return reproducible plan."""
    pack = load_pack(packs_dir / config.pack_id)
    dataset_path = Path(config.dataset_path).resolve()
    deps = check_vlm_train_dependencies()
    missing = [name for name, ok in deps.items() if not ok]

    dataset_exists = dataset_path.is_file()
    sample_count = 0
    if dataset_exists:
        with dataset_path.open("r", encoding="utf-8") as fh:
            sample_count = sum(1 for line in fh if line.strip())

    return {
        "stage": "train_vlm_lora",
        "pack_id": pack.pack_id,
        "base_model": config.base_model,
        "finetune_vision_layers": config.finetune_vision_layers,
        "dataset_path": str(dataset_path),
        "dataset_exists": dataset_exists,
        "sample_count": sample_count,
        "output_dir": str(Path(config.output_dir).resolve()),
        "dependencies": deps,
        "missing_dependencies": missing,
        "ready": dataset_exists and sample_count > 0 and len(missing) == 0,
        "install_hint": "pip install 'im-translator[vlm,train]'" if missing else "",
        "llamacpp_note": (
            "Remember: llama.cpp cannot dynamically load vision LoRA weights at runtime; "
            "either run with vlm_peft or merge adapter + export GGUF/mmproj."
        ),
        "config": config.model_dump(mode="json"),
    }
