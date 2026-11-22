"""Optional Conlang Text LoRA (`[Conlang] -> EN`) training planner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from im_translator.core.pack_loader import load_pack
from im_translator.training.schemas import TrainConlangLoRAConfig


def check_conlang_train_dependencies() -> dict[str, bool]:
    pkgs = ("torch", "transformers", "peft", "accelerate", "datasets")
    return {pkg: importlib.util.find_spec(pkg) is not None for pkg in pkgs}


def prepare_conlang_lora_training(
    config: TrainConlangLoRAConfig,
    packs_dir: Path,
) -> dict[str, Any]:
    """Validate parallel `conlang_lora.jsonl` dataset and build training summary."""
    pack = load_pack(packs_dir / config.pack_id)
    jsonl_path = Path(config.dataset_jsonl).resolve()
    deps = check_conlang_train_dependencies()
    missing = [name for name, ok in deps.items() if not ok]

    dataset_exists = jsonl_path.is_file()
    valid_pairs = 0
    if dataset_exists:
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("source") and obj.get("en"):
                    valid_pairs += 1

    recommendation = (
        "Dataset < 500 pairs: prefer GlossaryRAG + grammar.md without LoRA training."
        if valid_pairs < 500
        else "Dataset size suitable for Conlang Text LoRA + GlossaryRAG overlay."
    )

    return {
        "stage": "train_conlang_lora",
        "pack_id": pack.pack_id,
        "base_model": config.base_model,
        "dataset_jsonl": str(jsonl_path),
        "dataset_exists": dataset_exists,
        "valid_pairs": valid_pairs,
        "recommendation": recommendation,
        "output_dir": str(Path(config.output_dir).resolve()),
        "dependencies": deps,
        "missing_dependencies": missing,
        "ready": dataset_exists and valid_pairs > 0 and len(missing) == 0,
        "install_hint": "pip install 'im-translator[vlm,train]'" if missing else "",
        "config": config.model_dump(mode="json"),
    }
