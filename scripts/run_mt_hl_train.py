"""Train Qwen2.5-VL-3B-Instruct 4-bit NF4 QLoRA (vision + LLM) for packs/mt_hl and evaluate on val.jsonl."""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "hf_cache"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "models" / "torch_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

import torch

from im_translator.training.schemas import TrainVLMLoRAConfig
from im_translator.training.train_vlm_lora import run_vlm_lora_training

PACKS_DIR = PROJECT_ROOT / "packs"
TRAIN_JSONL = PROJECT_ROOT / "datasets" / "mt_hl" / "train.jsonl"
VAL_JSONL = PROJECT_ROOT / "datasets" / "mt_hl" / "val.jsonl"
ADAPTER_DIR = PACKS_DIR / "mt_hl" / "models" / "vlm_lora"
OUT_EVAL_JSON = PROJECT_ROOT / "outputs" / "mt_hl_val_eval.json"


def main() -> None:
    OUT_EVAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    cfg = TrainVLMLoRAConfig(
        pack_id="mt_hl",
        dataset_path=str(TRAIN_JSONL),
        output_dir=str(ADAPTER_DIR),
        base_model="Qwen/Qwen2.5-VL-3B-Instruct",
        epochs=1,
        max_steps=250,
        batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,
        lora_r=16,
        lora_alpha=32,
        load_in_4bit=True,
        finetune_vision_layers=True,
        min_pixels=64 * 28 * 28,
        max_pixels=192 * 28 * 28,
        save_steps=50,
    )

    try:
        res = run_vlm_lora_training(cfg, packs_dir=PACKS_DIR, val_dataset_path=VAL_JSONL)
    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM at batch_size=1, max_pixels=192*28*28; falling back to max_pixels=144*28*28.", flush=True)
        torch.cuda.empty_cache()
        cfg.max_pixels = 144 * 28 * 28
        res = run_vlm_lora_training(cfg, packs_dir=PACKS_DIR, val_dataset_path=VAL_JSONL)

    val = res.get("validation", {})
    all_samples = val.get("samples", [])
    compact = dict(res)
    compact["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else 0.0
    compact["sample_predictions_20"] = all_samples[:20]

    OUT_EVAL_JSON.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"DONE train+eval | peak_vram={compact['peak_vram_gb']}GB | "
        f"val_cer={val.get('cer'):.4f} | val_wer={val.get('wer'):.4f} | "
        f"exact={val.get('exact_match')}/{val.get('num_samples')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
