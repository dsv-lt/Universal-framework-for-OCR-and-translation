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


def _resolve_jsonl_path(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    if path.is_file():
        return path
    for candidate in ("train.jsonl", "labels.jsonl", "val.jsonl"):
        if (path / candidate).exists():
            return path / candidate
    raise FileNotFoundError(f"Could not locate JSONL dataset file in '{path}'.")


def _load_jsonl_samples(jsonl_path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    import json

    samples: list[dict[str, Any]] = []
    base_dir = jsonl_path.parent
    for raw_line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        item = json.loads(raw_line)
        img_rel = item.get("image")
        text = str(item.get("text", "")).strip()
        if not img_rel or not text:
            continue
        img_path = Path(img_rel) if Path(img_rel).is_absolute() else (base_dir / img_rel)
        if not img_path.exists():
            continue
        samples.append({"image_path": img_path, "text": text, "id": item.get("id", len(samples))})
        if max_samples is not None and len(samples) >= max_samples:
            break
    if not samples:
        raise ValueError(f"No valid (image, text) samples found in '{jsonl_path}'.")
    return samples


def _ensure_min_image_size(pil_img: Any, min_side: int = 28) -> Any:
    from PIL import Image as _PILImage

    w, h = pil_img.size
    if w >= min_side and h >= min_side:
        return pil_img
    scale = max(min_side / max(w, 1), min_side / max(h, 1))
    return pil_img.resize(
        (max(min_side, int(round(w * scale))), max(min_side, int(round(h * scale)))),
        _PILImage.Resampling.BICUBIC,
    )


def evaluate_vlm_ocr(
    model: Any,
    processor: Any,
    val_jsonl: Path,
    prompt_text: str,
    max_samples: int | None = None,
    max_new_tokens: int = 96,
) -> dict[str, Any]:
    """Evaluate Qwen2.5-VL OCR model on a held-out JSONL dataset using jiwer CER & WER."""
    import jiwer
    import torch
    from PIL import Image as _PILImage

    samples = _load_jsonl_samples(val_jsonl, max_samples=max_samples)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    refs: list[str] = []
    preds: list[str] = []
    details: list[dict[str, Any]] = []
    exact = 0

    for item in samples:
        with _PILImage.open(item["image_path"]) as im:
            pil_img = _ensure_min_image_size(im.convert("RGB"))

        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        chat_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat_text], images=[pil_img], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)

        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        in_len = inputs.input_ids.shape[1]
        trimmed = out_ids[0, in_len:]
        pred = processor.tokenizer.decode(trimmed, skip_special_tokens=True).strip()
        pred = "\n".join(" ".join(ln.split()) for ln in pred.splitlines() if ln.strip())
        ref = "\n".join(" ".join(ln.split()) for ln in item["text"].splitlines() if ln.strip())

        refs.append(ref)
        preds.append(pred if pred else " ")
        if pred == ref:
            exact += 1
        details.append({"id": item["id"], "image": str(item["image_path"]), "gt": ref, "pred": pred})

    cer = float(jiwer.cer(refs, preds))
    wer = float(jiwer.wer(refs, preds))
    return {
        "num_samples": len(samples),
        "exact_match": exact,
        "exact_match_rate": exact / max(1, len(samples)),
        "cer": cer,
        "wer": wer,
        "samples": details,
    }


def run_vlm_lora_training(
    config: TrainVLMLoRAConfig,
    packs_dir: Path,
    val_dataset_path: Path | None = None,
) -> dict[str, Any]:
    """Run real 4-bit NF4 QLoRA fine-tuning of Qwen2.5-VL (vision + LLM) for a Language Pack."""
    import gc
    import math
    import random
    import time

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from PIL import Image as _PILImage
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
        get_cosine_schedule_with_warmup,
    )

    from im_translator.core.pack_loader import load_pack

    pack = load_pack(packs_dir / config.pack_id)
    prompt_text = pack.manifest.recognition.vlm_prompt
    train_jsonl = _resolve_jsonl_path(config.dataset_path)
    train_samples = _load_jsonl_samples(train_jsonl)

    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = (pack.root_dir / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        if (config.load_in_4bit and use_cuda)
        else None
    )

    try:
        processor = AutoProcessor.from_pretrained(config.base_model, local_files_only=True)
    except OSError:
        processor = AutoProcessor.from_pretrained(config.base_model)
    processor.tokenizer.padding_side = "right"
    if hasattr(processor, "image_processor"):
        if hasattr(processor.image_processor, "size") and isinstance(processor.image_processor.size, dict):
            processor.image_processor.size = {"shortest_edge": config.min_pixels, "longest_edge": config.max_pixels}
        if hasattr(processor.image_processor, "min_pixels"):
            processor.image_processor.min_pixels = config.min_pixels
        if hasattr(processor.image_processor, "max_pixels"):
            processor.image_processor.max_pixels = config.max_pixels

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.base_model,
            dtype=dtype,
            device_map="auto" if use_cuda else None,
            quantization_config=bnb_config,
            local_files_only=True,
        )
    except OSError:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.base_model,
            dtype=dtype,
            device_map="auto" if use_cuda else None,
            quantization_config=bnb_config,
        )
    model.config.use_cache = False

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    def _require_grad_hook(_module: Any, _args: Any, output: Any) -> None:
        if isinstance(output, torch.Tensor) and output.is_floating_point():
            output.requires_grad_(True)

    if config.finetune_vision_layers and hasattr(model, "visual") and hasattr(model.visual, "patch_embed"):
        model.visual.patch_embed.register_forward_hook(_require_grad_hook)

    target_modules: str | list[str]
    if config.finetune_vision_layers:
        target_modules = r".*(visual\.merger\.mlp\.(0|2)|\.attn\.(qkv|proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
    else:
        target_modules = r".*(self_attn\.(q_proj|k_proj|v_proj|o_proj)|model\.(language_model\.)?layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj))$"

    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    vision_lora_params = [
        (n, p) for n, p in model.named_parameters() if p.requires_grad and "visual." in n
    ]
    llm_lora_params = [
        (n, p) for n, p in model.named_parameters() if p.requires_grad and "visual." not in n
    ]
    print(
        f"Trainable LoRA tensors -> vision: {len(vision_lora_params)}, llm: {len(llm_lora_params)}",
        flush=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=0.01)

    steps_per_epoch = max(1, math.ceil(len(train_samples) / config.batch_size / config.gradient_accumulation_steps))
    total_opt_steps = config.max_steps if config.max_steps is not None else steps_per_epoch * config.epochs
    warmup_steps = min(30, max(5, total_opt_steps // 10))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps)

    def _collate(batch_items: list[dict[str, Any]]) -> tuple[dict[str, Any], torch.Tensor]:
        imgs = []
        prompt_texts = []
        full_texts = []
        for item in batch_items:
            with _PILImage.open(item["image_path"]) as im:
                pil_img = _ensure_min_image_size(im.convert("RGB"))
            imgs.append(pil_img)
            u_msgs = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": prompt_text}],
                }
            ]
            f_msgs = u_msgs + [{"role": "assistant", "content": [{"type": "text", "text": item["text"]}]}]
            prompt_texts.append(processor.apply_chat_template(u_msgs, tokenize=False, add_generation_prompt=True))
            full_texts.append(processor.apply_chat_template(f_msgs, tokenize=False, add_generation_prompt=False))

        batch_enc = processor(text=full_texts, images=imgs, padding=True, return_tensors="pt")
        labels = batch_enc["input_ids"].clone()
        labels[batch_enc["attention_mask"] == 0] = -100

        for i, (p_txt, im) in enumerate(zip(prompt_texts, imgs)):
            p_enc = processor(text=[p_txt], images=[im], return_tensors="pt")
            p_len = int(p_enc["input_ids"].shape[1])
            labels[i, :p_len] = -100

        return batch_enc, labels

    rng = random.Random(42)
    model.train()
    optimizer.zero_grad()

    opt_step = 0
    accum_loss = 0.0
    accum_count = 0
    loss_history: list[dict[str, float]] = []
    vision_grad_verified = False
    t_start = time.time()
    t_step = t_start

    stop_training = False
    for epoch in range(max(1, config.epochs)):
        indices = list(range(len(train_samples)))
        rng.shuffle(indices)
        for start in range(0, len(indices), config.batch_size):
            batch_items = [train_samples[j] for j in indices[start : start + config.batch_size]]
            batch_enc, labels = _collate(batch_items)
            batch_enc = batch_enc.to(model.device)
            labels = labels.to(model.device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                out = model(**batch_enc, labels=labels)
                raw_loss = out.loss
                loss = raw_loss / config.gradient_accumulation_steps

            loss.backward()
            accum_loss += float(raw_loss.detach().item())
            accum_count += 1

            if accum_count >= config.gradient_accumulation_steps or (start + config.batch_size >= len(indices)):
                if not vision_grad_verified and config.finetune_vision_layers and vision_lora_params:
                    nonzero = sum(
                        1 for _, p in vision_lora_params if p.grad is not None and float(p.grad.abs().sum().item()) > 0.0
                    )
                    print(f"Step 1 vision LoRA nonzero-grad check: {nonzero}/{len(vision_lora_params)}", flush=True)
                    if nonzero == 0:
                        raise RuntimeError("Vision LoRA parameters received zero gradients!")
                    vision_grad_verified = True

                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                opt_step += 1
                avg_loss = accum_loss / max(1, accum_count)
                accum_loss = 0.0
                accum_count = 0

                now = time.time()
                if opt_step == 1 or opt_step % 10 == 0 or opt_step >= total_opt_steps:
                    dt = now - t_step
                    sec_per_step = dt / (1.0 if opt_step == 1 else 10.0)
                    lr_now = float(scheduler.get_last_lr()[0])
                    print(
                        f"[epoch={epoch + 1} step={opt_step}/{total_opt_steps}] "
                        f"loss={avg_loss:.4f} grad_norm={grad_norm:.3f} lr={lr_now:.2e} "
                        f"sec/step={sec_per_step:.2f} elapsed={now - t_start:.1f}s",
                        flush=True,
                    )
                    loss_history.append({"step": opt_step, "loss": avg_loss, "lr": lr_now, "sec_per_step": sec_per_step})
                    t_step = now

                if config.save_steps > 0 and opt_step % config.save_steps == 0:
                    model.save_pretrained(str(output_dir))
                    processor.save_pretrained(str(output_dir))

                if opt_step >= total_opt_steps:
                    stop_training = True
                    break
        if stop_training:
            break

    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    total_train_sec = time.time() - t_start

    result: dict[str, Any] = {
        "stage": "train_vlm_lora",
        "pack_id": config.pack_id,
        "base_model": config.base_model,
        "output_dir": str(output_dir),
        "train_samples": len(train_samples),
        "optimizer_steps": opt_step,
        "train_seconds": round(total_train_sec, 2),
        "vision_lora_tensors": len(vision_lora_params),
        "llm_lora_tensors": len(llm_lora_params),
        "loss_history": loss_history,
        "final_train_loss": loss_history[-1]["loss"] if loss_history else None,
    }

    if val_dataset_path is not None:
        val_jsonl = _resolve_jsonl_path(val_dataset_path)
        print(f"Evaluating fine-tuned model on {val_jsonl}...", flush=True)
        val_metrics = evaluate_vlm_ocr(model, processor, val_jsonl, prompt_text=prompt_text)
        result["validation"] = val_metrics

    del model, optimizer, scheduler
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    return result
