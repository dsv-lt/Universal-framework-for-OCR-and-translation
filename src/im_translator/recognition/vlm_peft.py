"""PyTorch + PEFT Vision-Language Model recognizer (Qwen2.5-VL + per-pack LoRA)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from im_translator.core.pack_loader import LoadedPack
from im_translator.core.schemas import RecognitionOutput, RecognizerBackend
from im_translator.detection.base import ModelNotInstalledError
from im_translator.recognition.base import TextRecognizer


class VLMPeftRecognizer(TextRecognizer):
    """VLM recognizer backed by Transformers + PEFT adapters with lazy imports."""

    backend_name = RecognizerBackend.VLM_PEFT

    def __init__(self, models_dir: Path, device: str = "auto") -> None:
        self.models_dir = models_dir
        self.device = device
        self._cache: dict[tuple[str, str | None], tuple[Any, Any]] = {}

    @property
    def is_available(self) -> bool:
        try:
            import peft  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self, base_model_id: str, adapter_path: Path | None) -> tuple[Any, Any]:
        cache_key = (base_model_id, str(adapter_path) if adapter_path else None)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "torch, transformers, and peft are required for VLMPeftRecognizer. "
                "Install with: pip install 'im-translator[vlm]'"
            ) from exc

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if (self.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"

        processor = AutoProcessor.from_pretrained(base_model_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
        self._cache[cache_key] = (model, processor)
        return model, processor

    def recognize(self, crops: list[np.ndarray], pack: LoadedPack) -> list[RecognitionOutput]:
        if not crops:
            return []

        base_model = pack.manifest.recognition.vlm_base_model or "Qwen/Qwen2.5-VL-3B-Instruct"
        adapter_path: Path | None = None
        if pack.manifest.recognition.vlm_adapter_path:
            adapter_path = pack.resolve_path(pack.manifest.recognition.vlm_adapter_path)
            if adapter_path is None or not adapter_path.exists():
                raise ModelNotInstalledError(
                    f"VLM LoRA adapter not found at '{adapter_path}' for pack '{pack.pack_id}'."
                )

        model, processor = self._load_model(base_model, adapter_path)
        prompt = pack.manifest.recognition.vlm_prompt

        outputs: list[RecognitionOutput] = []
        for crop in crops:
            pil_img = Image.fromarray(crop.astype(np.uint8)).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text_prompt],
                images=[pil_img],
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
            outputs.append(
                RecognitionOutput(
                    text=decoded,
                    confidence=None,
                    backend=self.backend_name,
                    raw_metadata={
                        "base_model": base_model,
                        "adapter_path": str(adapter_path) if adapter_path else None,
                    },
                )
            )
        return outputs
