"""Training configuration schemas for ONNX OCR, VLM LoRA, and Conlang Text LoRA."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrainOnnxOCRConfig(BaseModel):
    pack_id: str
    dataset_path: str
    output_dir: str
    architecture: str = "svtr_ctc"
    epochs: int = Field(default=30, ge=1, le=500)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    img_height: int = 32
    max_width: int = 320
    export_onnx: bool = True


class TrainVLMLoRAConfig(BaseModel):
    pack_id: str
    dataset_path: str
    output_dir: str
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    epochs: int = Field(default=3, ge=1, le=50)
    batch_size: int = Field(default=2, ge=1, le=64)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=128)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    lora_r: int = 16
    lora_alpha: int = 32
    load_in_4bit: bool = True
    finetune_vision_layers: bool = True
    max_steps: int | None = None
    min_pixels: int = 128 * 28 * 28
    max_pixels: int = 512 * 28 * 28
    save_steps: int = 100


class TrainConlangLoRAConfig(BaseModel):
    pack_id: str
    dataset_jsonl: str
    output_dir: str
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    epochs: int = Field(default=3, ge=1, le=50)
    batch_size: int = Field(default=4, ge=1, le=64)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=128)
    learning_rate: float = Field(default=1.5e-4, gt=0.0)
    lora_r: int = 16
    lora_alpha: int = 32
    load_in_4bit: bool = True
    target_lang: str = "en"
