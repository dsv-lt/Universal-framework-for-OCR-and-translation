"""Recognizer factory and runtime availability reporting."""

from __future__ import annotations

from typing import Any

from im_translator.core.schemas import RecognizerBackend
from im_translator.core.settings import AppSettings
from im_translator.recognition.base import TextRecognizer
from im_translator.recognition.onnx_ocr import OnnxOCRRecognizer
from im_translator.recognition.vlm_llamacpp import VLMLlamaCppRecognizer
from im_translator.recognition.vlm_peft import VLMPeftRecognizer


def get_recognizer(backend: RecognizerBackend, settings: AppSettings) -> TextRecognizer:
    """Instantiate a TextRecognizer for the requested backend."""
    if backend == RecognizerBackend.ONNX_OCR:
        return OnnxOCRRecognizer(models_dir=settings.models_path, device=settings.device)
    if backend == RecognizerBackend.VLM_PEFT:
        return VLMPeftRecognizer(models_dir=settings.models_path, device=settings.device)
    if backend == RecognizerBackend.VLM_LLAMACPP:
        return VLMLlamaCppRecognizer(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    raise ValueError(f"Unsupported recognizer backend: {backend}")


def list_recognizers_status(settings: AppSettings) -> list[dict[str, Any]]:
    """Return availability status of all recognizer backends."""
    statuses: list[dict[str, Any]] = []
    for backend in RecognizerBackend:
        rec = get_recognizer(backend, settings)
        statuses.append(
            {
                "backend": backend.value,
                "runtime_available": rec.is_available,
            }
        )
    return statuses
