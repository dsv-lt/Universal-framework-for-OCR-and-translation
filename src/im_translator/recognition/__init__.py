"""Recognition backends (ONNX OCR, VLM PEFT, VLM llama.cpp)."""

from im_translator.recognition.base import TextRecognizer
from im_translator.recognition.onnx_ocr import OnnxOCRRecognizer, crop_to_png_base64
from im_translator.recognition.registry import get_recognizer, list_recognizers_status
from im_translator.recognition.vlm_llamacpp import VLMLlamaCppRecognizer
from im_translator.recognition.vlm_peft import VLMPeftRecognizer

__all__ = [
    "OnnxOCRRecognizer",
    "TextRecognizer",
    "VLMLlamaCppRecognizer",
    "VLMPeftRecognizer",
    "crop_to_png_base64",
    "get_recognizer",
    "list_recognizers_status",
]
