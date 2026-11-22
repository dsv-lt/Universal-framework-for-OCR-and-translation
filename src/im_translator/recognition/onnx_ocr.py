"""Lightweight ONNX Runtime OCR recognizer (PARSeq / SVTR / CTC models)."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from im_translator.core.pack_loader import LoadedPack
from im_translator.core.schemas import RecognitionOutput, RecognizerBackend
from im_translator.detection.base import ModelNotInstalledError
from im_translator.detection.craft import _select_onnx_providers
from im_translator.recognition.base import TextRecognizer


def crop_to_png_base64(crop: np.ndarray) -> str:
    """Encode a uint8 RGB numpy crop as base64 PNG string."""
    img = Image.fromarray(crop.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class OnnxOCRRecognizer(TextRecognizer):
    """Line-level ONNX OCR recognizer with CTC greedy decoding."""

    backend_name = RecognizerBackend.ONNX_OCR

    def __init__(self, models_dir: Path, device: str = "auto", img_height: int = 32, max_width: int = 320) -> None:
        self.models_dir = models_dir
        self.device = device
        self.img_height = img_height
        self.max_width = max_width
        self._sessions: dict[str, Any] = {}

    @property
    def is_available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False

    def _resolve_model_path(self, pack: LoadedPack) -> Path:
        if pack.manifest.recognition.onnx_model_path:
            p = pack.resolve_path(pack.manifest.recognition.onnx_model_path)
            if p is not None and p.is_file():
                return p
            if p is not None:
                raise ModelNotInstalledError(f"ONNX OCR model not found at '{p}'.")
        default_path = (self.models_dir / "ocr" / f"{pack.pack_id}.onnx").resolve()
        if not default_path.is_file():
            raise ModelNotInstalledError(
                f"ONNX OCR model not found for pack '{pack.pack_id}'. "
                f"Expected '{default_path}' or recognition.onnx_model_path in manifest.yaml."
            )
        return default_path

    def _get_session(self, model_path: Path) -> Any:
        key = str(model_path)
        if key in self._sessions:
            return self._sessions[key]
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for OnnxOCRRecognizer. Install with: pip install 'im-translator[onnx]'"
            ) from exc
        providers = _select_onnx_providers(ort, self.device)
        session = ort.InferenceSession(key, providers=providers)
        self._sessions[key] = session
        return session

    def preprocess_crop(self, crop: np.ndarray, channels: int = 3) -> np.ndarray:
        """Aspect-ratio-preserving resize to target height and pad to max_width."""
        h, w = crop.shape[:2]
        scale = self.img_height / max(float(h), 1.0)
        new_w = max(1, min(self.max_width, int(round(w * scale))))
        resized = cv2.resize(crop, (new_w, self.img_height), interpolation=cv2.INTER_CUBIC)
        if channels == 1 and resized.ndim == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)[:, :, None]

        canvas = np.zeros((self.img_height, self.max_width, channels), dtype=np.float32)
        if resized.ndim == 2:
            resized = resized[:, :, None]
        canvas[:, :new_w, :] = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = np.transpose(canvas, (2, 0, 1))[None, ...]
        return tensor.astype(np.float32)

    @staticmethod
    def ctc_greedy_decode(logits: np.ndarray, charset: str, blank_idx: int = 0) -> tuple[str, float]:
        """Decode (T, C) CTC logits into text and mean token probability."""
        arr = np.asarray(logits, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        # Softmax along class axis
        shifted = arr - np.max(arr, axis=-1, keepdims=True)
        probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
        best_indices = np.argmax(probs, axis=-1)
        best_probs = np.max(probs, axis=-1)

        chars: list[str] = []
        confidences: list[float] = []
        prev = blank_idx
        for idx, prob in zip(best_indices.tolist(), best_probs.tolist()):
            if idx != blank_idx and idx != prev:
                char_pos = idx - 1 if blank_idx == 0 else idx
                if 0 <= char_pos < len(charset):
                    chars.append(charset[char_pos])
                    confidences.append(float(prob))
            prev = idx

        mean_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
        return "".join(chars), round(mean_conf, 4)

    def recognize(self, crops: list[np.ndarray], pack: LoadedPack) -> list[RecognitionOutput]:
        if not crops:
            return []
        model_path = self._resolve_model_path(pack)
        session = self._get_session(model_path)
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        shape = input_meta.shape
        channels = shape[1] if len(shape) == 4 and isinstance(shape[1], int) else 3

        results: list[RecognitionOutput] = []
        for crop in crops:
            tensor = self.preprocess_crop(crop, channels=channels)
            logits = session.run(None, {input_name: tensor})[0]
            text, conf = self.ctc_greedy_decode(logits, pack.manifest.script.charset)
            results.append(
                RecognitionOutput(
                    text=text,
                    confidence=conf,
                    backend=self.backend_name,
                    raw_metadata={"model_path": str(model_path)},
                )
            )
        return results
