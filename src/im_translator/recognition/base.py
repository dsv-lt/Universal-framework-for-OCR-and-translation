"""Abstract base interface for OCR / VLM text recognizers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from im_translator.core.pack_loader import LoadedPack
from im_translator.core.schemas import RecognitionOutput, RecognizerBackend


class TextRecognizer(ABC):
    """Unified recognizer contract shared by ONNX OCR, VLM PEFT, and VLM llama.cpp."""

    backend_name: RecognizerBackend

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Return True if runtime dependencies and endpoint/weights can be used."""

    @abstractmethod
    def recognize(self, crops: list[np.ndarray], pack: LoadedPack) -> list[RecognitionOutput]:
        """Recognize text from dewarped RGB uint8 crop images."""
