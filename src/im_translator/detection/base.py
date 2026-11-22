"""Abstract base interface for text detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from im_translator.core.schemas import DetectedRegion, DetectionConfig, DetectorType


class ModelNotInstalledError(FileNotFoundError):
    """Raised when a required detector or recognizer model weight file is missing."""


class TextDetector(ABC):
    """Base interface implemented by CRAFT and DBNet++ detectors."""

    detector_type: DetectorType

    def __init__(self, model_path: Path | None = None, device: str = "auto") -> None:
        self.model_path = model_path
        self.device = device

    @property
    def is_available(self) -> bool:
        return self.model_path is not None and self.model_path.is_file()

    @abstractmethod
    def detect(self, image: np.ndarray, config: DetectionConfig) -> list[DetectedRegion]:
        """Detect text polygons in an RGB or BGR uint8 numpy image."""
