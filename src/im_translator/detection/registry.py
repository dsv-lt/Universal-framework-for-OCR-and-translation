"""Detector factory and capability status helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from im_translator.core.paths import format_display_path
from im_translator.core.schemas import DetectorType
from im_translator.detection.base import TextDetector
from im_translator.detection.craft import CRAFTDetector
from im_translator.detection.dbnet import DBNetDetector


def get_detector(
    detector_type: DetectorType,
    models_dir: Path,
    device: str = "auto",
) -> TextDetector:
    """Instantiate a CRAFT or DBNet++ detector bound to `<models_dir>/detectors/`."""
    det_dir = models_dir / "detectors"
    if detector_type == DetectorType.CRAFT:
        return CRAFTDetector(model_path=det_dir / "craft.onnx", device=device)
    if detector_type == DetectorType.DBNET:
        return DBNetDetector(model_path=det_dir / "dbnet.onnx", device=device)
    raise ValueError(f"Unsupported detector type: {detector_type}")


def list_detectors_status(models_dir: Path) -> list[dict[str, Any]]:
    """Return installation & weight availability for all supported detectors."""
    try:
        import onnxruntime  # noqa: F401
        ort_installed = True
    except ImportError:
        ort_installed = False

    statuses: list[dict[str, Any]] = []
    for det_type in (DetectorType.CRAFT, DetectorType.DBNET):
        det = get_detector(det_type, models_dir)
        statuses.append(
            {
                "detector": det_type.value,
                "model_path": format_display_path(det.model_path),
                "weights_present": det.is_available,
                "onnxruntime_installed": ort_installed,
                "ready": bool(det.is_available and ort_installed),
            }
        )
    return statuses
