"""Text detection (CRAFT / DBNet++ ONNX) and perspective dewarping."""

from im_translator.detection.base import ModelNotInstalledError, TextDetector
from im_translator.detection.craft import CRAFTDetector
from im_translator.detection.dbnet import DBNetDetector
from im_translator.detection.dewarp import (
    detect_contours_fallback,
    dewarp_polygon,
    draw_regions_overlay,
    group_and_sort_regions,
    order_polygon_points,
)
from im_translator.detection.registry import get_detector, list_detectors_status

__all__ = [
    "CRAFTDetector",
    "DBNetDetector",
    "ModelNotInstalledError",
    "TextDetector",
    "detect_contours_fallback",
    "dewarp_polygon",
    "draw_regions_overlay",
    "get_detector",
    "group_and_sort_regions",
    "list_detectors_status",
    "order_polygon_points",
]
