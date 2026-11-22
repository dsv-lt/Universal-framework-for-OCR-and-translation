"""DBNet++ (Differentiable Binarization) ONNX detector."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from im_translator.core.paths import format_display_path
from im_translator.core.schemas import (
    DetectedRegion,
    DetectionConfig,
    DetectorType,
)
from im_translator.detection.base import ModelNotInstalledError, TextDetector
from im_translator.detection.craft import _select_onnx_providers
from im_translator.detection.dewarp import order_polygon_points


class DBNetDetector(TextDetector):
    """Single-pass DBNet / DBNet++ text detector using ONNX Runtime."""

    detector_type = DetectorType.DBNET

    def __init__(self, model_path: Path | None = None, device: str = "auto") -> None:
        super().__init__(model_path=model_path, device=device)
        self._session: Any = None

    def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session
        if not self.is_available:
            raise ModelNotInstalledError(
                f"DBNet++ ONNX model not found at '{format_display_path(self.model_path)}'. "
                "Place dbnet.onnx in your models_dir/detectors/ directory."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for DBNetDetector. Install with: pip install 'im-translator[onnx]'"
            ) from exc

        providers = _select_onnx_providers(ort, self.device)
        self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        return self._session

    @staticmethod
    def preprocess(image: np.ndarray, max_side: int = 1280) -> tuple[np.ndarray, float, float]:
        """Resize RGB image to multiple of 32, convert RGB->BGR (PaddleOCR/DBNet convention), and normalize."""
        h, w = image.shape[:2]
        scale = min(max_side / max(h, w), 1.0)
        target_h = max(32, int(math.ceil((h * scale) / 32.0) * 32))
        target_w = max(32, int(math.ceil((w * scale) / 32.0) * 32))

        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        # Convert RGB input to BGR as expected by standard PaddleOCR DBNet ONNX weights
        resized_bgr = resized[:, :, ::-1]
        img_f = resized_bgr.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (img_f - mean) / std
        tensor = np.transpose(norm, (2, 0, 1))[None, ...]
        return tensor.astype(np.float32), w / float(target_w), h / float(target_h)

    @staticmethod
    def extract_polygons_from_probmap(
        prob_map: np.ndarray,
        thresh: float = 0.30,
        box_thresh: float = 0.60,
        unclip_ratio: float = 1.60,
        min_area: int = 12,
        ratio_x: float = 1.0,
        ratio_y: float = 1.0,
    ) -> list[DetectedRegion]:
        """Extract and unclip oriented quadrilateral boxes from a DBNet probability map."""
        pmap = np.asarray(prob_map, dtype=np.float32)
        map_h, map_w = pmap.shape[:2]
        bitmap = (pmap >= thresh).astype(np.uint8)

        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        regions: list[DetectedRegion] = []

        for cnt in contours:
            if len(cnt) < 3:
                continue
            area = float(cv2.contourArea(cnt))
            perimeter = float(cv2.arcLength(cnt, True))
            if area < min_area or perimeter <= 0.0:
                continue

            # Compute fast probability score inside contour bbox
            bx, by, bw, bh = cv2.boundingRect(cnt)
            roi_mask = np.zeros((bh, bw), dtype=np.uint8)
            shifted = cnt.copy()
            shifted[:, :, 0] -= bx
            shifted[:, :, 1] -= by
            cv2.fillPoly(roi_mask, [shifted], 1)
            roi_prob = pmap[by : by + bh, bx : bx + bw]
            score = float(np.mean(roi_prob[roi_mask == 1]))
            if score < box_thresh:
                continue

            # DBNet unclip expansion on oriented bounding box
            distance = area * max(unclip_ratio - 1.0, 0.0) / perimeter
            rect = cv2.minAreaRect(cnt.astype(np.float32))
            center, (rw, rh), angle = rect
            expanded_rect = (center, (rw + 2.0 * distance, rh + 2.0 * distance), angle)
            box = cv2.boxPoints(expanded_rect)
            box[:, 0] = np.clip(box[:, 0], 0, max(map_w - 1, 1)) * ratio_x
            box[:, 1] = np.clip(box[:, 1], 0, max(map_h - 1, 1)) * ratio_y
            ordered = order_polygon_points(box)

            regions.append(
                DetectedRegion(
                    id=len(regions),
                    polygon=[(float(px), float(py)) for px, py in ordered],
                    confidence=round(float(np.clip(score, 0.0, 1.0)), 4),
                    detector=DetectorType.DBNET,
                )
            )

        return regions

    def detect(self, image: np.ndarray, config: DetectionConfig) -> list[DetectedRegion]:
        session = self._ensure_session()
        tensor, _, _ = self.preprocess(image)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: tensor})

        prob = np.asarray(outputs[0])
        if prob.ndim == 4:
            prob_map = prob[0, 0, :, :] if prob.shape[1] == 1 else prob[0, :, :, 0]
        elif prob.ndim == 3:
            prob_map = prob[0]
        else:
            raise ValueError(f"Unexpected DBNet ONNX output shape: {prob.shape}")

        map_h, map_w = prob_map.shape[:2]
        orig_h, orig_w = image.shape[:2]
        regions = self.extract_polygons_from_probmap(
            prob_map=prob_map,
            thresh=config.db_thresh,
            box_thresh=config.db_box_thresh,
            unclip_ratio=config.db_unclip_ratio,
            ratio_x=orig_w / float(map_w),
            ratio_y=orig_h / float(map_h),
        )
        for reg in regions:
            reg.granularity = config.granularity
        return regions
