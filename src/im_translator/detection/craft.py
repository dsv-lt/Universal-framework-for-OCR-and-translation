"""CRAFT (Character Region Awareness for Text Detection) ONNX detector."""

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
from im_translator.detection.dewarp import order_polygon_points


class CRAFTDetector(TextDetector):
    """Single-pass CRAFT text detector using ONNX Runtime."""

    detector_type = DetectorType.CRAFT

    def __init__(self, model_path: Path | None = None, device: str = "auto") -> None:
        super().__init__(model_path=model_path, device=device)
        self._session: Any = None

    def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session
        if not self.is_available:
            raise ModelNotInstalledError(
                f"CRAFT ONNX model not found at '{format_display_path(self.model_path)}'. "
                "Place craft.onnx in your models_dir/detectors/ directory."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for CRAFTDetector. Install with: pip install 'im-translator[onnx]'"
            ) from exc

        providers = _select_onnx_providers(ort, self.device)
        self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        return self._session

    @staticmethod
    def preprocess(image: np.ndarray, max_side: int = 1280) -> tuple[np.ndarray, float, float]:
        """Resize image to multiple of 32 and apply ImageNet normalization (NCHW float32)."""
        h, w = image.shape[:2]
        scale = min(max_side / max(h, w), 1.0)
        target_h = max(32, int(math.ceil((h * scale) / 32.0) * 32))
        target_w = max(32, int(math.ceil((w * scale) / 32.0) * 32))

        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_f = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (img_f - mean) / std
        tensor = np.transpose(norm, (2, 0, 1))[None, ...]
        ratio_x = w / float(target_w)
        ratio_y = h / float(target_h)
        return tensor.astype(np.float32), ratio_x, ratio_y

    @staticmethod
    def extract_polygons_from_heatmaps(
        region_score: np.ndarray,
        affinity_score: np.ndarray,
        text_threshold: float = 0.70,
        link_threshold: float = 0.40,
        low_text: float = 0.40,
        min_area: int = 10,
        ratio_x: float = 1.0,
        ratio_y: float = 1.0,
    ) -> list[DetectedRegion]:
        """Convert CRAFT region/affinity score maps into oriented quadrilateral regions."""
        reg = np.asarray(region_score, dtype=np.float32)
        aff = np.asarray(affinity_score, dtype=np.float32)
        img_h, img_w = reg.shape[:2]

        text_mask = (reg >= low_text).astype(np.uint8)
        link_mask = (aff >= link_threshold).astype(np.uint8)
        combined = np.clip(text_mask + link_mask, 0, 1).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=4)
        regions: list[DetectedRegion] = []

        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
            bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

            roi_labels = labels[y : y + bh, x : x + bw]
            roi_reg = reg[y : y + bh, x : x + bw]
            comp_mask = roi_labels == label_id
            if float(np.max(roi_reg[comp_mask])) < text_threshold:
                continue

            # Expand slightly around character centers to avoid clipping outer strokes
            niter = max(1, int(math.sqrt(area * min(bw, bh) / max(bw * bh, 1))))
            ksize = min(9, max(3, niter * 2 + 1))
            pad = ksize
            segmap = np.zeros((bh + 2 * pad, bw + 2 * pad), dtype=np.uint8)
            segmap[pad : pad + bh, pad : pad + bw][comp_mask] = 255
            # Remove pure link bridges when eroding/dilating? Standard CRAFT keeps combined dilated
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            segmap = cv2.dilate(segmap, kernel, iterations=1)

            ys, xs = np.where(segmap > 0)
            if len(xs) < 4:
                continue
            pts = np.column_stack((xs - pad + x, ys - pad + y)).astype(np.float32)
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)
            box[:, 0] = np.clip(box[:, 0], 0, max(img_w - 1, 1)) * ratio_x
            box[:, 1] = np.clip(box[:, 1], 0, max(img_h - 1, 1)) * ratio_y
            ordered = order_polygon_points(box)

            conf = float(np.clip(np.mean(roi_reg[comp_mask]), 0.0, 1.0))
            regions.append(
                DetectedRegion(
                    id=len(regions),
                    polygon=[(float(px), float(py)) for px, py in ordered],
                    confidence=round(conf, 4),
                    detector=DetectorType.CRAFT,
                )
            )

        return regions

    def detect(self, image: np.ndarray, config: DetectionConfig) -> list[DetectedRegion]:
        session = self._ensure_session()
        tensor, ratio_x, ratio_y = self.preprocess(image)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: tensor})

        score_map = np.asarray(outputs[0])
        # Support typical CRAFT ONNX shapes: (1, H/2, W/2, 2) or (1, 2, H/2, W/2)
        if score_map.ndim == 4 and score_map.shape[-1] == 2:
            region_score = score_map[0, :, :, 0]
            affinity_score = score_map[0, :, :, 1]
        elif score_map.ndim == 4 and score_map.shape[1] == 2:
            region_score = score_map[0, 0, :, :]
            affinity_score = score_map[0, 1, :, :]
        else:
            raise ValueError(f"Unexpected CRAFT ONNX output shape: {score_map.shape}")

        # CRAFT heatmaps are usually 1/2 of input resolution
        map_h, map_w = region_score.shape[:2]
        orig_h, orig_w = image.shape[:2]
        eff_ratio_x = orig_w / float(map_w)
        eff_ratio_y = orig_h / float(map_h)

        regions = self.extract_polygons_from_heatmaps(
            region_score=region_score,
            affinity_score=affinity_score,
            text_threshold=config.text_threshold,
            link_threshold=config.link_threshold,
            low_text=config.low_text,
            ratio_x=eff_ratio_x,
            ratio_y=eff_ratio_y,
        )
        for reg in regions:
            reg.granularity = config.granularity
        return regions


def _select_onnx_providers(ort_module: Any, device: str) -> list[str]:
    available = set(ort_module.get_available_providers())
    ordered: list[str] = []
    if device == "cuda" or (device == "auto" and "CUDAExecutionProvider" in available):
        ordered.append("CUDAExecutionProvider")
    if device == "dml" or (device == "auto" and "DmlExecutionProvider" in available):
        ordered.append("DmlExecutionProvider")
    if device == "mps" or (device == "auto" and "CoreMLExecutionProvider" in available):
        ordered.append("CoreMLExecutionProvider")
    ordered.append("CPUExecutionProvider")
    return [p for p in ordered if p in available] or ["CPUExecutionProvider"]
