"""Polygon geometry, perspective dewarping, occlusion-resilient grouping, and overlay rendering."""

from __future__ import annotations

import math

import cv2
import numpy as np

from im_translator.core.schemas import (
    DetectedRegion,
    DetectionGranularity,
    DetectorType,
    WritingDirection,
)


def order_polygon_points(pts: np.ndarray) -> np.ndarray:
    """Order polygon vertices canonically as [top-left, top-right, bottom-right, bottom-left]."""
    pts_arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(pts_arr) > 4:
        rect = cv2.minAreaRect(pts_arr)
        pts_arr = cv2.boxPoints(rect).astype(np.float32)
    elif len(pts_arr) < 4:
        raise ValueError(f"Expected at least 4 polygon points, got {len(pts_arr)}")

    cx, cy = pts_arr.mean(axis=0)
    angles = np.arctan2(pts_arr[:, 1] - cy, pts_arr[:, 0] - cx)
    cw_pts = pts_arr[np.argsort(angles)]

    # Top-left point minimizes (x + y)
    tl_idx = int(np.argmin(cw_pts[:, 0] + cw_pts[:, 1]))
    return np.roll(cw_pts, -tl_idx, axis=0).astype(np.float32)


def dewarp_polygon(
    image: np.ndarray,
    polygon: list[tuple[float, float]] | np.ndarray,
    padding: int = 4,
) -> np.ndarray:
    """Apply perspective transform to straighten a quadrilateral text region."""
    src = order_polygon_points(np.asarray(polygon, dtype=np.float32))
    tl, tr, br, bl = src

    width_top = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    dst_w = max(int(round(max(width_top, width_bottom))), 1)

    height_left = float(np.linalg.norm(bl - tl))
    height_right = float(np.linalg.norm(br - tr))
    dst_h = max(int(round(max(height_left, height_right))), 1)

    pad = max(int(padding), 0)
    dst = np.array(
        [
            [pad, pad],
            [pad + dst_w - 1, pad],
            [pad + dst_w - 1, pad + dst_h - 1],
            [pad, pad + dst_h - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image,
        matrix,
        (dst_w + 2 * pad, dst_h + 2 * pad),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _bbox(poly: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _merge_two_regions(a: DetectedRegion, b: DetectedRegion) -> DetectedRegion:
    combined = np.asarray(a.polygon + b.polygon, dtype=np.float32)
    rect = cv2.minAreaRect(combined)
    box = order_polygon_points(cv2.boxPoints(rect))
    return DetectedRegion(
        id=a.id,
        polygon=[(float(x), float(y)) for x, y in box],
        confidence=float((a.confidence + b.confidence) / 2.0),
        detector=a.detector,
        granularity=a.granularity,
        enabled=a.enabled and b.enabled,
    )


def _merge_horizontal_line(
    line_regions: list[DetectedRegion],
    merge_gap_ratio: float,
) -> list[DetectedRegion]:
    ordered = sorted(line_regions, key=lambda r: _bbox(r.polygon)[0])
    merged: list[DetectedRegion] = [ordered[0]]
    for current in ordered[1:]:
        prev = merged[-1]
        px1, py1, px2, py2 = _bbox(prev.polygon)
        cx1, cy1, cx2, cy2 = _bbox(current.polygon)
        ph = max(py2 - py1, 1.0)
        ch = max(cy2 - cy1, 1.0)
        h_min = min(ph, ch)
        h_max = max(ph, ch)
        gap = cx1 - px2
        if h_max <= 2.2 * h_min and gap <= merge_gap_ratio * h_min:
            merged[-1] = _merge_two_regions(prev, current)
        else:
            merged.append(current)
    return merged


def _merge_vertical_column(
    col_regions: list[DetectedRegion],
    merge_gap_ratio: float,
) -> list[DetectedRegion]:
    ordered = sorted(col_regions, key=lambda r: _bbox(r.polygon)[1])
    merged: list[DetectedRegion] = [ordered[0]]
    for current in ordered[1:]:
        prev = merged[-1]
        px1, py1, px2, py2 = _bbox(prev.polygon)
        cx1, cy1, cx2, cy2 = _bbox(current.polygon)
        pw = max(px2 - px1, 1.0)
        cw = max(cx2 - cx1, 1.0)
        w_min = min(pw, cw)
        w_max = max(pw, cw)
        gap = cy1 - py2
        if w_max <= 2.2 * w_min and gap <= merge_gap_ratio * w_min:
            merged[-1] = _merge_two_regions(prev, current)
        else:
            merged.append(current)
    return merged


def group_and_sort_regions(
    regions: list[DetectedRegion],
    direction: WritingDirection = WritingDirection.LTR,
    granularity: DetectionGranularity = DetectionGranularity.LINE,
    merge_gap_ratio: float = 1.5,
) -> list[DetectedRegion]:
    """Merge occlusion-split fragments and sort regions in natural reading order.

    For `ltr` / `rtl`, lines are ordered top-to-bottom and comparable-height fragments along
    the same line with horizontal gap <= `merge_gap_ratio * min_line_height` are merged when
    `granularity == LINE`. For `ttb`, vertical columns are ordered right-to-left and
    top-to-bottom within each column.
    """
    if not regions:
        return []

    if direction == WritingDirection.TTB:
        # Group into vertical columns based on X-overlap and comparable width
        sorted_by_x = sorted(regions, key=lambda r: -((_bbox(r.polygon)[0] + _bbox(r.polygon)[2]) / 2.0))
        columns: list[list[DetectedRegion]] = []
        for reg in sorted_by_x:
            x1, _, x2, _ = _bbox(reg.polygon)
            cx = (x1 + x2) / 2.0
            w = max(x2 - x1, 1.0)
            placed = False
            for col in columns:
                cx1, _, cx2, _ = _bbox(col[0].polygon)
                col_cx = (cx1 + cx2) / 2.0
                col_w = max(cx2 - cx1, 1.0)
                overlap = min(x2, cx2) - max(x1, cx1)
                same_scale = max(w, col_w) <= 2.2 * min(w, col_w)
                if same_scale and (overlap >= 0.40 * min(w, col_w) or abs(cx - col_cx) <= 0.50 * min(w, col_w)):
                    col.append(reg)
                    placed = True
                    break
            if not placed:
                columns.append([reg])

        # Sort columns right-to-left
        columns.sort(key=lambda col: -sum((_bbox(r.polygon)[0] + _bbox(r.polygon)[2]) / 2.0 for r in col) / len(col))
        output: list[DetectedRegion] = []
        for col in columns:
            if granularity == DetectionGranularity.LINE and merge_gap_ratio > 0:
                col = _merge_vertical_column(col, merge_gap_ratio)
            col.sort(key=lambda r: _bbox(r.polygon)[1])
            output.extend(col)
    else:
        # Group into horizontal lines based on Y-overlap and comparable height
        sorted_by_y = sorted(regions, key=lambda r: (_bbox(r.polygon)[1] + _bbox(r.polygon)[3]) / 2.0)
        lines: list[list[DetectedRegion]] = []
        for reg in sorted_by_y:
            _, y1, _, y2 = _bbox(reg.polygon)
            cy = (y1 + y2) / 2.0
            h = max(y2 - y1, 1.0)
            placed = False
            for line in lines:
                _, ly1, _, ly2 = _bbox(line[0].polygon)
                line_cy = (ly1 + ly2) / 2.0
                line_h = max(ly2 - ly1, 1.0)
                overlap = min(y2, ly2) - max(y1, ly1)
                same_scale = max(h, line_h) <= 2.2 * min(h, line_h)
                if same_scale and (overlap >= 0.40 * min(h, line_h) or abs(cy - line_cy) <= 0.50 * min(h, line_h)):
                    line.append(reg)
                    placed = True
                    break
            if not placed:
                lines.append([reg])

        # Sort lines top-to-bottom
        lines.sort(key=lambda line: sum((_bbox(r.polygon)[1] + _bbox(r.polygon)[3]) / 2.0 for r in line) / len(line))
        output = []
        for line in lines:
            if granularity == DetectionGranularity.LINE and merge_gap_ratio > 0:
                line = _merge_horizontal_line(line, merge_gap_ratio)
            if direction == WritingDirection.RTL:
                line.sort(key=lambda r: -_bbox(r.polygon)[2])
            else:
                line.sort(key=lambda r: _bbox(r.polygon)[0])
            output.extend(line)

    reindexed: list[DetectedRegion] = []
    for idx, reg in enumerate(output):
        reindexed.append(
            DetectedRegion(
                id=idx,
                polygon=reg.polygon,
                confidence=reg.confidence,
                detector=reg.detector,
                granularity=reg.granularity,
                enabled=reg.enabled,
            )
        )
    return reindexed


def draw_regions_overlay(image: np.ndarray, regions: list[DetectedRegion]) -> np.ndarray:
    """Draw numbered bounding polygons over a copy of an RGB/BGR image."""
    canvas = image.copy()
    for reg in regions:
        pts = np.asarray(reg.polygon, dtype=np.int32).reshape(-1, 1, 2)
        color = (56, 189, 248) if reg.enabled else (120, 120, 120)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        anchor = tuple(pts[0, 0].tolist())
        cv2.putText(
            canvas,
            f"#{reg.id}",
            (max(anchor[0], 4), max(anchor[1] - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def detect_contours_fallback(
    image: np.ndarray,
    detector_tag: DetectorType = DetectorType.CRAFT,
    granularity: DetectionGranularity = DetectionGranularity.LINE,
    min_area: int = 120,
) -> list[DetectedRegion]:
    """Heuristic adaptive-threshold contour detector used only when demo fallback is requested."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 11
    )

    h_img, w_img = gray.shape[:2]
    img_area = float(max(h_img * w_img, 1))

    # Clear thin outer frame margin so screen/page borders do not fuse across the canvas
    margin = max(4, int(min(h_img, w_img) * 0.015))
    if h_img > 2 * margin and w_img > 2 * margin:
        thresh[:margin, :] = 0
        thresh[-margin:, :] = 0
        thresh[:, :margin] = 0
        thresh[:, -margin:] = 0

    # Strip giant non-text border/frame/illustration strokes before morphological grouping
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if num_labels > 1:
        ws = stats[:, cv2.CC_STAT_WIDTH]
        hs = stats[:, cv2.CC_STAT_HEIGHT]
        drop_mask = np.zeros(num_labels, dtype=bool)
        drop_mask[1:] = (
            ((ws[1:] > 0.75 * w_img) & (hs[1:] > 0.20 * h_img))
            | (hs[1:] > 0.45 * h_img)
            | ((ws[1:].astype(np.float32) * hs[1:].astype(np.float32)) > 0.35 * img_area)
        )
        if np.any(drop_mask):
            thresh[drop_mask[labels]] = 0

    kw = max(int(w_img * (0.025 if granularity == DetectionGranularity.LINE else 0.035)), 9)
    kh = max(int(h_img * (0.006 if granularity == DetectionGranularity.LINE else 0.012)), 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    max_area_ratio = 0.45 if granularity == DetectionGranularity.LINE else 0.70
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[DetectedRegion] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw < 6 or rh < 6:
            continue
        box = order_polygon_points(cv2.boxPoints(rect))
        box[:, 0] = np.clip(box[:, 0], 0, max(w_img - 1, 1))
        box[:, 1] = np.clip(box[:, 1], 0, max(h_img - 1, 1))
        bbox_area = float((box[:, 0].max() - box[:, 0].min()) * (box[:, 1].max() - box[:, 1].min()))
        if bbox_area > max_area_ratio * img_area:
            continue
        regions.append(
            DetectedRegion(
                id=len(regions),
                polygon=[(float(x), float(y)) for x, y in box],
                confidence=round(min(0.95, 0.5 + math.log1p(area) / 20.0), 3),
                detector=detector_tag,
                granularity=granularity,
                enabled=True,
            )
        )
    return regions
