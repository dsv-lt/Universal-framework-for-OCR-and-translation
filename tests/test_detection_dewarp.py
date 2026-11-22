"""Tests for detection postprocessing (CRAFT/DBNet++), polygon ordering, grouping, and dewarping."""

from __future__ import annotations

import cv2
import numpy as np

from im_translator.core.schemas import (
    DetectedRegion,
    DetectionGranularity,
    DetectorType,
    WritingDirection,
)
from im_translator.detection.craft import CRAFTDetector
from im_translator.detection.dbnet import DBNetDetector
from im_translator.detection.dewarp import (
    detect_contours_fallback,
    dewarp_polygon,
    group_and_sort_regions,
    order_polygon_points,
)


def _rect_poly(x1: float, y1: float, x2: float, y2: float) -> list[tuple[float, float]]:
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def test_order_polygon_points_canonical_order() -> None:
    pts = np.array([[100.0, 50.0], [10.0, 20.0], [10.0, 50.0], [100.0, 20.0]], dtype=np.float32)
    ordered = order_polygon_points(pts)
    expected = np.array([[10.0, 20.0], [100.0, 20.0], [100.0, 50.0], [10.0, 50.0]], dtype=np.float32)
    assert np.allclose(ordered, expected)


def test_dewarp_polygon_straightens_tilted_box() -> None:
    canvas = np.full((240, 320, 3), 240, dtype=np.uint8)
    center = (160.0, 120.0)
    size = (120.0, 36.0)
    angle = -22.0
    box = cv2.boxPoints((center, size, angle)).astype(np.int32)
    cv2.fillPoly(canvas, [box], color=(25, 25, 25))

    crop = dewarp_polygon(canvas, [(float(x), float(y)) for x, y in box], padding=4)
    ch, cw = crop.shape[:2]
    assert cw > ch * 2
    assert crop.mean() < 120.0


def test_group_and_sort_regions_ltr_rtl_ttb_and_occlusion_merge() -> None:
    # Two fragments on line 1 separated by small occlusion gap (gap=20, height=30 -> 20 <= 1.5*30)
    r_left = DetectedRegion(id=0, polygon=_rect_poly(10, 20, 80, 50))
    r_right = DetectedRegion(id=1, polygon=_rect_poly(100, 22, 170, 52))
    # Distant separate column on same y (gap=200 > 1.5*30) -> should stay separate
    r_col2 = DetectedRegion(id=2, polygon=_rect_poly(370, 20, 440, 50))
    # Second line below
    r_line2 = DetectedRegion(id=3, polygon=_rect_poly(15, 90, 160, 120))

    ltr = group_and_sort_regions(
        [r_line2, r_col2, r_right, r_left],
        direction=WritingDirection.LTR,
        granularity=DetectionGranularity.LINE,
        merge_gap_ratio=1.5,
    )
    assert len(ltr) == 3
    # First region is merged left+right spanning x=10..170
    assert ltr[0].polygon[0][0] <= 12.0
    assert ltr[0].polygon[1][0] >= 168.0
    # Second region on line 1 is right column
    assert ltr[1].polygon[0][0] >= 365.0
    # Third region is line 2
    assert ltr[2].polygon[0][1] >= 85.0

    # RTL ordering (disable merge to test explicit order)
    rtl = group_and_sort_regions(
        [
            DetectedRegion(id=0, polygon=_rect_poly(10, 20, 60, 50)),
            DetectedRegion(id=1, polygon=_rect_poly(200, 20, 260, 50)),
        ],
        direction=WritingDirection.RTL,
        granularity=DetectionGranularity.BLOCK,
    )
    assert rtl[0].polygon[0][0] > rtl[1].polygon[0][0]

    # TTB ordering: right column before left column
    ttb = group_and_sort_regions(
        [
            DetectedRegion(id=0, polygon=_rect_poly(20, 20, 50, 140)),
            DetectedRegion(id=1, polygon=_rect_poly(200, 20, 230, 140)),
        ],
        direction=WritingDirection.TTB,
        granularity=DetectionGranularity.BLOCK,
    )
    assert ttb[0].polygon[0][0] > ttb[1].polygon[0][0]


def test_craft_heatmap_extraction() -> None:
    region = np.zeros((160, 240), dtype=np.float32)
    affinity = np.zeros((160, 240), dtype=np.float32)
    # Draw two character blobs connected by affinity bridge
    region[40:70, 30:55] = 0.92
    region[40:70, 65:90] = 0.90
    affinity[45:65, 50:70] = 0.85

    regions = CRAFTDetector.extract_polygons_from_heatmaps(
        region_score=region,
        affinity_score=affinity,
        text_threshold=0.70,
        link_threshold=0.40,
        low_text=0.40,
        min_area=20,
    )
    assert len(regions) == 1
    assert regions[0].detector == DetectorType.CRAFT
    assert regions[0].confidence > 0.7


def test_dbnet_probmap_extraction_and_unclip() -> None:
    prob = np.zeros((160, 240), dtype=np.float32)
    prob[50:80, 40:180] = 0.88

    regions = DBNetDetector.extract_polygons_from_probmap(
        prob_map=prob,
        thresh=0.30,
        box_thresh=0.60,
        unclip_ratio=1.60,
        min_area=20,
    )
    assert len(regions) == 1
    assert regions[0].detector == DetectorType.DBNET
    xs = [p[0] for p in regions[0].polygon]
    ys = [p[1] for p in regions[0].polygon]
    # Unclipped box should be strictly larger than core 140x30 region
    assert (max(xs) - min(xs)) > 140
    assert (max(ys) - min(ys)) > 30


def test_contour_fallback_finds_text_lines() -> None:
    img = np.full((180, 320, 3), 245, dtype=np.uint8)
    cv2.putText(img, "HELLO RUNE", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    regions = detect_contours_fallback(img)
    assert len(regions) >= 1
