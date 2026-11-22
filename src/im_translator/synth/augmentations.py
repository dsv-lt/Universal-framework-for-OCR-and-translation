"""Procedural realistic image augmentations for synthetic OCR/VLM training."""

from __future__ import annotations

import io
import math

import cv2
import numpy as np
from PIL import Image

from im_translator.core.schemas import AugmentationConfig, BackgroundStyle


def _make_background(
    width: int,
    height: int,
    style: BackgroundStyle,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create procedural RGB paper/parchment/leather texture."""
    if style == BackgroundStyle.CLEAN:
        val = int(rng.integers(242, 255))
        return np.full((height, width, 3), val, dtype=np.uint8)

    if style == BackgroundStyle.PARCHMENT:
        base = np.array([232, 214, 178], dtype=np.float32) + rng.uniform(-12, 12, size=3)
    elif style == BackgroundStyle.DARK_LEATHER:
        base = np.array([44, 34, 28], dtype=np.float32) + rng.uniform(-8, 8, size=3)
    else:  # NOISY_PAPER
        base = np.array([228, 226, 220], dtype=np.float32) + rng.uniform(-10, 10, size=3)

    canvas = np.tile(base, (height, width, 1))
    # Coarse grain + fine paper fibers
    coarse_h, coarse_w = max(4, height // 12), max(4, width // 12)
    coarse = rng.normal(0.0, 10.0, size=(coarse_h, coarse_w, 1)).astype(np.float32)
    coarse_up = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    if coarse_up.ndim == 2:
        coarse_up = coarse_up[:, :, None]
    fine = rng.normal(0.0, 4.5, size=(height, width, 1)).astype(np.float32)
    canvas = canvas + coarse_up + fine
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _apply_ink_jitter_and_elastic(
    rgba: np.ndarray,
    stroke_jitter: int,
    elastic_alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3]

    if stroke_jitter > 0:
        op = int(rng.integers(-1, 2))
        if op != 0:
            k = int(rng.integers(2, 2 + stroke_jitter))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            if op > 0:
                alpha = cv2.dilate(alpha, kernel, iterations=1)
            else:
                alpha = cv2.erode(alpha, kernel, iterations=1)
            rgba = rgba.copy()
            rgba[:, :, 3] = alpha

    if elastic_alpha > 0.1 and h >= 8 and w >= 8:
        dx = cv2.GaussianBlur(rng.uniform(-1.0, 1.0, (h, w)).astype(np.float32), (0, 0), sigmaX=6.0)
        dy = cv2.GaussianBlur(rng.uniform(-1.0, 1.0, (h, w)).astype(np.float32), (0, 0), sigmaX=6.0)
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = grid_x + dx * float(elastic_alpha) * 2.5
        map_y = grid_y + dy * float(elastic_alpha) * 2.5
        rgba = cv2.remap(rgba, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    return rgba


def _apply_geometric_warp(
    rgba: np.ndarray,
    cfg: AugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = rgba.shape[:2]
    pad = cfg.padding
    padded = cv2.copyMakeBorder(rgba, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    ph, pw = padded.shape[:2]

    # Shear + rotation
    angle = float(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg)) if cfg.rotation_deg > 0 else 0.0
    shear_rad = math.radians(float(rng.uniform(-cfg.shear_deg, cfg.shear_deg))) if cfg.shear_deg > 0 else 0.0
    rot_mat = cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), angle, 1.0)
    rot_mat[0, 1] += math.tan(shear_rad) * 0.7
    warped = cv2.warpAffine(
        padded,
        rot_mat,
        (pw, ph),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # Perspective distortion
    if cfg.perspective_strength > 0.005:
        s = cfg.perspective_strength
        src = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], dtype=np.float32)
        dx = pw * s * 0.5
        dy = ph * s * 0.5
        dst = src + np.array(
            [
                [rng.uniform(0, dx), rng.uniform(0, dy)],
                [-rng.uniform(0, dx), rng.uniform(0, dy)],
                [-rng.uniform(0, dx), -rng.uniform(0, dy)],
                [rng.uniform(0, dx), -rng.uniform(0, dy)],
            ],
            dtype=np.float32,
        )
        persp = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            warped,
            persp,
            (pw, ph),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
    return warped


def _recolor_ink_for_background(rgba: np.ndarray, style: BackgroundStyle, rng: np.random.Generator) -> np.ndarray:
    out = rgba.copy()
    mask = out[:, :, 3] > 0
    if not np.any(mask):
        return out
    if style == BackgroundStyle.DARK_LEATHER:
        # Gold/silver embossed ink on dark book cover
        tint = np.array([rng.integers(190, 235), rng.integers(170, 215), rng.integers(115, 165)], dtype=np.float32)
        out[:, :, :3] = np.clip(tint, 0, 255).astype(np.uint8)
    else:
        # Dark brown/charcoal ink variation
        tint = np.array([rng.integers(15, 55), rng.integers(12, 48), rng.integers(10, 42)], dtype=np.float32)
        out[:, :, :3] = np.clip(tint, 0, 255).astype(np.uint8)
    return out


def apply_augmentations(
    ink_rgba: Image.Image,
    cfg: AugmentationConfig,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Apply realistic manuscript/anime-frame augmentations to a transparent RGBA ink canvas."""
    gen = rng or np.random.default_rng(42)
    rgba = np.asarray(ink_rgba.convert("RGBA"), dtype=np.uint8)

    rgba = _recolor_ink_for_background(rgba, cfg.background, gen)
    rgba = _apply_ink_jitter_and_elastic(rgba, cfg.stroke_jitter, cfg.elastic_alpha, gen)

    # Optional strikethrough line on ink before geometric warp so it tilts naturally with text
    if cfg.strikethrough_prob > 0 and float(gen.random()) < cfg.strikethrough_prob:
        h, w = rgba.shape[:2]
        y1 = int(gen.integers(max(2, int(h * 0.35)), max(3, int(h * 0.65))))
        y2 = int(gen.integers(max(2, int(h * 0.35)), max(3, int(h * 0.65))))
        x1 = int(gen.integers(2, max(3, int(w * 0.25))))
        x2 = int(gen.integers(max(4, int(w * 0.7)), max(5, w - 2)))
        ink_col = tuple(int(c) for c in rgba[rgba[:, :, 3] > 30][0, :3]) if np.any(rgba[:, :, 3] > 30) else (25, 20, 18)
        cv2.line(rgba, (x1, y1), (x2, y2), (*ink_col, 230), thickness=max(2, h // 26), lineType=cv2.LINE_AA)

    warped_rgba = _apply_geometric_warp(rgba, cfg, gen)
    wh, ww = warped_rgba.shape[:2]

    bg = _make_background(ww, wh, cfg.background, gen)
    alpha = (warped_rgba[:, :, 3:4].astype(np.float32)) / 255.0
    fg_rgb = warped_rgba[:, :, :3].astype(np.float32)
    composed = fg_rgb * alpha + bg.astype(np.float32) * (1.0 - alpha)

    # Optional partial edge occlusion (hand/object shadow)
    if cfg.occlusion_prob > 0 and float(gen.random()) < cfg.occlusion_prob:
        occ_w = max(8, int(ww * gen.uniform(0.10, 0.22)))
        occ_h = max(8, int(wh * gen.uniform(0.45, 0.90)))
        from_left = bool(gen.integers(0, 2))
        ox = 0 if from_left else max(0, ww - occ_w)
        oy = int(gen.integers(0, max(1, wh - occ_h)))
        occ_col = np.array([gen.integers(55, 115), gen.integers(45, 95), gen.integers(40, 85)], dtype=np.float32)
        composed[oy : oy + occ_h, ox : ox + occ_w] = occ_col

    # Lighting gradient / glare
    if cfg.lighting_variation > 0.01:
        gx = np.linspace(-1.0, 1.0, ww, dtype=np.float32)
        gy = np.linspace(-1.0, 1.0, wh, dtype=np.float32)
        mx, my = np.meshgrid(gx, gy)
        theta = float(gen.uniform(0, 2 * math.pi))
        grad = (mx * math.cos(theta) + my * math.sin(theta)) * float(cfg.lighting_variation) * 55.0
        composed = composed + grad[:, :, None]

    composed = np.clip(composed, 0, 255).astype(np.uint8)

    if cfg.blur_sigma > 0.05:
        sigma = float(gen.uniform(0.05, cfg.blur_sigma))
        composed = cv2.GaussianBlur(composed, (0, 0), sigmaX=sigma)

    pil_out = Image.fromarray(composed, mode="RGB")
    if cfg.jpeg_quality < 98:
        buf = io.BytesIO()
        pil_out.save(buf, format="JPEG", quality=int(cfg.jpeg_quality))
        buf.seek(0)
        with Image.open(buf) as decoded:
            pil_out = decoded.convert("RGB")

    return pil_out
