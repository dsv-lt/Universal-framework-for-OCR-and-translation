"""Extract clean 52 upper/lower transparent RGBA glyphs + punctuation for `packs/mt_hl`."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from im_translator.core.paths import normalize_user_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLKIT_LETTERS = normalize_user_path(
    os.environ.get("MT_TOOLKIT_LETTERS", r"D:\Personal\download\basque_cipher_toolkit\letters")
)
PACK_DIR = PROJECT_ROOT / "packs" / "mt_hl"
GLYPHS_DIR = PACK_DIR / "glyphs"
SHEET_PATH = PROJECT_ROOT / "outputs" / "mt_hl_glyphs_sheet.png"


def _find_upper_start(arr: np.ndarray) -> int:
    """Find column index strictly to the right of the printed `Xx [y]` label."""
    h, w = arr.shape[:2]
    sub = arr[6 : h - 4, : min(w, 145)]
    gray = sub.mean(axis=2)
    mask = (gray < 125).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    bracket_x2 = -1
    for cid in range(1, num_labels):
        x, _, cw, chh, _ = stats[cid]
        # Printed ']' bracket is ~33px tall (sometimes merged with letter, cw <= 32) on the left text column
        if chh >= 24 and cw <= 32 and x >= 55 and x + cw <= 136:
            bracket_x2 = max(bracket_x2, int(x + cw))

    if bracket_x2 > 0:
        return bracket_x2 + 2
    return int(0.43 * w)


def _extract_patch_ink(patch_rgb: np.ndarray, is_right_patch: bool) -> Image.Image:
    """Extract clean dark calligraphy ink + detached dots onto a transparent RGBA sprite."""
    ph, pw = patch_rgb.shape[:2]
    # Remove top/bottom table borders and adjacent-row bleed
    top_trim = 6
    bot_trim = max(top_trim + 10, ph - 4)
    sub = patch_rgb[top_trim:bot_trim, :].copy()
    sh, sw = sub.shape[:2]

    r = sub[:, :, 0].astype(np.int16)
    g = sub[:, :, 1].astype(np.int16)
    b = sub[:, :, 2].astype(np.int16)
    is_blue_bg = (b - r) > 10

    luma = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)

    # Local rectangular morphological closing fills dark calligraphy strokes (<=14px)
    # while preserving sharp color/brightness steps between table margins and dark swatches.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    local_bg = cv2.morphologyEx(
        np.clip(luma, 0, 255).astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel_close,
        borderType=cv2.BORDER_REPLICATE,
    ).astype(np.float32)

    darkness = np.clip(local_bg - luma, 0.0, 255.0)
    blue_margin = cv2.dilate(is_blue_bg.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
    darkness[blue_margin] = 0.0

    # Dynamic threshold (handles both crisp black and softer grey strokes like W/F)
    valid_dark = darkness[~blue_margin]
    max_dark = float(np.max(valid_dark)) if len(valid_dark) else 0.0
    thresh = max(18.0, min(55.0, max_dark * 0.28))
    mask = (darkness >= thresh).astype(np.uint8)

    # Trim extreme left/right 2px edges to avoid cell split seams
    mask[:, :2] = 0
    if sw > 4:
        mask[:, -2:] = 0

    # Connected components cleanup: keep primary glyph + nearby diacritic/dot
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return Image.new("RGBA", (32, 48), (0, 0, 0, 0))

    # Pick primary glyph component favoring central, non-edge components
    best_idx = 1
    best_score = -1.0
    for cid in range(1, num_labels):
        x1 = int(stats[cid, cv2.CC_STAT_LEFT])
        y1 = int(stats[cid, cv2.CC_STAT_TOP])
        cw = int(stats[cid, cv2.CC_STAT_WIDTH])
        ch = int(stats[cid, cv2.CC_STAT_HEIGHT])
        area = float(stats[cid, cv2.CC_STAT_AREA])
        # Penalize stray neighbor components hugging outer top-right or right edge
        edge_penalty = 0.2 if (y1 <= 1 and x1 + cw >= sw - 3) or (is_right_patch and x1 + cw >= sw - 2 and ch < sh * 0.45) else 1.0
        if area * edge_penalty > best_score:
            best_score = area * edge_penalty
            best_idx = cid

    main_idx = best_idx
    areas = stats[1:, cv2.CC_STAT_AREA]
    main_area = int(stats[main_idx, cv2.CC_STAT_AREA])
    mx = int(stats[main_idx, cv2.CC_STAT_LEFT])
    my = int(stats[main_idx, cv2.CC_STAT_TOP])
    mw = int(stats[main_idx, cv2.CC_STAT_WIDTH])
    mh = int(stats[main_idx, cv2.CC_STAT_HEIGHT])
    main_x2 = mx + mw

    # Local paper warmth (R - B) immediately surrounding a component to distinguish adjacent swatches
    warmth_map = (r - b).astype(np.float32)

    def _comp_warmth(cid: int) -> float:
        cmask = (labels == cid).astype(np.uint8)
        ring = (cv2.dilate(cmask, np.ones((5, 5), dtype=np.uint8)) > 0) & (cmask == 0) & (~blue_margin)
        vals = warmth_map[ring]
        return float(np.median(vals)) if len(vals) > 0 else 0.0

    main_warmth = _comp_warmth(main_idx)

    keep = np.zeros(num_labels, dtype=bool)
    keep[main_idx] = True

    for cid in range(1, num_labels):
        if cid == main_idx:
            continue
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < 3:
            continue
        cx, cy = centroids[cid]
        x1 = int(stats[cid, cv2.CC_STAT_LEFT])
        y1 = int(stats[cid, cv2.CC_STAT_TOP])
        cw = int(stats[cid, cv2.CC_STAT_WIDTH])
        ch = int(stats[cid, cv2.CC_STAT_HEIGHT])

        # Drop thin straight paper-swatch border/corner shadow slivers (preserve round dots)
        if area < main_area * 0.25 and min(cw, ch) <= 3 and max(cw, ch) >= 2.2 * min(cw, ch):
            continue
        # Drop far-right stray cell overlaps or top-right adjacent swatch strokes (letter_S / letter_L)
        if x1 > main_x2 + 4 and (is_right_patch or x1 + cw >= sw - 3):
            continue
        if is_right_patch and x1 >= main_x2 - 2 and y1 < my + mh * 0.55 and area < main_area * 0.85:
            continue
        if is_right_patch and x1 >= main_x2 - 4 and abs(_comp_warmth(cid) - main_warmth) > 22.0:
            continue
        # Drop tiny corner specks hugging top-right / top-left borders
        if (y1 <= 2 or y1 + ch >= sh - 2) and (x1 <= 3 or x1 + cw >= sw - 3) and area < main_area * 0.35:
            continue
        # Drop thin horizontal/vertical border slivers hugging the outer edge
        if (x1 <= 2 or x1 + cw >= sw - 2) and area < main_area * 0.18:
            continue
        if (y1 <= 2 or y1 + ch >= sh - 2) and cw > ch * 2.2 and area < main_area * 0.35:
            continue

        # Keep strokes or dots reasonably close to the main glyph bounding box
        dx = max(0.0, mx - cx, cx - main_x2)
        dy = max(0.0, my - cy, cy - (my + mh))
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= max(14.0, sh * 0.28):
            keep[cid] = True

    clean_mask = keep[labels]
    ys, xs = np.where(clean_mask)
    if len(xs) == 0:
        return Image.new("RGBA", (32, 48), (0, 0, 0, 0))

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    # Soft alpha derived from normalized darkness inside kept mask
    crop_dark = darkness[y0:y1, x0:x1]
    crop_keep = clean_mask[y0:y1, x0:x1]
    norm_alpha = np.clip((crop_dark / max(max_dark * 0.72, 40.0)) * 255.0, 0.0, 255.0)
    alpha = np.where(crop_keep, np.maximum(norm_alpha, 150.0), 0.0).astype(np.uint8)

    # Dark calligraphy brown-black ink RGB
    rgba = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    rgba[:, :, 0] = 24
    rgba[:, :, 1] = 20
    rgba[:, :, 2] = 18
    rgba[:, :, 3] = alpha

    pad = 4
    padded = cv2.copyMakeBorder(rgba, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    return Image.fromarray(padded, mode="RGBA")


def _make_punctuation_sprites(glyphs_dir: Path) -> dict[str, str]:
    """Generate clean matching ink punctuation PNG sprites."""
    punct_map = {
        ".": "punct_period.png",
        ",": "punct_comma.png",
        ";": "punct_semicolon.png",
        ":": "punct_colon.png",
        "-": "punct_hyphen.png",
        "!": "punct_excl.png",
        "?": "punct_quest.png",
    }
    font_candidates = list(Path("/usr/share/fonts").rglob("DejaVuSerif*.ttf"))
    font = ImageFont.truetype(str(font_candidates[0]), size=48) if font_candidates else ImageFont.load_default()

    for ch, fname in punct_map.items():
        img = Image.new("RGBA", (28, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((6, 4), ch, font=font, fill=(24, 20, 18, 240))
        bbox = img.getbbox()
        if bbox:
            pad = 3
            cropped = img.crop(
                (max(0, bbox[0] - pad), max(0, bbox[1] - pad), min(img.width, bbox[2] + pad), min(img.height, bbox[3] + pad))
            )
            cropped.save(glyphs_dir / fname, format="PNG")
        else:
            img.save(glyphs_dir / fname, format="PNG")
    return punct_map


def extract_all() -> None:
    GLYPHS_DIR.mkdir(parents=True, exist_ok=True)
    SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)

    meta_glyphs: dict[str, dict[str, object]] = {}

    for code in range(ord("A"), ord("Z") + 1):
        up = chr(code)
        low = up.lower()
        src_path = TOOLKIT_LETTERS / f"letter_{up}.png"
        arr = np.asarray(Image.open(src_path).convert("RGB"), dtype=np.uint8)
        h, w = arr.shape[:2]

        # Locate split column with minimum ink density between upper and lower patches
        upper_start = _find_upper_start(arr)
        x_start = max(upper_start + 12, int(0.48 * w))
        roi_gray = arr[6 : h - 5, x_start:].mean(axis=2)
        s0 = max(4, int(0.66 * w) - x_start)
        s1 = min(roi_gray.shape[1] - 2, int(0.79 * w) - x_start)
        col_dark_sum = (roi_gray < 140).sum(axis=0)
        split_x = x_start + s0 + int(np.argmin(col_dark_sum[s0:s1]))

        upper_patch = arr[:, upper_start:split_x]
        lower_patch = arr[:, split_x:]

        lower_img = _extract_patch_ink(lower_patch, is_right_patch=True)
        lower_fname = f"lower_{low}.png"
        lower_img.save(GLYPHS_DIR / lower_fname, format="PNG")

        if up == "Q":
            # Uppercase Q is missing (empty placeholder box) in the source chart; reuse lower_q
            upper_img = lower_img.copy()
        else:
            upper_img = _extract_patch_ink(upper_patch, is_right_patch=False)
        upper_fname = f"upper_{up}.png"
        upper_img.save(GLYPHS_DIR / upper_fname, format="PNG")

        # Slight baseline offset for descenders like p, t
        low_offset = 0.16 if low in ("p", "t") else 0.0
        up_offset = 0.08 if up == "P" else 0.0

        meta_glyphs[up] = {
            "files": [upper_fname],
            "baseline_offset": up_offset,
            "scale": 1.05,
            "advance_ratio": None,
            "is_combining": False,
        }
        meta_glyphs[low] = {
            "files": [lower_fname],
            "baseline_offset": low_offset,
            "scale": 1.0,
            "advance_ratio": None,
            "is_combining": False,
        }

    punct_map = _make_punctuation_sprites(GLYPHS_DIR)
    for ch, fname in punct_map.items():
        if ch in (".", ","):
            offset = 0.65
            scale = 0.30
            adv = 0.35
        elif ch == "-":
            offset = 0.35
            scale = 0.25
            adv = 0.45
        elif ch in (";", ":"):
            offset = 0.25
            scale = 0.65
            adv = 0.38
        else:
            offset = 0.0
            scale = 0.90
            adv = 0.45
        meta_glyphs[ch] = {
            "files": [fname],
            "baseline_offset": offset,
            "scale": scale,
            "advance_ratio": adv,
            "is_combining": False,
        }

    meta = {
        "default_height": 64,
        "default_advance_ratio": 0.78,
        "space_width_ratio": 0.55,
        "line_height_ratio": 1.35,
        "glyphs": meta_glyphs,
    }
    with (PACK_DIR / "glyphs_meta.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False, allow_unicode=True)

    # Render contact sheet for visual verification
    cols = 13
    cell_w, cell_h = 110, 105
    rows = 4
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (242, 229, 200))
    draw = ImageDraw.Draw(sheet)

    ordered_keys = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    for idx, ch in enumerate(ordered_keys):
        r_idx = idx // cols
        c_idx = idx % cols
        x0 = c_idx * cell_w
        y0 = r_idx * cell_h
        draw.rectangle([x0 + 2, y0 + 2, x0 + cell_w - 3, y0 + cell_h - 3], outline=(180, 160, 130))
        draw.text((x0 + 6, y0 + 4), ch, fill=(30, 30, 30))

        fname = meta_glyphs[ch]["files"][0]  # type: ignore[index]
        spr = Image.open(GLYPHS_DIR / fname).convert("RGBA")
        # Fit into 90x66 preview slot
        scale = min(84.0 / max(spr.width, 1), 64.0 / max(spr.height, 1), 1.4)
        nw = max(2, int(round(spr.width * scale)))
        nh = max(2, int(round(spr.height * scale)))
        spr_res = spr.resize((nw, nh), Image.Resampling.BICUBIC)
        px = x0 + (cell_w - nw) // 2
        py = y0 + 26 + (72 - nh) // 2
        sheet.paste(spr_res, (px, py), spr_res)

    sheet.save(SHEET_PATH, format="PNG")
    print(f"Extracted {len(meta_glyphs)} glyphs to {GLYPHS_DIR} and saved contact sheet {SHEET_PATH}")


if __name__ == "__main__":
    extract_all()
