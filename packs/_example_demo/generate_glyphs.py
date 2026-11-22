"""Standalone generator for the `_example_demo` geometric rune PNG sprites and glyphs_meta.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image, ImageDraw


def generate_demo_glyphs(pack_dir: Path | None = None) -> Path:
    target_pack = (pack_dir or Path(__file__).resolve().parent).resolve()
    glyphs_dir = target_pack / "glyphs"
    glyphs_dir.mkdir(parents=True, exist_ok=True)

    nodes = [(12, 10), (36, 10), (12, 32), (36, 32), (12, 54), (36, 54)]
    edges = [(0, 1), (2, 3), (4, 5), (0, 3), (1, 2), (2, 5), (3, 4)]
    meta_glyphs: dict[str, dict[str, object]] = {}

    for idx in range(26):
        ch = chr(ord("a") + idx)
        img = Image.new("RGBA", (48, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        stem = (0, 4) if idx % 2 == 0 else (1, 5)
        draw.line([nodes[stem[0]], nodes[stem[1]]], fill=(20, 18, 16, 255), width=4)
        bitmask = idx + 1
        for bit_idx, (u, v) in enumerate(edges):
            if (bitmask >> bit_idx) & 1:
                draw.line([nodes[u], nodes[v]], fill=(20, 18, 16, 255), width=4)
        if idx >= 16:
            draw.ellipse((20, 2, 28, 10), fill=(20, 18, 16, 255))

        fname = f"lower_{ch}.png"
        img.save(glyphs_dir / fname, format="PNG")
        meta_glyphs[ch] = {
            "files": [fname],
            "baseline_offset": 0.0,
            "scale": 1.0,
            "advance_ratio": None,
            "is_combining": False,
        }

    meta = {
        "default_height": 64,
        "default_advance_ratio": 0.80,
        "space_width_ratio": 0.55,
        "line_height_ratio": 1.35,
        "glyphs": meta_glyphs,
    }
    with (target_pack / "glyphs_meta.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False, allow_unicode=True)
    return glyphs_dir


if __name__ == "__main__":
    out_dir = generate_demo_glyphs()
    print(f"Generated 26 demo rune sprites in {out_dir}")
