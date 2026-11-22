"""Sprite-based glyph renderer (`glyphs/` RGBA PNGs + `glyphs_meta.yaml`)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from im_translator.core.pack_loader import LoadedPack
from im_translator.core.schemas import (
    GlyphMetaEntry,
    GlyphsMetaConfig,
    WritingDirection,
)


class SpriteGlyphRenderer:
    """Render text lines onto a transparent RGBA canvas using PNG glyph sprites."""

    def __init__(
        self,
        glyphs_dir: Path,
        meta: GlyphsMetaConfig,
        case_sensitive: bool = False,
        direction: WritingDirection = WritingDirection.LTR,
    ) -> None:
        self.glyphs_dir = glyphs_dir
        self.meta = meta
        self.case_sensitive = case_sensitive
        self.direction = direction
        self._cache: dict[str, Image.Image] = {}

    @classmethod
    def from_pack(cls, pack: LoadedPack) -> "SpriteGlyphRenderer":
        if pack.manifest.script.glyphs_dir is None:
            raise ValueError(f"Pack '{pack.pack_id}' does not define script.glyphs_dir")
        glyphs_dir = (pack.path / pack.manifest.script.glyphs_dir).resolve()
        meta = pack.glyphs_meta or GlyphsMetaConfig()
        return cls(
            glyphs_dir=glyphs_dir,
            meta=meta,
            case_sensitive=pack.manifest.script.case_sensitive,
            direction=pack.manifest.script.direction,
        )

    def _lookup_entry(self, ch: str) -> GlyphMetaEntry | None:
        if ch in self.meta.glyphs:
            return self.meta.glyphs[ch]
        if not self.case_sensitive:
            if ch.lower() in self.meta.glyphs:
                return self.meta.glyphs[ch.lower()]
            if ch.upper() in self.meta.glyphs:
                return self.meta.glyphs[ch.upper()]
        return None

    def _load_sprite(self, rel_path: str) -> Image.Image:
        if rel_path not in self._cache:
            full_path = (self.glyphs_dir / rel_path).resolve()
            with Image.open(full_path) as img:
                self._cache[rel_path] = img.convert("RGBA")
        return self._cache[rel_path]

    def _pick_sprite(self, entry: GlyphMetaEntry, rng: np.random.Generator) -> Image.Image:
        idx = int(rng.integers(0, len(entry.files))) if len(entry.files) > 1 else 0
        sprite = self._load_sprite(entry.files[idx])
        target_h = max(4, int(round(self.meta.default_height * entry.scale)))
        scale = target_h / max(sprite.height, 1)
        target_w = max(2, int(round(sprite.width * scale)))
        if (target_w, target_h) != sprite.size:
            return sprite.resize((target_w, target_h), Image.Resampling.BICUBIC)
        return sprite.copy()

    def render_text(
        self,
        text: str,
        rng: np.random.Generator | None = None,
        letter_spacing_jitter: float = 0.08,
        padding: int = 16,
    ) -> Image.Image:
        """Render single or multi-line text onto a transparent RGBA image."""
        gen = rng or np.random.default_rng(42)
        lines = text.splitlines() or [text]
        base_h = self.meta.default_height
        line_step = max(base_h + 8, int(round(base_h * self.meta.line_height_ratio)))
        space_adv = max(4, int(round(base_h * self.meta.space_width_ratio)))
        default_adv = max(4, int(round(base_h * self.meta.default_advance_ratio)))

        placements: list[tuple[int, int, Image.Image]] = []

        if self.direction == WritingDirection.TTB:
            col_x = padding + (len(lines) - 1) * line_step
            for line in lines:
                cursor_y = padding
                last_box: tuple[int, int, int, int] | None = None
                for ch in line:
                    if ch.isspace():
                        cursor_y += space_adv
                        last_box = None
                        continue
                    entry = self._lookup_entry(ch)
                    if entry is None:
                        cursor_y += int(space_adv * 0.6)
                        continue
                    sprite = self._pick_sprite(entry, gen)
                    if entry.is_combining and last_box is not None:
                        bx, by, bw, bh = last_box
                        px = bx + int(round(bw * entry.anchor_offset[0]))
                        py = by + int(round(bh * entry.anchor_offset[1]))
                        placements.append((px, py, sprite))
                    else:
                        px = col_x + max(0, (base_h - sprite.width) // 2)
                        py = cursor_y
                        placements.append((px, py, sprite))
                        last_box = (px, py, sprite.width, sprite.height)
                        adv = entry.advance_ratio if entry.advance_ratio is not None else self.meta.default_advance_ratio
                        step = max(sprite.height + 2, int(round(base_h * adv)))
                        jitter = 1.0 + float(gen.uniform(-letter_spacing_jitter, letter_spacing_jitter))
                        cursor_y += max(4, int(round(step * jitter)))
                col_x -= line_step
        else:
            for line_idx, line in enumerate(lines):
                cursor_x = padding
                baseline_y = padding + line_idx * line_step
                last_box = None
                chars: Iterable[str] = reversed(line) if self.direction == WritingDirection.RTL else line
                for ch in chars:
                    if ch.isspace():
                        cursor_x += space_adv
                        last_box = None
                        continue
                    entry = self._lookup_entry(ch)
                    if entry is None:
                        cursor_x += int(space_adv * 0.6)
                        continue
                    sprite = self._pick_sprite(entry, gen)
                    if entry.is_combining and last_box is not None:
                        bx, by, bw, bh = last_box
                        anchor_y = by
                        if entry.anchor == "top":
                            anchor_y = by - sprite.height // 2
                        elif entry.anchor == "bottom":
                            anchor_y = by + bh - sprite.height // 2
                        elif entry.anchor == "center":
                            anchor_y = by + (bh - sprite.height) // 2
                        px = bx + (bw - sprite.width) // 2 + int(round(bw * entry.anchor_offset[0]))
                        py = anchor_y + int(round(bh * entry.anchor_offset[1]))
                        placements.append((px, py, sprite))
                    else:
                        px = cursor_x
                        py = baseline_y + int(round(base_h * entry.baseline_offset))
                        placements.append((px, py, sprite))
                        last_box = (px, py, sprite.width, sprite.height)
                        adv_ratio = entry.advance_ratio
                        step = int(round(base_h * adv_ratio)) if adv_ratio is not None else max(sprite.width + 3, default_adv)
                        jitter = 1.0 + float(gen.uniform(-letter_spacing_jitter, letter_spacing_jitter))
                        cursor_x += max(4, int(round(step * jitter)))

        if not placements:
            return Image.new("RGBA", (max(64, padding * 2), max(base_h + padding * 2, 64)), (0, 0, 0, 0))

        min_x = min(px for px, _, _ in placements)
        min_y = min(py for _, py, _ in placements)
        shift_x = max(0, padding - min_x)
        shift_y = max(0, padding - min_y)

        max_x = max(px + shift_x + spr.width for px, _, spr in placements)
        max_y = max(py + shift_y + spr.height for _, py, spr in placements)

        canvas = Image.new("RGBA", (max_x + padding, max_y + padding), (0, 0, 0, 0))
        for px, py, spr in placements:
            canvas.alpha_composite(spr, (px + shift_x, py + shift_y))
        return canvas
