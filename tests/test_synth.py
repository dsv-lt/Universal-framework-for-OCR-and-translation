"""Tests for sprite/font glyph rendering, augmentations, and dataset generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from im_translator.core.pack_loader import load_pack
from im_translator.core.schemas import (
    AugmentationConfig,
    BackgroundStyle,
    RendererType,
    WritingDirection,
)
from im_translator.synth.augmentations import apply_augmentations
from im_translator.synth.generator import generate_dataset, render_preview
from im_translator.synth.renderer_font import FontGlyphRenderer
from im_translator.synth.renderer_sprites import SpriteGlyphRenderer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_PACK = REPO_ROOT / "packs" / "_example_demo"


def test_sprite_glyph_renderer_directions() -> None:
    pack = load_pack(DEMO_PACK)
    for direction in (WritingDirection.LTR, WritingDirection.RTL, WritingDirection.TTB):
        renderer = SpriteGlyphRenderer(
            glyphs_dir=pack.path / "glyphs",
            meta=pack.glyphs_meta,  # type: ignore[arg-type]
            case_sensitive=False,
            direction=direction,
        )
        canvas = renderer.render_text("ab c\nde")
        assert canvas.mode == "RGBA"
        assert canvas.width > 30 and canvas.height > 30
        arr = np.asarray(canvas)
        assert np.any(arr[:, :, 3] > 0)


def test_font_glyph_renderer_with_system_font() -> None:
    fonts = list(Path("/usr/share/fonts").rglob("*.ttf"))
    if not fonts:
        pytest.skip("No TrueType fonts installed on system")

    renderer = FontGlyphRenderer(font_path=fonts[0], font_size=40)
    canvas = renderer.render_text("hello world")
    assert canvas.mode == "RGBA"
    assert np.any(np.asarray(canvas)[:, :, 3] > 0)


def test_augmentations_and_preview_bytes() -> None:
    pack = load_pack(DEMO_PACK)
    renderer = SpriteGlyphRenderer.from_pack(pack)
    ink = renderer.render_text("rune test")

    for style in BackgroundStyle:
        out = apply_augmentations(
            ink,
            AugmentationConfig(background=style, strikethrough_prob=1.0, occlusion_prob=1.0),
            rng=np.random.default_rng(7),
        )
        assert out.mode == "RGB"
        assert out.width > 40 and out.height > 40

    png_bytes = render_preview(pack, "aethelgard", renderer_override=RendererType.SPRITES, seed=123)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_generate_dataset_creates_images_and_jsonl(tmp_path: Path) -> None:
    pack = load_pack(DEMO_PACK)
    labels_path = generate_dataset(
        pack=pack,
        corpus_lines=["hello world", "magic runes", "silva"],
        output_dir=tmp_path / "ds",
        num_samples=6,
        mode="line",
        seed=99,
        proper_noun_ratio=0.5,
    )
    assert labels_path.is_file()
    lines = labels_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6
    for row in lines:
        rec = json.loads(row)
        assert (tmp_path / "ds" / rec["image"]).is_file()
        assert rec["pack_id"] == "_example_demo"
        assert len(rec["text"]) > 0
