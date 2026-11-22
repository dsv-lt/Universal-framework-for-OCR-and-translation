"""Synthetic preview and dataset generation orchestrator."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from im_translator.core.pack_loader import LoadedPack
from im_translator.core.schemas import AugmentationConfig, RendererType
from im_translator.synth.augmentations import apply_augmentations
from im_translator.synth.renderer_font import FontGlyphRenderer
from im_translator.synth.renderer_sprites import SpriteGlyphRenderer


def render_ink_canvas(
    pack: LoadedPack,
    text: str,
    renderer_override: RendererType | None = None,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Render raw transparent RGBA ink image using either sprites or font renderer."""
    renderer_type = renderer_override or pack.manifest.script.renderer
    if renderer_type == RendererType.SPRITES:
        renderer = SpriteGlyphRenderer.from_pack(pack)
        return renderer.render_text(text, rng=rng)
    if renderer_type == RendererType.FONT:
        renderer = FontGlyphRenderer.from_pack(pack)
        return renderer.render_text(text)
    raise ValueError(f"Unsupported renderer type: {renderer_type}")


def render_preview(
    pack: LoadedPack,
    text: str,
    renderer_override: RendererType | None = None,
    aug_cfg: AugmentationConfig | None = None,
    seed: int = 42,
) -> bytes:
    """Render an augmented synthetic preview image and return PNG bytes."""
    rng = np.random.default_rng(seed)
    cfg = aug_cfg or AugmentationConfig()
    ink = render_ink_canvas(pack, text, renderer_override=renderer_override, rng=rng)
    final_img = apply_augmentations(ink, cfg, rng=rng)
    buf = io.BytesIO()
    final_img.save(buf, format="PNG")
    return buf.getvalue()


def _make_pseudo_proper_noun(charset: str, rng: np.random.Generator) -> str:
    letters = [c for c in charset if c.isalpha()] or list("abcdefghijkmnpqrstuvwxyz")
    length = int(rng.integers(4, 9))
    word = "".join(letters[int(i)] for i in rng.integers(0, len(letters), size=length))
    return word.capitalize()


def generate_dataset(
    pack: LoadedPack,
    corpus_lines: list[str],
    output_dir: Path,
    num_samples: int = 100,
    aug_cfg: AugmentationConfig | None = None,
    renderer_override: RendererType | None = None,
    mode: str = "line",
    seed: int = 42,
    proper_noun_ratio: float = 0.15,
) -> Path:
    """Generate a synthetic OCR/VLM dataset (`images/` + `labels.jsonl`) on disk."""
    out_dir = output_dir.resolve()
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.jsonl"

    rng = np.random.default_rng(seed)
    cfg = aug_cfg or AugmentationConfig()

    base_lines = [line.strip() for line in corpus_lines if line.strip()]
    if not base_lines:
        base_lines = [
            "sample text",
            pack.manifest.script.charset[: min(12, len(pack.manifest.script.charset))],
        ]

    proper_nouns = [e.term for e in pack.glossary if e.is_proper_noun]

    with labels_path.open("w", encoding="utf-8") as labels_file:
        for idx in range(num_samples):
            if mode == "block":
                count = int(rng.integers(2, 5))
                selected = [base_lines[int(rng.integers(0, len(base_lines)))] for _ in range(count)]
                text = "\n".join(selected)
            else:
                text = base_lines[int(rng.integers(0, len(base_lines)))]

            if proper_noun_ratio > 0 and float(rng.random()) < proper_noun_ratio:
                noun = (
                    proper_nouns[int(rng.integers(0, len(proper_nouns)))]
                    if proper_nouns
                    else _make_pseudo_proper_noun(pack.manifest.script.charset, rng)
                )
                text = f"{noun} {text}" if float(rng.random()) < 0.5 else f"{text} {noun}"

            ink = render_ink_canvas(pack, text, renderer_override=renderer_override, rng=rng)
            img = apply_augmentations(ink, cfg, rng=rng)
            rel_img = f"images/sample_{idx:06d}.png"
            img.save(out_dir / rel_img, format="PNG")

            record = {
                "id": idx,
                "image": rel_img,
                "text": text,
                "pack_id": pack.pack_id,
                "mode": mode,
                "renderer": (renderer_override or pack.manifest.script.renderer).value,
            }
            labels_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return labels_path
