"""Vector font glyph renderer (`.ttf` / `.otf`) via Pillow (+ libraqm when available)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features

from im_translator.core.pack_loader import LoadedPack
from im_translator.core.schemas import WritingDirection


class FontGlyphRenderer:
    """Render text lines onto a transparent RGBA canvas using a TrueType/OpenType font."""

    def __init__(
        self,
        font_path: Path,
        font_size: int = 56,
        direction: WritingDirection = WritingDirection.LTR,
        ink_rgba: tuple[int, int, int, int] = (24, 20, 18, 255),
    ) -> None:
        self.font_path = font_path
        self.font_size = font_size
        self.direction = direction
        self.ink_rgba = ink_rgba
        self._has_raqm = bool(features.check("raqm"))

    @classmethod
    def from_pack(cls, pack: LoadedPack, font_size: int = 56) -> "FontGlyphRenderer":
        if not pack.manifest.script.font_path:
            raise ValueError(f"Pack '{pack.pack_id}' does not define script.font_path")
        font_path = (pack.path / pack.manifest.script.font_path).resolve()
        if not font_path.is_file():
            raise FileNotFoundError(f"Font file not found at '{font_path}'")
        return cls(
            font_path=font_path,
            font_size=font_size,
            direction=pack.manifest.script.direction,
        )

    def _load_font(self) -> ImageFont.FreeTypeFont:
        if self._has_raqm:
            try:
                return ImageFont.truetype(
                    str(self.font_path),
                    size=self.font_size,
                    layout_engine=ImageFont.Layout.RAQM,
                )
            except Exception:
                pass
        return ImageFont.truetype(str(self.font_path), size=self.font_size)

    def render_text(self, text: str, padding: int = 16) -> Image.Image:
        """Render single or multi-line text onto a transparent RGBA canvas."""
        font = self._load_font()
        lines = text.splitlines() or [text]
        if not any(lines):
            return Image.new("RGBA", (64, 64), (0, 0, 0, 0))

        dummy = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        draw_dummy = ImageDraw.Draw(dummy)

        if self.direction == WritingDirection.TTB:
            col_step = int(round(self.font_size * 1.35))
            char_step = int(round(self.font_size * 1.15))
            max_chars = max((len(line) for line in lines), default=1)
            width = padding * 2 + max(len(lines), 1) * col_step
            height = padding * 2 + max(max_chars, 1) * char_step
            canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            col_x = width - padding - col_step
            for line in lines:
                y = padding
                for ch in line:
                    draw.text((col_x, y), ch, font=font, fill=self.ink_rgba)
                    y += char_step
                col_x -= col_step
            return canvas

        prepared_lines = [line[::-1] if (self.direction == WritingDirection.RTL and not self._has_raqm) else line for line in lines]
        rendered_text = "\n".join(prepared_lines)
        kwargs: dict[str, object] = {}
        if self._has_raqm:
            kwargs["direction"] = "rtl" if self.direction == WritingDirection.RTL else "ltr"

        bbox = draw_dummy.multiline_textbbox((0, 0), rendered_text, font=font, spacing=8, **kwargs)
        tw = max(int(bbox[2] - bbox[0]), 8)
        th = max(int(bbox[3] - bbox[1]), 8)
        canvas = Image.new("RGBA", (tw + padding * 2, th + padding * 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.multiline_text(
            (padding - bbox[0], padding - bbox[1]),
            rendered_text,
            font=font,
            fill=self.ink_rgba,
            spacing=8,
            **kwargs,
        )
        return canvas
