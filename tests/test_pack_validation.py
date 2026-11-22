"""Tests for Language Pack manifest validation, loading, and template scaffolding."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from im_translator.core.pack_loader import (
    create_pack_from_templates,
    load_pack,
    validate_pack,
)
from im_translator.core.schemas import (
    LanguageClass,
    PackManifest,
    RendererType,
    ScriptConfig,
    WritingDirection,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_PACK = REPO_ROOT / "packs" / "_example_demo"
TEMPLATES_DIR = REPO_ROOT / "docs" / "templates"


def test_example_demo_pack_is_valid_and_postprocesses() -> None:
    report = validate_pack(DEMO_PACK)
    assert report.valid is True
    assert not any(issue.level == "error" for issue in report.issues)

    pack = load_pack(DEMO_PACK)
    assert pack.pack_id == "_example_demo"
    assert pack.manifest.script.renderer == RendererType.SPRITES
    assert len(pack.glossary) >= 2
    assert "ROT13" in pack.grammar_text

    res = pack.apply_postprocess("uryyb jbeyq")
    assert res.text == "hello world"
    assert res.original_script == "uryyb jbeyq"


def test_script_config_validates_renderer_requirements() -> None:
    with pytest.raises(ValidationError):
        ScriptConfig(renderer=RendererType.SPRITES, glyphs_dir=None)

    with pytest.raises(ValidationError):
        ScriptConfig(renderer=RendererType.FONT, font_path=None)

    cfg_font = ScriptConfig(renderer=RendererType.FONT, font_path="custom.ttf")
    assert cfg_font.font_path == "custom.ttf"


def test_validate_pack_reports_missing_files_and_invalid_yaml(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_pack"
    empty_dir.mkdir()
    report_empty = validate_pack(empty_dir)
    assert report_empty.valid is False
    assert any(i.file == "manifest.yaml" and i.level == "error" for i in report_empty.issues)

    bad_yaml_dir = tmp_path / "bad_yaml"
    bad_yaml_dir.mkdir()
    (bad_yaml_dir / "manifest.yaml").write_text("::: [unclosed", encoding="utf-8")
    assert validate_pack(bad_yaml_dir).valid is False

    # Valid manifest pointing to missing font.ttf
    font_pack = tmp_path / "missing_font"
    font_pack.mkdir()
    manifest = PackManifest(
        id="missing_font",
        name="Missing Font",
        script=ScriptConfig(renderer=RendererType.FONT, font_path="nonexistent.ttf"),
    )
    (font_pack / "manifest.yaml").write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    report_font = validate_pack(font_pack)
    assert report_font.valid is False
    assert any(i.file == "nonexistent.ttf" and i.level == "error" for i in report_font.issues)


def test_create_pack_from_templates_sprites_and_font(tmp_path: Path) -> None:
    # Sprites pack without postprocess
    sprites_dir = create_pack_from_templates(
        packs_dir=tmp_path,
        pack_id="test_sprites",
        name="Test Sprites",
        language_class=LanguageClass.CONLANG,
        renderer=RendererType.SPRITES,
        direction=WritingDirection.LTR,
        enable_postprocess=False,
        templates_dir=TEMPLATES_DIR,
    )
    assert (sprites_dir / "manifest.yaml").is_file()
    assert (sprites_dir / "glyphs").is_dir()
    assert (sprites_dir / "glyphs_meta.yaml").is_file()
    assert not (sprites_dir / "postprocess.py").exists()

    # Font pack with postprocess
    font_dir = create_pack_from_templates(
        packs_dir=tmp_path,
        pack_id="test_font",
        name="Test Font",
        language_class=LanguageClass.CIPHER,
        renderer=RendererType.FONT,
        direction=WritingDirection.RTL,
        enable_postprocess=True,
        templates_dir=TEMPLATES_DIR,
    )
    assert (font_dir / "postprocess.py").is_file()
    # Attach a real system TTF if present to confirm full font validation passes
    ttf_candidates = list(Path("/usr/share/fonts").rglob("*.ttf"))
    if ttf_candidates:
        shutil.copyfile(ttf_candidates[0], font_dir / "font.ttf")
        rep = validate_pack(font_dir)
        assert rep.valid is True
