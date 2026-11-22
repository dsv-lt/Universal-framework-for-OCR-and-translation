"""Language Pack discovery, loading, validation, and template scaffolding."""

from __future__ import annotations

import importlib.util
import inspect
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from PIL import Image, ImageFont
from pydantic import ValidationError

from im_translator.core.schemas import (
    GlossaryEntry,
    GlyphsMetaConfig,
    LanguageClass,
    PackManifest,
    PostprocessResult,
    RendererType,
    ValidationIssue,
    ValidationReport,
    WritingDirection,
)
from im_translator.core.settings import detect_project_root

PostprocessFn = Callable[[str, dict[str, Any] | None], dict[str, Any]]


@dataclass
class LoadedPack:
    """Fully loaded Language Pack with resolved paths and optional assets."""

    path: Path
    manifest: PackManifest
    glyphs_meta: GlyphsMetaConfig | None = None
    glossary: list[GlossaryEntry] = field(default_factory=list)
    grammar_text: str = ""
    postprocess_fn: PostprocessFn | None = None

    @property
    def pack_id(self) -> str:
        return self.manifest.id

    def resolve_path(self, rel_or_abs: str | None) -> Path | None:
        if not rel_or_abs:
            return None
        candidate = Path(rel_or_abs)
        if candidate.is_absolute():
            return candidate
        return (self.path / candidate).resolve()

    def apply_postprocess(self, raw_text: str, context: dict[str, Any] | None = None) -> PostprocessResult:
        if self.postprocess_fn is None:
            return PostprocessResult(text=raw_text, original_script=raw_text)
        result = self.postprocess_fn(raw_text, context or {})
        if isinstance(result, PostprocessResult):
            return result
        if isinstance(result, dict):
            return PostprocessResult.model_validate(result)
        raise TypeError("postprocess.transform() must return a dict or PostprocessResult")


def discover_packs(packs_dir: Path) -> list[Path]:
    """List pack directories containing a manifest.yaml file."""
    if not packs_dir.exists():
        return []
    result: list[Path] = []
    for child in sorted(packs_dir.iterdir()):
        if child.is_dir() and (child / "manifest.yaml").is_file():
            result.append(child)
    return result


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping/object")
    return data


def _load_glossary_entries(glossary_path: Path) -> list[GlossaryEntry]:
    with glossary_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict) and "entries" in raw:
        raw = raw["entries"]
    if not isinstance(raw, list):
        raise ValueError("glossary.json must be a JSON array or an object with 'entries' list")
    return [GlossaryEntry.model_validate(item) for item in raw]


def _load_postprocess_fn(module_path: Path) -> PostprocessFn:
    spec = importlib.util.spec_from_file_location(f"im_translator_pack_{module_path.stem}_{id(module_path)}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "transform", None)
    if fn is None or not callable(fn):
        raise AttributeError(f"{module_path} must define a callable 'transform(raw_text, context)' function")
    return fn


def load_pack(pack_dir: Path) -> LoadedPack:
    """Load a Language Pack from disk without ignoring structural errors."""
    pack_dir = pack_dir.resolve()
    manifest_path = pack_dir / "manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.yaml in {pack_dir}")

    manifest = PackManifest.model_validate(_load_yaml_dict(manifest_path))

    glyphs_meta: GlyphsMetaConfig | None = None
    if manifest.script.renderer == RendererType.SPRITES and manifest.script.glyphs_meta:
        meta_path = (pack_dir / manifest.script.glyphs_meta).resolve()
        if meta_path.is_file():
            glyphs_meta = GlyphsMetaConfig.model_validate(_load_yaml_dict(meta_path))

    glossary: list[GlossaryEntry] = []
    if manifest.translation.glossary_path:
        glossary_path = (pack_dir / manifest.translation.glossary_path).resolve()
        if glossary_path.is_file():
            glossary = _load_glossary_entries(glossary_path)

    grammar_text = ""
    if manifest.translation.grammar_path:
        grammar_path = (pack_dir / manifest.translation.grammar_path).resolve()
        if grammar_path.is_file():
            grammar_text = grammar_path.read_text(encoding="utf-8")

    postprocess_fn: PostprocessFn | None = None
    if manifest.postprocess.enabled and manifest.postprocess.module_path:
        mod_path = (pack_dir / manifest.postprocess.module_path).resolve()
        if mod_path.is_file():
            postprocess_fn = _load_postprocess_fn(mod_path)

    return LoadedPack(
        path=pack_dir,
        manifest=manifest,
        glyphs_meta=glyphs_meta,
        glossary=glossary,
        grammar_text=grammar_text,
        postprocess_fn=postprocess_fn,
    )


def _validate_sprites(pack_dir: Path, manifest: PackManifest, issues: list[ValidationIssue]) -> None:
    glyphs_dir = (pack_dir / (manifest.script.glyphs_dir or "glyphs")).resolve()
    meta_rel = manifest.script.glyphs_meta or "glyphs_meta.yaml"
    meta_path = (pack_dir / meta_rel).resolve()

    if not glyphs_dir.is_dir():
        issues.append(ValidationIssue(level="error", file=str(manifest.script.glyphs_dir), message="Glyphs directory does not exist."))
        return
    if not meta_path.is_file():
        issues.append(ValidationIssue(level="error", file=meta_rel, message="Missing glyphs_meta.yaml required for sprites renderer."))
        return

    try:
        meta = GlyphsMetaConfig.model_validate(_load_yaml_dict(meta_path))
    except (ValidationError, ValueError, yaml.YAMLError) as exc:
        issues.append(ValidationIssue(level="error", file=meta_rel, message=f"Invalid glyphs_meta.yaml: {exc}"))
        return

    if not meta.glyphs:
        issues.append(ValidationIssue(level="warning", file=meta_rel, message="No glyph entries defined in glyphs_meta.yaml."))

    for symbol, entry in meta.glyphs.items():
        for rel_file in entry.files:
            sprite_path = (glyphs_dir / rel_file).resolve()
            if not sprite_path.is_file():
                issues.append(ValidationIssue(level="error", file=f"{manifest.script.glyphs_dir}/{rel_file}", message=f"Missing sprite file for glyph '{symbol}'."))
                continue
            try:
                with Image.open(sprite_path) as img:
                    if img.width < 2 or img.height < 2:
                        issues.append(ValidationIssue(level="error", file=rel_file, message=f"Sprite '{rel_file}' dimensions too small ({img.width}x{img.height})."))
                    elif "A" not in img.getbands():
                        issues.append(ValidationIssue(level="warning", file=rel_file, message=f"Sprite '{rel_file}' has no alpha channel; transparent RGBA PNG is recommended."))
            except Exception as exc:
                issues.append(ValidationIssue(level="error", file=rel_file, message=f"Failed to read sprite image '{rel_file}': {exc}"))

    missing_charset = [ch for ch in manifest.script.charset if ch not in meta.glyphs and not ch.isspace()]
    if missing_charset:
        preview = "".join(missing_charset[:12])
        suffix = "..." if len(missing_charset) > 12 else ""
        issues.append(
            ValidationIssue(
                level="warning",
                file=meta_rel,
                message=f"Charset characters missing from glyphs_meta.yaml ({len(missing_charset)}): '{preview}{suffix}'.",
            )
        )


def _validate_font(pack_dir: Path, manifest: PackManifest, issues: list[ValidationIssue]) -> None:
    font_rel = manifest.script.font_path or "font.ttf"
    font_path = (pack_dir / font_rel).resolve()
    if not font_path.is_file():
        issues.append(ValidationIssue(level="error", file=font_rel, message=f"Font file '{font_rel}' does not exist."))
        return
    try:
        ImageFont.truetype(str(font_path), size=32)
    except Exception as exc:
        issues.append(ValidationIssue(level="error", file=font_rel, message=f"Pillow failed to load font '{font_rel}': {exc}"))


def validate_pack(pack_dir: Path) -> ValidationReport:
    """Validate a Language Pack directory and return structured issues."""
    pack_dir = pack_dir.resolve()
    pack_id = pack_dir.name
    issues: list[ValidationIssue] = []

    manifest_path = pack_dir / "manifest.yaml"
    if not manifest_path.is_file():
        issues.append(ValidationIssue(level="error", file="manifest.yaml", message="Missing manifest.yaml."))
        return ValidationReport(pack_id=pack_id, pack_dir=str(pack_dir), valid=False, issues=issues)

    try:
        manifest = PackManifest.model_validate(_load_yaml_dict(manifest_path))
        pack_id = manifest.id
    except (ValidationError, ValueError, yaml.YAMLError) as exc:
        issues.append(ValidationIssue(level="error", file="manifest.yaml", message=f"Invalid manifest.yaml: {exc}"))
        return ValidationReport(pack_id=pack_id, pack_dir=str(pack_dir), valid=False, issues=issues)

    if manifest.id != pack_dir.name:
        issues.append(
            ValidationIssue(
                level="warning",
                file="manifest.yaml",
                message=f"manifest.id ('{manifest.id}') differs from folder name ('{pack_dir.name}').",
            )
        )

    if manifest.script.renderer == RendererType.SPRITES:
        _validate_sprites(pack_dir, manifest, issues)
    else:
        _validate_font(pack_dir, manifest, issues)

    if manifest.postprocess.enabled:
        mod_rel = manifest.postprocess.module_path or "postprocess.py"
        mod_path = (pack_dir / mod_rel).resolve()
        if not mod_path.is_file():
            issues.append(ValidationIssue(level="error", file=mod_rel, message="postprocess.enabled is true, but postprocess module file is missing."))
        else:
            try:
                fn = _load_postprocess_fn(mod_path)
                sig = inspect.signature(fn)
                if len(sig.parameters) < 1:
                    issues.append(ValidationIssue(level="error", file=mod_rel, message="transform() must accept at least (raw_text, context=None)."))
            except Exception as exc:
                issues.append(ValidationIssue(level="error", file=mod_rel, message=f"Invalid postprocess module: {exc}"))
    else:
        issues.append(ValidationIssue(level="info", file="manifest.yaml", message="Intermediate postprocess layer is disabled (direct OCR -> Translator)."))

    if manifest.translation.glossary_path:
        g_path = (pack_dir / manifest.translation.glossary_path).resolve()
        if not g_path.is_file():
            issues.append(ValidationIssue(level="warning", file=manifest.translation.glossary_path, message="Glossary file configured in manifest was not found."))
        else:
            try:
                _load_glossary_entries(g_path)
            except Exception as exc:
                issues.append(ValidationIssue(level="error", file=manifest.translation.glossary_path, message=f"Invalid glossary JSON: {exc}"))

    if manifest.translation.grammar_path:
        gr_path = (pack_dir / manifest.translation.grammar_path).resolve()
        if not gr_path.is_file():
            issues.append(ValidationIssue(level="warning", file=manifest.translation.grammar_path, message="Grammar markdown file configured in manifest was not found."))

    has_errors = any(item.level == "error" for item in issues)
    return ValidationReport(pack_id=pack_id, pack_dir=str(pack_dir), valid=not has_errors, issues=issues)


def create_pack_from_templates(
    packs_dir: Path,
    pack_id: str,
    name: str,
    language_class: LanguageClass = LanguageClass.CONLANG,
    renderer: RendererType = RendererType.SPRITES,
    direction: WritingDirection = WritingDirection.LTR,
    enable_postprocess: bool = False,
    templates_dir: Path | None = None,
) -> Path:
    """Scaffold a new Language Pack directory from docs/templates/."""
    pack_dir = (packs_dir / pack_id).resolve()
    if pack_dir.exists():
        raise FileExistsError(f"Pack directory already exists: {pack_dir}")

    t_dir = templates_dir or (detect_project_root() / "docs" / "templates")
    pack_dir.mkdir(parents=True, exist_ok=False)

    manifest = PackManifest(
        id=pack_id,
        name=name,
        description=f"{name} language pack ({language_class.value}).",
        language_class=language_class,
        source_lang_code=pack_id,
        intermediate_lang_code="eu" if language_class == LanguageClass.CIPHER else None,
    )
    manifest.script.direction = direction
    manifest.script.renderer = renderer
    manifest.postprocess.enabled = enable_postprocess

    with (pack_dir / "manifest.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest.model_dump(mode="json"), fh, sort_keys=False, allow_unicode=True)

    if renderer == RendererType.SPRITES:
        (pack_dir / "glyphs").mkdir(parents=True, exist_ok=True)
        meta_tpl = t_dir / "glyphs_meta.template.yaml"
        if meta_tpl.is_file():
            shutil.copyfile(meta_tpl, pack_dir / "glyphs_meta.yaml")
        else:
            meta = GlyphsMetaConfig()
            with (pack_dir / "glyphs_meta.yaml").open("w", encoding="utf-8") as fh:
                yaml.safe_dump(meta.model_dump(mode="json"), fh, sort_keys=False, allow_unicode=True)

    glossary_tpl = t_dir / "glossary.template.json"
    if glossary_tpl.is_file():
        shutil.copyfile(glossary_tpl, pack_dir / "glossary.json")
    else:
        (pack_dir / "glossary.json").write_text("[]\n", encoding="utf-8")

    grammar_tpl = t_dir / "grammar.template.md"
    if grammar_tpl.is_file():
        grammar_content = grammar_tpl.read_text(encoding="utf-8")
        grammar_content = grammar_content.replace("<LANGUAGE_NAME>", name).replace("<pack_id>", pack_id)
        (pack_dir / "grammar.md").write_text(grammar_content, encoding="utf-8")
    else:
        (pack_dir / "grammar.md").write_text(f"# Grammar Guide for {name} ({pack_id})\n", encoding="utf-8")

    if enable_postprocess:
        post_tpl = t_dir / "postprocess.template.py"
        if post_tpl.is_file():
            shutil.copyfile(post_tpl, pack_dir / "postprocess.py")

    return pack_dir
