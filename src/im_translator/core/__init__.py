"""Core schemas, settings, pack loader, and orchestration pipeline."""

from typing import TYPE_CHECKING, Any

from im_translator.core.pack_loader import (
    LoadedPack,
    create_pack_from_templates,
    discover_packs,
    load_pack,
    validate_pack,
)
from im_translator.core.schemas import (
    AugmentationConfig,
    BackgroundStyle,
    DetectedRegion,
    DetectionConfig,
    DetectionGranularity,
    DetectorType,
    GlossaryEntry,
    GlyphMetaEntry,
    GlyphsMetaConfig,
    LanguageClass,
    PackManifest,
    PipelineResult,
    PostprocessConfig,
    PostprocessResult,
    RecognitionConfig,
    RecognitionOutput,
    RecognizerBackend,
    RegionResult,
    RendererType,
    ScriptConfig,
    ScriptType,
    TranslationConfig,
    ValidationIssue,
    ValidationReport,
    WritingDirection,
)
from im_translator.core.settings import AppSettings, detect_project_root, load_settings, save_settings

__all__ = [
    "AppSettings",
    "AugmentationConfig",
    "BackgroundStyle",
    "DetectedRegion",
    "DetectionConfig",
    "DetectionGranularity",
    "DetectorType",
    "GlossaryEntry",
    "GlyphMetaEntry",
    "GlyphsMetaConfig",
    "LanguageClass",
    "LoadedPack",
    "PackManifest",
    "PipelineResult",
    "PostprocessConfig",
    "PostprocessResult",
    "RecognitionConfig",
    "RecognitionOutput",
    "RecognizerBackend",
    "RegionResult",
    "RendererType",
    "ScriptConfig",
    "ScriptType",
    "TranslationConfig",
    "TranslationPipeline",
    "ValidationIssue",
    "ValidationReport",
    "WritingDirection",
    "create_pack_from_templates",
    "detect_project_root",
    "discover_packs",
    "load_pack",
    "load_settings",
    "save_settings",
    "validate_pack",
]

if TYPE_CHECKING:
    from im_translator.core.pipeline import TranslationPipeline


def __getattr__(name: str) -> Any:
    if name == "TranslationPipeline":
        from im_translator.core.pipeline import TranslationPipeline

        return TranslationPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
