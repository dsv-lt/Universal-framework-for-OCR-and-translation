"""Pydantic v2 schemas and enums for im_Translator."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class LanguageClass(str, Enum):
    CIPHER = "cipher"
    CONLANG = "conlang"
    NATURAL = "natural"


class WritingDirection(str, Enum):
    LTR = "ltr"
    RTL = "rtl"
    TTB = "ttb"


class RendererType(str, Enum):
    SPRITES = "sprites"
    FONT = "font"


class ScriptType(str, Enum):
    ALPHABET = "alphabet"
    ABUGIDA = "abugida"
    ABJAD = "abjad"
    SYLLABARY = "syllabary"
    LOGOGRAPHY = "logography"
    CURSIVE = "cursive"


class DetectorType(str, Enum):
    CRAFT = "craft"
    DBNET = "dbnet"


class DetectionGranularity(str, Enum):
    LINE = "line"
    BLOCK = "block"


class RecognizerBackend(str, Enum):
    ONNX_OCR = "onnx_ocr"
    VLM_PEFT = "vlm_peft"
    VLM_LLAMACPP = "vlm_llamacpp"


class BackgroundStyle(str, Enum):
    CLEAN = "clean"
    PARCHMENT = "parchment"
    DARK_LEATHER = "dark_leather"
    NOISY_PAPER = "noisy_paper"


class GlyphMetaEntry(BaseModel):
    files: list[str] = Field(min_length=1)
    baseline_offset: float = 0.0
    scale: float = Field(default=1.0, gt=0.0)
    advance_ratio: float | None = Field(default=None, gt=0.0)
    is_combining: bool = False
    anchor: Literal["top", "bottom", "center"] | None = None
    anchor_offset: tuple[float, float] = (0.0, 0.0)


class GlyphsMetaConfig(BaseModel):
    default_height: int = Field(default=64, ge=8, le=1024)
    default_advance_ratio: float = Field(default=0.85, gt=0.0)
    space_width_ratio: float = Field(default=0.55, gt=0.0)
    line_height_ratio: float = Field(default=1.35, gt=0.5)
    glyphs: dict[str, GlyphMetaEntry] = Field(default_factory=dict)


class ScriptConfig(BaseModel):
    direction: WritingDirection = WritingDirection.LTR
    renderer: RendererType = RendererType.SPRITES
    glyphs_dir: str | None = "glyphs"
    glyphs_meta: str | None = "glyphs_meta.yaml"
    font_path: str | None = "font.ttf"
    script_type: ScriptType = ScriptType.ALPHABET
    charset: str = "abcdefghijklmnopqrstuvwxyz"
    case_sensitive: bool = False

    @model_validator(mode="after")
    def validate_renderer_paths(self) -> "ScriptConfig":
        if self.renderer == RendererType.SPRITES:
            if not self.glyphs_dir:
                raise ValueError("script.glyphs_dir must be set when script.renderer == 'sprites'")
            if not self.glyphs_meta:
                raise ValueError("script.glyphs_meta must be set when script.renderer == 'sprites'")
        elif self.renderer == RendererType.FONT:
            if not self.font_path:
                raise ValueError("script.font_path must be set when script.renderer == 'font'")
        return self


class DetectionConfig(BaseModel):
    default_detector: DetectorType = DetectorType.CRAFT
    granularity: DetectionGranularity = DetectionGranularity.LINE
    text_threshold: float = Field(default=0.70, ge=0.05, le=0.99)
    link_threshold: float = Field(default=0.40, ge=0.05, le=0.99)
    low_text: float = Field(default=0.40, ge=0.05, le=0.99)
    db_thresh: float = Field(default=0.30, ge=0.05, le=0.99)
    db_box_thresh: float = Field(default=0.60, ge=0.05, le=0.99)
    db_unclip_ratio: float = Field(default=1.60, ge=1.0, le=5.0)


class RecognitionConfig(BaseModel):
    default_backend: RecognizerBackend = RecognizerBackend.ONNX_OCR
    onnx_model_path: str | None = None
    vlm_base_model: str | None = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_adapter_path: str | None = None
    vlm_prompt: str = "Transcribe the text shown in the image accurately. Output only the transcribed text."


class PostprocessConfig(BaseModel):
    enabled: bool = False
    module_path: str | None = "postprocess.py"


class TranslationConfig(BaseModel):
    pivot_via_en: bool = True
    glossary_path: str | None = "glossary.json"
    grammar_path: str | None = "grammar.md"
    conlang_lora_path: str | None = None
    lora_base_model: str | None = None


class PackManifest(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    name: str = Field(min_length=1)
    version: str = "0.1.0"
    description: str = ""
    language_class: LanguageClass = LanguageClass.CONLANG
    source_lang_code: str = "src"
    intermediate_lang_code: str | None = None
    script: ScriptConfig = Field(default_factory=ScriptConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    recognition: RecognitionConfig = Field(default_factory=RecognitionConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)


class GlossaryEntry(BaseModel):
    term: str = Field(min_length=1)
    en: str = Field(min_length=1)
    pos: str = ""
    variants: list[str] = Field(default_factory=list)
    is_proper_noun: bool = False
    notes: str = ""


class DetectedRegion(BaseModel):
    id: int
    polygon: list[tuple[float, float]] = Field(min_length=4)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    detector: DetectorType = DetectorType.CRAFT
    granularity: DetectionGranularity = DetectionGranularity.LINE
    enabled: bool = True


class PostprocessResult(BaseModel):
    text: str
    original_script: str = ""
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecognitionOutput(BaseModel):
    text: str
    confidence: float | None = None
    backend: RecognizerBackend
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class RegionResult(BaseModel):
    region: DetectedRegion
    crop_base64: str | None = None
    raw_text: str = ""
    confidence: float | None = None
    postprocessed_text: str = ""
    postprocess_notes: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    pack_id: str
    detector: DetectorType
    recognizer_backend: RecognizerBackend
    postprocess_applied: bool = False
    regions: list[RegionResult] = Field(default_factory=list)
    raw_full_text: str = ""
    postprocessed_full_text: str = ""
    en_text: str = ""
    target_lang: str = "ru"
    target_text: str = ""
    glossary_hits: list[GlossaryEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    level: Literal["error", "warning", "info"]
    file: str
    message: str


class ValidationReport(BaseModel):
    pack_id: str
    pack_dir: str
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class AugmentationConfig(BaseModel):
    background: BackgroundStyle = BackgroundStyle.PARCHMENT
    rotation_deg: float = Field(default=8.0, ge=0.0, le=45.0)
    perspective_strength: float = Field(default=0.08, ge=0.0, le=0.35)
    shear_deg: float = Field(default=5.0, ge=0.0, le=25.0)
    elastic_alpha: float = Field(default=2.0, ge=0.0, le=12.0)
    stroke_jitter: int = Field(default=1, ge=0, le=3)
    blur_sigma: float = Field(default=0.6, ge=0.0, le=3.0)
    strikethrough_prob: float = Field(default=0.15, ge=0.0, le=1.0)
    occlusion_prob: float = Field(default=0.15, ge=0.0, le=1.0)
    lighting_variation: float = Field(default=0.25, ge=0.0, le=0.8)
    jpeg_quality: int = Field(default=85, ge=20, le=100)
    padding: int = Field(default=18, ge=0, le=128)
