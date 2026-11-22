"""FastAPI Web UI and REST API server for im_Translator."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from im_translator import __version__
from im_translator.core.pack_loader import (
    create_pack_from_templates,
    discover_packs,
    load_pack,
    validate_pack,
)
from im_translator.core.pipeline import TranslationPipeline
from im_translator.core.schemas import (
    AugmentationConfig,
    DetectedRegion,
    DetectionGranularity,
    DetectorType,
    LanguageClass,
    RecognizerBackend,
    RendererType,
    WritingDirection,
)
from im_translator.core.paths import format_display_path, normalize_user_path
from im_translator.core.settings import (
    AppSettings,
    configure_runtime_caches,
    detect_project_root,
    load_settings,
    save_settings,
)
from im_translator.detection.base import ModelNotInstalledError
from im_translator.detection.dewarp import (
    detect_contours_fallback,
    dewarp_polygon,
    group_and_sort_regions,
)
from im_translator.detection.registry import get_detector, list_detectors_status
from im_translator.recognition.onnx_ocr import crop_to_png_base64
from im_translator.recognition.registry import list_recognizers_status
from im_translator.synth.generator import render_preview
from im_translator.training.schemas import (
    TrainConlangLoRAConfig,
    TrainOnnxOCRConfig,
    TrainVLMLoRAConfig,
)
from im_translator.training.train_conlang_lora import prepare_conlang_lora_training
from im_translator.training.train_onnx_ocr import prepare_onnx_ocr_training
from im_translator.training.train_vlm_lora import prepare_vlm_lora_training
from im_translator.translation.client import OpenAICompatibleClient

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ValidatePackRequest(BaseModel):
    pack_id: Optional[str] = None
    pack_dir: Optional[str] = None


class CreatePackRequest(BaseModel):
    pack_id: str
    name: str
    language_class: LanguageClass = LanguageClass.CONLANG
    renderer: RendererType = RendererType.SPRITES
    direction: WritingDirection = WritingDirection.LTR
    enable_postprocess: bool = False


class SynthPreviewRequest(BaseModel):
    pack_id: str
    text: str = "hello world"
    renderer_override: Optional[RendererType] = None
    augmentation: AugmentationConfig = AugmentationConfig()
    seed: int = 42


class TestLLMRequest(BaseModel):
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None


class TrainPlanRequest(BaseModel):
    mode: str
    pack_id: str
    dataset_path: str
    output_dir: str
    base_model: Optional[str] = None
    epochs: int = 10


def _read_upload_rgb(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}") from exc


def create_app(settings_override: AppSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(title="im_Translator", version=__version__)
    initial_settings = settings_override or load_settings()
    configure_runtime_caches(initial_settings)
    application.state.settings = initial_settings

    if STATIC_DIR.is_dir():
        application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def get_settings() -> AppSettings:
        return application.state.settings

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=404, detail="Web UI index.html not found.")
        return FileResponse(index_path)

    @application.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @application.get("/api/capabilities")
    def capabilities() -> dict[str, Any]:
        s = get_settings()
        return {
            "detectors": list_detectors_status(s.models_path),
            "recognizers": list_recognizers_status(s),
            "settings": s.model_dump(mode="json"),
        }

    @application.get("/api/packs")
    def list_packs_endpoint() -> dict[str, Any]:
        s = get_settings()
        items: list[dict[str, Any]] = []
        pack_dirs = sorted(
            discover_packs(s.packs_path),
            key=lambda p: (p.name.startswith("_"), p.name.lower()),
        )
        for pdir in pack_dirs:
            report = validate_pack(pdir)
            try:
                pack = load_pack(pdir)
                m = pack.manifest
                items.append(
                    {
                        "id": m.id,
                        "name": m.name,
                        "version": m.version,
                        "description": m.description,
                        "language_class": m.language_class.value,
                        "renderer": m.script.renderer.value,
                        "direction": m.script.direction.value,
                        "default_detector": m.detection.default_detector.value,
                        "granularity": m.detection.granularity.value,
                        "default_backend": m.recognition.default_backend.value,
                        "postprocess_enabled": m.postprocess.enabled,
                        "has_postprocess_module": pack.postprocess_fn is not None,
                        "valid": report.valid,
                        "issue_count": len(report.issues),
                    }
                )
            except Exception as exc:
                items.append(
                    {
                        "id": pdir.name,
                        "name": pdir.name,
                        "valid": False,
                        "issue_count": len(report.issues),
                        "error": str(exc),
                    }
                )
        return {"packs_dir": format_display_path(s.packs_path), "packs": items}

    @application.get("/api/packs/{pack_id}")
    def get_pack_detail(pack_id: str) -> dict[str, Any]:
        s = get_settings()
        pdir = (s.packs_path / pack_id).resolve()
        if not pdir.is_dir():
            raise HTTPException(status_code=404, detail=f"Pack '{pack_id}' not found.")
        report = validate_pack(pdir)
        try:
            pack = load_pack(pdir)
            return {
                "manifest": pack.manifest.model_dump(mode="json"),
                "validation": report.model_dump(mode="json"),
                "glossary_count": len(pack.glossary),
                "grammar_preview": pack.grammar_text[:1200],
            }
        except Exception as exc:
            return {
                "manifest": None,
                "validation": report.model_dump(mode="json"),
                "error": str(exc),
            }

    @application.post("/api/packs/validate")
    def validate_pack_endpoint(req: ValidatePackRequest) -> dict[str, Any]:
        s = get_settings()
        if req.pack_dir:
            pdir = normalize_user_path(req.pack_dir, base_dir=s.packs_path)
        elif req.pack_id:
            pdir = (s.packs_path / req.pack_id).resolve()
        else:
            raise HTTPException(status_code=400, detail="Provide either pack_id or pack_dir.")
        return validate_pack(pdir).model_dump(mode="json")

    @application.post("/api/packs/create")
    def create_pack_endpoint(req: CreatePackRequest) -> dict[str, Any]:
        s = get_settings()
        try:
            pdir = create_pack_from_templates(
                packs_dir=s.packs_path,
                pack_id=req.pack_id,
                name=req.name,
                language_class=req.language_class,
                renderer=req.renderer,
                direction=req.direction,
                enable_postprocess=req.enable_postprocess,
                templates_dir=s.templates_path,
            )
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        report = validate_pack(pdir)
        return {"pack_dir": format_display_path(pdir), "validation": report.model_dump(mode="json")}

    @application.post("/api/detect")
    async def detect_endpoint(
        file: UploadFile = File(...),
        pack_id: str = Form("_example_demo"),
        detector: DetectorType = Form(DetectorType.CRAFT),
        granularity: DetectionGranularity = Form(DetectionGranularity.LINE),
        text_threshold: float = Form(0.70),
        link_threshold: float = Form(0.40),
        low_text: float = Form(0.40),
        db_thresh: float = Form(0.30),
        db_box_thresh: float = Form(0.60),
        db_unclip_ratio: float = Form(1.60),
        demo_contour_fallback: bool = Form(True),
    ) -> dict[str, Any]:
        s = get_settings()
        raw_bytes = await file.read()
        rgb = _read_upload_rgb(raw_bytes)
        h, w = rgb.shape[:2]

        pdir = (s.packs_path / pack_id).resolve()
        if pdir.is_dir():
            pack = load_pack(pdir)
            det_cfg = pack.manifest.detection.model_copy(deep=True)
            direction = pack.manifest.script.direction
        else:
            from im_translator.core.schemas import DetectionConfig

            det_cfg = DetectionConfig()
            direction = WritingDirection.LTR

        det_cfg.default_detector = detector
        det_cfg.granularity = granularity
        det_cfg.text_threshold = text_threshold
        det_cfg.link_threshold = link_threshold
        det_cfg.low_text = low_text
        det_cfg.db_thresh = db_thresh
        det_cfg.db_box_thresh = db_box_thresh
        det_cfg.db_unclip_ratio = db_unclip_ratio

        det_impl = get_detector(detector, s.models_path, device=s.device)
        used_fallback = False
        if det_impl.is_available:
            try:
                raw_regions = det_impl.detect(rgb, det_cfg)
            except ImportError as exc:
                if not demo_contour_fallback:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                raw_regions = detect_contours_fallback(rgb, detector_tag=detector, granularity=granularity)
                used_fallback = True
        elif demo_contour_fallback:
            raw_regions = detect_contours_fallback(rgb, detector_tag=detector, granularity=granularity)
            used_fallback = True
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Detector ONNX weights missing at '{format_display_path(det_impl.model_path)}'. Enable contour fallback or install weights.",
            )

        sorted_regions = group_and_sort_regions(raw_regions, direction=direction, granularity=granularity)
        crops = [crop_to_png_base64(dewarp_polygon(rgb, r.polygon)) for r in sorted_regions]

        return {
            "width": w,
            "height": h,
            "detector": detector.value,
            "used_fallback": used_fallback,
            "regions": [r.model_dump(mode="json") for r in sorted_regions],
            "crops": crops,
        }

    @application.post("/api/pipeline/run")
    async def pipeline_run_endpoint(
        file: UploadFile = File(...),
        pack_id: str = Form(...),
        detector: Optional[DetectorType] = Form(None),
        backend: Optional[RecognizerBackend] = Form(None),
        use_postprocess: Optional[bool] = Form(None),
        target_lang: str = Form("ru"),
        skip_translation: bool = Form(False),
        demo_contour_fallback: bool = Form(False),
        regions_json: Optional[str] = Form(None),
    ) -> dict[str, Any]:
        s = get_settings()
        pdir = (s.packs_path / pack_id).resolve()
        if not pdir.is_dir():
            raise HTTPException(status_code=404, detail=f"Pack '{pack_id}' not found.")
        pack = load_pack(pdir)
        rgb = _read_upload_rgb(await file.read())

        regions_override: list[DetectedRegion] | None = None
        if regions_json:
            try:
                parsed = json.loads(regions_json)
                regions_override = [DetectedRegion.model_validate(item) for item in parsed]
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid regions_json: {exc}") from exc

        pipeline = TranslationPipeline(settings=s)
        try:
            result = pipeline.run(
                image=rgb,
                pack=pack,
                detector_type=detector,
                recognizer_backend=backend,
                use_postprocess=use_postprocess,
                target_lang=target_lang,
                regions_override=regions_override,
                skip_translation=skip_translation,
                demo_contour_fallback=demo_contour_fallback,
            )
        except (ModelNotInstalledError, ImportError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.model_dump(mode="json")

    @application.post("/api/synth/preview")
    def synth_preview_endpoint(req: SynthPreviewRequest) -> Response:
        s = get_settings()
        pdir = (s.packs_path / req.pack_id).resolve()
        if not pdir.is_dir():
            raise HTTPException(status_code=404, detail=f"Pack '{req.pack_id}' not found.")
        pack = load_pack(pdir)
        try:
            png_bytes = render_preview(
                pack=pack,
                text=req.text,
                renderer_override=req.renderer_override,
                aug_cfg=req.augmentation,
                seed=req.seed,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=png_bytes, media_type="image/png")

    @application.post("/api/train/plan")
    def train_plan_endpoint(req: TrainPlanRequest) -> dict[str, Any]:
        s = get_settings()
        root_dir = detect_project_root()
        ds_resolved = str(normalize_user_path(req.dataset_path, base_dir=root_dir))
        out_resolved = str(normalize_user_path(req.output_dir, base_dir=root_dir))
        try:
            if req.mode == "onnx_ocr":
                cfg = TrainOnnxOCRConfig(
                    pack_id=req.pack_id,
                    dataset_path=ds_resolved,
                    output_dir=out_resolved,
                    epochs=req.epochs,
                )
                return prepare_onnx_ocr_training(cfg, s.packs_path)
            if req.mode == "vlm_lora":
                cfg_vlm = TrainVLMLoRAConfig(
                    pack_id=req.pack_id,
                    dataset_path=ds_resolved,
                    output_dir=out_resolved,
                    base_model=req.base_model or "Qwen/Qwen2.5-VL-3B-Instruct",
                    epochs=req.epochs,
                )
                return prepare_vlm_lora_training(cfg_vlm, s.packs_path)
            if req.mode == "conlang_lora":
                cfg_txt = TrainConlangLoRAConfig(
                    pack_id=req.pack_id,
                    dataset_jsonl=ds_resolved,
                    output_dir=out_resolved,
                    base_model=req.base_model or "Qwen/Qwen2.5-7B-Instruct",
                    epochs=req.epochs,
                )
                return prepare_conlang_lora_training(cfg_txt, s.packs_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise HTTPException(status_code=400, detail=f"Unsupported training mode: {req.mode}")

    @application.get("/api/settings")
    def get_settings_endpoint() -> dict[str, Any]:
        s = get_settings()
        formatted = s.model_copy(
            update={
                "packs_dir": format_display_path(s.packs_path),
                "models_dir": format_display_path(s.models_path),
                "datasets_dir": format_display_path(s.datasets_path),
            }
        )
        return formatted.model_dump(mode="json")

    @application.put("/api/settings")
    def update_settings_endpoint(new_settings: AppSettings) -> dict[str, Any]:
        formatted = new_settings.model_copy(
            update={
                "packs_dir": format_display_path(new_settings.packs_path),
                "models_dir": format_display_path(new_settings.models_path),
                "datasets_dir": format_display_path(new_settings.datasets_path),
            }
        )
        configure_runtime_caches(formatted)
        application.state.settings = formatted
        saved_to = save_settings(formatted)
        return {"saved_to": format_display_path(saved_to), "settings": formatted.model_dump(mode="json")}

    @application.post("/api/settings/test-llm")
    def test_llm_endpoint(req: Optional[TestLLMRequest] = None) -> dict[str, Any]:
        s = get_settings()
        client = OpenAICompatibleClient(
            base_url=(req.base_url if req and req.base_url else s.llm_base_url),
            api_key=(req.api_key if req and req.api_key else s.llm_api_key),
            model=(req.model if req and req.model else s.llm_model),
            timeout=10.0,
        )
        return client.health_check()

    return application


app = create_app()
