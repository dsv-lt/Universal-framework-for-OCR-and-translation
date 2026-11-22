"""End-to-end orchestration pipeline: Detect -> Dewarp -> Recognize -> [Postprocess] -> Pivot Translate."""

from __future__ import annotations

import numpy as np

from im_translator.core.pack_loader import LoadedPack
from im_translator.core.schemas import (
    DetectedRegion,
    DetectionGranularity,
    DetectorType,
    PipelineResult,
    RecognizerBackend,
    RegionResult,
)
from im_translator.core.settings import AppSettings
from im_translator.detection.dewarp import (
    detect_contours_fallback,
    dewarp_polygon,
    group_and_sort_regions,
)
from im_translator.detection.registry import get_detector
from im_translator.recognition.onnx_ocr import crop_to_png_base64
from im_translator.recognition.registry import get_recognizer
from im_translator.translation.client import OpenAICompatibleClient
from im_translator.translation.pivot import PivotTranslator


class TranslationPipeline:
    """Modular image-to-translation orchestrator."""

    def __init__(
        self,
        settings: AppSettings,
        llm_client: OpenAICompatibleClient | None = None,
    ) -> None:
        self.settings = settings
        self.llm_client = llm_client

    def _get_llm_client(self) -> OpenAICompatibleClient:
        if self.llm_client is not None:
            return self.llm_client
        return OpenAICompatibleClient(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
        )

    def detect_only(
        self,
        image: np.ndarray,
        pack: LoadedPack,
        detector_type: DetectorType | None = None,
        granularity: DetectionGranularity | None = None,
        demo_contour_fallback: bool = False,
    ) -> list[DetectedRegion]:
        """Run text detection and geometric reading-order grouping without OCR."""
        det_type = detector_type or pack.manifest.detection.default_detector
        det_cfg = pack.manifest.detection.model_copy(deep=True)
        if granularity is not None:
            det_cfg.granularity = granularity

        detector = get_detector(det_type, self.settings.models_path, device=self.settings.device)
        if not detector.is_available and demo_contour_fallback:
            raw_regions = detect_contours_fallback(
                image,
                detector_tag=det_type,
                granularity=det_cfg.granularity,
            )
        else:
            raw_regions = detector.detect(image, det_cfg)

        return group_and_sort_regions(
            raw_regions,
            direction=pack.manifest.script.direction,
            granularity=det_cfg.granularity,
        )

    def run(
        self,
        image: np.ndarray,
        pack: LoadedPack,
        detector_type: DetectorType | None = None,
        recognizer_backend: RecognizerBackend | None = None,
        use_postprocess: bool | None = None,
        target_lang: str | None = None,
        regions_override: list[DetectedRegion] | None = None,
        skip_translation: bool = False,
        demo_contour_fallback: bool = False,
    ) -> PipelineResult:
        """Execute full OCR + optional intermediate layer + pivot translation."""
        det_type = detector_type or pack.manifest.detection.default_detector
        rec_backend = recognizer_backend or pack.manifest.recognition.default_backend
        tgt_lang = target_lang or self.settings.default_target_lang
        warnings: list[str] = []

        if regions_override is not None:
            regions = regions_override
        else:
            regions = self.detect_only(
                image=image,
                pack=pack,
                detector_type=det_type,
                demo_contour_fallback=demo_contour_fallback,
            )

        active_regions = [r for r in regions if r.enabled]
        if not active_regions:
            return PipelineResult(
                pack_id=pack.pack_id,
                detector=det_type,
                recognizer_backend=rec_backend,
                postprocess_applied=False,
                target_lang=tgt_lang,
                warnings=["No active text regions detected or selected."],
            )

        crops = [dewarp_polygon(image, reg.polygon) for reg in active_regions]
        recognizer = get_recognizer(rec_backend, self.settings)
        rec_outputs = recognizer.recognize(crops, pack)

        want_postprocess = pack.manifest.postprocess.enabled if use_postprocess is None else use_postprocess
        can_postprocess = want_postprocess and pack.postprocess_fn is not None
        if want_postprocess and pack.postprocess_fn is None:
            warnings.append("Postprocess toggle is ON, but this pack does not define postprocess.py.")

        region_results: list[RegionResult] = []
        raw_lines: list[str] = []
        post_lines: list[str] = []

        for reg, crop, rec_out in zip(active_regions, crops, rec_outputs):
            raw_text = rec_out.text
            raw_lines.append(raw_text)
            if can_postprocess:
                post_res = pack.apply_postprocess(
                    raw_text,
                    context={"pack_id": pack.pack_id, "region_id": reg.id, "confidence": rec_out.confidence},
                )
                post_text = post_res.text
                post_notes = post_res.notes
            else:
                post_text = raw_text
                post_notes = []

            post_lines.append(post_text)
            region_results.append(
                RegionResult(
                    region=reg,
                    crop_base64=crop_to_png_base64(crop),
                    raw_text=raw_text,
                    confidence=rec_out.confidence,
                    postprocessed_text=post_text,
                    postprocess_notes=post_notes,
                )
            )

        raw_full = "\n".join(raw_lines)
        post_full = "\n".join(post_lines)
        en_text = ""
        target_text = ""
        glossary_hits = []

        if not skip_translation and post_full.strip():
            translator = PivotTranslator(self._get_llm_client())
            try:
                t_res = translator.translate(post_full, pack=pack, target_lang=tgt_lang)
                en_text = t_res.en_text
                target_text = t_res.target_text
                glossary_hits = t_res.glossary_hits
            except Exception as exc:
                warnings.append(f"Translation stage failed (check LLM server URL in Settings): {exc}")

        return PipelineResult(
            pack_id=pack.pack_id,
            detector=det_type,
            recognizer_backend=rec_backend,
            postprocess_applied=can_postprocess,
            regions=region_results,
            raw_full_text=raw_full,
            postprocessed_full_text=post_full,
            en_text=en_text,
            target_lang=tgt_lang,
            target_text=target_text,
            glossary_hits=glossary_hits,
            warnings=warnings,
        )
