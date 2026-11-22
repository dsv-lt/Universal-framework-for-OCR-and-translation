"""Typer CLI entrypoint for im_Translator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import typer
import uvicorn
from PIL import Image
from rich.console import Console
from rich.table import Table

from im_translator.core.pack_loader import (
    create_pack_from_templates,
    discover_packs,
    load_pack,
    validate_pack,
)
from im_translator.core.paths import format_display_path, normalize_user_path
from im_translator.core.pipeline import TranslationPipeline
from im_translator.core.schemas import (
    AugmentationConfig,
    BackgroundStyle,
    DetectorType,
    LanguageClass,
    RecognizerBackend,
    RendererType,
    WritingDirection,
)
from im_translator.core.settings import configure_runtime_caches, load_settings
from im_translator.synth.generator import generate_dataset, render_preview

app = typer.Typer(
    name="im-translator",
    help="Universal OCR + Translation framework for fictional, ciphered, and rare scripts.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host address."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, help="Enable auto-reload during development."),
) -> None:
    """Start the FastAPI Web UI and REST API server."""
    settings = load_settings()
    configure_runtime_caches(settings)
    console.print(f"[cyan]Starting im_Translator Web UI at http://{host}:{port}[/cyan]")
    uvicorn.run("im_translator.app.server:app", host=host, port=port, reload=reload)


@app.command("list-packs")
def list_packs(
    packs_dir: Optional[str] = typer.Option(None, help="Directory containing Language Packs."),
) -> None:
    """List discovered Language Packs and their validation status."""
    settings = load_settings()
    target_dir = normalize_user_path(packs_dir) if packs_dir else settings.packs_path
    pack_paths = sorted(
        discover_packs(target_dir),
        key=lambda p: (p.name.startswith("_"), p.name.lower()),
    )

    table = Table(title=f"Language Packs ({format_display_path(target_dir)})")
    table.add_column("ID", style="bold cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Class")
    table.add_column("Renderer")
    table.add_column("Detector")
    table.add_column("Postprocess")
    table.add_column("Status")

    if not pack_paths:
        console.print(f"[yellow]No packs found in {format_display_path(target_dir)}[/yellow]")
        return

    for pdir in pack_paths:
        report = validate_pack(pdir)
        try:
            pack = load_pack(pdir)
            m = pack.manifest
            table.add_row(
                m.id,
                m.name,
                m.language_class.value,
                m.script.renderer.value,
                m.detection.default_detector.value,
                "enabled" if m.postprocess.enabled else "off",
                "[green]VALID[/green]" if report.valid else "[red]INVALID[/red]",
            )
        except Exception as exc:
            table.add_row(pdir.name, "-", "-", "-", "-", "-", f"[red]ERROR: {exc}[/red]")

    console.print(table)


@app.command("validate-pack")
def validate_pack_cmd(
    pack_path: str = typer.Argument(..., help="Path to Language Pack directory."),
) -> None:
    """Validate a Language Pack directory against im_Translator schemas."""
    report = validate_pack(normalize_user_path(pack_path))
    table = Table(title=f"Validation Report: {report.pack_id}")
    table.add_column("Level")
    table.add_column("File")
    table.add_column("Message")

    for issue in report.issues:
        color = {"error": "red", "warning": "yellow", "info": "cyan"}.get(issue.level, "white")
        table.add_row(f"[{color}]{issue.level.upper()}[/{color}]", issue.file, issue.message)

    if report.issues:
        console.print(table)

    if report.valid:
        console.print(f"[green]Pack '{report.pack_id}' is VALID.[/green]")
    else:
        console.print(f"[red]Pack '{report.pack_id}' is INVALID.[/red]")
        raise typer.Exit(code=1)


@app.command("init-pack")
def init_pack(
    pack_id: str = typer.Argument(..., help="Lowercase pack ID (folder name)."),
    name: Optional[str] = typer.Option(None, help="Display name for the Language Pack."),
    language_class: LanguageClass = typer.Option(LanguageClass.CONLANG, "--class", help="cipher | conlang | natural"),
    renderer: RendererType = typer.Option(RendererType.SPRITES, help="sprites | font"),
    direction: WritingDirection = typer.Option(WritingDirection.LTR, help="ltr | rtl | ttb"),
    enable_postprocess: bool = typer.Option(False, help="Include postprocess.py template."),
    packs_dir: Optional[str] = typer.Option(None, help="Target packs directory."),
) -> None:
    """Scaffold a new Language Pack from docs/templates/."""
    settings = load_settings()
    target_dir = normalize_user_path(packs_dir) if packs_dir else settings.packs_path
    display_name = name or pack_id.replace("_", " ").title()
    pack_dir = create_pack_from_templates(
        packs_dir=target_dir,
        pack_id=pack_id,
        name=display_name,
        language_class=language_class,
        renderer=renderer,
        direction=direction,
        enable_postprocess=enable_postprocess,
        templates_dir=settings.templates_path,
    )
    console.print(f"[green]Created pack scaffold at {format_display_path(pack_dir)}[/green]")


@app.command("synth-preview")
def synth_preview(
    pack_id: str = typer.Argument(..., help="Language Pack ID."),
    text: str = typer.Argument(..., help="Text string to render."),
    output: str = typer.Option("preview.png", "-o", "--output", help="Output PNG path."),
    renderer: Optional[RendererType] = typer.Option(None, help="Override renderer (sprites | font)."),
    background: BackgroundStyle = typer.Option(BackgroundStyle.PARCHMENT, help="Background texture style."),
    seed: int = typer.Option(42, help="Random seed."),
    packs_dir: Optional[str] = typer.Option(None, help="Packs directory."),
) -> None:
    """Render a synthetic preview PNG using a Language Pack's sprites or font."""
    settings = load_settings()
    target_packs = normalize_user_path(packs_dir) if packs_dir else settings.packs_path
    pack = load_pack(target_packs / pack_id)
    out_path = normalize_user_path(output)
    png_bytes = render_preview(
        pack=pack,
        text=text,
        renderer_override=renderer,
        aug_cfg=AugmentationConfig(background=background),
        seed=seed,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png_bytes)
    console.print(f"[green]Saved synthetic preview to {format_display_path(out_path)}[/green]")


@app.command("synth-generate")
def synth_generate(
    pack_id: str = typer.Argument(..., help="Language Pack ID."),
    output_dir: str = typer.Option(..., "-o", "--output-dir", help="Dataset output directory."),
    num_samples: int = typer.Option(100, "-n", "--num-samples", help="Number of synthetic samples."),
    corpus_file: Optional[str] = typer.Option(None, help="Optional UTF-8 text corpus file (one line per sample)."),
    renderer: Optional[RendererType] = typer.Option(None, help="Override renderer (sprites | font)."),
    mode: str = typer.Option("line", help="line | block"),
    seed: int = typer.Option(42, help="Random seed."),
    proper_noun_ratio: float = typer.Option(0.15, help="Fraction of samples injected with proper nouns."),
    packs_dir: Optional[str] = typer.Option(None, help="Packs directory."),
) -> None:
    """Generate a synthetic OCR/VLM dataset (`images/` + `labels.jsonl`)."""
    settings = load_settings()
    target_packs = normalize_user_path(packs_dir) if packs_dir else settings.packs_path
    pack = load_pack(target_packs / pack_id)

    corpus_lines: list[str] = []
    if corpus_file is not None:
        corpus_lines = normalize_user_path(corpus_file).read_text(encoding="utf-8").splitlines()

    labels_path = generate_dataset(
        pack=pack,
        corpus_lines=corpus_lines,
        output_dir=normalize_user_path(output_dir),
        num_samples=num_samples,
        renderer_override=renderer,
        mode=mode,
        seed=seed,
        proper_noun_ratio=proper_noun_ratio,
    )
    console.print(f"[green]Generated {num_samples} samples at {format_display_path(labels_path)}[/green]")


@app.command("infer")
def infer_cmd(
    image_path: str = typer.Argument(..., help="Input image path."),
    pack_id: str = typer.Option(..., "--pack", "--lang", help="Language Pack ID."),
    detector: Optional[DetectorType] = typer.Option(None, help="Detector override (craft | dbnet)."),
    backend: Optional[RecognizerBackend] = typer.Option(None, help="Recognizer backend override."),
    no_postprocess: bool = typer.Option(False, help="Disable intermediate postprocess.py layer."),
    target_lang: Optional[str] = typer.Option(None, help="Target translation language (e.g. ru, en)."),
    skip_translation: bool = typer.Option(False, help="Run detection + OCR + postprocess only."),
    demo_contour_fallback: bool = typer.Option(False, help="Use contour fallback if ONNX detector weights are missing."),
    packs_dir: Optional[str] = typer.Option(None, help="Packs directory."),
) -> None:
    """Run full Detect -> Dewarp -> Recognize -> [Postprocess] -> Translate pipeline."""
    settings = load_settings()
    configure_runtime_caches(settings)
    target_packs = normalize_user_path(packs_dir) if packs_dir else settings.packs_path
    pack = load_pack(target_packs / pack_id)

    with Image.open(normalize_user_path(image_path)) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)

    pipeline = TranslationPipeline(settings=settings)
    result = pipeline.run(
        image=rgb,
        pack=pack,
        detector_type=detector,
        recognizer_backend=backend,
        use_postprocess=False if no_postprocess else None,
        target_lang=target_lang,
        skip_translation=skip_translation,
        demo_contour_fallback=demo_contour_fallback,
    )
    console.print_json(json.dumps(result.model_dump(mode="json", exclude={"regions": {"__all__": {"crop_base64"}}}), ensure_ascii=False))


@app.command("train-vlm")
def train_vlm_cmd(
    pack: str = typer.Option(..., "--pack", "--lang", help="Language Pack ID."),
    dataset: str = typer.Option(..., "--dataset", help="Training JSONL or directory."),
    val_dataset: Optional[str] = typer.Option(None, "--val-dataset", help="Optional held-out validation JSONL."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="LoRA adapter output directory."),
    base_model: str = typer.Option("Qwen/Qwen2.5-VL-3B-Instruct", "--base-model", help="Base Qwen2.5-VL model ID."),
    epochs: int = typer.Option(2, "--epochs", help="Maximum training epochs."),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", help="Optional cap on optimizer steps."),
    batch_size: int = typer.Option(2, "--batch-size", help="Micro-batch size per GPU step."),
    grad_accum: int = typer.Option(8, "--grad-accum", help="Gradient accumulation steps."),
    lr: float = typer.Option(2e-4, "--lr", help="Peak AdamW learning rate."),
    lora_r: int = typer.Option(16, "--lora-r", help="LoRA rank."),
    lora_alpha: int = typer.Option(32, "--lora-alpha", help="LoRA alpha."),
    no_vision: bool = typer.Option(False, "--no-vision", help="Disable vision encoder LoRA."),
    packs_dir: Optional[str] = typer.Option(None, "--packs-dir", help="Packs directory."),
) -> None:
    """Fine-tune Qwen2.5-VL (4-bit NF4 QLoRA) on a Language Pack dataset."""
    from im_translator.training.schemas import TrainVLMLoRAConfig
    from im_translator.training.train_vlm_lora import run_vlm_lora_training

    settings = load_settings()
    configure_runtime_caches(settings)
    pdir = normalize_user_path(packs_dir) if packs_dir else settings.packs_path
    out_path = normalize_user_path(output_dir) if output_dir else (pdir / pack / "models" / "vlm_lora")

    cfg = TrainVLMLoRAConfig(
        pack_id=pack,
        dataset_path=str(normalize_user_path(dataset)),
        output_dir=str(out_path),
        base_model=base_model,
        epochs=epochs,
        max_steps=max_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        finetune_vision_layers=not no_vision,
    )
    val_path = normalize_user_path(val_dataset) if val_dataset else None
    summary = run_vlm_lora_training(cfg, packs_dir=pdir, val_dataset_path=val_path)
    if "validation" in summary and isinstance(summary["validation"], dict):
        summary["validation"] = {k: v for k, v in summary["validation"].items() if k != "samples"}
    console.print_json(json.dumps(summary, ensure_ascii=False))
