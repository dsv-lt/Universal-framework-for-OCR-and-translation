"""Regression tests for isolated module imports, CTC decoding, CLI commands, and train/LLM API."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from im_translator.app.server import create_app
from im_translator.cli.main import app as cli_app
from im_translator.core.settings import AppSettings
from im_translator.recognition.onnx_ocr import OnnxOCRRecognizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_PACK = REPO_ROOT / "packs" / "_example_demo"


@pytest.mark.parametrize(
    "module_name",
    [
        "im_translator.detection.craft",
        "im_translator.detection.dbnet",
        "im_translator.recognition.onnx_ocr",
        "im_translator.recognition.vlm_peft",
        "im_translator.recognition.vlm_llamacpp",
        "im_translator.translation.pivot",
        "im_translator.translation.rag",
        "im_translator.synth.generator",
        "im_translator.core.pipeline",
        "im_translator.app.server",
        "im_translator.cli.main",
    ],
)
def test_isolated_module_import_no_circular_error(module_name: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Failed importing {module_name}: {proc.stderr}"


def test_core_lazy_export_translation_pipeline() -> None:
    from im_translator.core import TranslationPipeline
    from im_translator.core.pipeline import TranslationPipeline as DirectPipeline

    assert TranslationPipeline is DirectPipeline


def test_onnx_ocr_ctc_greedy_decode_and_preprocess(tmp_path: Path) -> None:
    logits = np.full((6, 27), -10.0, dtype=np.float32)
    # blank=0, a=1..z=26 -> sequence [h, h, blank, i, blank, i] -> "hii"
    logits[0, 8] = 5.0
    logits[1, 8] = 5.0
    logits[2, 0] = 5.0
    logits[3, 9] = 5.0
    logits[4, 0] = 5.0
    logits[5, 9] = 5.0

    text, conf = OnnxOCRRecognizer.ctc_greedy_decode(logits, "abcdefghijklmnopqrstuvwxyz")
    assert text == "hii"
    assert conf > 0.99

    rec = OnnxOCRRecognizer(models_dir=tmp_path)
    tensor = rec.preprocess_crop(np.zeros((24, 100, 3), dtype=np.uint8), channels=3)
    assert tensor.shape == (1, 3, 32, 320)
    assert tensor.dtype == np.float32


def test_cli_end_to_end_commands(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("IM_TRANSLATOR_HOME", str(REPO_ROOT))
    monkeypatch.setenv("IM_TRANSLATOR_CONFIG", str(tmp_path / "config.yaml"))
    runner = CliRunner()

    res_list = runner.invoke(cli_app, ["list-packs", "--packs-dir", str(REPO_ROOT / "packs")])
    assert res_list.exit_code == 0
    assert "_example_demo" in res_list.stdout

    res_val_ok = runner.invoke(cli_app, ["validate-pack", str(DEMO_PACK)])
    assert res_val_ok.exit_code == 0

    res_val_miss = runner.invoke(cli_app, ["validate-pack", str(tmp_path / "nonexistent")])
    assert res_val_miss.exit_code == 1

    packs_out = tmp_path / "packs"
    res_init = runner.invoke(
        cli_app,
        [
            "init-pack",
            "cli_test_pack",
            "--name",
            "CLI Test Pack",
            "--class",
            "conlang",
            "--renderer",
            "sprites",
            "--packs-dir",
            str(packs_out),
        ],
    )
    assert res_init.exit_code == 0
    assert (packs_out / "cli_test_pack" / "manifest.yaml").is_file()

    preview_path = tmp_path / "cli_preview.png"
    res_prev = runner.invoke(
        cli_app,
        [
            "synth-preview",
            "_example_demo",
            "uryyb jbeyq",
            "-o",
            str(preview_path),
            "--packs-dir",
            str(REPO_ROOT / "packs"),
        ],
    )
    assert res_prev.exit_code == 0
    assert preview_path.is_file() and preview_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    ds_line = tmp_path / "ds_line"
    res_gen_line = runner.invoke(
        cli_app,
        [
            "synth-generate",
            "_example_demo",
            "-o",
            str(ds_line),
            "-n",
            "4",
            "--mode",
            "line",
            "--packs-dir",
            str(REPO_ROOT / "packs"),
        ],
    )
    assert res_gen_line.exit_code == 0
    line_rows = (ds_line / "labels.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(line_rows) == 4

    ds_block = tmp_path / "ds_block"
    res_gen_block = runner.invoke(
        cli_app,
        [
            "synth-generate",
            "_example_demo",
            "-o",
            str(ds_block),
            "-n",
            "3",
            "--mode",
            "block",
            "--packs-dir",
            str(REPO_ROOT / "packs"),
        ],
    )
    assert res_gen_block.exit_code == 0
    block_rows = (ds_block / "labels.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(block_rows) == 3
    assert "\n" in json.loads(block_rows[0])["text"]


def test_api_train_plan_and_offline_llm_test(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("IM_TRANSLATOR_CONFIG", str(tmp_path / "config.yaml"))
    packs_dir = tmp_path / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(DEMO_PACK, packs_dir / "_example_demo")

    labels_file = tmp_path / "labels.jsonl"
    labels_file.write_text(
        json.dumps({"image": "images/sample_000000.png", "text": "hello"}) + "\n",
        encoding="utf-8",
    )

    settings = AppSettings(
        packs_dir=str(packs_dir),
        models_dir=str(tmp_path / "models"),
        datasets_dir=str(tmp_path / "datasets"),
    )
    client = TestClient(create_app(settings_override=settings))

    for mode, ds_path in [
        ("onnx_ocr", str(labels_file)),
        ("vlm_lora", str(labels_file)),
        ("conlang_lora", str(REPO_ROOT / "docs" / "templates" / "conlang_lora.template.jsonl")),
    ]:
        res = client.post(
            "/api/train/plan",
            json={
                "mode": mode,
                "pack_id": "_example_demo",
                "dataset_path": ds_path,
                "output_dir": str(tmp_path / f"out_{mode}"),
                "epochs": 3,
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["stage"] == f"train_{mode}"
        assert body["dataset_exists"] is True

    llm_res = client.post(
        "/api/settings/test-llm",
        json={"base_url": "http://127.0.0.1:19999/v1"},
    )
    assert llm_res.status_code == 200
    assert llm_res.json()["ok"] is False
