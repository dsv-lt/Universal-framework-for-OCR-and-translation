"""FastAPI Web UI and REST API endpoint tests."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from im_translator.app.server import create_app
from im_translator.core.settings import AppSettings

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_PACK = REPO_ROOT / "packs" / "_example_demo"


def _make_client(tmp_path: Path) -> TestClient:
    packs_dir = tmp_path / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(DEMO_PACK, packs_dir / "_example_demo")

    settings = AppSettings(
        packs_dir=str(packs_dir),
        models_dir=str(tmp_path / "models"),
        datasets_dir=str(tmp_path / "datasets"),
    )
    app = create_app(settings_override=settings)
    return TestClient(app)


def test_app_endpoints_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("IM_TRANSLATOR_CONFIG", str(tmp_path / "config.yaml"))
    client = _make_client(tmp_path)

    # Web UI root
    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "im_Translator" in r_index.text

    # Health & capabilities
    assert client.get("/api/health").json()["status"] == "ok"
    caps = client.get("/api/capabilities").json()
    assert {d["detector"] for d in caps["detectors"]} == {"craft", "dbnet"}
    assert {r["backend"] for r in caps["recognizers"]} == {"onnx_ocr", "vlm_peft", "vlm_llamacpp"}

    # Packs list & detail & validate
    packs_res = client.get("/api/packs").json()
    assert len(packs_res["packs"]) == 1
    assert packs_res["packs"][0]["id"] == "_example_demo"
    assert packs_res["packs"][0]["valid"] is True

    detail = client.get("/api/packs/_example_demo").json()
    assert detail["validation"]["valid"] is True

    val_res = client.post("/api/packs/validate", json={"pack_id": "_example_demo"}).json()
    assert val_res["valid"] is True

    # Create pack via API
    create_res = client.post(
        "/api/packs/create",
        json={
            "pack_id": "new_conlang",
            "name": "New Conlang",
            "language_class": "conlang",
            "renderer": "sprites",
            "direction": "ltr",
            "enable_postprocess": False,
        },
    )
    assert create_res.status_code == 200
    assert (tmp_path / "packs" / "new_conlang" / "manifest.yaml").is_file()

    # Synth preview returns valid PNG bytes
    preview_res = client.post(
        "/api/synth/preview",
        json={"pack_id": "_example_demo", "text": "hello world", "seed": 7},
    )
    assert preview_res.status_code == 200
    assert preview_res.headers["content-type"] == "image/png"
    png_bytes = preview_res.content
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    # Detection endpoint (using contour fallback since ONNX weights aren't present)
    for det_name in ("craft", "dbnet"):
        det_res = client.post(
            "/api/detect",
            files={"file": ("sample.png", png_bytes, "image/png")},
            data={
                "pack_id": "_example_demo",
                "detector": det_name,
                "granularity": "line",
                "demo_contour_fallback": "true",
            },
        )
        assert det_res.status_code == 200
        det_json = det_res.json()
        assert det_json["detector"] == det_name
        assert det_json["used_fallback"] is True
        assert len(det_json["regions"]) >= 1

    # Settings GET / PUT
    cur_settings = client.get("/api/settings").json()
    cur_settings["default_target_lang"] = "en"
    put_res = client.put("/api/settings", json=cur_settings)
    assert put_res.status_code == 200
    assert client.get("/api/settings").json()["default_target_lang"] == "en"
