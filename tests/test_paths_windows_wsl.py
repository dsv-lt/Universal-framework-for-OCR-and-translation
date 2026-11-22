"""Tests for cross-platform Windows <-> WSL2 path normalization and display formatting."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from im_translator.app.server import create_app
from im_translator.core import paths as path_utils
from im_translator.core.settings import AppSettings, configure_runtime_caches


def test_win_and_wsl_string_conversion() -> None:
    assert path_utils.win_to_wsl_path(r"D:\ai\im_Translator\packs") == "/mnt/d/ai/im_Translator/packs"
    assert path_utils.win_to_wsl_path("C:/Users/demo/file.png") == "/mnt/c/Users/demo/file.png"
    assert path_utils.win_to_wsl_path(r"E:") == "/mnt/e"
    assert path_utils.win_to_wsl_path("datasets/mt_hl") == "datasets/mt_hl"

    assert path_utils.wsl_to_win_path("/mnt/d/ai/im_Translator/packs") == r"D:\ai\im_Translator\packs"
    assert path_utils.wsl_to_win_path("/mnt/d") == "D:\\"
    assert path_utils.wsl_to_win_path("/home/user/project") == "/home/user/project"


def test_app_settings_accepts_windows_paths_on_wsl(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(path_utils, "is_wsl", lambda: True)
    settings = AppSettings(
        packs_dir=r"D:\ai\im_Translator\packs",
        models_dir=r"D:\ai\im_Translator\models",
        datasets_dir=r"D:\ai\im_Translator\datasets",
    )
    assert settings.packs_path == Path("/mnt/d/ai/im_Translator/packs").resolve()
    assert settings.models_path == Path("/mnt/d/ai/im_Translator/models").resolve()
    assert settings.datasets_path == Path("/mnt/d/ai/im_Translator/datasets").resolve()

    assert path_utils.format_display_path("/mnt/d/ai/im_Translator/models") == r"D:\ai\im_Translator\models"
    assert path_utils.format_display_path(tmp_path) == str(tmp_path)

    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("TORCH_HOME", raising=False)
    configure_runtime_caches(settings)
    assert path_utils.os.environ["HF_HOME"] == "/mnt/d/ai/im_Translator/models/hf_cache"


def test_api_settings_and_pack_ordering(monkeypatch, tmp_path: Path) -> None:
    packs = tmp_path / "packs"
    (packs / "_demo").mkdir(parents=True)
    (packs / "mt_hl").mkdir(parents=True)
    (packs / "alpha").mkdir(parents=True)

    monkeypatch.setenv("IM_TRANSLATOR_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.setattr(path_utils, "is_wsl", lambda: True)

    app = create_app(
        AppSettings(
            packs_dir=str(packs),
            models_dir=str(tmp_path / "models"),
            datasets_dir=str(tmp_path / "datasets"),
        )
    )
    client = TestClient(app)

    res = client.put(
        "/api/settings",
        json={
            "packs_dir": r"D:\ai\im_Translator\packs",
            "models_dir": r"D:\ai\im_Translator\models",
            "datasets_dir": r"D:\ai\im_Translator\datasets",
            "device": "auto",
            "llm_base_url": "http://127.0.0.1:8080/v1",
            "llm_api_key": "not-needed",
            "llm_model": "local-model",
            "default_target_lang": "ru",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["settings"]["packs_dir"] == r"D:\ai\im_Translator\packs"
    assert body["settings"]["models_dir"] == r"D:\ai\im_Translator\models"

    packs_resp = client.get("/api/packs")
    assert packs_resp.status_code == 200
    ids = [p["id"] for p in packs_resp.json()["packs"]]
    if "mt_hl" in ids and "_example_demo" in ids:
        assert ids.index("mt_hl") < ids.index("_example_demo")
