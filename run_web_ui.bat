@echo off
setlocal EnableExtensions
title im_Translator Web UI

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "IM_TRANSLATOR_HOME=%ROOT%"
set "HF_HOME=%ROOT%\models\hf_cache"
set "TORCH_HOME=%ROOT%\models\torch_cache"

echo Starting im_Translator Web UI at http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000

if exist "%ROOT%\.venv\Scripts\im-translator.exe" (
    "%ROOT%\.venv\Scripts\im-translator.exe" serve --host 127.0.0.1 --port 8000
) else (
    wsl.exe -e bash -lc "WSL_ROOT=$(wslpath '%ROOT%'); export IM_TRANSLATOR_HOME=\"$WSL_ROOT\" HF_HOME=\"$WSL_ROOT/models/hf_cache\" TORCH_HOME=\"$WSL_ROOT/models/torch_cache\"; if [ -d \"$HF_HOME/hub\" ]; then export HF_HUB_OFFLINE=1; fi; BIN=\"$HOME/.venvs/im_translator/bin/im-translator\"; if [ ! -x \"$BIN\" ]; then BIN=im-translator; fi; exec \"$BIN\" serve --host 127.0.0.1 --port 8000"
)
pause
