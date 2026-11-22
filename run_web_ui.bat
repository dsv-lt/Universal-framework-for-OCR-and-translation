@echo off
setlocal
title im_Translator Web UI
echo Starting im_Translator Web UI at http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
wsl.exe -e bash -lc "export IM_TRANSLATOR_HOME=/mnt/d/ai/im_Translator && /home/tshatmatis/.venvs/im_translator/bin/im-translator serve --host 127.0.0.1 --port 8000"
pause
