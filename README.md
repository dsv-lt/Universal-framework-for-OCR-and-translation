# im_Translator

**im_Translator** — универсальный модульный open-source фреймворк для распознавания (OCR / VLM) и перевода вымышленных, зашифрованных и редких письменностей с использованием **Языковых пакетов (Language Packs)**.

> Статус: **Phase 1 (Core, Templates, CLI, Web UI, Synthesis & Detectors Architecture)** — без предустановленных тяжёлых весов. Работает из коробки на лёгких зависимостях, а ONNX / VLM / Train-бэкенды подключаются по мере необходимости.

---

## Ключевая архитектура

Пайплайн разделён на независимые переносимые стадии, чтобы проект одинаково гибко работал на CPU, NVIDIA CUDA, AMD DirectML и Apple Silicon:

```text
[Изображение]
   │
   ▼ 1. Детекция и выпрямление (CRAFT / DBNet++ ONNX + warpPerspective)
[Выпрямленные области / строки]
   │
   ▼ 2. Распознавание письменности (--lang, ONNX OCR / VLM PEFT / VLM llama.cpp)
[Сырой текст / транскрипция]
   │
   ├─► (Опционально) Промежуточный слой: packs/<lang>/postprocess.py
   │
   ▼ 3. Перевод [Source] → EN → [Target]
       (любой OpenAI-совместимый API: llama-server, Ollama, LM Studio, vLLM, облако)
       + RAG по glossary.json + grammar.md + (опционально) Conlang Text LoRA
```

Подробнее:
- [docs/architecture.md](docs/architecture.md) — полная спецификация архитектуры и форматов.
- [docs/pack_guide.md](docs/pack_guide.md) — пошаговое руководство по созданию своего Language Pack из шаблонов (`docs/templates/`).
- [docs/translation_rag_lora.md](docs/translation_rag_lora.md) — руководство по пивот-переводу через EN, заполнению `glossary.json`/`grammar.md` и выбору между RAG и Text LoRA.

---

## Поддерживаемые классы языков и рендереры

- **Классы языков (`class`):**
  - `cipher` — шифры и альтернативные письменности реальных языков (например, `[HL] → Basq`).
  - `conlang` — полноценные вымышленные языки со своей грамматикой и лексикой.
  - `natural` — малоресурсные и исторические естественные языки.
- **Типы рендера символов для синтетики (`script.renderer`):**
  - `sprites` — папка `glyphs/` с прозрачными PNG-спрайтами (поддерживает несколько рукописных вариантов одной буквы и диакритические якоря) + `glyphs_meta.yaml`.
  - `font` — векторный шрифт `.ttf` / `.otf` (включая кернинг, лигатуры, RTL/TTB через Pillow / HarfBuzz).
- **Детекторы (`detection.detector`):**
  - `craft` — CRAFT ONNX (тепловые карты символов и связок — хорошо видит незнакомые шрифты).
  - `dbnet` — DBNet++ ONNX (быстрая сегментация текстовых областей с `unclip`).

---

## Быстрый старт

### 1. Установка (WSL2 / Linux / macOS / Windows)

> **Важно для WSL2 + диск `D:`:** проект и тяжёлые модели/датасеты удобно хранить на `/mnt/d/ai/im_Translator`, а Python virtualenv лучше создавать на Linux ext4 (например, `~/.venvs/im_translator`), чтобы избежать медленной работы 9p/NTFS с симлинками.

```bash
# Базовое окружение (CLI, Web UI, генератор синтетики, RAG, валидатор)
uv venv ~/.venvs/im_translator --python 3.12
source ~/.venvs/im_translator/bin/activate
uv pip install -e ".[dev]"

# Опциональные расширения:
# uv pip install -e ".[onnx]"          # ONNX Runtime для CRAFT / DBNet++ / ONNX OCR
# uv pip install -e ".[vlm,train]"     # PyTorch + Transformers + PEFT + обучение
```

### 2. Команды CLI

```bash
# Посмотреть доступные Language Packs
im-translator list-packs

# При первом клоне: сгенерировать демо-глифы для packs/_example_demo/
python packs/_example_demo/generate_glyphs.py

# Проверить пакет (на примере встроенного демо-пакета)
im-translator validate-pack packs/_example_demo

# Создать заготовку нового пакета из шаблонов
im-translator init-pack my_lang --name "My Language" --class conlang --renderer sprites

# Сгенерировать превью синтетики
im-translator synth-preview _example_demo "hello world" -o preview.png

# Запустить локальный Web UI + REST API (http://127.0.0.1:8000)
im-translator serve --host 127.0.0.1 --port 8000
# Или из Windows: run_web_ui.bat
```

### 3. Разделы Web UI

1. **Recognize & Translate** — загрузка кадра, переключение детектора `CRAFT` / `DBNet++`, интерактивное выделение/отключение полигонов, выбор OCR-бэкенда, тумблер промежуточного слоя (`postprocess.py`), пошаговый вывод (`Raw → Postprocessed → EN → Target`).
2. **Pack Studio** — просмотр и валидация пакетов, мастер создания нового пакета с выбором `sprites` (`glyphs/`) или `font` (`.ttf`).
3. **Synth & Train** — живое превью синтетики со слайдерами аугментаций (перспектива, наклон, размытие, зачёркивания, перекрытия, фон), генерация команд для создания датасета и обучения.
4. **Settings** — пути (`packs_dir`, `models_dir`, `datasets_dir`), устройство, URL OpenAI-совместимого LLM-сервера (`llama-server`, `Ollama`, и т.д.).

---

## Тестирование

```bash
pytest -v
```