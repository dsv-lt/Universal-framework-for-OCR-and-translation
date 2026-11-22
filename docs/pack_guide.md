# Руководство по созданию Language Pack

Эта инструкция описывает, как создать свой языковой пакет в папке `packs/<pack_id>/` с нуля или через **Pack Studio** в Web UI.

Готовые шаблоны с подробными комментариями лежат в [`docs/templates/`](templates/).

---

## Быстрый старт (через CLI или Web UI)

1. Создайте заготовку пакета командой:
   ```bash
   im-translator init-pack my_lang \
     --name "My Conlang" \
     --class conlang \
     --renderer sprites \
     --direction ltr
   ```
   *(Либо воспользуйтесь формой **Create New Pack** на вкладке **Pack Studio** в Web UI.)*
2. Настройте файлы пакета по шагам ниже.
3. Проверьте корректность пакета:
   ```bash
   im-translator validate-pack packs/my_lang
   ```

---

## Шаг 1. Заполнение `manifest.yaml`

Скопируйте `docs/templates/manifest.template.yaml` в `packs/<pack_id>/manifest.yaml`.

Ключевые поля:
- `id`: идентификатор папки пакета (строчные латинские буквы, цифры, `_`, `-`).
- `language_class`:
  - `cipher` — шифр/алфавит реального языка (нужен `postprocess.py` и желателен `intermediate_lang_code`, например `eu`).
  - `conlang` — вымышленный язык со своей грамматикой/лексикой.
  - `natural` — редкий/исторический естественный язык.
- `script.direction`: `ltr` (слева направо), `rtl` (справа налево), `ttb` (сверху вниз).
- `script.renderer`: `sprites` или `font`.
- `script.charset`: строка всех уникальных символов, которые должен предсказывать OCR и рендерить генератор синтетики.
- `detection.default_detector`: `craft` (лучше для непривычных/рукописных глифов) или `dbnet` (быстрее для ровных строк/блоков).
- `detection.granularity`: `line` (для `onnx_ocr`) или `block` (для `vlm_peft` / `vlm_llamacpp`).
- `postprocess.enabled`: `true`, если у языка есть детерминированный промежуточный слой (`postprocess.py`), иначе `false`.

---

## Шаг 2. Выбор способа рендера: `glyphs/` или `font.ttf`

### Вариант А: `renderer: sprites` (`glyphs/` + `glyphs_meta.yaml`)
Подходит, если `.ttf`-шрифта нет и символы вырезаны из кадров/таблиц:
1. Поместите прозрачные **RGBA PNG** (тёмные/белые чернила на **полностью прозрачном фоне**, альфа-канал — маска чернил) в `packs/<pack_id>/glyphs/`.
   - Стандартная высота холста — около `64–128 px` (прозрачные поля по краям желательно аккуратно обрезать или учесть в `baseline_offset`).
   - Можно добавить несколько вариантов одного глифа (`a_1.png`, `a_2.png`) для имитации живого почерка.
2. Заполните `glyphs_meta.yaml` по шаблону `docs/templates/glyphs_meta.template.yaml`:
   - `baseline_offset`: сдвиг вниз/вверх в долях высоты (например, `0.25` для букв с нижним выносным элементом `g`, `p`, `y`; `-0.1` для приподнятых).
   - `scale`: индивидуальный масштаб, если символ вырезался крупнее/мельче остальных.
   - `advance_ratio`: ширина шага до следующей буквы.
   - Для **абугид и диакритики**: укажите `is_combining: true`, `anchor: "top" | "bottom" | "center"` и `anchor_offset: [dx, dy]` — генератор наложит знак на предыдущий базовый символ.

> **Совет по именам файлов на Windows/NTFS:** файловая система Windows не различает `A.png` и `a.png`! Для заглавных и строчных букв обязательно используйте разные имена, например `upper_A.png` и `lower_a.png` (и пропишите их в `glyphs_meta.yaml`).

### Вариант Б: `renderer: font` (`font.ttf` / `.otf`)
Подходит для Тенгвара, Ауребеша, клингонского и любых письменностей с готовым векторным шрифтом:
1. Положите файл шрифта в `packs/<pack_id>/font.ttf` (или `.otf`).
2. В `manifest.yaml` укажите:
   ```yaml
   script:
     renderer: font
     font_path: font.ttf
   ```
3. Проверьте отображение через вкладку **Synth & Train** или `im-translator synth-preview <pack_id> "test string"`.

---

## Шаг 3. Опциональный промежуточный слой (`postprocess.py`)

Нужен, если между распознаванием символов и переводом есть детерминированное преобразование (шифр подстановки, декодер чисел, очистка разметки).
- Если слой **не нужен** (большинство конлангов и редких языков) — просто оставьте `postprocess.enabled: false` и не создавайте файл.
- Если слой **нужен**:
  1. Скопируйте `docs/templates/postprocess.template.py` в `packs/<pack_id>/postprocess.py`.
  2. Реализуйте функцию с фиксированной сигнатурой:
     ```python
     def transform(raw_text: str, context: dict) -> dict:
         return {
             "text": decoded_text,
             "original_script": raw_text,
             "notes": ["..."],
             "metadata": {},
         }
     ```
  3. Пользователь в Web UI или CLI сможет в любой момент отключить этот слой тумблером (например, для отладки сырого вывода OCR).

---

## Шаг 4. Словарь и грамматика (`glossary.json` и `grammar.md`)

Даже если вы переводите через общую LLM (без отдельной текстовой LoRA), правильно заполненные `glossary.json` и `grammar.md` резко снижают ошибки на именах собственных, магии и редкой грамматике. Подробнее см. в [docs/translation_rag_lora.md](translation_rag_lora.md).

---

## Шаг 5. Валидация пакета

Команда `validate-pack` (и кнопка **Validate Pack** в Web UI) автоматически проверяет:
- синтаксис и схему `manifest.yaml`;
- существование всех файлов спрайтов из `glyphs_meta.yaml` и наличие альфа-канала / валидных размеров (или загрузку `font.ttf`);
- синтаксис `postprocess.py` и наличие функции `transform`;
- валидность JSON-структуры `glossary.json` и читаемость `grammar.md`.
