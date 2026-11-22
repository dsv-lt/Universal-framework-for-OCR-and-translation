# Grammar & Style Guide for `<LANGUAGE_NAME>` (`<pack_id>`)

> Заполните этот файл кратко (300–800 токенов). Он передаётся в системный промпт переводчика на шаге `[Source] → [EN]`.

## 1. Typology & Word Order
- **Word order:** (например: `SOV` / `SVO` / `VSO`)
- **Morphology:** (например: agglutinative suffixes / fusional / isolating)
- **Script notes:** (например: case-insensitive, numerals use Roman-like additive notation, `[?]` marks unreadable occluded glyphs, `~~word~~` marks crossed-out text)

## 2. Key Affixes & Particles
| Morpheme | Function | Example |
| :--- | :--- | :--- |
| `-a` | Definite / singular marker | `ur` → `ura` (the water) |
| `-ak` | Plural / ergative marker | `urak` (the waters) |
| `-en` | Genitive (`of`) | `Rudeusen` (of Rudeus) |

## 3. Translation Rules
1. Never translate or "correct" proper nouns listed as `is_proper_noun: true` in `glossary.json`.
2. Preserve crossed-out spans (`~~...~~`) in the English translation.
3. If an unknown token looks like an unattested proper name or spell incantation, transliterate it faithfully rather than guessing an English word.

## 4. Few-Shot Reference Examples (`Source → EN`)
- **Source:** `Example source sentence one.`
  **EN:** `Example English translation one.`
- **Source:** `Example source sentence two.`
  **EN:** `Example English translation two.`
