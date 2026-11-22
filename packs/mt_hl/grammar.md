# Grammar & Translation Guide for Mushoku Tensei Human Language (`mt_hl`)

## 1. Script & Cipher Structure
- **Underlying Language:** Real **Basque (`eu` / Euskara)** encoded in Human Language (`[HL]`) calligraphy via a reciprocal consonant Atbash cipher (`b↔z, c↔y, d↔x, f↔w, g↔v, h↔t, j↔s, k↔r, l↔q, m↔p, n↔n`, vowels `a, e, i, o, u` unchanged).
- **OCR Output:** The specialized OCR/VLM outputs the decoded **Basque (`Basq`)** text directly.
- **Numerals:** Additive Roman-style sequences using `[HL]` glyphs (`k=1(I), e=5(V), r=10(X), s=50(L), z=100(C), w=500(D), g=1000(M)`), e.g. `kkk` (`Basq: rrr`) = 3, `srk` (`Basq: jkr`) = 61.

## 2. Basque Typology & Affixes
- **Word Order:** `SOV` (Subject–Object–Verb), agglutinative postpositional suffixes.
- **Key Suffixes:**
  - `-a` (definite singular), `-ak` (plural / absolutive), `-ek` (ergative plural)
  - `-en` / `-aren` (genitive "of"), `-ari` (dative "to")
  - `-tik` / `-tatik` (ablative "from"), `-ra` (allative "to / for"), `-an` (inessive "in / on"), `-gatik` ("because of")
  - `-tzen` (imperfective / "-ing"), `-tzeko` / `-teko` ("in order to"), `-tzea` (gerund "-tion"), `-tuta` (participle "-ed")
  - `-ko` / `-zko` (attributive / material adjective), `-tasun` (abstract noun "-ness")

## 3. Translation Rules
1. Never alter or translate *Mushoku Tensei* proper nouns (`Rudeus`, `Greyrat`, `Roxy`, `Sylphiette`, `Eris`, `Ruijerd`, `Orsted`, `Hitogami`, `Fittoa`, `Asura`, etc.).
2. Preserve crossed-out spans (`~~...~~`) and unreadable markers (`[?]`).
3. Keep magic rank terminology consistent: `oinarrizko` = Elementary, `bitartekoa` = Intermediate, `aurreratua` = Advanced, `santua` = Saint, `erregea` = King, `enperadorea` = Emperor, `jainkozkoa` = God-tier.
4. If the input contains partial or noisy OCR fragments (e.g., from occluded anime frames), translate only what is legible and avoid inventing missing sentences.

## 4. Reference Examples (`Basq → EN`)
- **Source:** `Kristal magikoak`
  **EN:** `Magic crystals`
- **Source:** `Magia Graduen berez`
  **EN:** `Of the Ranks of Magic (by nature)`
- **Source:** `Zirkulu magikoa aktibatzeko behar da.`
  **EN:** `It is needed to activate the magic circle.`
