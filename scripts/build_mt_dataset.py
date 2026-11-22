"""Build held-out synthetic Basque + Mushoku Tensei dataset (3,600 train / 200 val) for `packs/mt_hl`."""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "hf_cache"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "models" / "torch_cache"))

from datasets import load_dataset

from im_translator.core.pack_loader import load_pack
from im_translator.core.schemas import AugmentationConfig
from im_translator.synth.generator import generate_dataset

PACK_DIR = PROJECT_ROOT / "packs" / "mt_hl"
DATASET_DIR = PROJECT_ROOT / "datasets" / "mt_hl"

ALLOWED_RE = re.compile(r"^[A-Za-z .,;:!?-]+$")
CAMEL_RE = re.compile(r"[a-z][A-Z]")

DOMAIN_CORPUS = [
    "Rudeus Greyrat Asura Erresumako magoa da.",
    "Sylphiette eta Roxy Migurdia magoak dira.",
    "Eris Boreas Greyrat Fittoa eskualdekoa da.",
    "Paul Greyrat eta Zenith Lilia etxean daude.",
    "Orsted Dragoi Jainkoa indartsua da.",
    "Perugius Dola zeruko gotorlekuan bizi da.",
    "Ruijerd Superdia gerlaria bidaian dabil.",
    "Kishirika Kishirisu deabru enperadorea da.",
    "Nanahoshi Shizuka beste mundu batetik etorri zen.",
    "Zanoba Shirone eta Cliff Grimoire lagunak dira.",
    "Ghislaine Dedoldia ezpata erregea da.",
    "Ranoa Unibertsitatea hiri handian dago.",
    "Magia zirkulu zaharra harri gainean dago.",
    "Sua eta ura elkartzen dira sorginkeria honetan.",
    "Argi beltza zerutik jaisten da gau ilunean.",
    "Etxe zaharreko giltza burdinazkoa eta astuna da.",
    "Urre txanpon eta zilar prezioak igotzen dira.",
    "Teleportazio kalamitatea Fittoa eskualdean gertatu zen.",
    "Abenturazaleen gremioa kontratu berriak banatzen ditu.",
    "Iparraldeko basoan munstro arriskutsuak bizi dira.",
    "Deabru kontinentean bidea oso gogorra eta luzea da.",
    "Millis kontinenteko eliza legeak zorrotzak dira.",
    "Ezpata estiloa indarra eta abiadura behar ditu.",
    "Liburu zahar hau hizkuntza sekretuan idatzita dago.",
    "Quagmire ezizena oso ezaguna bihurtu zen.",
    "Vortex bortitzak itsasontzia eraman zuen urrunera.",
    "Xanadu eta Buena herriak mapan agertzen dira.",
    "Kristal distiratsuak magia gordetzen du barruan.",
    "Prezioa: rrr zilar txanpon eta kk brontze.",
    "Urtea: wwww eeee rrr kk.",
    "Sari handia: zzz urre txanpon bizirik edo hilda!",
    "Kontuz: zirkulu debekatua - ez ukitu harria!",
]

# Extra lexical building blocks ensuring balanced rare-letter & punctuation exposure
BASQUE_LEXICON = [
    "etxe", "ur", "su", "lur", "haize", "argi", "ilun", "magia", "sorginkeria",
    "harri", "kristal", "zirkulu", "botere", "indar", "ezpata", "liburu", "bide",
    "mendi", "baso", "itsaso", "zeru", "izar", "eguzki", "ilargi", "gizon", "emakume",
    "haur", "errege", "jaun", "andre", "lagun", "etsai", "gerlari", "mago", "maisua",
    "ikasle", "herri", "hiri", "erresuma", "gaztelu", "dorre", "atea", "giltza",
    "urre", "zilar", "brontze", "txanpon", "diru", "prezio", "kontratu", "gutun",
    "berri", "zahar", "handi", "txiki", "on", "txar", "beltz", "zuri", "gorri",
    "urdin", "berde", "altu", "baxu", "luze", "labur", "azkar", "motel", "bero", "hotz",
    "izan", "egon", "egin", "joan", "etorri", "ikusi", "entzun", "esan", "jakin",
    "ikasi", "idatzi", "irakurri", "babestu", "eraso", "sendatu", "bilatu", "aurkitu",
    "eman", "hartu", "ekarri", "eraman", "sortu", "apurtu", "zabaldu", "itxi",
    "hau", "hori", "hura", "hauek", "eta", "edo", "baina", "ez", "bai", "oso",
    "beti", "inoiz", "hemen", "hor", "han", "orain", "gaur", "bihar", "atzo",
    "Fittoa", "Asura", "Ranoa", "Millis", "Sharia", "Roa", "Buena", "Rudeus",
    "Sylphiette", "Roxy", "Eris", "Orsted", "Perugius", "Zanoba", "Cliff",
    "Quagmire", "Vortex", "Kishirika", "Nanahoshi", "Ghislaine", "Ruijerd",
    "Zenith", "Paul", "Lilia", "Norn", "Aisha", "Badigadi", "Ariel",
]


def _normalize_basque_ascii(text: str) -> str:
    """Fold accented Latin characters into pure ASCII [A-Za-z] + basic punctuation."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_chars = []
    for ch in nfkd:
        if unicodedata.combining(ch):
            continue
        if ch in "«»\"'()[]{}/*@#$%^&+=_~`\\|<>":
            continue
        if ch in "–—":
            ch = "-"
        ascii_chars.append(ch)
    cleaned = " ".join("".join(ascii_chars).split())
    return cleaned.strip(" -")


def _extract_opus_clauses(max_lines: int = 4500) -> list[str]:
    ds = load_dataset("Helsinki-NLP/opus-100", "en-eu", split="train", streaming=True)
    seen: set[str] = set()
    clauses: list[str] = []

    for row in ds:
        raw = row["translation"].get("eu", "")
        norm = _normalize_basque_ascii(raw)
        if not norm or not ALLOWED_RE.match(norm):
            continue
        if CAMEL_RE.search(norm) or "kgm" in norm.lower() or "http" in norm.lower():
            continue

        # Split overly long sentences into natural 2-7 word clauses
        words = norm.split()
        if len(words) < 2:
            continue
        if len(words) > 8:
            # Break on punctuation or every 5-6 words
            parts = re.split(r"(?<=[,;:.!?])\s+", norm)
            chunks = []
            for p in parts:
                p_words = p.split()
                if 2 <= len(p_words) <= 8:
                    chunks.append(p)
                elif len(p_words) > 8:
                    chunks.append(" ".join(p_words[:6]))
        else:
            chunks = [norm]

        for ch in chunks:
            ch = " ".join(ch.split())
            if 8 <= len(ch) <= 48 and 2 <= len(ch.split()) <= 6 and ch not in seen:
                seen.add(ch)
                clauses.append(ch)
                if len(clauses) >= max_lines:
                    return clauses
    return clauses


def _synthesize_rare_and_lex_lines(count: int, seed: int = 123) -> list[str]:
    rng = random.Random(seed)
    puncts = [".", ".", ".", ",", ";", ":", "!", "?", "-"]
    rare_words = [
        "Quagmire", "Vortex", "Wig", "Xanadu", "Zanoba", "Sylphiette", "Cliff",
        "Kishirika", "Roxy", "Jewel", "Queen", "kzk", "rrr", "eee", "zzz", "ww", "gg",
    ]
    out: list[str] = []
    for _ in range(count):
        n_words = rng.randint(2, 6)
        words = [rng.choice(BASQUE_LEXICON) for _ in range(n_words)]
        if rng.random() < 0.45:
            words[rng.randrange(n_words)] = rng.choice(rare_words)
        if rng.random() < 0.70 and words[0][0].islower():
            words[0] = words[0].capitalize()
        line = " ".join(words)
        if rng.random() < 0.65:
            p = rng.choice(puncts)
            if p == "-":
                mid = max(1, len(words) // 2)
                line = " ".join(words[:mid]) + " - " + " ".join(words[mid:])
            else:
                line = line.rstrip(".,;:!?") + p
        out.append(line)
    return out


def _shard_worker(args: tuple) -> list[dict]:
    corpus_lines, aug_dict, count, mode, seed, shard_dir_str = args
    shard_dir = Path(shard_dir_str)
    pack = load_pack(PACK_DIR)
    aug = AugmentationConfig(**aug_dict)
    lbl = generate_dataset(
        pack=pack,
        corpus_lines=corpus_lines,
        output_dir=shard_dir,
        num_samples=count,
        aug_cfg=aug,
        mode=mode,
        seed=seed,
        proper_noun_ratio=0.18,
    )
    rows = []
    for raw in lbl.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            item = json.loads(raw)
            item["_src_path"] = str(shard_dir / item["image"])
            rows.append(item)
    return rows


def _build_split(
    lines_corpus: list[str],
    aug_cfg: AugmentationConfig,
    n_line: int,
    n_block: int,
    split_name: str,
    seed: int,
    tmp_root: Path,
) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    aug_dict = aug_cfg.model_dump()
    tasks = []
    # 4 line shards + 2 block shards in parallel on local Linux /tmp
    for idx_s, cnt in enumerate(_split_counts(n_line, 4)):
        tasks.append((lines_corpus, aug_dict, cnt, "line", seed + idx_s * 17, str(tmp_root / f"{split_name}_l{idx_s}")))
    for idx_s, cnt in enumerate(_split_counts(n_block, 2)):
        tasks.append((lines_corpus, aug_dict, cnt, "block", seed + 1000 + idx_s * 19, str(tmp_root / f"{split_name}_b{idx_s}")))

    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for res_rows in pool.map(_shard_worker, tasks):
            records.extend(res_rows)

    rng = random.Random(seed + 99)
    rng.shuffle(records)

    images_dir = DATASET_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    copy_pairs: list[tuple[str, str]] = []
    out_records: list[dict] = []
    for idx, item in enumerate(records):
        src_p = item.pop("_src_path")
        dst_rel = f"images/{split_name}_{idx:06d}.png"
        copy_pairs.append((src_p, str(DATASET_DIR / dst_rel)))
        item["id"] = idx
        item["image"] = dst_rel
        item["split"] = split_name
        out_records.append(item)

    with ThreadPoolExecutor(max_workers=8) as io_pool:
        list(io_pool.map(lambda pair: shutil.copyfile(pair[0], pair[1]), copy_pairs))

    return out_records


def _split_counts(total: int, shards: int) -> list[int]:
    q, r = divmod(total, shards)
    return [q + (1 if i < r else 0) for i in range(shards) if q + (1 if i < r else 0) > 0]


def main() -> None:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    tmp_root = Path("/tmp/mt_hl_build")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    print("Loading packs/mt_hl ...", flush=True)
    _ = load_pack(PACK_DIR)

    print("Streaming clean Basque clauses from Helsinki-NLP/opus-100 (en-eu)...", flush=True)
    opus_clauses = _extract_opus_clauses(max_lines=4000)
    print(f"Collected {len(opus_clauses)} clean OPUS Basque clauses.", flush=True)

    lex_lines = _synthesize_rare_and_lex_lines(count=1500, seed=77)
    unique_pool = list(dict.fromkeys(opus_clauses + DOMAIN_CORPUS + lex_lines))
    rng = random.Random(2026)
    rng.shuffle(unique_pool)

    # Strictly disjoint split between train and val corpora
    split_idx = int(len(unique_pool) * 0.90)
    base_train = unique_pool[:split_idx]
    val_corpus = unique_pool[split_idx:]
    # Boost in-domain MT sentences within train only
    train_corpus = base_train + [s for s in DOMAIN_CORPUS if s in set(base_train)] * 10
    print(f"Corpus sizes -> train pool: {len(train_corpus)} ({len(base_train)} unique), val pool: {len(val_corpus)}", flush=True)

    aug_cfg = AugmentationConfig(
        rotation_deg=3.0,
        perspective_strength=0.04,
        shear_deg=3.0,
        elastic_alpha=1.4,
        stroke_jitter=1,
        blur_sigma=0.6,
        strikethrough_prob=0.03,
        occlusion_prob=0.0,
        lighting_variation=0.22,
        jpeg_quality=84,
        padding=44,
    )

    # 3,600 train (70% line = 2520, 30% block = 1080); 200 val (140 line, 60 block)
    train_rows = _build_split(train_corpus, aug_cfg, n_line=2520, n_block=1080, split_name="train", seed=42, tmp_root=tmp_root)
    val_rows = _build_split(val_corpus, aug_cfg, n_line=140, n_block=60, split_name="val", seed=4242, tmp_root=tmp_root)
    shutil.rmtree(tmp_root, ignore_errors=True)

    (DATASET_DIR / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + "\n",
        encoding="utf-8",
    )
    (DATASET_DIR / "val.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val_rows) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(train_rows)} train samples to {DATASET_DIR / 'train.jsonl'}")
    print(f"Saved {len(val_rows)} val samples to {DATASET_DIR / 'val.jsonl'}")


if __name__ == "__main__":
    main()
