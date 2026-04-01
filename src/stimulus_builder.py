"""
stimulus_builder.py
~~~~~~~~~~~~~~~~~~~
Seeds the SCM warmth/competence probe set from existing benchmarks.

Sources
-------
- CrowS-Pairs EN  (gender, profession, nationality)   → EN native items
- SHADES EN + FR  (nationality)                        → EN + FR native items
- Manual BG       (all three groups)                   → BG placeholder rows

Output
------
data/stimuli_seed.csv   — full seed set, ready for human annotation

Column schema
-------------
item_id            : unique key, e.g. EN-G-001  (lang-group-seq)
parallel_group_id  : shared across languages for the same concept, e.g. G-001
language           : en | fr | bg
origin             : parallel | native
                      parallel = same item translated across all 3 languages
                      native   = culture-specific to one language
dimension          : warmth | competence | needs_review
target_group       : gender | nationality | profession
target             : specific group (e.g. "woman", "French", "engineer")
sent_stereotype    : the stereotypical sentence
sent_anti_stereotype: the anti-stereotypical sentence
source             : dataset this was seeded from
validated          : False until a human has reviewed and corrected it
notes              : free-text annotation hints
"""

import io
import os
import pathlib
import sys
import urllib.request

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

COLS = [
    "item_id", "parallel_group_id",
    "language", "origin",
    "dimension",
    "target_group", "target",
    "sent_stereotype", "sent_anti_stereotype",
    "source", "validated", "notes",
]

# EN CrowS-Pairs integer → string bias_type map (verified against dataset examples)
CP_EN_BIAS = {
    0: "race", 1: "socioeconomic", 2: "gender",
    3: "disability", 4: "nationality", 5: "sexual-orientation",
    6: "physical-appearance", 7: "religion", 8: "age",
}

rows = []
_counters: dict = {}


def _next_id(lang: str, group: str) -> str:
    key = f"{lang}-{group}"
    _counters[key] = _counters.get(key, 0) + 1
    return f"{lang.upper()}-{group}-{_counters[key]:03d}"


def add(
    lang: str, origin: str, dimension: str,
    target_group: str, target: str,
    sent_stereo: str, sent_anti: str,
    source: str, notes: str = "",
) -> None:
    grp = {"gender": "G", "nationality": "N", "profession": "P"}[target_group]
    iid = _next_id(lang, grp)
    rows.append({
        "item_id":              iid,
        "parallel_group_id":    iid[3:],   # strip lang prefix: EN-G-001 → G-001
        "language":             lang,
        "origin":               origin,
        "dimension":            dimension,
        "target_group":         target_group,
        "target":               target,
        "sent_stereotype":      sent_stereo.strip(),
        "sent_anti_stereotype": sent_anti.strip(),
        "source":               source,
        "validated":            False,
        "notes":                notes,
    })


# ── 1. CrowS-Pairs EN ────────────────────────────────────────────────────────
print("Loading CrowS-Pairs EN …")
try:
    from datasets import load_dataset
    cp_en = load_dataset("crows_pairs", split="test",
                         trust_remote_code=True).to_pandas()
    cp_en["bias_type_str"] = cp_en["bias_type"].map(CP_EN_BIAS)

    # Gender (bias_type=2)
    for _, r in cp_en[cp_en["bias_type"] == 2].head(40).iterrows():
        add("en", "native", "needs_review",
            "gender", "woman/man",
            r["sent_more"], r["sent_less"],
            "crows_pairs_en",
            "Label warmth or competence; confirm target gender")

    # Socioeconomic → profession proxy (bias_type=1)
    for _, r in cp_en[cp_en["bias_type"] == 1].head(30).iterrows():
        add("en", "native", "needs_review",
            "profession", "",
            r["sent_more"], r["sent_less"],
            "crows_pairs_en",
            "Extract target profession from sentence; label warmth or competence")

    # Nationality (bias_type=4)
    for _, r in cp_en[cp_en["bias_type"] == 4].head(30).iterrows():
        add("en", "native", "needs_review",
            "nationality", "",
            r["sent_more"], r["sent_less"],
            "crows_pairs_en",
            "Extract target nationality; label warmth or competence")

    print(f"  CrowS-Pairs EN: {len(cp_en[cp_en['bias_type'].isin([1,2,4])])} source rows extracted")
except Exception as exc:
    print(f"  CrowS-Pairs EN failed: {exc}")


# ── 2. SHADES EN + FR (nationality pairs) ───────────────────────────────────
print("Loading SHADES …")
_SHADES_BASE = (
    "https://huggingface.co/datasets/"
    "bigscience-catalogue-data/bias-shades/raw/main"
)

for lang in ("en", "fr"):
    try:
        url = f"{_SHADES_BASE}/shades_nationality_{lang}.csv"
        hdrs = ({"Authorization": f"Bearer {HF_TOKEN}"}
                if HF_TOKEN else {"User-Agent": "Mozilla/5.0"})
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=30) as resp:
            sh = pd.read_csv(io.BytesIO(resp.read()))

        # Pair stereotype + anti-stereotype per nation entity
        paired = 0
        for nation, grp in sh.groupby("nation_entity"):
            stereos  = grp[grp["is_stereotype"] == "yes"]["sentence"].tolist()
            astereos = grp[grp["is_stereotype"] == "no"]["sentence"].tolist()
            for s, a in zip(stereos[:3], astereos[:3]):
                # SHADES nationality items map to warmth (trust/liking dimension)
                add(lang, "native", "warmth",
                    "nationality", nation,
                    s, a,
                    f"shades_{lang}", "")
                paired += 1
        print(f"  SHADES {lang.upper()}: {paired} pairs extracted")
    except Exception as exc:
        print(f"  SHADES {lang} failed: {exc}")


# ── 3. FR gender + profession placeholders ──────────────────────────────────
# CrowS-Pairs multilingual contains EN text; genuine FR items need authoring.
print("Adding FR placeholders …")
FR_PLACEHOLDERS = [
    # (dimension, target_group, target, sent_stereotype, sent_anti_stereotype)
    ("warmth",     "gender",     "femme/homme",
     "TODO: warmth stereotype sentence about a woman in French",
     "TODO: warmth anti-stereotype sentence about a woman in French"),
    ("competence", "gender",     "femme/homme",
     "TODO: competence stereotype sentence about a woman in French",
     "TODO: competence anti-stereotype sentence in French"),
    ("warmth",     "gender",     "femme/homme",
     "TODO: warmth stereotype about a man in French",
     "TODO: warmth anti-stereotype about a man in French"),
    ("competence", "profession", "ingénieure/infirmière",
     "TODO: competence stereotype about an engineer (FR)",
     "TODO: competence anti-stereotype (FR)"),
    ("warmth",     "profession", "infirmière/PDG",
     "TODO: warmth stereotype about a nurse (FR)",
     "TODO: warmth anti-stereotype (FR)"),
    ("competence", "profession", "PDG/femme de ménage",
     "TODO: competence stereotype about a CEO (FR)",
     "TODO: competence anti-stereotype (FR)"),
    ("warmth",     "nationality", "Français(e)",
     "TODO: warmth stereotype about French people, from a French perspective",
     "TODO: warmth anti-stereotype (FR)"),
    ("competence", "nationality", "Américain(e)",
     "TODO: competence stereotype about Americans, from a French perspective",
     "TODO: competence anti-stereotype (FR)"),
]
for dim, tg, target, s, a in FR_PLACEHOLDERS:
    add("fr", "native", dim, tg, target, s, a,
        "manual_fr", "NEEDS HUMAN AUTHORING in French")


# ── 4. BG placeholders ──────────────────────────────────────────────────────
print("Adding BG placeholders …")
BG_PLACEHOLDERS = [
    ("warmth",     "gender",      "жена/мъж",
     "TODO: топлота стереотип за жена на български",
     "TODO: топлота анти-стереотип за жена на български"),
    ("competence", "gender",      "жена/мъж",
     "TODO: компетентност стереотип за жена на български",
     "TODO: компетентност анти-стереотип на български"),
    ("warmth",     "gender",      "мъж/жена",
     "TODO: топлота стереотип за мъж на български",
     "TODO: топлота анти-стереотип за мъж на български"),
    ("competence", "profession",  "инженер/инженерка",
     "TODO: компетентност стереотип за инженер (BG)",
     "TODO: компетентност анти-стереотип (BG)"),
    ("warmth",     "profession",  "медицинска сестра",
     "TODO: топлота стереотип за медицинска сестра (BG)",
     "TODO: топлота анти-стереотип (BG)"),
    ("competence", "profession",  "директор/директорка",
     "TODO: компетентност стереотип за директор (BG)",
     "TODO: компетентност анти-стереотип (BG)"),
    ("warmth",     "nationality", "българин/българка",
     "TODO: топлота стереотип за българи",
     "TODO: топлота анти-стереотип (BG)"),
    ("warmth",     "nationality", "французин/французойка",
     "TODO: топлота стереотип за французи, от българска гледна точка",
     "TODO: топлота анти-стереотип (BG)"),
    ("competence", "nationality", "американец/американка",
     "TODO: компетентност стереотип за американци, от българска гледна точка",
     "TODO: компетентност анти-стереотип (BG)"),
    ("warmth",     "nationality", "англичанин/англичанка",
     "TODO: топлота стереотип за британци, от българска гледна точка",
     "TODO: топлота анти-стереотип (BG)"),
]
for dim, tg, target, s, a in BG_PLACEHOLDERS:
    add("bg", "native", dim, tg, target, s, a,
        "manual_bg", "NEEDS HUMAN AUTHORING in Bulgarian")


# ── Save ─────────────────────────────────────────────────────────────────────
out = DATA / "stimuli_seed.csv"
if out.exists():
    sys.exit(
        f"ERROR: {out} already exists.\n"
        "stimulus_builder.py seeds a new CSV from scratch and would overwrite\n"
        "human-validated data. Delete the file manually if you truly want to reseed."
    )

df = pd.DataFrame(rows, columns=COLS)
df.to_csv(out, index=False, encoding="utf-8-sig")

print(f"\n{'-'*50}")
print(f"Saved {len(df)} items -> {out.relative_to(ROOT)}")
print()
print(df.groupby(["language", "target_group"])["item_id"].count()
        .rename("count").to_string())
print()
needs_review = (df["dimension"] == "needs_review").sum()
needs_authoring = df["notes"].str.startswith("NEEDS").sum()
print(f"Validated      : {df['validated'].sum()} / {len(df)}")
print(f"Needs review   : {needs_review}  (dimension label missing)")
print(f"Needs authoring: {needs_authoring}  (placeholder rows)")
