"""
stimulus_expander.py
~~~~~~~~~~~~~~~~~~~~
Expands data/stimuli_seed.csv with additional items from:

  1. CrowS-Pairs EN  — full sets (gender, profession, nationality), removing
                       the head() caps used in stimulus_builder.py
  2. EuroGEST BG     — 778 Masculine/Feminine gender pairs in Bulgarian
  3. EuroGEST FR     — 552 Masculine/Feminine gender pairs in French

Existing rows are preserved as-is (validated items are not touched).
New rows are appended with validated=False so they can be reviewed.

Dimension is inferred automatically from the English Source sentence
via keyword matching; ambiguous items are labelled needs_review.
"""

import io
import os
import pathlib
import urllib.request
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

ROOT     = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "stimuli_seed.csv"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ── Load existing set ────────────────────────────────────────────────────────
existing = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
existing["validated"] = existing["validated"].map(
    lambda x: True if str(x).strip().lower() in ("true", "1") else False
)
print(f"Existing items: {len(existing)}  (all preserved)")

# Build per-group counters so new IDs continue from where we left off
_counters: dict = {}
for iid in existing["item_id"]:
    # format: EN-G-042  →  lang=EN, grp=G, seq=42
    parts = iid.split("-")
    if len(parts) == 3:
        key = f"{parts[0].lower()}-{parts[1]}"
        _counters[key] = max(_counters.get(key, 0), int(parts[2]))

new_rows: list = []


def _next_id(lang: str, grp: str) -> str:
    key = f"{lang.lower()}-{grp}"
    _counters[key] = _counters.get(key, 0) + 1
    return f"{lang.upper()}-{grp}-{_counters[key]:03d}"


def add(lang, origin, dimension, target_group, target,
        sent_stereo, sent_anti, source, notes=""):
    grp = {"gender": "G", "nationality": "N", "profession": "P"}[target_group]
    iid = _next_id(lang, grp)
    new_rows.append({
        "item_id":              iid,
        "parallel_group_id":    iid[3:],
        "language":             lang,
        "origin":               origin,
        "dimension":            dimension,
        "target_group":         target_group,
        "target":               target,
        "sent_stereotype":      str(sent_stereo).strip(),
        "sent_anti_stereotype": str(sent_anti).strip(),
        "source":               source,
        "validated":            False,
        "notes":                notes,
    })


# ── Dimension classifier (keyword matching on English source) ────────────────
_COMPETENCE_KW = frozenset([
    "lead", "leadership", "leader", "skill", "skilled", "engineer",
    "engineering", "math", "mathematics", "logic", "logical", "intelligent",
    "intelligence", "smart", "professional", "career", "expert", "capable",
    "ability", "efficient", "independent", "decisive", "rational",
    "analytical", "problem", "solve", "fix", "repair", "build", "drive",
    "driving", "athletic", "strong", "strength", "achieve", "achievement",
    "success", "succeed", "work", "job", "business", "compete", "win",
    "technology", "science", "computer", "code", "coding", "brave",
    "bravery", "courageous", "courage", "confident", "confidence",
    "ambition", "ambitious", "assertive", "direct", "heroic", "decisive",
    "manage", "manage", "manager", "strategy", "strategic", "analyze",
])

_WARMTH_KW = frozenset([
    "emotion", "emotional", "cry", "crying", "feel", "feeling", "love",
    "care", "caring", "nurture", "nurturing", "empathy", "empathetic",
    "sensitive", "kind", "kindness", "friendly", "warm", "warmth",
    "support", "supportive", "help", "helpful", "family", "friend",
    "friendship", "relationship", "romantic", "gentle", "soft", "tender",
    "compassion", "compassionate", "trust", "trustworthy", "honest",
    "moral", "passionate", "affection", "affectionate", "social",
    "sociable", "generous", "charming", "cheerful", "comforting",
    "nurture", "patient", "patience", "polite", "cooperative",
])


def infer_dimension(source_en: str) -> tuple[str, str]:
    """
    Returns (dimension, stereotype_gender) where:
      dimension          : 'warmth' | 'competence' | 'needs_review'
      stereotype_gender  : 'masculine' | 'feminine'
        — for competence items, masculine sentence = stereotypical
        — for warmth items,     feminine sentence  = stereotypical
    """
    words = set(source_en.lower().split())
    c_score = len(words & _COMPETENCE_KW)
    w_score = len(words & _WARMTH_KW)

    if c_score > w_score:
        return "competence", "masculine"   # man=competent stereotype
    elif w_score > c_score:
        return "warmth", "feminine"        # woman=warm stereotype
    else:
        return "needs_review", "masculine"


# ── 1. CrowS-Pairs EN — full sets (no head() caps) ──────────────────────────
print("\nLoading CrowS-Pairs EN (full) ...")
CP_EN_BIAS = {
    0: "race", 1: "socioeconomic", 2: "gender",
    3: "disability", 4: "nationality", 5: "sexual-orientation",
    6: "physical-appearance", 7: "religion", 8: "age",
}

try:
    from datasets import load_dataset

    cp_en = load_dataset("crows_pairs", split="test",
                         trust_remote_code=True).to_pandas()

    # IDs already in the CSV  — skip those exact sentence pairs
    existing_stereos = set(existing["sent_stereotype"].str.strip())

    added = {"gender": 0, "profession": 0, "nationality": 0}
    for bt, tg, dim, target in [
        (2, "gender",      "needs_review", "woman/man"),
        (1, "profession",  "needs_review", ""),
        (4, "nationality", "needs_review", ""),
    ]:
        for _, r in cp_en[cp_en["bias_type"] == bt].iterrows():
            if r["sent_more"].strip() in existing_stereos:
                continue           # already in the set
            add("en", "native", dim, tg, target,
                r["sent_more"], r["sent_less"],
                "crows_pairs_en",
                "Expanded batch — dimension/target needs human review")
            existing_stereos.add(r["sent_more"].strip())
            added[tg] += 1

    print(f"  Added: gender={added['gender']}  "
          f"profession={added['profession']}  "
          f"nationality={added['nationality']}")
except Exception as exc:
    print(f"  CrowS-Pairs EN failed: {exc}")


# ── 2. EuroGEST BG — Masculine/Feminine gender pairs ────────────────────────
print("\nLoading EuroGEST BG ...")
try:
    from datasets import load_dataset

    bg = load_dataset(
        "utter-project/EuroGEST", split="Bulgarian",
        token=HF_TOKEN or None, trust_remote_code=True,
    ).to_pandas()

    usable = bg.dropna(subset=["Masculine", "Feminine"]).copy()
    added_bg = 0
    for _, r in usable.iterrows():
        dim, stereo_gender = infer_dimension(str(r["Source"]))
        if stereo_gender == "masculine":
            s, a = str(r["Masculine"]), str(r["Feminine"])
        else:
            s, a = str(r["Feminine"]), str(r["Masculine"])

        add("bg", "native", dim,
            "gender", "жена/мъж",
            s, a,
            "eurogest_bg",
            "" if dim != "needs_review" else "Dimension ambiguous — review Source: " + str(r["Source"])[:60])
        added_bg += 1

    print(f"  EuroGEST BG: {added_bg} gender pairs added "
          f"({usable.shape[0]} available)")
    dims = pd.Series([infer_dimension(str(r["Source"]))[0]
                      for _, r in usable.iterrows()]).value_counts()
    print(f"  Auto-labelled: {dims.to_dict()}")
except Exception as exc:
    print(f"  EuroGEST BG failed: {exc}")


# ── 3. EuroGEST FR — Masculine/Feminine gender pairs ────────────────────────
print("\nLoading EuroGEST FR ...")
try:
    from datasets import load_dataset

    fr = load_dataset(
        "utter-project/EuroGEST", split="French",
        token=HF_TOKEN or None, trust_remote_code=True,
    ).to_pandas()

    usable_fr = fr.dropna(subset=["Masculine", "Feminine"]).copy()
    added_fr = 0
    for _, r in usable_fr.iterrows():
        dim, stereo_gender = infer_dimension(str(r["Source"]))
        if stereo_gender == "masculine":
            s, a = str(r["Masculine"]), str(r["Feminine"])
        else:
            s, a = str(r["Feminine"]), str(r["Masculine"])

        add("fr", "native", dim,
            "gender", "femme/homme",
            s, a,
            "eurogest_fr",
            "" if dim != "needs_review" else "Dimension ambiguous — review Source: " + str(r["Source"])[:60])
        added_fr += 1

    print(f"  EuroGEST FR: {added_fr} gender pairs added "
          f"({usable_fr.shape[0]} available)")
    dims_fr = pd.Series([infer_dimension(str(r["Source"]))[0]
                         for _, r in usable_fr.iterrows()]).value_counts()
    print(f"  Auto-labelled: {dims_fr.to_dict()}")
except Exception as exc:
    print(f"  EuroGEST FR failed: {exc}")


# ── Merge and save ────────────────────────────────────────────────────────────
# Use explicit column list so notes is never silently dropped even if the
# existing CSV was saved without it (e.g. after user edits in Excel).
OUTPUT_COLS = [
    "item_id", "parallel_group_id", "language", "origin",
    "dimension", "target_group", "target",
    "sent_stereotype", "sent_anti_stereotype",
    "source", "validated", "notes",
]
# Ensure existing df has a notes column before concat
if "notes" not in existing.columns:
    existing["notes"] = ""

new_df  = pd.DataFrame(new_rows, columns=OUTPUT_COLS)
final   = pd.concat([existing[OUTPUT_COLS], new_df], ignore_index=True)

final.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

print(f"\n{'-'*55}")
print(f"Total items  : {len(final)}  "
      f"({len(existing)} existing + {len(new_df)} new)")
print()
print(final.groupby(["language", "target_group"])["item_id"].count()
      .rename("count").to_string())
print()
print(f"Validated    : {final['validated'].sum()} / {len(final)}")
print(f"Needs review : {(final['dimension'] == 'needs_review').sum()}")
print(f"Warmth       : {(final['dimension'] == 'warmth').sum()}")
print(f"Competence   : {(final['dimension'] == 'competence').sum()}")
