"""
patch_add_notes.py
~~~~~~~~~~~~~~~~~~
One-time patch: adds a `notes` column to data/stimuli_seed.csv and
populates it with useful annotation context:

  - eurogest_bg / eurogest_fr : "EN Source: <original English sentence>"
                                (+ "Dimension ambiguous" prefix for needs_review)
  - crows_pairs_en validated  : "Label warmth or competence; confirm target"
  - crows_pairs_en expanded   : "Expanded batch -- dimension/target needs human review"
  - shades_*                  : "" (already fully labelled)
  - manual_bg / manual_fr     : "NEEDS HUMAN AUTHORING"

The script is safe to re-run: it always rebuilds notes from scratch using
the authoritative sources, never stacking annotations on top of each other.
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

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

# ── Start with empty notes column ────────────────────────────────────────────
df["notes"] = ""

# ── 1. EuroGEST BG — join English Source back in ─────────────────────────────
print("\nLoading EuroGEST BG for Source lookup ...")
try:
    from datasets import load_dataset

    bg = load_dataset(
        "utter-project/EuroGEST", split="Bulgarian",
        token=HF_TOKEN or None, trust_remote_code=True,
    ).to_pandas()

    # Build lookup: (sent_stereotype, sent_anti_stereotype) -> Source
    # The expander sets stereo=Masculine for competence/needs_review,
    # stereo=Feminine for warmth — so we index both orderings.
    bg_lookup: dict = {}
    for _, r in bg.dropna(subset=["Masculine", "Feminine", "Source"]).iterrows():
        m = str(r["Masculine"]).strip()
        f = str(r["Feminine"]).strip()
        src = str(r["Source"]).strip()
        bg_lookup[(m, f)] = src   # competence / needs_review key
        bg_lookup[(f, m)] = src   # warmth key

    mask = df["source"] == "eurogest_bg"
    hits = 0
    for idx in df[mask].index:
        key = (df.at[idx, "sent_stereotype"], df.at[idx, "sent_anti_stereotype"])
        src = bg_lookup.get(key, "")
        if src:
            prefix = ("Dimension ambiguous -- review: "
                      if df.at[idx, "dimension"] == "needs_review" else "EN Source: ")
            df.at[idx, "notes"] = prefix + src[:120]
            hits += 1

    print(f"  EuroGEST BG: {hits}/{mask.sum()} rows annotated with Source")
except Exception as exc:
    print(f"  EuroGEST BG failed: {exc}")


# ── 2. EuroGEST FR — join English Source back in ─────────────────────────────
print("\nLoading EuroGEST FR for Source lookup ...")
try:
    from datasets import load_dataset

    fr = load_dataset(
        "utter-project/EuroGEST", split="French",
        token=HF_TOKEN or None, trust_remote_code=True,
    ).to_pandas()

    fr_lookup: dict = {}
    for _, r in fr.dropna(subset=["Masculine", "Feminine", "Source"]).iterrows():
        m = str(r["Masculine"]).strip()
        f = str(r["Feminine"]).strip()
        src = str(r["Source"]).strip()
        fr_lookup[(m, f)] = src
        fr_lookup[(f, m)] = src

    mask = df["source"] == "eurogest_fr"
    hits = 0
    for idx in df[mask].index:
        key = (df.at[idx, "sent_stereotype"], df.at[idx, "sent_anti_stereotype"])
        src = fr_lookup.get(key, "")
        if src:
            prefix = ("Dimension ambiguous -- review: "
                      if df.at[idx, "dimension"] == "needs_review" else "EN Source: ")
            df.at[idx, "notes"] = prefix + src[:120]
            hits += 1

    print(f"  EuroGEST FR: {hits}/{mask.sum()} rows annotated with Source")
except Exception as exc:
    print(f"  EuroGEST FR failed: {exc}")


# ── 3. CrowS-Pairs EN ─────────────────────────────────────────────────────────
# Validated original items (from stimulus_builder.py, bias_type head() batches)
mask_val = (df["source"] == "crows_pairs_en") & (df["validated"] == True)
df.loc[mask_val, "notes"] = "Label warmth or competence; confirm target gender/profession/nationality"
print(f"\nCrowS-Pairs EN validated: {mask_val.sum()} rows noted")

# Expanded items (from stimulus_expander.py, full sets)
mask_exp = (df["source"] == "crows_pairs_en") & (df["validated"] == False)
df.loc[mask_exp, "notes"] = "Expanded batch -- dimension and target need human review"
print(f"CrowS-Pairs EN expanded:  {mask_exp.sum()} rows noted")


# ── 4. Manual placeholders ────────────────────────────────────────────────────
for src_val in ("manual_bg", "manual_fr"):
    mask = df["source"] == src_val
    df.loc[mask, "notes"] = "NEEDS HUMAN AUTHORING"
    print(f"Manual ({src_val}): {mask.sum()} rows noted")


# ── Save ──────────────────────────────────────────────────────────────────────
# Enforce column order with notes at the end
COLS = [
    "item_id", "parallel_group_id", "language", "origin",
    "dimension", "target_group", "target",
    "sent_stereotype", "sent_anti_stereotype",
    "source", "validated", "notes",
]
df = df[COLS]
df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

print(f"\nSaved {len(df)} rows with notes column -> {CSV_PATH.relative_to(ROOT)}")
print(f"  Rows with non-empty notes : {(df['notes'] != '').sum()}")
print(f"  Rows with empty notes     : {(df['notes'] == '').sum()}")
print()
print("Sample notes by source:")
for src in df["source"].unique():
    sample = df[df["source"] == src]["notes"].iloc[0]
    print(f"  [{src}] {sample[:80]}")
