"""
patch_add_notes.py
~~~~~~~~~~~~~~~~~~
One-time patch: adds/repairs the `notes` column in data/stimuli_seed.csv
by joining English Source sentences back from EuroGEST and adding context
hints for CrowS-Pairs and manual items.

Safe to re-run: only writes notes for rows where notes are currently empty,
so GPT-generated annotations from annotate_needs_review.py are preserved.

Sources updated:
  eurogest_bg / eurogest_fr  : "EN Source: <English sentence>"
                               "Dimension ambiguous -- review: ..." for needs_review
  crows_pairs_en validated   : annotation hint
  crows_pairs_en expanded    : annotation hint
  manual_bg / manual_fr      : "NEEDS HUMAN AUTHORING"
  shades_*                   : left empty (fully labelled, no context needed)
"""

import os
import pathlib
import sys

import pandas as pd

from validate_csv import load_validated, validate, CSV_PATH

ROOT     = pathlib.Path(__file__).resolve().parent.parent
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def _build_eurogest_lookup(language: str) -> dict:
    from datasets import load_dataset

    split_name = {"bg": "Bulgarian", "fr": "French"}[language]
    ds = load_dataset(
        "utter-project/EuroGEST", split=split_name,
        token=HF_TOKEN or None, trust_remote_code=True,
    ).to_pandas()

    lookup: dict = {}
    for _, r in ds.dropna(subset=["Masculine", "Feminine", "Source"]).iterrows():
        m   = str(r["Masculine"]).strip()
        f   = str(r["Feminine"]).strip()
        src = str(r["Source"]).strip()
        lookup[(m, f)] = src
        lookup[(f, m)] = src
    return lookup


def main() -> None:
    df = load_validated()
    print(f"Loaded {len(df)} rows.")

    updated = 0

    # ── EuroGEST BG + FR ─────────────────────────────────────────────────────
    for lang in ("bg", "fr"):
        source_val = f"eurogest_{lang}"
        print(f"\nLoading EuroGEST {lang.upper()} for Source lookup ...")
        try:
            lookup = _build_eurogest_lookup(lang)
            mask   = df["source"] == source_val
            hits   = 0
            for idx in df[mask].index:
                # Only fill if note is currently empty
                if str(df.at[idx, "notes"]).strip():
                    continue
                key = (df.at[idx, "sent_stereotype"], df.at[idx, "sent_anti_stereotype"])
                src = lookup.get(key, "")
                if not src:
                    continue
                prefix = (
                    "Dimension ambiguous -- review: "
                    if df.at[idx, "dimension"] == "needs_review"
                    else "EN Source: "
                )
                df.at[idx, "notes"] = prefix + src[:120]
                hits    += 1
                updated += 1
            print(f"  {source_val}: {hits} rows annotated with Source")
        except Exception as exc:
            print(f"  EuroGEST {lang.upper()} failed: {exc}", file=sys.stderr)

    # ── CrowS-Pairs EN ────────────────────────────────────────────────────────
    for mask, note in [
        (
            (df["source"] == "crows_pairs_en") & df["validated"].astype(bool),
            "Label warmth or competence; confirm target gender/profession/nationality",
        ),
        (
            (df["source"] == "crows_pairs_en") & ~df["validated"].astype(bool),
            "Expanded batch -- dimension and target need human review",
        ),
    ]:
        empty_mask = mask & (df["notes"].str.strip() == "")
        df.loc[empty_mask, "notes"] = note
        n = empty_mask.sum()
        updated += n
        print(f"CrowS-Pairs EN: {n} empty notes filled")

    # ── Manual placeholders ───────────────────────────────────────────────────
    for src_val in ("manual_bg", "manual_fr"):
        empty_mask = (df["source"] == src_val) & (df["notes"].str.strip() == "")
        df.loc[empty_mask, "notes"] = "NEEDS HUMAN AUTHORING"
        n = empty_mask.sum()
        updated += n
        print(f"Manual ({src_val}): {n} empty notes filled")

    validate(df, str(CSV_PATH))
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"\nDone. {updated} notes added/updated (existing non-empty notes preserved).")
    print(f"Rows with non-empty notes : {(df['notes'].str.strip() != '').sum()}")
    print(f"Rows with empty notes     : {(df['notes'].str.strip() == '').sum()}")


if __name__ == "__main__":
    main()
