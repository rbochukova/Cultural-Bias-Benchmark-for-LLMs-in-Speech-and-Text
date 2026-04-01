"""
stimulus_expander.py
~~~~~~~~~~~~~~~~~~~~
Expands data/stimuli_seed.csv with additional items from:

  1. CrowS-Pairs EN  -- full sets (gender, profession, nationality)
  2. EuroGEST BG     -- 778 Masculine/Feminine gender pairs in Bulgarian
  3. EuroGEST FR     -- 552 Masculine/Feminine gender pairs in French

Existing rows are preserved as-is. New rows are appended with validated=False.
Dimension is inferred from the English Source sentence via keyword matching;
ambiguous items are labelled needs_review.
"""

import os
import pathlib
import re
import sys

import pandas as pd

from validate_csv import load_validated, validate, CSV_PATH

ROOT     = pathlib.Path(__file__).resolve().parent.parent
HF_TOKEN = os.environ.get("HF_TOKEN", "")

OUTPUT_COLS = [
    "item_id", "parallel_group_id", "language", "origin",
    "dimension", "target_group", "target",
    "sent_stereotype", "sent_anti_stereotype",
    "source", "validated", "notes",
]

_GROUP_LETTER = {"gender": "G", "nationality": "N", "profession": "P"}

# ── Dimension keyword sets ────────────────────────────────────────────────────
_COMPETENCE_KW = frozenset([
    "lead", "leadership", "leader", "skill", "skilled", "engineer",
    "engineering", "math", "mathematics", "logic", "logical", "intelligent",
    "intelligence", "smart", "professional", "career", "expert", "capable",
    "ability", "efficient", "independent", "rational", "analytical",
    "problem", "solve", "fix", "repair", "build", "drive", "driving",
    "athletic", "strong", "strength", "achieve", "achievement", "success",
    "succeed", "work", "job", "business", "compete", "win", "technology",
    "science", "computer", "code", "coding", "brave", "bravery",
    "courageous", "courage", "confident", "confidence", "ambition",
    "ambitious", "assertive", "heroic", "decisive", "manager", "strategy",
    "strategic", "analyze",
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
    "patient", "patience", "polite", "cooperative",
])


def infer_dimension(source_en: str) -> tuple[str, str]:
    """
    Returns (dimension, stereotype_gender):
      dimension        : 'warmth' | 'competence' | 'needs_review'
      stereotype_gender: 'masculine' | 'feminine'
        competence -> masculine sentence is stereotypical
        warmth     -> feminine sentence is stereotypical
    """
    # Strip punctuation before tokenising so "skilled," matches "skilled"
    tokens = set(re.sub(r"[^\w\s]", " ", source_en.lower()).split())
    c_score = len(tokens & _COMPETENCE_KW)
    w_score = len(tokens & _WARMTH_KW)

    if c_score > w_score:
        return "competence", "masculine"
    elif w_score > c_score:
        return "warmth", "feminine"
    else:
        return "needs_review", "masculine"


class _Expander:
    """Encapsulates the mutable ID counter and row accumulator."""

    def __init__(self, existing: pd.DataFrame) -> None:
        self._counters: dict[str, int] = {}
        self.new_rows:  list[dict]     = []

        for iid in existing["item_id"]:
            parts = str(iid).split("-")
            if len(parts) != 3:
                continue
            try:
                seq = int(parts[2])
            except ValueError:
                print(f"WARNING: malformed item_id '{iid}' -- skipped in counter init",
                      file=sys.stderr)
                continue
            key = f"{parts[0].lower()}-{parts[1]}"
            self._counters[key] = max(self._counters.get(key, 0), seq)

    def _next_id(self, lang: str, grp: str) -> str:
        key = f"{lang.lower()}-{grp}"
        self._counters[key] = self._counters.get(key, 0) + 1
        return f"{lang.upper()}-{grp}-{self._counters[key]:03d}"

    def add(
        self, lang: str, origin: str, dimension: str,
        target_group: str, target: str,
        sent_stereo: str, sent_anti: str,
        source: str, notes: str = "",
    ) -> None:
        grp = _GROUP_LETTER[target_group]
        iid = self._next_id(lang, grp)
        self.new_rows.append({
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


def _load_crows_pairs_en(expander: _Expander, existing_stereos: set) -> None:
    from datasets import load_dataset

    cp = load_dataset("crows_pairs", split="test", trust_remote_code=True).to_pandas()
    added = {"gender": 0, "profession": 0, "nationality": 0}

    for bias_type, target_group, target in [
        (2, "gender",      "woman/man"),
        (1, "profession",  ""),
        (4, "nationality", ""),
    ]:
        for _, r in cp[cp["bias_type"] == bias_type].iterrows():
            stereo = r["sent_more"].strip()
            if stereo in existing_stereos:
                continue
            expander.add(
                "en", "native", "needs_review", target_group, target,
                stereo, r["sent_less"],
                "crows_pairs_en",
                "Expanded batch -- dimension/target needs human review",
            )
            existing_stereos.add(stereo)
            added[target_group] += 1

    print(f"  Added: gender={added['gender']}  "
          f"profession={added['profession']}  nationality={added['nationality']}")


def _load_eurogest(expander: _Expander, language: str) -> None:
    """Load one EuroGEST language split and add gender pairs."""
    from datasets import load_dataset

    split_name = {"bg": "Bulgarian", "fr": "French"}[language]
    default_target = {"bg": "zhena/mazh", "fr": "femme/homme"}[language]

    ds = load_dataset(
        "utter-project/EuroGEST", split=split_name,
        token=HF_TOKEN or None, trust_remote_code=True,
    ).to_pandas()

    usable = ds.dropna(subset=["Masculine", "Feminine"]).copy()
    added = 0

    for _, r in usable.iterrows():
        dim, stereo_gender = infer_dimension(str(r.get("Source", "")))
        if stereo_gender == "masculine":
            s, a = str(r["Masculine"]), str(r["Feminine"])
        else:
            s, a = str(r["Feminine"]), str(r["Masculine"])

        source_text = str(r.get("Source", ""))[:120]
        notes = (
            f"Dimension ambiguous -- review: {source_text}"
            if dim == "needs_review"
            else f"EN Source: {source_text}"
        )

        expander.add(
            language, "native", dim,
            "gender", default_target,
            s, a,
            f"eurogest_{language}",
            notes,
        )
        added += 1

    dims = pd.Series(
        [infer_dimension(str(r.get("Source", "")))[0] for _, r in usable.iterrows()]
    ).value_counts()
    print(f"  EuroGEST {language.upper()}: {added} pairs added. "
          f"Auto-labelled: {dims.to_dict()}")


def main() -> None:
    existing = load_validated()
    print(f"Existing items: {len(existing)}  (all preserved)")

    expander = _Expander(existing)
    existing_stereos = set(existing["sent_stereotype"].str.strip())

    print("\nLoading CrowS-Pairs EN (full) ...")
    try:
        _load_crows_pairs_en(expander, existing_stereos)
    except Exception as exc:
        print(f"  CrowS-Pairs EN failed: {exc}", file=sys.stderr)

    for lang in ("bg", "fr"):
        print(f"\nLoading EuroGEST {lang.upper()} ...")
        try:
            _load_eurogest(expander, lang)
        except Exception as exc:
            print(f"  EuroGEST {lang.upper()} failed: {exc}", file=sys.stderr)

    new_df = pd.DataFrame(expander.new_rows, columns=OUTPUT_COLS)
    final  = pd.concat([existing[OUTPUT_COLS], new_df], ignore_index=True)

    validate(final, str(CSV_PATH))
    final.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"\n{'-' * 55}")
    print(f"Total: {len(final)}  ({len(existing)} existing + {len(new_df)} new)")
    print()
    print(final.groupby(["language", "target_group"])["item_id"].count()
          .rename("count").to_string())
    print()
    print(f"Validated    : {final['validated'].sum()} / {len(final)}")
    print(f"Needs review : {(final['dimension'] == 'needs_review').sum()}")
    print(f"Warmth       : {(final['dimension'] == 'warmth').sum()}")
    print(f"Competence   : {(final['dimension'] == 'competence').sum()}")


if __name__ == "__main__":
    main()
