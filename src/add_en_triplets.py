"""
add_en_triplets.py
~~~~~~~~~~~~~~~~~~
Generates English versions of the 87 FR↔BG parallel items, completing
FR↔BG↔EN triplets.

Method
──────
1. For each of the 87 parallel groups, take the French stereotype and
   anti-stereotype sentences.
2. Translate both to English via GPT-4o-mini, preserving:
   - First-person voice
   - The gendered form (masculine / feminine) that appears in the FR source
3. Assign:
   - item_id  : EN-G-NNN  (continuing from max existing EN-G-NNN)
   - language : en
   - origin   : parallel
   - parallel_group_id : same PG-NNN as the FR/BG pair
   - target   : man/woman  or  woman/man  (EN equivalent of homme/femme / femme/homme)
   - dimension : copied from the FR item (or from fidelity dim_agree column)
   - validated : True
4. Appends to stimuli_seed.csv (idempotent: skips groups already present).

Resume-safe: checks for existing EN items with PG-* parallel_group_ids.

Usage:
    OPENAI_API_KEY=sk-... python src/add_en_triplets.py
    OPENAI_API_KEY=sk-... python src/add_en_triplets.py --dry-run
"""

import argparse
import os
import pathlib
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT      = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH  = ROOT / "data" / "stimuli_seed.csv"
FIDELITY  = ROOT / "data" / "parallel_fidelity.csv"

# FR target → EN target
TARGET_MAP = {
    "homme/femme":  "man/woman",
    "femme/homme":  "woman/man",
}

TRANSLATION_SYSTEM = (
    "You are a precise translator. "
    "Translate French first-person sentences to English. "
    "Preserve:\n"
    "  - first-person voice (I am / I have / I think ...)\n"
    "  - the grammatical gender of the speaker "
    "(if the French uses a masculine form like 'fatigué', "
    "use 'tired' in a context that implies a male speaker; "
    "if feminine like 'fatiguée', imply a female speaker)\n"
    "  - the same level of formality and sentence length\n"
    "Return ONLY the translated sentence, nothing else."
)


def _translate_batch(client, texts: list[str]) -> list[str]:
    """Translate a list of FR sentences to EN using GPT-4o-mini, one by one."""
    results = []
    for text in texts:
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[
                        {"role": "system", "content": TRANSLATION_SYSTEM},
                        {"role": "user",   "content": text},
                    ],
                )
                results.append(resp.choices[0].message.content.strip())
                break
            except Exception as exc:
                if attempt == 2:
                    print(f"\n    WARN: translation failed for '{text[:40]}': {exc}")
                    results.append("")
                time.sleep(2)
    return results


def _next_en_g_num(df: pd.DataFrame) -> int:
    """Return the next available EN-G-NNN number."""
    import re
    en_g = df[df["item_id"].str.match(r"EN-G-\d+", na=False)]
    if en_g.empty:
        return 263
    nums = en_g["item_id"].str.extract(r"EN-G-(\d+)")[0].astype(int)
    return int(nums.max()) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EN parallel triplets")
    parser.add_argument("--dry-run", action="store_true",
                        help="Translate and print first 3 pairs without saving")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit("ERROR: OPENAI_API_KEY not set.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key) if api_key else None

    df       = pd.read_csv(CSV_PATH, encoding="utf-8")
    fid_df   = pd.read_csv(FIDELITY, encoding="utf-8")

    # Which PG groups already have an EN item?
    already_done = set(
        df[
            (df["language"] == "en") &
            df["parallel_group_id"].str.startswith("PG-", na=False)
        ]["parallel_group_id"]
    )
    print(f"EN triplets already in CSV : {len(already_done)}")

    # Collect FR items for each parallel group
    par_fr = df[
        (df["language"] == "fr") &
        df["parallel_group_id"].str.startswith("PG-", na=False)
    ].copy()

    to_process = par_fr[~par_fr["parallel_group_id"].isin(already_done)]
    print(f"Groups to translate        : {len(to_process)}")

    if len(to_process) == 0:
        print("Nothing to do. All EN triplets already present.")
        return

    if args.dry_run:
        sample = to_process.head(3)
        print("\n=== DRY RUN — first 3 groups ===")
        for _, row in sample.iterrows():
            print(f"\n[{row['parallel_group_id']}] FR item: {row['item_id']}")
            print(f"  Target FR: {row['target']} → EN: {TARGET_MAP.get(row['target'], row['target'])}")
            print(f"  S (FR): {row['sent_stereotype'][:80]}")
            print(f"  A (FR): {row['sent_anti_stereotype'][:80]}")
            if client:
                en_s = _translate_batch(client, [str(row["sent_stereotype"])])[0]
                en_a = _translate_batch(client, [str(row["sent_anti_stereotype"])])[0]
                print(f"  S (EN): {en_s[:80]}")
                print(f"  A (EN): {en_a[:80]}")
        return

    # Translate all stereotype sentences, then all anti-stereotype sentences
    print("\nTranslating stereotypical sentences ...")
    stereo_texts = to_process["sent_stereotype"].astype(str).tolist()
    en_stereo    = _translate_batch(client, stereo_texts)

    print("Translating anti-stereotypical sentences ...")
    anti_texts   = to_process["sent_anti_stereotype"].astype(str).tolist()
    en_anti      = _translate_batch(client, anti_texts)

    # Build new rows
    next_num = _next_en_g_num(df)
    new_rows = []
    for (_, row), s_en, a_en in zip(to_process.iterrows(), en_stereo, en_anti):
        if not s_en.strip() or not a_en.strip():
            print(f"  SKIP {row['parallel_group_id']}: empty translation")
            continue
        en_target = TARGET_MAP.get(str(row["target"]), "man/woman")
        new_rows.append({
            "item_id":             f"EN-G-{next_num:03d}",
            "parallel_group_id":   row["parallel_group_id"],
            "language":            "en",
            "origin":              "parallel",
            "dimension":           row["dimension"],
            "target_group":        "gender",
            "target":              en_target,
            "sent_stereotype":     s_en,
            "sent_anti_stereotype": a_en,
            "source":              "eurogest_fr_translated",
            "validated":           True,
            "notes":               f"Translated from {row['item_id']} via GPT-4o-mini",
        })
        next_num += 1

    if not new_rows:
        print("No new rows to add.")
        return

    # Filter out degenerate items where S and A are identical after translation.
    # This happens because English adjectives lack grammatical gender marking
    # (e.g., FR "passionné"/"passionnée" → EN "passionate"/"passionate").
    # Such items cannot function as forced-choice pairs.
    valid_rows  = [r for r in new_rows if r["sent_stereotype"].strip() != r["sent_anti_stereotype"].strip()]
    skipped     = len(new_rows) - len(valid_rows)
    if skipped:
        print(f"\nSkipped {skipped} degenerate EN items (S == A after translation).")
        print("This is expected for items where the FR stereotype/anti-stereotype")
        print("distinction is purely grammatical gender on adjectives (lost in EN).")

    if not valid_rows:
        print("No valid EN triplets to add (all were degenerate).")
        return

    new_df   = pd.DataFrame(valid_rows)
    combined = pd.concat([df, new_df], ignore_index=True)
    combined.to_csv(CSV_PATH, index=False, encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"Added {len(valid_rows)} valid EN items to stimuli_seed.csv")
    if valid_rows:
        print(f"New EN-G numbers: {valid_rows[0]['item_id']} — {valid_rows[-1]['item_id']}")
    print(f"  ({skipped} degenerate pairs skipped — pronoun/gender-marking not expressible in EN)")

    # Quick summary
    updated = pd.read_csv(CSV_PATH, encoding="utf-8")
    print(f"\nTotal items in CSV: {len(updated)}")
    print("EN parallel items:", len(updated[
        (updated["language"] == "en") &
        updated["parallel_group_id"].str.startswith("PG-", na=False)
    ]))


if __name__ == "__main__":
    main()
