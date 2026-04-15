"""
add_en_gender_parallels.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Completes the PG-NNN gender parallel series by adding English items for the
74 PG- groups that currently have only FR + BG legs.

French gender items use first-person voice with gendered morphology
(e.g. "Je suis passionnée" vs "Je suis passionné").  Since English lacks
grammatical gender agreement, the translation converts to third-person and
inserts she/he pronouns to preserve the male/female contrast.

Translation rules fed to GPT-4o-mini:
  - Translate French → English
  - Convert first-person ("Je...") to third-person
  - Feminine morphology (e.g. -ée, -ive, -ve, -nte, -ère endings) → she/her
  - Masculine morphology (default / -é, -if, -f, -nt, -er endings) → he/him
  - Preserve meaning and sentence length

New item IDs continue from the current EN-G-NNN maximum.
Resume-safe: skips PG- groups that already have an EN item.

Usage:
    OPENAI_API_KEY=sk-... python src/add_en_gender_parallels.py
    OPENAI_API_KEY=sk-... python src/add_en_gender_parallels.py --dry-run
"""

import argparse
import os
import pathlib
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT     = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "stimuli_seed.csv"

SYSTEM_PROMPT = (
    "You are a precise translator from French to English specialising in gender-bias research sentences.\n\n"
    "Translation rules:\n"
    "1. Convert first-person French ('Je suis...', 'J'étais...', 'J'ai...') to THIRD person English.\n"
    "2. Detect the grammatical gender of the subject from French morphology:\n"
    "   - Feminine markers: adjective endings -ée, -ive, -ve, -nte, -ère, -sse, -elle, past participles -ée, -ues, -ies.\n"
    "     Use 'she/her/hers' for the subject.\n"
    "   - Masculine markers (default or -é, -if, -f, -nt, -er, -eur endings).\n"
    "     Use 'he/him/his' for the subject.\n"
    "3. Preserve the original meaning, register, and approximate sentence length.\n"
    "4. Do NOT add explanations. Return ONLY the translated English sentence."
)


def _max_num(df: pd.DataFrame, prefix: str) -> int:
    ids = df[df["item_id"].str.startswith(prefix, na=False)]["item_id"]
    if ids.empty:
        return 0
    nums = ids.str.extract(re.escape(prefix) + r"(\d+)")[0].astype(int)
    return int(nums.max())


def _translate(client, text: str) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            if attempt == 2:
                print(f"\n    WARN: translation failed: {exc}")
                return ""
            time.sleep(2)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add EN items for PG- gender groups missing an English leg"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show first 5 items without calling the API")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit("ERROR: OPENAI_API_KEY not set.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key) if api_key else None

    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    # Identify PG- groups that have FR but no EN
    pg_fr = df[(df["parallel_group_id"].str.startswith("PG-", na=False)) & (df["language"] == "fr")]
    pg_en_groups = set(
        df[(df["parallel_group_id"].str.startswith("PG-", na=False)) & (df["language"] == "en")
        ]["parallel_group_id"]
    )
    to_translate = pg_fr[~pg_fr["parallel_group_id"].isin(pg_en_groups)].copy()

    print(f"PG- groups already with EN : {len(pg_en_groups)}")
    print(f"PG- groups needing EN      : {len(to_translate)}")

    if len(to_translate) == 0:
        print("Nothing to do.")
        return

    if args.dry_run:
        print(f"\n=== DRY RUN — first 5 items ===")
        for _, row in to_translate.head(5).iterrows():
            print(f"\n  [{row['parallel_group_id']}] {row['item_id']} dim={row['dimension']}")
            print(f"    FR S: {row['sent_stereotype']}")
            print(f"    FR A: {row['sent_anti_stereotype']}")
        return

    next_num = _max_num(df, "EN-G-") + 1
    new_rows = []
    skipped  = 0

    print(f"\nTranslating {len(to_translate)} FR items → EN ...")
    for i, (_, row) in enumerate(to_translate.iterrows(), 1):
        s_en = _translate(client, str(row["sent_stereotype"]).strip())
        a_en = _translate(client, str(row["sent_anti_stereotype"]).strip())

        if not s_en or not a_en or s_en == a_en:
            skipped += 1
        else:
            new_rows.append({
                "item_id":              f"EN-G-{next_num:03d}",
                "parallel_group_id":    row["parallel_group_id"],
                "language":             "en",
                "origin":               "parallel",
                "dimension":            row["dimension"],
                "target_group":         "gender",
                "target":               row["target"],
                "sent_stereotype":      s_en,
                "sent_anti_stereotype": a_en,
                "source":               f"fr_gender_translated",
                "validated":            True,
                "notes":                f"Translated from {row['item_id']} via GPT-4o-mini",
            })
            next_num += 1

        if i % 10 == 0 or i == len(to_translate):
            print(f"\r  {i}/{len(to_translate)}  added={len(new_rows)}  skipped={skipped}",
                  end="", flush=True)
        time.sleep(0.05)

    print()

    if not new_rows:
        print("No valid items produced.")
        return

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"Added {len(new_rows)} EN gender items (skipped {skipped})")
    print(f"Total items in CSV: {len(df)}")
    print(f"PG- groups now with EN: "
          f"{df[(df['parallel_group_id'].str.startswith('PG-', na=False)) & (df['language']=='en')]['parallel_group_id'].nunique()}")


if __name__ == "__main__":
    main()
