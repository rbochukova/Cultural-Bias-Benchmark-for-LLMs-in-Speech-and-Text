"""
add_en_roma_items.py
~~~~~~~~~~~~~~~~~~~~
Completes the ROM-NNN nationality parallel series by adding English items.

ROM-NNN already has FR items (Rom/Français context) and BG items
(Ром/Българин context).  This script creates EN counterparts using a
Roma/British framing — the UK has well-documented Roma/Traveller communities
and is the natural Anglophone cultural context.

Translation approach:
  - Take the FR ROM-NNN stereotype sentence as the semantic base
  - Translate to English
  - Replace references to French national group with "British"
  - Resulting target: "Roma/British"

New item IDs continue from the current EN-N-NNN maximum.
Resume-safe: skips ROM-NNN groups that already have an EN item.

Usage:
    OPENAI_API_KEY=sk-... python src/add_en_roma_items.py
    OPENAI_API_KEY=sk-... python src/add_en_roma_items.py --dry-run
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
    "You are a precise translator from French to English for social-psychology research.\n\n"
    "Translation rules:\n"
    "1. Translate the French sentence to natural English.\n"
    "2. The sentences describe stereotypes about Roma communities compared to a majority group.\n"
    "   - Replace any reference to French people (les Français, un Français, le Français, etc.) "
    "with the British equivalent (the British, a British person, the British person, etc.).\n"
    "   - Replace any reference to Roma (les Roms, un Rom, le Rom, etc.) "
    "with 'the Roma' / 'a Roma person' / 'the Roma person' as appropriate.\n"
    "3. Preserve the third-person narrative style.\n"
    "4. Preserve the sentence structure and register.\n"
    "5. Return ONLY the translated English sentence, nothing else."
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
        description="Add EN items for ROM- Roma parallel groups"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview first 5 items without calling the API")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit("ERROR: OPENAI_API_KEY not set.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key) if api_key else None

    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    # FR ROM- items (use as translation source)
    rom_fr = df[
        (df["parallel_group_id"].str.startswith("ROM-", na=False)) &
        (df["language"] == "fr")
    ].copy()

    # Skip groups that already have EN
    rom_en_groups = set(
        df[(df["parallel_group_id"].str.startswith("ROM-", na=False)) & (df["language"] == "en")
        ]["parallel_group_id"]
    )
    to_translate = rom_fr[~rom_fr["parallel_group_id"].isin(rom_en_groups)]

    print(f"ROM- groups with FR items     : {len(rom_fr)}")
    print(f"ROM- groups already with EN   : {len(rom_en_groups)}")
    print(f"ROM- groups needing EN        : {len(to_translate)}")

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

    next_num = _max_num(df, "EN-N-") + 1
    new_rows = []
    skipped  = 0

    print(f"\nTranslating {len(to_translate)} FR ROM items → EN (Roma/British) ...")
    for i, (_, row) in enumerate(to_translate.iterrows(), 1):
        s_en = _translate(client, str(row["sent_stereotype"]).strip())
        a_en = _translate(client, str(row["sent_anti_stereotype"]).strip())

        if not s_en or not a_en or s_en == a_en:
            skipped += 1
        else:
            new_rows.append({
                "item_id":              f"EN-N-{next_num:03d}",
                "parallel_group_id":    row["parallel_group_id"],
                "language":             "en",
                "origin":               "parallel",
                "dimension":            row["dimension"],
                "target_group":         "nationality",
                "target":               "Roma/British",
                "sent_stereotype":      s_en,
                "sent_anti_stereotype": a_en,
                "source":               "fr_roma_translated",
                "validated":            True,
                "notes":                f"Translated from {row['item_id']} via GPT-4o-mini; target adapted to Roma/British",
            })
            next_num += 1

        if i % 5 == 0 or i == len(to_translate):
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
    print(f"Added {len(new_rows)} EN Roma items (skipped {skipped})")
    print(f"Total items in CSV: {len(df)}")


if __name__ == "__main__":
    main()
