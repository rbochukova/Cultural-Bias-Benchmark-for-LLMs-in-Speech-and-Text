"""
add_translated_items.py
~~~~~~~~~~~~~~~~~~~~~~~
Translates EN nationality (EN-N-*) and EN profession (EN-P-*) items into
French and Bulgarian, creating EN↔FR↔BG parallel triplets for both dimensions.

Parallel group IDs assigned:
  Nationality : PN-001 .. PN-N   (separate series from gender PG-*)
  Profession  : PP-001 .. PP-N

New item IDs continue from the current maximum in each series:
  FR-N-033+,  BG-N-005+
  FR-P-004+,  BG-P-004+

The EN source items are also updated with the new parallel_group_id so all
three legs of each triplet are linked.

Degenerate pairs (S == A after translation) are skipped and reported.

Resume-safe: checks which EN items already have a parallel_group_id set;
skips those groups. Run twice without issue.

Usage:
    OPENAI_API_KEY=sk-... python src/add_translated_items.py
    OPENAI_API_KEY=sk-... python src/add_translated_items.py --dim profession
    OPENAI_API_KEY=sk-... python src/add_translated_items.py --dry-run
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

TRANSLATION_SYSTEM = {
    "fr": (
        "You are a precise translator. Translate English sentences to French. "
        "Rules:\n"
        "- Preserve all proper nouns, nationality names, and profession names exactly "
        "as they are — do not substitute them with French equivalents unless the name "
        "itself needs to change (e.g. 'New Yorker' → 'New-Yorkais')\n"
        "- Preserve the third-person narrative style\n"
        "- Keep the same sentence length and register\n"
        "- Return ONLY the translated sentence, nothing else"
    ),
    "bg": (
        "You are a precise translator. Translate English sentences to Bulgarian. "
        "Rules:\n"
        "- Preserve all proper nouns, nationality names, and profession names "
        "(transliterate if needed, e.g. 'Mexican' → 'мексиканец')\n"
        "- Preserve the third-person narrative style\n"
        "- Keep the same sentence length and register\n"
        "- Use Cyrillic script\n"
        "- Return ONLY the translated sentence, nothing else"
    ),
}

LANG_PREFIX = {"fr": "FR", "bg": "BG"}
GROUP_PREFIX = {"nationality": "PN", "profession": "PP"}
ITEM_CODE    = {"nationality": "N",  "profession": "P"}


def _max_num(df: pd.DataFrame, prefix: str) -> int:
    ids = df[df["item_id"].str.startswith(prefix, na=False)]["item_id"]
    if ids.empty:
        return 0
    nums = ids.str.extract(re.escape(prefix) + r"(\d+)")[0].astype(int)
    return int(nums.max())


def _max_group_num(df: pd.DataFrame, group_prefix: str) -> int:
    ids = df[df["parallel_group_id"].str.startswith(group_prefix + "-", na=False)]["parallel_group_id"]
    if ids.empty:
        return 0
    nums = ids.str.extract(re.escape(group_prefix) + r"-(\d+)")[0].astype(int)
    return int(nums.max())


def _translate(client, text: str, lang: str) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": TRANSLATION_SYSTEM[lang]},
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


def _translate_batch(client, texts: list, lang: str) -> list:
    results = []
    for i, text in enumerate(texts, 1):
        t = _translate(client, text, lang)
        results.append(t)
        if i % 10 == 0 or i == len(texts):
            print(f"\r    {i}/{len(texts)}", end="", flush=True)
    print()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate EN-N/EN-P to FR and BG")
    parser.add_argument("--dim", default="both",
                        choices=["nationality", "profession", "both"],
                        help="Which target_group to translate (default: both)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show first 3 items per group without calling API")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit("ERROR: OPENAI_API_KEY not set.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key) if api_key else None

    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    dims = (["nationality", "profession"] if args.dim == "both"
            else [args.dim])

    total_added = 0

    for tg in dims:
        gp  = GROUP_PREFIX[tg]   # PN or PP
        ic  = ITEM_CODE[tg]      # N or P

        # EN source items for this target_group (not already linked)
        en_items = df[
            (df["language"] == "en") &
            (df["target_group"] == tg)
        ].copy()

        # Items that already have a parallel_group_id for this series
        already_linked = en_items[
            en_items["parallel_group_id"].str.startswith(gp + "-", na=False)
        ]["item_id"].tolist()
        to_process = en_items[~en_items["item_id"].isin(already_linked)]

        print(f"\n{'='*60}")
        print(f"Target group: {tg}  ({len(en_items)} EN items)")
        print(f"Already linked: {len(already_linked)}")
        print(f"To process: {len(to_process)}")

        if len(to_process) == 0:
            print("  Nothing to do.")
            continue

        if args.dry_run:
            print(f"\n=== DRY RUN — first 3 {tg} items ===")
            for _, row in to_process.head(3).iterrows():
                print(f"\n  [{row['item_id']}] dim={row['dimension']} target={row['target']}")
                print(f"    S: {str(row['sent_stereotype'])[:80]}")
                print(f"    A: {str(row['sent_anti_stereotype'])[:80]}")
            continue

        # Determine starting numbers
        next_group  = _max_group_num(df, gp) + 1
        next_fr_num = _max_num(df, f"FR-{ic}-") + 1
        next_bg_num = _max_num(df, f"BG-{ic}-") + 1

        # Translate for each target language
        new_rows = []
        en_updates = {}  # item_id → new parallel_group_id

        for lang in ["fr", "bg"]:
            lang_prefix = LANG_PREFIX[lang]
            next_num    = next_fr_num if lang == "fr" else next_bg_num

            print(f"\n  Translating {len(to_process)} stereo sentences → {lang.upper()} ...")
            stereo_texts = to_process["sent_stereotype"].astype(str).tolist()
            stereo_trans = _translate_batch(client, stereo_texts, lang)

            print(f"  Translating {len(to_process)} anti sentences → {lang.upper()} ...")
            anti_texts   = to_process["sent_anti_stereotype"].astype(str).tolist()
            anti_trans   = _translate_batch(client, anti_texts, lang)

            skipped = 0
            for i, (_, row) in enumerate(to_process.iterrows()):
                s_tr = stereo_trans[i].strip()
                a_tr = anti_trans[i].strip()

                if not s_tr or not a_tr:
                    skipped += 1
                    continue
                if s_tr == a_tr:
                    skipped += 1
                    continue

                group_num = next_group + (list(to_process.index).index(row.name))
                pgid      = f"{gp}-{group_num:03d}"

                new_rows.append({
                    "item_id":              f"{lang_prefix}-{ic}-{next_num:03d}",
                    "parallel_group_id":    pgid,
                    "language":             lang,
                    "origin":               "parallel",
                    "dimension":            row["dimension"],
                    "target_group":         tg,
                    "target":               row["target"],
                    "sent_stereotype":      s_tr,
                    "sent_anti_stereotype": a_tr,
                    "source":               f"en_{tg}_translated",
                    "validated":            True,
                    "notes":                f"Translated from {row['item_id']} via GPT-4o-mini",
                })
                en_updates[row["item_id"]] = pgid
                next_num += 1

            if lang == "fr":
                next_fr_num = next_num
            else:
                next_bg_num = next_num

            print(f"    Added: {len(new_rows) - skipped}  Skipped: {skipped}")

        if not new_rows:
            print("  No valid rows to add.")
            continue

        # Update EN items with their new parallel_group_id
        for en_iid, pgid in en_updates.items():
            df.loc[df["item_id"] == en_iid, "parallel_group_id"] = pgid
            df.loc[df["item_id"] == en_iid, "origin"] = "parallel"

        # Append new rows
        new_df   = pd.DataFrame(new_rows)
        df       = pd.concat([df, new_df], ignore_index=True)
        total_added += len(new_rows)

        print(f"\n  {tg}: added {len(new_rows)} translated items "
              f"({len(en_updates)} EN items now linked)")

    if not args.dry_run and total_added > 0:
        df.to_csv(CSV_PATH, index=False, encoding="utf-8")
        print(f"\n{'='*60}")
        print(f"Total new items added: {total_added}")
        print(f"Saved: {CSV_PATH.name}")

        updated = pd.read_csv(CSV_PATH, encoding="utf-8")
        print(f"Total items in CSV: {len(updated)}")
        print("\nParallel item counts by language x target_group:")
        par = updated[updated["origin"] == "parallel"]
        print(par.groupby(["language", "target_group"])["item_id"].count().to_string())


if __name__ == "__main__":
    main()
