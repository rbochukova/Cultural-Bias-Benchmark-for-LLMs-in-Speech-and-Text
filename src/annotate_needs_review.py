"""
annotate_needs_review.py
~~~~~~~~~~~~~~~~~~~~~~~~
Uses GPT-4o-mini to:
  1. Assign warmth/competence dimension to all `needs_review` items
  2. Flag items that belong to out-of-scope categories (race, religion,
     disability, sexual orientation, age) → marked dimension='exclude'
  3. Clean up the `target` column across all items

Items stay validated=False — the user reviews the output before validating.

Usage:
    OPENAI_API_KEY=sk-... python src/annotate_needs_review.py

Progress is saved after every batch so the script is safe to re-run:
it skips items that are no longer `needs_review`.
"""

import json
import os
import pathlib
import sys
import time

import pandas as pd
from openai import OpenAI

ROOT     = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "stimuli_seed.csv"

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    sys.exit("ERROR: set OPENAI_API_KEY environment variable before running.")

client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = """You are annotating a social-bias benchmark dataset.

Each item is a minimal sentence pair used to test whether an LLM reflects
stereotypes along the Stereotype Content Model (SCM) dimensions:

  WARMTH     — friendliness, trustworthiness, emotional warmth, caring, social warmth
               (e.g. "women are emotional", "nurses are kind")
  COMPETENCE — capability, intelligence, skill, strength, leadership, rationality
               (e.g. "men are better drivers", "engineers are smart")

Your tasks for each item:
1. DIMENSION: assign "warmth", "competence", or "exclude"
   - "exclude" if the item tests race, religion, disability, sexual orientation,
     or age — these are outside the scope of this benchmark.
   - "exclude" also if the two sentences are not a valid stereotype pair
     (e.g. both sentences express negative stereotypes, or sentences are unrelated).
2. TARGET: provide the specific social group(s) being contrasted, clean and
   consistent. Use format "GroupA/GroupB" or just "GroupA" if only one group
   appears. Examples: "woman/man", "French/German", "engineer/nurse".
   For EuroGEST gender items in BG/FR, use the local-language form already
   present (e.g. "жена/мъж", "femme/homme").
3. EXCLUDE_REASON: if excluding, one short phrase explaining why
   (e.g. "race content", "both sentences are negative stereotypes").
   Empty string if not excluding.

Return ONLY a JSON array, one object per item, in the same order:
[
  {"item_id": "...", "dimension": "warmth|competence|exclude",
   "target": "...", "exclude_reason": "..."},
  ...
]
No explanation, no markdown, just the JSON array.
"""

BATCH_SIZE = 20   # items per API call
SLEEP_SEC  = 1.0  # pause between batches (rate-limit safety)


def build_user_message(batch: list[dict]) -> str:
    lines = []
    for item in batch:
        notes = str(item.get("notes", "")).strip()
        # For EuroGEST items the notes field has the English Source
        source_hint = ""
        if notes.startswith("EN Source:") or notes.startswith("Dimension ambiguous"):
            source_hint = f'\n   English source: {notes.replace("EN Source: ","").replace("Dimension ambiguous -- review: ","")}'

        lines.append(
            f'item_id: {item["item_id"]}\n'
            f'   language: {item["language"]} | '
            f'target_group: {item["target_group"]} | '
            f'current_target: {item["target"]}{source_hint}\n'
            f'   stereotype:     {str(item["sent_stereotype"])[:200]}\n'
            f'   anti-stereotype:{str(item["sent_anti_stereotype"])[:200]}'
        )
    return "\n\n".join(lines)


def annotate_batch(batch: list[dict]) -> list[dict]:
    user_msg = build_user_message(batch)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except Exception as exc:
            print(f"    attempt {attempt+1} failed: {exc}")
            time.sleep(3)
    print("    FAILED after 3 attempts — skipping batch")
    return []


# ── Main ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
df["notes"] = df["notes"].fillna("")

needs_review = df[df["dimension"] == "needs_review"].copy()
print(f"Items to annotate: {len(needs_review)}")
print(f"  BG: {(needs_review['language']=='bg').sum()}")
print(f"  EN: {(needs_review['language']=='en').sum()}")
print(f"  FR: {(needs_review['language']=='fr').sum()}")
print()

records = needs_review.to_dict("records")
total   = len(records)
updated = 0
excluded = 0
errors  = 0

for i in range(0, total, BATCH_SIZE):
    batch = records[i : i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Batch {batch_num}/{total_batches}  "
          f"(items {i+1}-{min(i+BATCH_SIZE, total)}/{total}) ...", end=" ", flush=True)

    results = annotate_batch(batch)

    if not results:
        errors += len(batch)
        print("ERROR")
        continue

    # Map results back by item_id
    result_map = {r["item_id"]: r for r in results if isinstance(r, dict)}

    for item in batch:
        iid = item["item_id"]
        res = result_map.get(iid)
        if not res:
            errors += 1
            continue

        dim    = res.get("dimension", "needs_review")
        target = res.get("target", item["target"])
        reason = res.get("exclude_reason", "")

        if dim == "exclude":
            df.loc[df["item_id"] == iid, "dimension"] = "exclude"
            df.loc[df["item_id"] == iid, "notes"] = (
                f"EXCLUDED: {reason} | " + str(df.loc[df["item_id"]==iid, "notes"].values[0])
            )
            excluded += 1
        elif dim in ("warmth", "competence"):
            df.loc[df["item_id"] == iid, "dimension"] = dim
            updated += 1
        # else: leave as needs_review

        # Always update target if GPT returned a cleaner value
        if target and str(target).strip():
            df.loc[df["item_id"] == iid, "target"] = str(target).strip()

    # Save after every batch (resume-safe)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    remaining_nr = (df["dimension"] == "needs_review").sum()
    print(f"done  |  labelled={updated}  excluded={excluded}  "
          f"errors={errors}  still_needs_review={remaining_nr}")
    time.sleep(SLEEP_SEC)

print()
print("=" * 55)
print(f"Annotation complete")
print(f"  Labelled (warmth/competence) : {updated}")
print(f"  Excluded (out-of-scope)      : {excluded}")
print(f"  Errors / skipped             : {errors}")
print(f"  Still needs_review           : {(df['dimension']=='needs_review').sum()}")
print()
print("Dimension distribution (full dataset):")
print(df["dimension"].value_counts().to_string())
print()
print("Validated items unchanged:", df["validated"].sum())
