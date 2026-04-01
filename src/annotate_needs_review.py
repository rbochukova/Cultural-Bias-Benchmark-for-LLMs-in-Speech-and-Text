"""
annotate_needs_review.py
~~~~~~~~~~~~~~~~~~~~~~~~
Uses GPT-4o-mini to:
  1. Assign warmth/competence dimension to all `needs_review` items
  2. Flag out-of-scope items (race, religion, disability, sexual orientation,
     age) -> dimension='exclude'
  3. Clean up the `target` column

Items stay validated=False. The user reviews the output before validating.

Usage:
    OPENAI_API_KEY=sk-... python src/annotate_needs_review.py

Resume-safe: on restart the script re-reads the CSV and only processes items
still marked needs_review. Progress is written after every batch.

Set OPENAI_API_KEY in a .env file (see .env.example) or as an env variable.
"""

import json
import os
import pathlib
import re
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

ROOT     = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "stimuli_seed.csv"

BATCH_SIZE  = 20
VALID_DIMS  = {"warmth", "competence", "exclude"}
MAX_TARGET_LEN = 80

SYSTEM_PROMPT = """You are annotating a social-bias benchmark dataset.

Each item is a minimal sentence pair used to test whether an LLM reflects
stereotypes along the Stereotype Content Model (SCM) dimensions:

  WARMTH     -- friendliness, trustworthiness, emotional warmth, caring
                (e.g. "women are emotional", "nurses are kind")
  COMPETENCE -- capability, intelligence, skill, strength, leadership
                (e.g. "men are better drivers", "engineers are smart")

Your tasks for each item:
1. DIMENSION: assign "warmth", "competence", or "exclude"
   - "exclude" if the item tests race, religion, disability, sexual orientation,
     or age -- outside the scope of this benchmark.
   - "exclude" if the two sentences are not a valid stereotype pair
     (e.g. both sentences express negative stereotypes, or are unrelated).
2. TARGET: the specific social group(s) contrasted. Use "GroupA/GroupB" or
   "GroupA". Examples: "woman/man", "French/German", "engineer/nurse".
   For EuroGEST gender items in BG/FR use the local form already present
   (e.g. "zhena/mazh", "femme/homme"). Keep it short, under 60 characters.
3. EXCLUDE_REASON: if excluding, one short phrase (e.g. "race content").
   Empty string otherwise.

Return ONLY a JSON array, one object per item, in input order:
[
  {"item_id": "...", "dimension": "warmth|competence|exclude",
   "target": "...", "exclude_reason": "..."},
  ...
]
No markdown, no explanation, just the JSON array."""


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences if GPT wraps the JSON in them."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _validate_result(res: object) -> bool:
    """Return True only if res is a well-formed annotation dict."""
    if not isinstance(res, dict):
        return False
    if res.get("dimension") not in VALID_DIMS:
        return False
    target = str(res.get("target", "")).strip()
    if not target or len(target) > MAX_TARGET_LEN:
        return False
    # Reject if GPT echoed the example placeholder from the prompt
    if target in ("GroupA/GroupB", "GroupA"):
        return False
    return True


def _build_user_message(batch: list[dict]) -> str:
    parts = []
    for item in batch:
        notes = str(item.get("notes", "")).strip()
        source_hint = ""
        if notes.startswith("EN Source:"):
            source_hint = "\n   English source: " + notes[len("EN Source: "):]
        elif notes.startswith("Dimension ambiguous"):
            source_hint = "\n   English source: " + notes[notes.find(": ") + 2:]

        parts.append(
            f'item_id: {item["item_id"]}\n'
            f'   language: {item["language"]} | '
            f'target_group: {item["target_group"]} | '
            f'current_target: {item["target"]}{source_hint}\n'
            f'   stereotype:      {str(item["sent_stereotype"])[:200]}\n'
            f'   anti-stereotype: {str(item["sent_anti_stereotype"])[:200]}'
        )
    return "\n\n".join(parts)


def _annotate_batch(client: OpenAI, batch: list[dict]) -> list[dict]:
    """Call GPT-4o-mini for a batch. Returns list of validated result dicts."""
    user_msg = _build_user_message(batch)
    backoff = 2
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
            )
            raw     = _strip_fences(resp.choices[0].message.content)
            parsed  = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected JSON array, got {type(parsed)}")
            return [r for r in parsed if _validate_result(r)]
        except RateLimitError:
            print(f"\n    rate limited -- waiting {backoff}s ...", end=" ")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
        except Exception as exc:
            print(f"\n    attempt {attempt + 1}/4 failed: {exc}", end=" ")
            time.sleep(3)
    print("\n    FAILED after 4 attempts -- batch skipped")
    return []


def _apply_results(
    df: pd.DataFrame,
    batch: list[dict],
    results: list[dict],
) -> tuple[int, int, int]:
    """
    Write GPT results back into df in-place.
    Returns (labelled, excluded, errors) counts.
    """
    result_map = {r["item_id"]: r for r in results}
    labelled = excluded = errors = 0

    for item in batch:
        iid = item["item_id"]

        # Confirm the item is still needs_review (resume safety)
        current_dim = df.loc[df["item_id"] == iid, "dimension"]
        if current_dim.empty or current_dim.values[0] != "needs_review":
            continue

        # Confirm not accidentally validated
        current_val = df.loc[df["item_id"] == iid, "validated"]
        if current_val.empty or bool(current_val.values[0]):
            errors += 1
            print(f"\n    SKIP {iid}: validated=True but dimension=needs_review -- bug in data")
            continue

        res = result_map.get(iid)
        if not res:
            errors += 1
            continue

        dim    = res["dimension"]
        target = str(res["target"]).strip()
        reason = str(res.get("exclude_reason", "")).strip()

        # Update dimension
        df.loc[df["item_id"] == iid, "dimension"] = dim

        # Prepend exclude reason to existing notes without clobbering them
        if dim == "exclude" and reason:
            existing_note = str(df.loc[df["item_id"] == iid, "notes"].values[0])
            df.loc[df["item_id"] == iid, "notes"] = (
                f"EXCLUDED: {reason} | {existing_note}".strip(" |")
            )
            excluded += 1
        else:
            labelled += 1

        # Update target only if GPT returned something clean
        if target:
            df.loc[df["item_id"] == iid, "target"] = target

    return labelled, excluded, errors


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set. Add it to .env or the environment.")

    client = OpenAI(api_key=api_key)

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df["notes"]     = df["notes"].fillna("")
    df["validated"] = df["validated"].map(
        lambda x: True if str(x).strip().lower() in ("true", "1") else False
    )

    needs_review = df[df["dimension"] == "needs_review"]
    total = len(needs_review)
    print(f"Items to annotate: {total}")
    print(f"  EN: {(needs_review['language'] == 'en').sum()}")
    print(f"  FR: {(needs_review['language'] == 'fr').sum()}")
    print(f"  BG: {(needs_review['language'] == 'bg').sum()}")
    print()

    records       = needs_review.to_dict("records")
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    labelled_total = excluded_total = errors_total = 0

    for i in range(0, total, BATCH_SIZE):
        batch     = records[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        # Skip items already processed in a previous run
        batch = [
            item for item in batch
            if df.loc[df["item_id"] == item["item_id"], "dimension"].values[0] == "needs_review"
        ]
        if not batch:
            print(f"Batch {batch_num}/{total_batches} -- all already processed, skipping")
            continue

        print(
            f"Batch {batch_num}/{total_batches}  "
            f"(items {i + 1}-{min(i + BATCH_SIZE, total)}/{total}) ...",
            end=" ", flush=True,
        )

        results = _annotate_batch(client, batch)
        lab, exc, err = _apply_results(df, batch, results)
        labelled_total += lab
        excluded_total += exc
        errors_total   += err

        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

        remaining = (df["dimension"] == "needs_review").sum()
        print(
            f"done | labelled={labelled_total}  excluded={excluded_total}  "
            f"errors={errors_total}  remaining={remaining}"
        )
        time.sleep(1.0)

    print()
    print("=" * 55)
    print(f"Annotation complete")
    print(f"  Labelled (warmth/competence) : {labelled_total}")
    print(f"  Excluded (out-of-scope)      : {excluded_total}")
    print(f"  Errors / skipped             : {errors_total}")
    print(f"  Still needs_review           : {(df['dimension'] == 'needs_review').sum()}")
    print()
    print("Dimension distribution:")
    print(df["dimension"].value_counts().to_string())
    print()
    print(f"Validated items unchanged: {df['validated'].sum()}")


if __name__ == "__main__":
    main()
