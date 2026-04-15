"""
add_stereoset_profession.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ingests English profession items from StereoSet (Nadeem et al., 2021)
into stimuli_seed.csv as new EN source items.

StereoSet intrasentence format: each item has a context sentence with a BLANK
plus three continuations (stereotype, anti-stereotype, unrelated). The minimal
pair is constructed by substituting the BLANK with the stereotype or
anti-stereotype fill-in phrase, yielding two full sentences.

Because StereoSet fill-ins are single words or short phrases (not full
sentences), GPT-4o-mini is used to:
  1. Construct natural full sentences from context + fill-in
  2. Assign SCM dimension (warmth or competence)
  3. Assign a readable target label (e.g. "engineer/artist")

Items where the two constructed sentences are identical after normalisation
are skipped (degenerate pairs). Items already in the CSV (by sent_stereotype
match) are also skipped.

After ingestion, run add_translated_items.py --dim profession to generate
FR + BG translations.

Usage:
    OPENAI_API_KEY=sk-... python src/add_stereoset_profession.py
    OPENAI_API_KEY=sk-... python src/add_stereoset_profession.py --dry-run
    OPENAI_API_KEY=sk-... python src/add_stereoset_profession.py --limit 50
"""

import argparse
import io
import json
import os
import pathlib
import re
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT     = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "stimuli_seed.csv"

SS_URL = (
    "https://huggingface.co/datasets/McGill-NLP/stereoset/resolve/main/"
    "data/intrasentence.json"
)
# Fallback: parquet
SS_PARQUET = (
    "https://huggingface.co/datasets/McGill-NLP/stereoset/resolve/main/"
    "data/intrasentence-00000-of-00001.parquet"
)

CONSTRUCT_SYSTEM = (
    "You are a social-psychology researcher preparing minimal-pair sentences for a bias study.\n\n"
    "You receive a context sentence containing the word BLANK and two fill-in phrases: "
    "one stereotypical and one anti-stereotypical.\n\n"
    "Your task:\n"
    "1. Replace BLANK with each fill-in phrase to produce two natural English sentences.\n"
    "   If BLANK is mid-sentence, integrate the phrase smoothly (adjust articles if needed).\n"
    "2. Assign the SCM dimension:\n"
    "   'competence' = intelligence, skill, capability, productivity, education, status\n"
    "   'warmth'     = friendliness, trustworthiness, social threat, morality, emotionality\n"
    "3. Extract a short target label (e.g. 'engineer/artist', 'nurse/doctor').\n"
    "   The stereotyped profession comes first.\n\n"
    "Return ONLY valid JSON:\n"
    '{"sent_stereotype": "...", "sent_anti_stereotype": "...", '
    '"dimension": "warmth|competence", "target": "profession1/profession2"}'
)


def _fetch_stereoset() -> list[dict]:
    """Download StereoSet intrasentence data. Returns list of raw items."""
    print("Fetching StereoSet intrasentence from HuggingFace ...", flush=True)
    try:
        with urllib.request.urlopen(SS_URL, timeout=30) as r:
            raw = json.loads(r.read().decode("utf-8"))
        # Format: {"data": {"intrasentence": [...]}}  or just list
        if isinstance(raw, dict):
            data = raw.get("data", raw)
            if isinstance(data, dict):
                data = data.get("intrasentence", list(data.values())[0])
        else:
            data = raw
        print(f"  Downloaded {len(data)} items")
        return data
    except Exception as e:
        print(f"  JSON fetch failed ({e}), trying HuggingFace datasets library ...")
        try:
            from datasets import load_dataset
            ds = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation",
                              trust_remote_code=True)
            items = []
            for row in ds:
                items.append({
                    "id":        row["id"],
                    "target":    row["target"],
                    "bias_type": row["bias_type"],
                    "context":   row["context"],
                    "sentences": row["sentences"],
                })
            print(f"  Loaded {len(items)} items via datasets library")
            return items
        except Exception as e2:
            sys.exit(f"Could not load StereoSet: {e2}")


def _extract_pairs(items: list[dict]) -> list[dict]:
    """Extract (context, stereo_fill, anti_fill) for profession items."""
    LABEL_STEREOTYPE     = 1
    LABEL_ANTI           = 0

    pairs = []
    for item in items:
        if item.get("bias_type") != "profession":
            continue
        context   = item.get("context", "")
        sentences = item.get("sentences", {})

        # sentences may be dict with lists or list of dicts
        if isinstance(sentences, dict):
            sents      = sentences.get("sentence", [])
            gold_labels = sentences.get("gold_label", [])
        elif isinstance(sentences, list):
            sents       = [s.get("sentence", "") for s in sentences]
            gold_labels = [s.get("gold_label", -1) for s in sentences]
        else:
            continue

        stereo_fill = anti_fill = None
        for sent, label in zip(sents, gold_labels):
            if int(label) == LABEL_STEREOTYPE and stereo_fill is None:
                stereo_fill = sent
            elif int(label) == LABEL_ANTI and anti_fill is None:
                anti_fill = sent

        if stereo_fill and anti_fill and stereo_fill != anti_fill:
            pairs.append({
                "id":          item["id"],
                "target_raw":  item.get("target", ""),
                "context":     context,
                "stereo_fill": stereo_fill,
                "anti_fill":   anti_fill,
            })
    return pairs


def _classify(client, context: str, stereo_fill: str, anti_fill: str,
              target_raw: str) -> dict | None:
    prompt = (
        f"Context (with BLANK): {context}\n"
        f"Stereotypical fill-in: {stereo_fill}\n"
        f"Anti-stereotypical fill-in: {anti_fill}\n"
        f"Profession group: {target_raw}"
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": CONSTRUCT_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )
            data = json.loads(resp.choices[0].message.content)
            s  = data.get("sent_stereotype",      "").strip()
            a  = data.get("sent_anti_stereotype",  "").strip()
            d  = data.get("dimension",             "").strip().lower()
            t  = data.get("target",                "").strip()
            if s and a and s != a and d in ("warmth", "competence") and t:
                return {"sent_stereotype": s, "sent_anti_stereotype": a,
                        "dimension": d, "target": t}
            return None
        except Exception as exc:
            if attempt == 2:
                print(f"\n    WARN: classify failed: {exc}")
                return None
            time.sleep(2)
    return None


def _max_num(df: pd.DataFrame, prefix: str) -> int:
    ids  = df[df["item_id"].str.startswith(prefix, na=False)]["item_id"]
    nums = ids.str.extract(re.escape(prefix) + r"(\d+)")[0].dropna().astype(int)
    return int(nums.max()) if not nums.empty else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest StereoSet profession items as EN source items"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit",   type=int, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit("ERROR: OPENAI_API_KEY not set.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key) if api_key else None

    raw_items = _fetch_stereoset()
    pairs     = _extract_pairs(raw_items)
    print(f"Profession pairs extracted: {len(pairs)}")

    # Load existing CSV and deduplicate by sent_stereotype
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    existing_stereo = set(df["sent_stereotype"].astype(str).str.strip())

    # Also deduplicate by context to avoid near-duplicates
    existing_contexts = set()
    if "notes" in df.columns:
        notes = df["notes"].dropna().astype(str)
        existing_contexts = set(notes[notes.str.startswith("StereoSet:")].str.replace("StereoSet:", "").str.strip())

    pairs = [p for p in pairs
             if p["context"].strip() not in existing_contexts]
    print(f"After removing already-ingested contexts: {len(pairs)}")

    if args.limit:
        pairs = pairs[:args.limit]
        print(f"Limiting to {args.limit} items")

    if args.dry_run:
        print(f"\n=== DRY RUN — first 5 pairs ===")
        for p in pairs[:5]:
            print(f"\n  context   : {p['context']}")
            print(f"  stereo    : {p['stereo_fill']}")
            print(f"  anti      : {p['anti_fill']}")
            print(f"  profession: {p['target_raw']}")
        return

    if len(pairs) == 0:
        print("Nothing to add.")
        return

    next_num = _max_num(df, "EN-P-") + 1
    new_rows = []
    skipped  = 0

    print(f"\nConstructing + classifying {len(pairs)} StereoSet profession pairs ...")
    for i, p in enumerate(pairs, 1):
        meta = _classify(client, p["context"], p["stereo_fill"],
                         p["anti_fill"], p["target_raw"])
        if meta is None:
            skipped += 1
        elif meta["sent_stereotype"].strip() in existing_stereo:
            skipped += 1
        else:
            new_rows.append({
                "item_id":              f"EN-P-{next_num:03d}",
                "parallel_group_id":    None,
                "language":             "en",
                "origin":               "native",
                "dimension":            meta["dimension"],
                "target_group":         "profession",
                "target":               meta["target"],
                "sent_stereotype":      meta["sent_stereotype"],
                "sent_anti_stereotype": meta["sent_anti_stereotype"],
                "source":               "stereoset_en",
                "validated":            True,
                "notes":                f"StereoSet:{p['context'][:60]}",
            })
            existing_stereo.add(meta["sent_stereotype"].strip())
            next_num += 1

        if i % 10 == 0 or i == len(pairs):
            print(f"\r  {i}/{len(pairs)}  added={len(new_rows)}  skipped={skipped}",
                  end="", flush=True)
        time.sleep(0.05)

    print()

    if not new_rows:
        print("No valid items produced.")
        return

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"Added {len(new_rows)} EN profession items from StereoSet (skipped {skipped})")
    print(f"Total items in CSV: {len(df)}")
    print(f"\nNext: python src/add_translated_items.py --dim profession")


if __name__ == "__main__":
    main()
