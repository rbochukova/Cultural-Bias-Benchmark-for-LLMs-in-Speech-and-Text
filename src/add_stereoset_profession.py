"""
Ingests English profession items from StereoSet into stimuli_seed.csv as new EN source items.
Items where the two constructed sentences are identical after normalisation are skipped. Items already in the CSV are also skipped.
"""

import argparse
import json
import pathlib
import re
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

ROOT     = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "stimuli_seed.csv"

SS_URL = (
    "https://huggingface.co/datasets/McGill-NLP/stereoset/resolve/main/"
    "data/intrasentence.json"
)


def _fetch_stereoset() -> list[dict]:
    """Download StereoSet intrasentence data. Returns list of raw items."""
    try:
        with urllib.request.urlopen(SS_URL, timeout=30) as r:
            raw = json.loads(r.read().decode("utf-8"))
        if isinstance(raw, dict):
            data = raw.get("data", raw)
            if isinstance(data, dict):
                data = data.get("intrasentence", list(data.values())[0])
        else:
            data = raw
        print(f"  Downloaded {len(data)} items")
        return data
    except Exception as e:
        print(f"  JSON fetch failed ({e}), trying HuggingFace datasets library")
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
    """Extract (context, stereo_sent, anti_sent) for profession items."""
    LABEL_STEREOTYPE = 1
    LABEL_ANTI       = 0

    pairs = []
    for item in items:
        if item.get("bias_type") != "profession":
            continue
        context   = item.get("context", "")
        sentences = item.get("sentences", {})

        if isinstance(sentences, dict):
            sents       = sentences.get("sentence", [])
            gold_labels = sentences.get("gold_label", [])
        elif isinstance(sentences, list):
            sents       = [s.get("sentence", "") for s in sentences]
            gold_labels = [s.get("gold_label", -1) for s in sentences]
        else:
            continue

        stereo_sent = anti_sent = None
        for sent, label in zip(sents, gold_labels):
            if int(label) == LABEL_STEREOTYPE and stereo_sent is None:
                stereo_sent = sent
            elif int(label) == LABEL_ANTI and anti_sent is None:
                anti_sent = sent

        if stereo_sent and anti_sent and stereo_sent != anti_sent:
            pairs.append({
                "id":          item["id"],
                "target_raw":  item.get("target", ""),
                "context":     context,
                "stereo_sent": stereo_sent,
                "anti_sent":   anti_sent,
            })
    return pairs


def _max_num(df: pd.DataFrame, prefix: str) -> int:
    ids  = df[df["item_id"].str.startswith(prefix, na=False)]["item_id"]
    nums = ids.str.extract(re.escape(prefix) + r"(\d+)")[0].dropna().astype(int)
    return int(nums.max()) if not nums.empty else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest StereoSet profession items as EN source items"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch data and print first 5 items without saving")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Process at most N items (for testing)")
    args = parser.parse_args()

    raw_items = _fetch_stereoset()
    pairs     = _extract_pairs(raw_items)
    print(f"Profession pairs extracted: {len(pairs)}")

    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    existing_stereo = set(df["sent_stereotype"].astype(str).str.strip())
    pairs = [p for p in pairs if p["stereo_sent"].strip() not in existing_stereo]

    if args.limit:
        pairs = pairs[:args.limit]
        print(f"Limiting to {args.limit} items")

    if args.dry_run:
        for p in pairs[:5]:
            print(f"\n  S: {p['stereo_sent']}")
            print(f"  A: {p['anti_sent']}")
            print(f"  profession: {p['target_raw']}")
        return

    if not pairs:
        print("Nothing to add.")
        return

    next_num = _max_num(df, "EN-P-") + 1
    new_rows = []

    for p in pairs:
        new_rows.append({
            "item_id":              f"EN-P-{next_num:03d}",
            "parallel_group_id":    None,
            "language":             "en",
            "origin":               "native",
            "dimension":            "",
            "target_group":         "profession",
            "target":               "",
            "sent_stereotype":      p["stereo_sent"].strip(),
            "sent_anti_stereotype": p["anti_sent"].strip(),
            "source":               "stereoset_en",
            "validated":            False,
        })
        next_num += 1

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"Added {len(new_rows)} EN profession items")


if __name__ == "__main__":
    main()
