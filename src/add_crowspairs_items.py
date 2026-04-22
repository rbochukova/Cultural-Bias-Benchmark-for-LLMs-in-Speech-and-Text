"""
Ingests English nationality items from the CrowS-Pairs dataset into stimuli_seed.csv.
Items where stereotype == anti-stereotype after normalisation are skipped. Duplicate sentences are also skipped.
dimension and target are left blank for manual annotation after ingestion.
"""

import argparse
import io
import pathlib
import re
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

ROOT     = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "stimuli_seed.csv"

CROWSPAIRS_URL = (
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/"
    "crows_pairs_anonymized.csv"
)


def _fetch_crowspairs() -> pd.DataFrame:
    print("Fetching CrowS-Pairs from GitHub", flush=True)
    with urllib.request.urlopen(CROWSPAIRS_URL, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    print(f"  Downloaded {len(df)} items", flush=True)
    return df


def _max_num(df: pd.DataFrame, prefix: str) -> int:
    ids = df[df["item_id"].str.startswith(prefix, na=False)]["item_id"]
    if ids.empty:
        return 0
    nums = ids.str.extract(re.escape(prefix) + r"(\d+)")[0].astype(int)
    return int(nums.max())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest CrowS-Pairs nationality items as EN source items"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch data and print first 5 items without saving")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N items (for testing)")
    args = parser.parse_args()

    cp  = _fetch_crowspairs()
    nat = cp[cp["bias_type"] == "nationality"].copy()
    print(f"Nationality items in CrowS-Pairs: {len(nat)}")

    nat["sent_stereotype"] = nat.apply(
        lambda r: r["sent_more"] if r["stereo_antistereo"] == "stereo" else r["sent_less"],
        axis=1,
    )
    nat["sent_anti_stereotype"] = nat.apply(
        lambda r: r["sent_less"] if r["stereo_antistereo"] == "stereo" else r["sent_more"],
        axis=1,
    )

    nat = nat[nat["sent_stereotype"] != nat["sent_anti_stereotype"]].copy()

    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    existing_stereo = set(df["sent_stereotype"].astype(str).str.strip())
    nat = nat[~nat["sent_stereotype"].str.strip().isin(existing_stereo)].reset_index(drop=True)

    if args.limit:
        nat = nat.head(args.limit)
        print(f"Limiting to {args.limit} items")

    if args.dry_run:
        for _, row in nat.head(5).iterrows():
            print(f"\n  S: {row['sent_stereotype']}")
            print(f"  A: {row['sent_anti_stereotype']}")
        return

    if len(nat) == 0:
        print("Nothing to add.")
        return

    next_num = _max_num(df, "EN-N-") + 1
    new_rows = []

    for _, row in nat.iterrows():
        new_rows.append({
            "item_id":              f"EN-N-{next_num:03d}",
            "parallel_group_id":    None,
            "language":             "en",
            "origin":               "native",
            "dimension":            "",      
            "target_group":         "nationality",
            "target":               "",      
            "sent_stereotype":      row["sent_stereotype"].strip(),
            "sent_anti_stereotype": row["sent_anti_stereotype"].strip(),
            "source":               "crows_pairs_en",
            "validated":            False,
        })
        next_num += 1

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"Added {len(new_rows)} EN nationality items")

if __name__ == "__main__":
    main()
