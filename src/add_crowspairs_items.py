"""
add_crowspairs_items.py
~~~~~~~~~~~~~~~~~~~~~~~
Ingests English nationality items from the CrowS-Pairs dataset
(Nangia et al., 2020) into stimuli_seed.csv as new EN-N source items.

Source: https://github.com/nyu-mll/crows-pairs
  bias_type == 'nationality' → target_group='nationality'

For each item, GPT-4o-mini is used to:
  1. Classify dimension: warmth or competence
  2. Extract the stereotyped group pair (e.g. "Mexican/American")

Items where stereotype == anti-stereotype after normalisation are skipped.
Duplicate sentences (same sent_stereotype as an existing item) are also skipped.

After running this script, execute:
    python src/add_translated_items.py --dim nationality
to generate FR + BG translations of the new EN items via the existing pipeline.

Resume-safe: items whose sent_stereotype already appears in the CSV are skipped.

Usage:
    OPENAI_API_KEY=sk-... python src/add_crowspairs_items.py
    OPENAI_API_KEY=sk-... python src/add_crowspairs_items.py --dry-run
    OPENAI_API_KEY=sk-... python src/add_crowspairs_items.py --limit 50
"""

import argparse
import io
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

CROWSPAIRS_URL = (
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/"
    "crows_pairs_anonymized.csv"
)

CLASSIFY_SYSTEM = (
    "You are a social-psychology researcher annotating stereotype sentences.\n\n"
    "Given a sentence pair (stereotyped vs neutral version), output a JSON object with:\n"
    "  \"dimension\": \"warmth\" or \"competence\"\n"
    "    - warmth: social trust, friendliness, threat, dangerousness, criminality, community belonging\n"
    "    - competence: intelligence, capability, work ethic, education, skill, productivity\n"
    "  \"target\": the stereotyped group vs comparison group, e.g. \"Mexican/American\" "
    "or \"Muslim/Christian\". Use the most specific label visible in the sentence. "
    "If only one group is mentioned, use \"[group]/[majority]\", e.g. \"Iraqi/Canadian\".\n\n"
    "Return ONLY valid JSON, no explanation. Example:\n"
    "{\"dimension\": \"warmth\", \"target\": \"Iraqi/Canadian\"}"
)


def _fetch_crowspairs() -> pd.DataFrame:
    print(f"Fetching CrowS-Pairs from GitHub ...", flush=True)
    with urllib.request.urlopen(CROWSPAIRS_URL, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    print(f"  Downloaded {len(df)} items", flush=True)
    return df


def _classify(client, sent_stereo: str, sent_anti: str) -> dict | None:
    """Ask GPT-4o-mini for dimension + target. Returns dict or None."""
    prompt = f"Stereotyped: {sent_stereo}\nNeutral: {sent_anti}"
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": CLASSIFY_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )
            import json
            data = json.loads(resp.choices[0].message.content)
            dim = data.get("dimension", "").strip().lower()
            tgt = data.get("target", "").strip()
            if dim not in ("warmth", "competence") or not tgt:
                return None
            return {"dimension": dim, "target": tgt}
        except Exception as exc:
            if attempt == 2:
                print(f"\n    WARN: classify failed: {exc}")
                return None
            time.sleep(2)
    return None


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

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit("ERROR: OPENAI_API_KEY not set.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key) if api_key else None

    # Fetch CrowS-Pairs
    cp = _fetch_crowspairs()

    # Filter to nationality items
    nat = cp[cp["bias_type"] == "nationality"].copy()
    print(f"Nationality items in CrowS-Pairs: {len(nat)}")

    # Ensure sent_more is the stereotyped sentence
    # stereo_antistereo column: 'stereo' means sent_more is the stereotype
    nat["sent_stereotype"]      = nat.apply(
        lambda r: r["sent_more"] if r["stereo_antistereo"] == "stereo" else r["sent_less"],
        axis=1,
    )
    nat["sent_anti_stereotype"] = nat.apply(
        lambda r: r["sent_less"] if r["stereo_antistereo"] == "stereo" else r["sent_more"],
        axis=1,
    )

    # Drop degenerate pairs
    nat = nat[nat["sent_stereotype"] != nat["sent_anti_stereotype"]].copy()
    print(f"After removing degenerate pairs: {len(nat)}")

    # Load existing CSV to skip duplicates
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    existing_stereo = set(df["sent_stereotype"].astype(str).str.strip())
    nat = nat[~nat["sent_stereotype"].str.strip().isin(existing_stereo)].reset_index(drop=True)
    print(f"After removing duplicates with existing: {len(nat)}")

    if args.limit:
        nat = nat.head(args.limit)
        print(f"Limiting to {args.limit} items")

    if args.dry_run:
        print(f"\n=== DRY RUN — first 5 items ===")
        for _, row in nat.head(5).iterrows():
            print(f"\n  S: {row['sent_stereotype']}")
            print(f"  A: {row['sent_anti_stereotype']}")
        return

    if len(nat) == 0:
        print("Nothing to add.")
        return

    next_num = _max_num(df, "EN-N-") + 1
    new_rows = []
    skipped  = 0

    print(f"\nClassifying {len(nat)} items (dimension + target) via GPT-4o-mini ...")
    for i, (_, row) in enumerate(nat.iterrows(), 1):
        meta = _classify(client,
                         row["sent_stereotype"].strip(),
                         row["sent_anti_stereotype"].strip())
        if meta is None:
            skipped += 1
        else:
            new_rows.append({
                "item_id":              f"EN-N-{next_num:03d}",
                "parallel_group_id":    None,          # will be assigned by add_translated_items.py
                "language":             "en",
                "origin":               "native",
                "dimension":            meta["dimension"],
                "target_group":         "nationality",
                "target":               meta["target"],
                "sent_stereotype":      row["sent_stereotype"].strip(),
                "sent_anti_stereotype": row["sent_anti_stereotype"].strip(),
                "source":               "crows_pairs_en",
                "validated":            True,
                "notes":                "From CrowS-Pairs (Nangia et al., 2020)",
            })
            next_num += 1

        if i % 10 == 0 or i == len(nat):
            print(f"\r  {i}/{len(nat)}  added={len(new_rows)}  skipped={skipped}",
                  end="", flush=True)
        time.sleep(0.05)

    print()

    if not new_rows:
        print("No valid items produced.")
        return

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"Added {len(new_rows)} EN nationality items (skipped {skipped})")
    print(f"Total items in CSV: {len(df)}")
    print(f"\nNext step: run  python src/add_translated_items.py --dim nationality")
    print(f"to generate FR + BG translations of the new EN-N items.")


if __name__ == "__main__":
    main()
