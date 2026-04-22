"""
Ingests English gender-profession items from WinoBias into stimuli_seed.csv as new EN source items.
dimension and target are left blank for manual annotation after ingestion
"""

import argparse
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

ROOT     = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "stimuli_seed.csv"


def _load_winobias() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load type1_pro and type1_anti splits. Returns (pro_df, anti_df)."""
    print("Loading WinoBias via HuggingFace datasets library ...", flush=True)
    try:
        from datasets import load_dataset
        pro  = load_dataset("uclanlp/wino_bias", "type1_pro",
                            trust_remote_code=True)
        anti = load_dataset("uclanlp/wino_bias", "type1_anti",
                            trust_remote_code=True)

        pro_df  = pd.concat([pro["validation"].to_pandas(),
                             pro["test"].to_pandas()], ignore_index=True)
        anti_df = pd.concat([anti["validation"].to_pandas(),
                             anti["test"].to_pandas()], ignore_index=True)

        print(f"  type1_pro : {len(pro_df)} rows")
        print(f"  type1_anti: {len(anti_df)} rows")
        return pro_df, anti_df
    except Exception as e:
        sys.exit(f"Could not load WinoBias: {e}\n"
                 f"Install: pip install datasets")


def _tokens_to_sentence(tokens: list) -> str:
    """Join tokens into a sentence, handling punctuation spacing."""
    if not tokens:
        return ""
    result = tokens[0]
    for tok in tokens[1:]:
        if tok in {",", ".", "!", "?", ";", ":", "'s", "n't", "'re", "'ve",
                   "'ll", "'m", "'d"}:
            result += tok
        else:
            result += " " + tok
    return result.strip()


def _build_pairs(pro_df: pd.DataFrame, anti_df: pd.DataFrame) -> list[dict]:
    """
    Match pro/anti rows by positional index. The two splits share the same sentences with only gendered pronouns swapped
    """
    PRONOUNS = {"he", "she", "him", "her", "his", "hers", "himself", "herself"}

    pro_rows  = pro_df.reset_index(drop=True)
    anti_rows = anti_df.reset_index(drop=True)
    n = min(len(pro_rows), len(anti_rows))

    pairs = []
    for i in range(n):
        pro_tokens  = list(pro_rows.loc[i, "tokens"])
        anti_tokens = list(anti_rows.loc[i, "tokens"])

        pro_sent  = _tokens_to_sentence(pro_tokens)
        anti_sent = _tokens_to_sentence(anti_tokens)

        if pro_sent == anti_sent:
            continue
        if len(pro_tokens) != len(anti_tokens):
            continue
        diffs = [(p, a) for p, a in zip(pro_tokens, anti_tokens) if p.lower() != a.lower()]
        if not diffs:
            continue
        if not all(p.lower() in PRONOUNS and a.lower() in PRONOUNS for p, a in diffs):
            continue

        pairs.append({"pro_sent": pro_sent, "anti_sent": anti_sent})

    seen, unique = set(), []
    for p in pairs:
        if p["pro_sent"] not in seen:
            seen.add(p["pro_sent"])
            unique.append(p)

    return unique


def _max_num(df: pd.DataFrame, prefix: str) -> int:
    ids  = df[df["item_id"].str.startswith(prefix, na=False)]["item_id"]
    nums = ids.str.extract(re.escape(prefix) + r"(\d+)")[0].dropna().astype(int)
    return int(nums.max()) if not nums.empty else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest WinoBias type1 gender-profession items as EN source items"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch data and print first 5 items without saving")
    args = parser.parse_args()

    pro_df, anti_df = _load_winobias()
    pairs = _build_pairs(pro_df, anti_df)

    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    existing_stereo = set(df["sent_stereotype"].astype(str).str.strip())
    pairs = [p for p in pairs if p["pro_sent"].strip() not in existing_stereo]

    if args.dry_run:
        for p in pairs[:5]:
            print(f"\n  S: {p['pro_sent']}")
            print(f"  A: {p['anti_sent']}")
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
            "sent_stereotype":      p["pro_sent"],
            "sent_anti_stereotype": p["anti_sent"],
            "source":               "winobias_en",
            "validated":            False,
        })
        next_num += 1

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"Added {len(new_rows)} EN profession items from WinoBias")

if __name__ == "__main__":
    main()
