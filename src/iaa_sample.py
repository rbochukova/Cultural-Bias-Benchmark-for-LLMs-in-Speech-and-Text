"""Draw a stratified subsample for inter-annotator agreement (IAA) on the SCM warmth/competence labels, and emit a blind coding sheet for a 2nd annotator.
"""

import argparse
import pandas as pd

SEED = 42
N_SAMPLE = 150


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="data/stimuli_seed.csv")
    ap.add_argument("--sheet", default="data/iaa_coding_sheet.csv")
    ap.add_argument("--key", default="data/iaa_key.csv")
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    args = ap.parse_args()

    df = pd.read_csv(args.infile)
    df = df[df["validated"]].copy()
    # Stratify by language x target_group
    df["stratum"] = df["language"] + "|" + df["target_group"]
    frac = args.n / len(df)
    sample = (
        df.groupby("stratum", group_keys=False)
        .apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=SEED),
               include_groups=False)
        .reset_index(drop=True)
    )

    # Shuffle A/B presentation per item.
    rows_sheet, rows_key = [], []
    for _, r in sample.iterrows():
        flip = (hash((r["item_id"], SEED)) % 2) == 0
        opt_a = r["sent_anti_stereotype"] if flip else r["sent_stereotype"]
        opt_b = r["sent_stereotype"] if flip else r["sent_anti_stereotype"]
        gold_dir = "B" if flip else "A"
        rows_sheet.append({
            "item_id": r["item_id"],
            "language": r["language"],
            "option_A": opt_a,
            "option_B": opt_b,
            "dimension_R2": "",   # annotator fills: warmth/competence
            "direction_R2": "",   # annotator fills: A/B
        })
        rows_key.append({
            "item_id": r["item_id"],
            "dimension_gold": r["dimension"],
            "direction_gold": gold_dir,
        })

    pd.DataFrame(rows_sheet).to_csv(args.sheet, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_key).to_csv(args.key, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(rows_sheet)} items -> {args.sheet} (blind) and {args.key} (gold)")


if __name__ == "__main__":
    main()
