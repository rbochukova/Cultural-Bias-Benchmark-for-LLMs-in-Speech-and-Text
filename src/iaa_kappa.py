"""Compute Cohen's kappa for inter-annotator agreement on the SCM labels.

Reads the completed blind coding sheet (annotator 2) and the gold key
(annotator 1), and reports raw agreement + Cohen's kappa with a bootstrap 95% CI for both labels: SCM dimension and stereotype direction.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

SEED = 42
N_BOOT = 2000


def boot_ci(a, b, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    a, b = np.asarray(a), np.asarray(b)
    stats = []
    idx = np.arange(len(a))
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(set(a[s])) < 2 or len(set(b[s])) < 2:
            continue
        stats.append(cohen_kappa_score(a[s], b[s]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return lo, hi


def report(name, gold, r2):
    k = cohen_kappa_score(gold, r2)
    agree = float((np.asarray(gold) == np.asarray(r2)).mean())
    lo, hi = boot_ci(gold, r2)
    print(f"{name:12s}  raw agreement={agree:.3f}  kappa={k:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n={len(gold)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="data/iaa_coding_sheet.csv")
    ap.add_argument("--key", default="data/iaa_key.csv")
    args = ap.parse_args()

    sheet = pd.read_csv(args.sheet)
    key = pd.read_csv(args.key)
    m = sheet.merge(key, on="item_id", validate="one_to_one")
    m = m.dropna(subset=["dimension_R2", "direction_R2"])

    report("dimension", m["dimension_gold"].str.strip().str.lower(),
           m["dimension_R2"].str.strip().str.lower())
    report("direction", m["direction_gold"].str.strip().str.upper(),
           m["direction_R2"].str.strip().str.upper())


if __name__ == "__main__":
    main()
