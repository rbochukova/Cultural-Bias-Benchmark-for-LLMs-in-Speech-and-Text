"""
score.py
~~~~~~~~
Computes BiasScore, significance tests, and RQ1 analyses.

Analyses included
─────────────────
1.  Overall BiasScore with 95 % bootstrap CI
2.  By language × dimension (Bonferroni AND Benjamini–Hochberg FDR correction)
3.  Cohen's h effect size for every cell
4.  A/B position-balance verification (internal validity check)
5.  By origin: native vs parallel
6.  Parallel analysis — full 87 pairs and high-fidelity 78-pair subset
7.  Cue-based subgroup analysis (explicit-cue vs behavioural-expression pairs)
8.  Cross-language agreement on parallel items
9.  Item-level review of the 9 outlier (non-HF) parallel pairs
10. Logit-scale (continuous preference) analysis
11. ASR attribution (speech condition, if available)
12. Parallel fidelity methods table (summary of quality-control metrics)

Usage:
    python src/score.py
    python src/score.py --text-model gpt-4o-mini --asr-model large-v3 --lang fr
    python src/score.py --output reports/results.csv
"""

import argparse
import pathlib
import sys
import warnings

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

ROOT        = pathlib.Path(__file__).resolve().parent.parent
TEXT_DIR    = ROOT / "data" / "results" / "text"
SPEECH_DIR  = ROOT / "data" / "results" / "speech"
FIDELITY    = ROOT / "data" / "parallel_fidelity.csv"
STIMULI     = ROOT / "data" / "stimuli_seed.csv"


# ── Statistics helpers ────────────────────────────────────────────────────────

def cohen_h(bias_score: float) -> float:
    """Effect size for a proportion vs null 0.5. Returns nan if bs is nan."""
    if pd.isna(bias_score):
        return float("nan")
    return float(2 * np.arcsin(np.sqrt(bias_score)) - 2 * np.arcsin(np.sqrt(0.5)))


def bootstrap_ci(series: pd.Series, n: int = 5000, alpha: float = 0.05) -> tuple:
    """Two-sided bootstrap CI for the mean. Returns (lower, upper)."""
    if len(series) == 0:
        return (float("nan"), float("nan"))
    rng  = np.random.default_rng(42)
    vals = series.values.astype(float)
    boot = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    lo   = float(np.percentile(boot, 100 * alpha / 2))
    hi   = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return (round(lo, 4), round(hi, 4))


def bias_score_row(chosen_stereo: pd.Series) -> dict:
    """
    Given a boolean Series of stereotype choices return a stats dict.
    Includes BiasScore, 95 % CI, binomial p-value, Cohen's h.
    """
    from scipy.stats import binomtest
    n    = len(chosen_stereo)
    k    = int(chosen_stereo.sum())
    bs   = k / n if n > 0 else float("nan")
    ci   = bootstrap_ci(chosen_stereo) if n > 0 else (float("nan"), float("nan"))
    if n > 0:
        p_val = binomtest(k, n, p=0.5, alternative="two-sided").pvalue
    else:
        p_val = float("nan")
    return {
        "N":         n,
        "k_stereo":  k,
        "BiasScore": round(bs, 4),
        "CI_lo":     ci[0],
        "CI_hi":     ci[1],
        "p_value":   round(p_val, 4),
        "cohen_h":   round(cohen_h(bs), 4),
    }


def summarise(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """BiasScore table grouped by group_cols."""
    rows = []
    for keys, grp in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(bias_score_row(grp["chose_stereotype"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def bonferroni(df: pd.DataFrame, n_tests: int) -> pd.DataFrame:
    df = df.copy()
    threshold = 0.05 / n_tests
    df["sig_bonferroni"] = df["p_value"] < threshold
    df["bonferroni_threshold"] = round(threshold, 6)
    return df


def fdr_bh(df: pd.DataFrame) -> pd.DataFrame:
    """Add Benjamini–Hochberg FDR column to a summarise() result."""
    df = df.copy()
    pvals = df["p_value"].values
    n     = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    thresholds = ranks * 0.05 / n
    # BH: reject if p_i <= (i/m)*0.05 for each rank i
    sorted_p  = pvals[order]
    sorted_th = thresholds[order]
    # Cumulative max from right to preserve monotonicity
    reject_sorted = np.zeros(n, dtype=bool)
    below = sorted_p <= sorted_th
    if below.any():
        last_reject = np.max(np.where(below))
        reject_sorted[:last_reject + 1] = True
    reject = np.empty(n, dtype=bool)
    reject[order] = reject_sorted
    df["sig_fdr_bh"] = reject
    df["fdr_bh_threshold"] = [round(thresholds[i], 6) for i in range(n)]
    return df


# ── A/B position-balance check ────────────────────────────────────────────────

def ab_balance_check(df: pd.DataFrame) -> None:
    """
    Print the fraction of items where the stereotypical sentence was in
    position A.  Should be ≈ 0.50 if the MD5 seed is unbiased.
    """
    n_total  = len(df)
    n_A_is_S = df["A_is_stereotype"].sum()
    frac     = n_A_is_S / n_total if n_total > 0 else float("nan")

    print(f"\n── A/B Position-Balance Check ──")
    print(f"  Total items       : {n_total}")
    print(f"  Stereo assigned A : {n_A_is_S}  ({100*frac:.1f}%)")
    print(f"  Stereo assigned B : {n_total - n_A_is_S}  ({100*(1-frac):.1f}%)")
    if abs(frac - 0.5) < 0.03:
        print(f"  Balance OK (within 3 pp of 50/50)")
    else:
        print(f"  WARNING: imbalance exceeds 3 pp — check _item_seed() logic")

    # Also check whether A-preference inflates BiasScore
    chose_A_rate = df["chose_A"].mean()
    print(f"  Rate model chose A: {100*chose_A_rate:.1f}%  "
          f"(deviation from 50%: {abs(chose_A_rate - 0.5)*100:.1f} pp)")


# ── Logit-scale (continuous preference) analysis ──────────────────────────────

def logit_scale_analysis(df: pd.DataFrame) -> None:
    """
    Analyse the raw logprob difference as a continuous preference signal.
    logit_diff > 0  → model preferred the stereotypical sentence
    logit_diff < 0  → model preferred the anti-stereotypical sentence
    """
    # Sign the logprob difference by stereotype direction
    df = df.copy()
    df["logit_diff"] = np.where(
        df["A_is_stereotype"],
        df["logprob_A"] - df["logprob_B"],
        df["logprob_B"] - df["logprob_A"],
    )

    print(f"\n── Logit-Scale (Continuous Preference) Analysis ──")
    print(f"  logit_diff = logprob(stereo) − logprob(anti)")
    print(f"  Positive = model preferred stereotypical sentence")
    print(f"  N = {len(df)}")
    print(f"  Mean   : {df['logit_diff'].mean():.4f}")
    print(f"  Median : {df['logit_diff'].median():.4f}")
    print(f"  Std    : {df['logit_diff'].std():.4f}")
    print(f"  Min    : {df['logit_diff'].min():.4f}")
    print(f"  Max    : {df['logit_diff'].max():.4f}")
    pct_pos = (df["logit_diff"] > 0).mean()
    print(f"  Fraction with logit_diff > 0 : {100*pct_pos:.1f}%  (= BiasScore)")

    print(f"\n  By language × dimension (mean logit_diff):")
    tbl = df.groupby(["language", "dimension"])["logit_diff"].agg(["mean", "std", "count"])
    tbl.columns = ["mean", "std", "N"]
    tbl = tbl.round(4)
    print(tbl.to_string())

    # Test whether mean logit_diff differs from 0 per cell (one-sample t-test)
    from scipy.stats import ttest_1samp
    print(f"\n  One-sample t-test: H0: mean(logit_diff) = 0")
    for (lang, dim), grp in df.groupby(["language", "dimension"]):
        vals = grp["logit_diff"].dropna()
        if len(vals) < 5:
            continue
        t, p = ttest_1samp(vals, 0)
        print(f"  {lang}/{dim}: t={t:.3f}  p={p:.4f}  mean={vals.mean():.4f}")

    return df


# ── Cue-based subgroup analysis ───────────────────────────────────────────────

def cue_subgroup_analysis(df: pd.DataFrame, fidelity_df: pd.DataFrame) -> None:
    """
    Split parallel pairs into:
      - explicit-cue pairs (SCM trait word present in both FR and BG)
      - behavioural-expression pairs (cue absent in at least one language)
    Compare BiasScore across subgroups.
    """
    if fidelity_df is None or fidelity_df.empty:
        print("  (fidelity CSV not found — cue subgroup analysis skipped)")
        return

    explicit = fidelity_df[
        fidelity_df["cue_preserved_fr"] & fidelity_df["cue_preserved_bg"]
    ]["parallel_group_id"].tolist()
    behavioural = fidelity_df[
        ~(fidelity_df["cue_preserved_fr"] & fidelity_df["cue_preserved_bg"])
    ]["parallel_group_id"].tolist()

    par = df[df["parallel_group_id"].str.startswith("PG-", na=False)].copy()

    sub_exp  = par[par["parallel_group_id"].isin(explicit)]
    sub_beh  = par[par["parallel_group_id"].isin(behavioural)]

    print(f"\n── Cue-Based Subgroup Analysis (parallel items only) ──")
    print(f"  Explicit-cue pairs (SCM word in both langs) : {len(explicit)}")
    print(f"  Behavioural-expression pairs                : {len(behavioural)}")

    for label, sub in [("Explicit-cue", sub_exp), ("Behavioural", sub_beh)]:
        if len(sub) == 0:
            continue
        tbl = summarise(sub, ["language", "dimension"])
        tbl = fdr_bh(tbl)
        print(f"\n  {label} ({len(sub)} items, {len(sub)//2} pairs):")
        print(tbl[["language", "dimension", "N", "BiasScore", "CI_lo", "CI_hi",
                    "p_value", "cohen_h", "sig_fdr_bh"]].to_string(index=False))


# ── Outlier parallel pair review ─────────────────────────────────────────────

def outlier_review(fidelity_df: pd.DataFrame, stimuli_df: pd.DataFrame) -> None:
    """
    Print the 9 parallel pairs excluded from the high-fidelity subset
    (dimension or direction mismatch), with their actual sentences.
    """
    if fidelity_df is None or fidelity_df.empty:
        return

    outliers = fidelity_df[~fidelity_df["high_fidelity"]]
    print(f"\n── Outlier Parallel Pairs (non-HF, N={len(outliers)}) ──")
    print(f"  These pairs failed the high-fidelity criterion (dim or direction mismatch)")

    stim = stimuli_df.set_index("item_id") if stimuli_df is not None else None

    for _, r in outliers.iterrows():
        reasons = []
        if not r["dim_agree"]:       reasons.append("dim_mismatch")
        if not r["direction_agree"]: reasons.append("dir_mismatch")
        print(f"\n  {r['parallel_group_id']}  [{', '.join(reasons)}]")
        for col, lang in [("fr_item_id", "FR"), ("bg_item_id", "BG")]:
            iid = r[col]
            if stim is not None and iid in stim.index:
                row  = stim.loc[iid]
                dim  = row["dimension"]
                tgt  = row["target"]
                s    = str(row["sent_stereotype"])[:80]
                a    = str(row["sent_anti_stereotype"])[:80]
                print(f"    {lang} [{iid}] dim={dim} target={tgt}")
                print(f"      S: {s}")
                print(f"      A: {a}")
            else:
                print(f"    {lang} [{iid}]")


# ── Binary vs continuous comparison table ────────────────────────────────────

def binary_vs_continuous(df: pd.DataFrame,
                         group_cols: list = None) -> pd.DataFrame:
    """
    Side-by-side comparison of BiasScore (binary) and logit-diff (continuous)
    for each cell defined by group_cols (default: language × dimension).

    Columns returned
    ────────────────
    Binary side  : N, BiasScore, BS_p, BS_sig_fdr, cohen_h
    Continuous   : ld_mean, ld_sd, ld_t, ld_p, ld_sig_fdr
    Interpretation: agreement flag + note
    """
    from scipy.stats import ttest_1samp, binomtest

    if group_cols is None:
        group_cols = ["language", "dimension"]

    df = df.copy()
    df["logit_diff"] = np.where(
        df["A_is_stereotype"],
        df["logprob_A"] - df["logprob_B"],
        df["logprob_B"] - df["logprob_A"],
    )

    rows = []
    for keys, grp in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))

        # Binary BiasScore
        n  = len(grp)
        k  = int(grp["chose_stereotype"].sum())
        bs = k / n if n > 0 else float("nan")
        bp = binomtest(k, n, p=0.5, alternative="two-sided").pvalue if n > 0 else float("nan")

        # Continuous logit-diff
        ld      = grp["logit_diff"].dropna()
        ld_mean = float(ld.mean())
        ld_sd   = float(ld.std())
        if len(ld) >= 5:
            t_stat, tp = ttest_1samp(ld, 0)
        else:
            t_stat, tp = float("nan"), float("nan")

        row.update({
            "N":          n,
            "BiasScore":  round(bs, 4),
            "BS_p":       round(bp, 4),
            "cohen_h":    round(cohen_h(bs), 4),
            "ld_mean":    round(ld_mean, 4),
            "ld_sd":      round(ld_sd, 4),
            "ld_t":       round(t_stat, 3),
            "ld_p":       round(tp, 4),
        })
        rows.append(row)

    tbl = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

    # FDR on both p-value columns
    tbl["BS_sig_fdr"]  = _fdr_bh_list(tbl["BS_p"].tolist())
    tbl["ld_sig_fdr"]  = _fdr_bh_list(tbl["ld_p"].tolist())

    # Agreement / divergence flag
    def _interp(r):
        bs_sig = r["BS_sig_fdr"]
        ld_sig = r["ld_sig_fdr"]
        if bs_sig and ld_sig:
            return "both significant"
        if not bs_sig and not ld_sig:
            return "both null"
        if ld_sig and not bs_sig:
            return "continuous only *"   # continuous more sensitive
        return "binary only"

    tbl["agreement"] = tbl.apply(_interp, axis=1)
    return tbl


def _fdr_bh_list(pvals: list) -> list:
    """BH FDR at alpha=0.05, returns list of bool."""
    n      = len(pvals)
    order  = np.argsort(pvals)
    ranks  = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    thresholds   = ranks * 0.05 / n
    sorted_p     = np.array(pvals)[order]
    sorted_th    = thresholds[order]
    reject_s     = np.zeros(n, dtype=bool)
    below        = sorted_p <= sorted_th
    if below.any():
        reject_s[:np.max(np.where(below)) + 1] = True
    reject       = np.empty(n, dtype=bool)
    reject[order] = reject_s
    return list(reject)


# ── ASR attribution ───────────────────────────────────────────────────────────

def asr_attribution(text_df: pd.DataFrame, speech_df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose the text↔speech BiasScore gap into ASR-error contribution
    and residual modality effect.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sklearn_ok = True
    except ImportError:
        sklearn_ok = False
        warnings.warn("scikit-learn not installed — logistic regression skipped")

    merged = text_df[["item_id", "language", "dimension", "chose_stereotype"]].merge(
        speech_df[["item_id", "chose_stereotype", "wer_S", "wer_A"]],
        on="item_id", suffixes=("_text", "_speech")
    )
    merged["wer_mean"] = (merged["wer_S"] + merged["wer_A"]) / 2

    rows = []
    for (lang, dim), grp in merged.groupby(["language", "dimension"]):
        perfect   = grp[grp["wer_mean"] == 0]
        imperfect = grp[grp["wer_mean"] > 0]

        def gap(g):
            if len(g) == 0:
                return float("nan")
            return g["chose_stereotype_speech"].mean() - g["chose_stereotype_text"].mean()

        row = {
            "language":           lang,
            "dimension":          dim,
            "N_total":            len(grp),
            "N_wer0":             len(perfect),
            "N_wer_gt0":          len(imperfect),
            "BiasScore_text":     round(grp["chose_stereotype_text"].mean(), 4),
            "BiasScore_speech":   round(grp["chose_stereotype_speech"].mean(), 4),
            "gap_overall":        round(gap(grp), 4),
            "gap_wer0":           round(gap(perfect), 4),
            "gap_wer_gt0":        round(gap(imperfect), 4),
            "asr_contribution":   round(
                (gap(grp) - gap(perfect)) if not pd.isna(gap(perfect)) else float("nan"), 4
            ),
        }

        if sklearn_ok and len(grp) >= 10:
            try:
                n   = len(grp)
                X_t = np.column_stack([np.zeros(n), np.zeros(n)])
                X_s = np.column_stack([np.ones(n), grp["wer_mean"].values])
                X   = np.vstack([X_t, X_s])
                y   = np.concatenate([
                    grp["chose_stereotype_text"].values.astype(int),
                    grp["chose_stereotype_speech"].values.astype(int),
                ])
                scaler = StandardScaler()
                X_sc   = scaler.fit_transform(X)
                lr     = LogisticRegression(max_iter=500).fit(X_sc, y)
                row["lr_coef_modality"] = round(float(lr.coef_[0][0]), 4)
                row["lr_coef_wer"]      = round(float(lr.coef_[0][1]), 4)
            except Exception:
                row["lr_coef_modality"] = float("nan")
                row["lr_coef_wer"]      = float("nan")
        else:
            row["lr_coef_modality"] = float("nan")
            row["lr_coef_wer"]      = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BiasScore and RQ1 analyses")
    parser.add_argument("--text-model",  default="gpt-4o-mini")
    parser.add_argument("--asr-model",   default="large-v3")
    parser.add_argument("--lang",        default=None)
    parser.add_argument("--output",      default=None,
                        help="Optional path to write full results table as CSV")
    args = parser.parse_args()

    safe_llm = args.text_model.replace("/", "-")
    safe_asr = args.asr_model.replace("/", "-")

    text_path   = TEXT_DIR   / f"{safe_llm}_results.csv"
    speech_path = SPEECH_DIR / f"{safe_asr}_{safe_llm}_results.csv"

    if not text_path.exists():
        sys.exit(
            f"Text results not found: {text_path}\n"
            f"Run: python src/inference_text.py --model {args.text_model}"
        )

    text_df = pd.read_csv(text_path, encoding="utf-8")
    if args.lang:
        text_df = text_df[text_df["language"] == args.lang]

    stimuli_df   = pd.read_csv(STIMULI, encoding="utf-8") if STIMULI.exists() else None
    fidelity_df  = pd.read_csv(FIDELITY, encoding="utf-8") if FIDELITY.exists() else None

    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print(f"RQ1 TEXT CONDITION  — model: {args.text_model}")
    print("=" * 65)

    # ── 1. A/B position-balance ───────────────────────────────────────────────
    ab_balance_check(text_df)

    # ── 2. Overall BiasScore ─────────────────────────────────────────────────
    print(f"\n── [1] Overall ──")
    overall = bias_score_row(text_df["chose_stereotype"])
    print(f"  N={overall['N']}  BiasScore={overall['BiasScore']:.3f}  "
          f"[{overall['CI_lo']:.3f},{overall['CI_hi']:.3f}]  "
          f"p={overall['p_value']:.4f}  h={overall['cohen_h']:.4f}")

    # ── 3. By language × dimension ────────────────────────────────────────────
    print(f"\n── [2] By language × dimension ──")
    tbl = summarise(text_df, ["language", "dimension"])
    tbl = bonferroni(tbl, n_tests=len(tbl))
    tbl = fdr_bh(tbl)
    print(tbl[["language", "dimension", "N", "BiasScore", "CI_lo", "CI_hi",
               "p_value", "cohen_h", "sig_bonferroni", "sig_fdr_bh"]].to_string(index=False))
    print(f"  Bonferroni threshold: {0.05/len(tbl):.4f}  |  FDR 5% (BH)")

    # ── 4. By target (top 10) ─────────────────────────────────────────────────
    print(f"\n── [3] By target (top 10 by |BiasScore − 0.5|) ──")
    tbl_target = summarise(text_df, ["language", "dimension", "target"])
    tbl_target["deviation"] = (tbl_target["BiasScore"] - 0.5).abs()
    top10 = (tbl_target.nlargest(10, "deviation")
             .drop(columns="deviation"))
    print(top10[["language", "dimension", "target", "N", "BiasScore",
                 "CI_lo", "CI_hi", "p_value", "cohen_h"]].to_string(index=False))

    # ── 5. By origin: native vs parallel ──────────────────────────────────────
    print(f"\n── [4] By origin ──")
    orig_tbl = summarise(text_df, ["origin"])
    orig_tbl = fdr_bh(orig_tbl)
    print(orig_tbl[["origin", "N", "BiasScore", "CI_lo", "CI_hi",
                     "p_value", "cohen_h", "sig_fdr_bh"]].to_string(index=False))

    # ── 6. Parallel analysis (full 87 pairs) ──────────────────────────────────
    par = text_df[text_df["parallel_group_id"].str.startswith("PG-", na=False)].copy()
    if len(par) > 0:
        print(f"\n── [5a] Full parallel set ({par['parallel_group_id'].nunique()} groups) ──")
        par_tbl = summarise(par, ["language", "dimension"])
        par_tbl = bonferroni(par_tbl, n_tests=len(par_tbl))
        par_tbl = fdr_bh(par_tbl)
        print(par_tbl[["language", "dimension", "N", "BiasScore", "CI_lo", "CI_hi",
                        "p_value", "cohen_h", "sig_bonferroni", "sig_fdr_bh"]].to_string(index=False))

    # ── 7. High-fidelity subset (78 pairs) ────────────────────────────────────
    if fidelity_df is not None:
        hf_ids = set(fidelity_df[fidelity_df["high_fidelity"]]["fr_item_id"].tolist() +
                     fidelity_df[fidelity_df["high_fidelity"]]["bg_item_id"].tolist())
        hf = text_df[text_df["item_id"].isin(hf_ids)]
        if len(hf) > 0:
            print(f"\n── [5b] High-fidelity subset ({fidelity_df['high_fidelity'].sum()} pairs) ──")
            hf_tbl = summarise(hf, ["language", "dimension"])
            hf_tbl = bonferroni(hf_tbl, n_tests=len(hf_tbl))
            hf_tbl = fdr_bh(hf_tbl)
            print(hf_tbl[["language", "dimension", "N", "BiasScore", "CI_lo", "CI_hi",
                           "p_value", "cohen_h", "sig_bonferroni", "sig_fdr_bh"]].to_string(index=False))

    # ── 8. Cross-language agreement on parallel items ─────────────────────────
    if fidelity_df is not None and len(par) > 0:
        hf_groups = set(fidelity_df[fidelity_df["high_fidelity"]]["parallel_group_id"])
        hf_par    = par[par["parallel_group_id"].isin(hf_groups)]
        if len(hf_par) > 0:
            print(f"\n── [6] Cross-language agreement (HF parallel pairs) ──")
            fr_choices = (hf_par[hf_par["language"] == "fr"]
                          .set_index("parallel_group_id")["chose_stereotype"])
            bg_choices = (hf_par[hf_par["language"] == "bg"]
                          .set_index("parallel_group_id")["chose_stereotype"])
            common = fr_choices.index.intersection(bg_choices.index)
            agree  = (fr_choices.loc[common] == bg_choices.loc[common]).mean()
            print(f"  Pairs compared : {len(common)}")
            print(f"  Same choice    : {100*agree:.1f}%")
            print(f"  Different      : {100*(1-agree):.1f}%")

            # By dimension
            for dim in ["warmth", "competence"]:
                fr_d = hf_par[(hf_par["language"] == "fr") & (hf_par["dimension"] == dim)].set_index("parallel_group_id")["chose_stereotype"]
                bg_d = hf_par[(hf_par["language"] == "bg") & (hf_par["dimension"] == dim)].set_index("parallel_group_id")["chose_stereotype"]
                com_d = fr_d.index.intersection(bg_d.index)
                if len(com_d) == 0:
                    continue
                ag_d = (fr_d.loc[com_d] == bg_d.loc[com_d]).mean()
                print(f"    {dim}: {len(com_d)} pairs, {100*ag_d:.1f}% same")

    # ── 9. Cue-based subgroup analysis ───────────────────────────────────────
    cue_subgroup_analysis(text_df, fidelity_df)

    # ── 10. Logit-scale analysis ──────────────────────────────────────────────
    logit_scale_analysis(text_df)

    # ── 10b. Binary vs continuous side-by-side ────────────────────────────────
    print(f"\n── [10b] Binary (BiasScore) vs Continuous (logit-diff) Comparison ──")
    print(f"  Reference: Nangia et al. (2020) CrowS-Pairs use binary BiasScore.")
    print(f"  logit_diff = log P(stereo) − log P(anti)  [log-odds of stereo preference]")
    print(f"  Binary test: binomial, H0: p=0.5  |  Continuous test: one-sample t, H0: mean=0")
    print(f"  Both corrected with BH-FDR at alpha=0.05")
    print(f"  'continuous only *' = continuous more sensitive (magnitude captured, sign was not enough)")
    print()
    comp = binary_vs_continuous(text_df)
    print(comp[["language", "dimension", "N",
                "BiasScore", "BS_p", "BS_sig_fdr", "cohen_h",
                "ld_mean", "ld_sd", "ld_t", "ld_p", "ld_sig_fdr",
                "agreement"]].to_string(index=False))

    print(f"\n  Parallel items — full 87 pairs:")
    par_full = text_df[text_df["parallel_group_id"].str.startswith("PG-", na=False)]
    if len(par_full) > 0:
        comp_par = binary_vs_continuous(par_full)
        print(comp_par[["language", "dimension", "N",
                         "BiasScore", "BS_p", "BS_sig_fdr",
                         "ld_mean", "ld_t", "ld_p", "ld_sig_fdr",
                         "agreement"]].to_string(index=False))

    # ── 11. Outlier review ────────────────────────────────────────────────────
    outlier_review(fidelity_df, stimuli_df)

    # ── 12. Speech condition (if available) ───────────────────────────────────
    if speech_path.exists():
        speech_df = pd.read_csv(speech_path, encoding="utf-8")
        if args.lang:
            speech_df = speech_df[speech_df["language"] == args.lang]

        print("\n" + "=" * 65)
        print(f"SPEECH CONDITION  — ASR: {args.asr_model}  LLM: {args.text_model}")
        print("=" * 65)

        overall_s = bias_score_row(speech_df["chose_stereotype"])
        print(f"\n── Overall ──")
        print(f"  N={overall_s['N']}  BiasScore={overall_s['BiasScore']:.3f}  "
              f"[{overall_s['CI_lo']:.3f},{overall_s['CI_hi']:.3f}]  "
              f"p={overall_s['p_value']:.4f}  h={overall_s['cohen_h']:.4f}")

        tbl_s = summarise(speech_df, ["language", "dimension"])
        tbl_s = fdr_bh(tbl_s)
        print(f"\n── By language × dimension ──")
        print(tbl_s[["language", "dimension", "N", "BiasScore", "CI_lo", "CI_hi",
                      "p_value", "cohen_h", "sig_fdr_bh"]].to_string(index=False))

        print("\n" + "=" * 65)
        print("ASR ATTRIBUTION ANALYSIS")
        print("=" * 65)
        attr = asr_attribution(text_df, speech_df)
        print(attr.to_string(index=False))

        print("\n── ASR WER summary ──")
        print(f"  Mean WER_S : {speech_df['wer_S'].mean():.3f}")
        print(f"  Mean WER_A : {speech_df['wer_A'].mean():.3f}")
        pct_perfect = 100 * (speech_df["wer_S"] == 0).mean()
        print(f"  Items with WER_S=0 : {pct_perfect:.1f}%")
    else:
        print(f"\n(Speech results not found — run inference_speech.py to add speech condition)")

    # ── 13. Optional full CSV output ──────────────────────────────────────────
    if args.output:
        frames = [text_df.assign(modality="text")]
        if speech_path.exists():
            frames.append(pd.read_csv(speech_path, encoding="utf-8").assign(modality="speech"))
        combined = pd.concat(frames, ignore_index=True)
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\nFull results written to: {args.output}")


if __name__ == "__main__":
    main()
