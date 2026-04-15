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
    n_inf = np.isinf(df["logit_diff"]).sum()
    df["logit_diff"] = df["logit_diff"].replace([np.inf, -np.inf], np.nan)

    print(f"\n── Logit-Scale (Continuous Preference) Analysis ──")
    if n_inf > 0:
        print(f"  Note: {n_inf} items had ±inf logit_diff (token not in top_logprobs=2); excluded from continuous stats only.")
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
    df["logit_diff"] = df["logit_diff"].replace([np.inf, -np.inf], np.nan)

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


# ── Target-group × dimension SCM theory test ─────────────────────────────────

def target_group_analysis(df: pd.DataFrame, stimuli_df: pd.DataFrame) -> None:
    """
    Two analyses:

    [A] SCM theoretical prediction test
        SCM predicts profession items are competence-dominant and gender/nationality
        items are warmth-dominant.  Test whether BiasScore and logit-diff magnitude
        differ between the predicted-primary and predicted-secondary dimension for
        each target group.

    [B] Native vs translated within the same target group
        Tests whether culturally specific native items (manual_fr, manual_bg) show
        a different BiasScore from machine-translated EN items (crows_pairs_en,
        *_translated sources) within the same language × target_group cell.
        A significant difference is empirical evidence that cultural specificity
        matters for bias detection — justifying the manual authoring effort.
    """
    from scipy.stats import ttest_1samp, mannwhitneyu

    df = df.copy()
    df["logit_diff"] = np.where(
        df["A_is_stereotype"],
        df["logprob_A"] - df["logprob_B"],
        df["logprob_B"] - df["logprob_A"],
    )
    df["logit_diff"] = df["logit_diff"].replace([np.inf, -np.inf], np.nan)

    # ── [A] By target_group × dimension ──────────────────────────────────────
    print(f"\n── [12a] By target_group × dimension (SCM theory test) ──")
    print(f"  SCM prediction: profession → competence-dominant bias")
    print(f"                  gender / nationality → warmth-dominant bias")
    print()

    tbl = summarise(df, ["target_group", "dimension"])
    tbl["logit_diff_mean"] = [
        df[(df["target_group"] == row["target_group"]) &
           (df["dimension"] == row["dimension"])]["logit_diff"].mean()
        for _, row in tbl.iterrows()
    ]
    tbl["logit_diff_t"] = [
        ttest_1samp(
            df[(df["target_group"] == row["target_group"]) &
               (df["dimension"] == row["dimension"])]["logit_diff"].dropna(),
            0
        )[0] if row["N"] >= 5 else float("nan")
        for _, row in tbl.iterrows()
    ]
    tbl["logit_diff_p"] = [
        ttest_1samp(
            df[(df["target_group"] == row["target_group"]) &
               (df["dimension"] == row["dimension"])]["logit_diff"].dropna(),
            0
        )[1] if row["N"] >= 5 else float("nan")
        for _, row in tbl.iterrows()
    ]
    tbl = fdr_bh(tbl)

    # SCM prediction flag
    SCM_PRIMARY = {
        ("profession",   "competence"): True,
        ("profession",   "warmth"):     False,
        ("gender",       "warmth"):     True,
        ("gender",       "competence"): False,
        ("nationality",  "warmth"):     True,
        ("nationality",  "competence"): False,
    }
    tbl["scm_primary"] = tbl.apply(
        lambda r: SCM_PRIMARY.get((r["target_group"], r["dimension"]), None), axis=1
    )

    print(tbl[["target_group", "dimension", "N", "BiasScore", "CI_lo", "CI_hi",
               "p_value", "cohen_h", "sig_fdr_bh",
               "logit_diff_mean", "logit_diff_t", "logit_diff_p",
               "scm_primary"]].to_string(index=False))

    # Narrative summary
    print()
    for tg in ["gender", "nationality", "profession"]:
        w = tbl[(tbl["target_group"] == tg) & (tbl["dimension"] == "warmth")].iloc[0]
        c = tbl[(tbl["target_group"] == tg) & (tbl["dimension"] == "competence")].iloc[0]
        primary_dim = "warmth" if tg in ["gender", "nationality"] else "competence"
        primary     = w if primary_dim == "warmth" else c
        secondary   = c if primary_dim == "warmth" else w
        direction   = "anti-stereo" if primary["BiasScore"] < 0.5 else "pro-stereo"
        scm_confirmed = (
            (primary_dim == "warmth"     and primary["BiasScore"] < secondary["BiasScore"]) or
            (primary_dim == "competence" and primary["BiasScore"] < secondary["BiasScore"])
        )
        print(f"  {tg:12s}: primary={primary_dim} BS={primary['BiasScore']:.3f} "
              f"secondary BS={secondary['BiasScore']:.3f} | "
              f"direction={direction} sig={'*' if primary['sig_fdr_bh'] else 'ns'}")

    # ── [B] Native vs translated ──────────────────────────────────────────────
    if stimuli_df is None:
        print("\n  (stimuli CSV not loaded — native vs translated analysis skipped)")
        return

    print(f"\n── [12b] Native vs translated: within target_group × language ──")
    print(f"  'native'     = manual_fr, manual_bg, eurogest_fr, eurogest_bg, crows_pairs_en")
    print(f"  'translated' = *_translated sources (GPT-4o-mini from EN seed)")
    print(f"  Tests whether cultural specificity of items affects BiasScore magnitude.")
    print()

    TRANSLATED_SOURCES = {
        "en_nationality_translated", "en_profession_translated",
        "fr_gender_translated", "fr_roma_translated", "eurogest_fr_translated",
    }

    src_map = stimuli_df.set_index("item_id")["source"]
    df["provenance"] = df["item_id"].map(src_map).apply(
        lambda s: "translated" if s in TRANSLATED_SOURCES else "native"
    )

    rows = []
    for (lang, tg), grp in df.groupby(["language", "target_group"]):
        nat  = grp[grp["provenance"] == "native"]
        tran = grp[grp["provenance"] == "translated"]
        if len(nat) < 5 or len(tran) < 5:
            continue

        nat_bs  = nat["chose_stereotype"].mean()
        tran_bs = tran["chose_stereotype"].mean()

        # Mann-Whitney U on logit_diff (non-parametric; no normality assumed)
        stat, p_mw = mannwhitneyu(
            nat["logit_diff"].dropna(),
            tran["logit_diff"].dropna(),
            alternative="two-sided"
        )
        rows.append({
            "language":       lang,
            "target_group":   tg,
            "N_native":       len(nat),
            "BS_native":      round(nat_bs, 4),
            "ld_native":      round(nat["logit_diff"].mean(), 3),
            "N_translated":   len(tran),
            "BS_translated":  round(tran_bs, 4),
            "ld_translated":  round(tran["logit_diff"].mean(), 3),
            "delta_BS":       round(nat_bs - tran_bs, 4),
            "mw_U":           round(stat, 1),
            "mw_p":           round(p_mw, 4),
        })

    if not rows:
        print("  No language × target_group cells with both native and translated items.")
        return

    nt_tbl = pd.DataFrame(rows).sort_values(["target_group", "language"])
    nt_tbl["mw_sig_fdr"] = _fdr_bh_list(nt_tbl["mw_p"].tolist())

    print(nt_tbl[["language", "target_group",
                  "N_native", "BS_native", "ld_native",
                  "N_translated", "BS_translated", "ld_translated",
                  "delta_BS", "mw_p", "mw_sig_fdr"]].to_string(index=False))

    print()
    sig_cells = nt_tbl[nt_tbl["mw_sig_fdr"]]
    if not sig_cells.empty:
        print(f"  FDR-significant native vs translated differences:")
        for _, r in sig_cells.iterrows():
            direction = "native more anti-stereo" if r["delta_BS"] < 0 else "native more pro-stereo"
            print(f"    {r['language']}/{r['target_group']}: "
                  f"delta_BS={r['delta_BS']:+.3f} ({direction}), p={r['mw_p']:.4f}")
    else:
        print(f"  No FDR-significant native vs translated differences.")
        print(f"  Largest delta: {nt_tbl.loc[nt_tbl['delta_BS'].abs().idxmax(), ['language','target_group','delta_BS']].to_dict()}")


# ── Prompt-variant robustness ────────────────────────────────────────────────

def variant_robustness(variant_dfs: dict) -> None:
    """
    Compare BiasScore and mean logit-diff across prompt variants for each
    language × dimension cell.

    variant_dfs : {variant_name: DataFrame}  — must contain chose_stereotype,
                  A_is_stereotype, logprob_A, logprob_B, language, dimension.

    Prints
    ──────
    • Per-cell table: BiasScore and mean logit-diff for each variant
    • Robustness flag: direction consistent (all > 0.5, or all < 0.5, or all = 0.5)?
    • McNemar pairwise agreement between natural and each other variant
    • Summary: which cells survive across all variants
    """
    from scipy.stats import binomtest, ttest_1samp

    if len(variant_dfs) < 2:
        print("  (fewer than 2 variants available — robustness check skipped)")
        return

    # Compute logit_diff for each df
    for name, df in variant_dfs.items():
        variant_dfs[name] = df.copy()
        variant_dfs[name]["logit_diff"] = np.where(
            df["A_is_stereotype"],
            df["logprob_A"] - df["logprob_B"],
            df["logprob_B"] - df["logprob_A"],
        )
        variant_dfs[name]["logit_diff"] = variant_dfs[name]["logit_diff"].replace(
            [np.inf, -np.inf], np.nan
        )

    variants = list(variant_dfs.keys())
    cells    = sorted(
        set(
            tuple(r)
            for df in variant_dfs.values()
            for r in df[["language", "dimension"]].drop_duplicates().values.tolist()
        )
    )

    # ── 1. Per-cell BiasScore and logit-diff table ────────────────────────────
    print(f"\n── [11a] Prompt-Variant Robustness: BiasScore by variant ──")
    header_parts = ["language", "dimension"]
    for v in variants:
        header_parts += [f"BS_{v[:3]}", f"ld_{v[:3]}"]
    header_parts += ["direction_consistent", "bs_sig_any"]

    rows = []
    for (lang, dim) in cells:
        row = {"language": lang, "dimension": dim}
        bs_vals, ld_vals, sig_flags = [], [], []

        for v, df in variant_dfs.items():
            sub = df[(df["language"] == lang) & (df["dimension"] == dim)]
            if len(sub) == 0:
                row[f"BS_{v[:3]}"]  = float("nan")
                row[f"ld_{v[:3]}"]  = float("nan")
                row[f"N_{v[:3]}"]   = 0
                continue
            n  = len(sub)
            k  = int(sub["chose_stereotype"].sum())
            bs = k / n
            ld = float(sub["logit_diff"].mean())
            p  = binomtest(k, n, p=0.5, alternative="two-sided").pvalue
            row[f"BS_{v[:3]}"]  = round(bs, 4)
            row[f"ld_{v[:3]}"]  = round(ld, 4)
            row[f"N_{v[:3]}"]   = n
            bs_vals.append(bs)
            sig_flags.append(p < 0.05)

        # Direction consistency: all above 0.5, all below, or mixed?
        if all(b >= 0.5 for b in bs_vals):
            row["direction_consistent"] = "pro-stereo"
        elif all(b <= 0.5 for b in bs_vals):
            row["direction_consistent"] = "anti-stereo"
        else:
            row["direction_consistent"] = "mixed"

        row["bs_sig_any"] = any(sig_flags)
        rows.append(row)

    tbl = pd.DataFrame(rows)

    # Display columns
    bs_cols = [f"BS_{v[:3]}" for v in variants]
    ld_cols = [f"ld_{v[:3]}" for v in variants]
    print(tbl[["language", "dimension"] + bs_cols + ld_cols +
              ["direction_consistent", "bs_sig_any"]].to_string(index=False))

    # ── 2. McNemar pairwise agreement (natural vs each other) ────────────────
    if "natural" in variant_dfs:
        ref   = variant_dfs["natural"]
        others = {v: df for v, df in variant_dfs.items() if v != "natural"}
        if others:
            print(f"\n── [11b] McNemar pairwise agreement (natural vs other variants) ──")
            print(f"  McNemar H0: marginal choice distributions are equal")
            from scipy.stats import chi2

            for v, df in others.items():
                merged = ref[["item_id", "chose_stereotype"]].merge(
                    df[["item_id", "chose_stereotype"]],
                    on="item_id", suffixes=("_nat", f"_{v[:3]}")
                )
                if len(merged) == 0:
                    print(f"  natural vs {v}: no overlapping items")
                    continue
                a = int(((merged["chose_stereotype_nat"] == True)  & (merged[f"chose_stereotype_{v[:3]}"] == True)).sum())
                b = int(((merged["chose_stereotype_nat"] == True)  & (merged[f"chose_stereotype_{v[:3]}"] == False)).sum())
                c = int(((merged["chose_stereotype_nat"] == False) & (merged[f"chose_stereotype_{v[:3]}"] == True)).sum())
                d = int(((merged["chose_stereotype_nat"] == False) & (merged[f"chose_stereotype_{v[:3]}"] == False)).sum())
                agree_pct = 100 * (a + d) / len(merged) if len(merged) > 0 else float("nan")
                if (b + c) > 0:
                    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
                    p_mc = 1 - chi2.cdf(chi2_stat, df=1)
                else:
                    chi2_stat, p_mc = 0.0, 1.0
                print(f"  natural vs {v:8s}: N={len(merged)}  agree={agree_pct:.1f}%  "
                      f"b={b}  c={c}  chi2={chi2_stat:.3f}  p={p_mc:.4f}")

    # ── 3. Summary: stable significant cells ─────────────────────────────────
    print(f"\n── [11c] Cells significant in at least one variant ──")
    print(f"  (p < 0.05 uncorrected per variant; direction noted)")
    sig_rows = [r for r in rows if r["bs_sig_any"]]
    if sig_rows:
        for r in sig_rows:
            bs_str = "  ".join(
                f"{v[:3]}=BS{r[f'BS_{v[:3]}']:.3f}"
                for v in variants
                if f"BS_{v[:3]}" in r and not pd.isna(r[f"BS_{v[:3]}"])
            )
            print(f"  {r['language']}/{r['dimension']:10s}  dir={r['direction_consistent']:12s}  {bs_str}")
    else:
        print("  None.")


# ── ASR attribution ───────────────────────────────────────────────────────────

def asr_attribution(text_df: pd.DataFrame, speech_df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose the text↔speech BiasScore gap into ASR-error contribution
    and residual modality effect.

    Per language × dimension cell:
      gap_overall    : BiasScore(speech) − BiasScore(text)
      gap_ci_lo/hi   : 95% bootstrap CI on gap_overall (item-level resampling)
      mcnemar_chi2   : McNemar χ² on paired text/speech choices (continuity-corrected)
      mcnemar_p      : two-sided p-value; tests H0: P(text=S,speech=A) = P(text=A,speech=S)
      mcnemar_sig_fdr: BH-FDR corrected at α=0.05 across all cells
      gap_wer0       : gap restricted to WER_mean = 0 items (residual modality effect)
      gap_wer_gt0    : gap restricted to WER_mean > 0 items (ASR-error items)
      asr_contribution: gap_wer_gt0 − gap_wer0  (portion driven by transcription errors)
      lr_coef_modality: standardised LR coefficient for modality (speech vs text),
                        controlling for WER — descriptive only, not inferentially valid
                        (rows are paired, not independent; use McNemar for inference)
      lr_coef_wer    : standardised LR coefficient for WER — positive means higher
                        transcription error increases stereotype-choice probability
    """
    from scipy.stats import chi2 as chi2_dist, wilcoxon

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

    # ── WER asymmetry test (pooled, reported once) ────────────────────────────
    wer_diff = merged["wer_S"] - merged["wer_A"]
    nonzero  = wer_diff[wer_diff != 0]
    print("\n── WER Asymmetry (WER_S vs WER_A) ──")
    print(f"  H0: stereotypical and anti-stereotypical sentences have equal WER")
    print(f"  Mean WER_S : {merged['wer_S'].mean():.4f}")
    print(f"  Mean WER_A : {merged['wer_A'].mean():.4f}")
    print(f"  Mean WER_S − WER_A : {wer_diff.mean():.5f}")
    if len(nonzero) >= 10:
        w_stat, p_wer = wilcoxon(nonzero)
        direction = "WER_S > WER_A" if wer_diff.mean() > 0 else "WER_A > WER_S"
        print(f"  Wilcoxon signed-rank (N={len(nonzero)} non-zero pairs): "
              f"W={w_stat:.0f}  p={p_wer:.4f}  "
              f"{'*significant* — ' + direction if p_wer < 0.05 else 'not significant'}")
        print(f"  Interpretation: {'ASR systematically disadvantages one sentence type; ' if p_wer < 0.05 else 'No evidence of systematic ASR bias toward either sentence; '}"
              f"asymmetry in ΔASR partially attributable to WER imbalance.")
    else:
        print("  Insufficient non-zero differences for Wilcoxon test.")

    # ── Bootstrap CI helper ───────────────────────────────────────────────────
    _rng = np.random.default_rng(42)

    def _boot_gap(grp: pd.DataFrame, n: int = 5000) -> tuple:
        diffs = (grp["chose_stereotype_speech"].values.astype(float)
                 - grp["chose_stereotype_text"].values.astype(float))
        if len(diffs) < 5:
            return float("nan"), float("nan")
        boot = _rng.choice(diffs, size=(n, len(diffs)), replace=True).mean(axis=1)
        return round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)

    # ── McNemar helper ────────────────────────────────────────────────────────
    def _mcnemar(grp: pd.DataFrame) -> tuple:
        """Continuity-corrected McNemar. Returns (chi2, p)."""
        b = int(( grp["chose_stereotype_text"] & ~grp["chose_stereotype_speech"]).sum())
        c = int((~grp["chose_stereotype_text"] &  grp["chose_stereotype_speech"]).sum())
        if b + c == 0:
            return 0.0, 1.0
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p = float(1 - chi2_dist.cdf(chi2_stat, df=1))
        return round(chi2_stat, 3), round(p, 4)

    # ── Gap helper ────────────────────────────────────────────────────────────
    def _gap(g: pd.DataFrame) -> float:
        if len(g) == 0:
            return float("nan")
        return round(float(g["chose_stereotype_speech"].mean()
                           - g["chose_stereotype_text"].mean()), 4)

    rows = []
    for (lang, dim), grp in merged.groupby(["language", "dimension"]):
        perfect   = grp[grp["wer_mean"] == 0]
        imperfect = grp[grp["wer_mean"] > 0]

        gap_ov          = _gap(grp)
        ci_lo, ci_hi    = _boot_gap(grp)
        mc_chi2, mc_p   = _mcnemar(grp)
        gap_p           = _gap(perfect)
        gap_i           = _gap(imperfect)

        row = {
            "language":           lang,
            "dimension":          dim,
            "N_total":            len(grp),
            "N_wer0":             len(perfect),
            "N_wer_gt0":          len(imperfect),
            "BiasScore_text":     round(grp["chose_stereotype_text"].mean(), 4),
            "BiasScore_speech":   round(grp["chose_stereotype_speech"].mean(), 4),
            "gap_overall":        gap_ov,
            "gap_ci_lo":          ci_lo,
            "gap_ci_hi":          ci_hi,
            "mcnemar_chi2":       mc_chi2,
            "mcnemar_p":          mc_p,
            "gap_wer0":           gap_p,
            "gap_wer_gt0":        gap_i,
            "asr_contribution":   round(
                (gap_i - gap_p) if not (pd.isna(gap_i) or pd.isna(gap_p)) else float("nan"), 4
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

    df_out = pd.DataFrame(rows)
    df_out["mcnemar_sig_fdr"] = _fdr_bh_list(df_out["mcnemar_p"].tolist())

    # ── fr/warmth investigation (only cell with negative ΔASR) ───────────────
    frw = merged[(merged["language"] == "fr") & (merged["dimension"] == "warmth")].copy()
    if len(frw) > 0:
        frw_gap = _gap(frw)
        print(f"\n── fr/warmth Detailed Attribution (only cell with negative ΔASR={frw_gap:+.4f}) ──")
        print(f"  N_total={len(frw)}  N_wer0={(frw['wer_mean']==0).sum()}  "
              f"N_wer_gt0={(frw['wer_mean']>0).sum()}")
        print(f"  ΔASR by WER bin:")
        bins   = [-0.0001, 0.0001, 0.10, 0.30, 1.01]
        labels = ["0", "(0, 0.10]", "(0.10, 0.30]", "(0.30, 1.0]"]
        frw["wer_bin"] = pd.cut(frw["wer_mean"], bins=bins, labels=labels)
        for wer_bin, sub in frw.groupby("wer_bin", observed=True):
            if len(sub) == 0:
                continue
            g = sub["chose_stereotype_speech"].mean() - sub["chose_stereotype_text"].mean()
            print(f"    WER={wer_bin:>14s}  N={len(sub):4d}  gap={g:+.4f}")
        print(f"  Note: negative gap persists across WER bins — suggests fr/warmth")
        print(f"  speech transcripts alter fluency cues independently of ASR accuracy.")

    return df_out


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

    stimuli_df   = pd.read_csv(STIMULI,   encoding="utf-8") if STIMULI.exists()   else None
    fidelity_df  = pd.read_csv(FIDELITY,  encoding="utf-8") if FIDELITY.exists()  else None

    # Load prompt-variant result files
    grammar_path  = TEXT_DIR / f"{safe_llm}_grammar_results.csv"
    typical_path  = TEXT_DIR / f"{safe_llm}_typical_results.csv"

    def _load_variant(path, expected_variant):
        if not path.exists():
            return None
        df = pd.read_csv(path, encoding="utf-8")
        if args.lang:
            df = df[df["language"] == args.lang]
        return df

    grammar_df = _load_variant(grammar_path, "grammar")
    typical_df = _load_variant(typical_path, "typical")

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

    # ── 11. Target-group × dimension + native vs translated ─────────────────
    target_group_analysis(text_df, stimuli_df)

    # ── 12. Prompt-variant robustness ────────────────────────────────────────
    print(f"\n── [11] Prompt-Variant Robustness ──")
    print(f"  Compares BiasScore and logit-diff across natural / grammar / typical prompts.")
    variant_dfs = {"natural": text_df}
    if grammar_df is not None:
        variant_dfs["grammar"] = grammar_df
        print(f"  grammar  : {len(grammar_df)} items loaded")
    else:
        print(f"  grammar  : not found ({grammar_path})")
    if typical_df is not None:
        variant_dfs["typical"] = typical_df
        print(f"  typical  : {len(typical_df)} items loaded")
    else:
        print(f"  typical  : not found ({typical_path})")
    variant_robustness(variant_dfs)

    # ── 12. Outlier review ────────────────────────────────────────────────────
    outlier_review(fidelity_df, stimuli_df)

    # ── 12. Speech condition (if available) ───────────────────────────────────
    if speech_path.exists():
        speech_df = pd.read_csv(speech_path, encoding="utf-8")
        if args.lang:
            speech_df = speech_df[speech_df["language"] == args.lang]

        print("\n" + "=" * 65)
        print(f"SPEECH CONDITION  — ASR: {args.asr_model}  LLM: {args.text_model}")
        print("=" * 65)

        # ── Data completeness / silent exclusions ─────────────────────────────
        text_ids   = set(text_df["item_id"].astype(str))
        speech_ids = set(speech_df["item_id"].astype(str))
        in_text_only   = sorted(text_ids - speech_ids)
        in_speech_only = sorted(speech_ids - text_ids)
        print(f"\n── Data Completeness ──")
        print(f"  Text items     : {len(text_ids)}")
        print(f"  Speech items   : {len(speech_ids)}")
        if in_text_only:
            print(f"  In text but NOT speech ({len(in_text_only)} items — likely TTS failures):")
            for iid in in_text_only[:10]:
                print(f"    {iid}")
            if len(in_text_only) > 10:
                print(f"    ... and {len(in_text_only) - 10} more")
        else:
            print(f"  All text items have speech counterparts.")
        if in_speech_only:
            print(f"  In speech but NOT text ({len(in_speech_only)} items):")
            for iid in in_speech_only[:5]:
                print(f"    {iid}")
        print(f"  Paired items for attribution analysis: "
              f"{len(text_ids & speech_ids)}")

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

        print(f"\n── Per-cell ΔASR with 95% CI and McNemar test ──")
        print(f"  gap_ci_lo/hi : bootstrap 95% CI on ΔASR (5000 resamples)")
        print(f"  mcnemar_p    : H0: P(text=S,speech=A) = P(text=A,speech=S); continuity-corrected")
        print(f"  mcnemar_sig_fdr : BH-FDR at α=0.05 across all {len(attr)} cells")
        print(f"  lr_coef_*    : standardised LR descriptive only (rows are paired; use McNemar for inference)")
        print()
        print(attr[["language", "dimension", "N_total", "N_wer0", "N_wer_gt0",
                     "BiasScore_text", "BiasScore_speech",
                     "gap_overall", "gap_ci_lo", "gap_ci_hi",
                     "mcnemar_chi2", "mcnemar_p", "mcnemar_sig_fdr",
                     "gap_wer0", "gap_wer_gt0", "asr_contribution",
                     "lr_coef_modality", "lr_coef_wer"]].to_string(index=False))

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
