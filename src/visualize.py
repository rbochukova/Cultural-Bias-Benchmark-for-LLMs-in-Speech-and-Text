"""
visualize.py
~~~~~~~~~~~~
Generates all RQ1 figures for the thesis.

Figures produced
─────────────────
1.  biasbars.png           — BiasScore ± 95% CI by language × dimension
                             (Bonferroni and FDR significance markers)
2.  parallel_scatter.png   — FR vs BG logit-diff per parallel group
3.  origin_bars.png        — Native vs parallel BiasScore comparison
4.  logit_dist.png         — Distribution of logit_diff by language
5.  cue_comparison.png     — Explicit-cue vs behavioural subgroup BiasScore
6.  target_group_analysis.png — Heatmap by target_group × lang/dim + native vs translated dot plot
7.  variant_robustness.png         — BiasScore / logit-diff / delta across natural|grammar|typical
8.  model_comparison.png           — Three-way: gpt-4o-mini natural | grammar | mDeBERTa PLL
9a. speech_comparison.png          — Text vs speech BiasScore + ΔASR (RQ2)
9b. speech_variant_robustness.png  — ΔASR across natural|grammar|typical speech variants (robustness)

All figures are saved to reports/figures/ (created if absent).

Usage:
    python src/visualize.py
    python src/visualize.py --model gpt-4o-mini --no-show
"""

import argparse
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

ROOT        = pathlib.Path(__file__).resolve().parent.parent
TEXT_DIR    = ROOT / "data" / "results" / "text"
SPEECH_DIR  = ROOT / "data" / "results" / "speech"
FIDELITY    = ROOT / "data" / "parallel_fidelity.csv"
FIG_DIR     = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── colour palette ────────────────────────────────────────────────────────────
LANG_COLORS = {"en": "#4C72B0", "fr": "#DD8452", "bg": "#55A868"}
DIM_COLORS  = {"warmth": "#C44E52", "competence": "#4C72B0"}
DIM_MARKERS = {"warmth": "o", "competence": "s"}


# ── stats helpers (duplicated small subset from score.py) ─────────────────────

def _cohen_h(bs: float) -> float:
    if pd.isna(bs):
        return float("nan")
    return float(2 * np.arcsin(np.sqrt(bs)) - 2 * np.arcsin(np.sqrt(0.5)))


def _bootstrap_ci(series: pd.Series, n: int = 5000) -> tuple:
    if len(series) == 0:
        return (float("nan"), float("nan"))
    rng  = np.random.default_rng(42)
    vals = series.values.astype(float)
    boot = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def _bias_stats(series: pd.Series) -> dict:
    from scipy.stats import binomtest
    n  = len(series)
    k  = int(series.sum())
    bs = k / n if n > 0 else float("nan")
    ci = _bootstrap_ci(series)
    p  = binomtest(k, n, p=0.5, alternative="two-sided").pvalue if n > 0 else float("nan")
    return {"bs": bs, "ci_lo": ci[0], "ci_hi": ci[1], "p": p, "n": n}


def _fdr_bh(pvals: list) -> list:
    """Return BH-corrected reject booleans for alpha=0.05."""
    n      = len(pvals)
    order  = np.argsort(pvals)
    ranks  = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    thresholds = ranks * 0.05 / n
    sorted_p   = np.array(pvals)[order]
    sorted_th  = thresholds[order]
    reject_s   = np.zeros(n, dtype=bool)
    below      = sorted_p <= sorted_th
    if below.any():
        reject_s[:np.max(np.where(below)) + 1] = True
    reject       = np.empty(n, dtype=bool)
    reject[order] = reject_s
    return list(reject)


# ── Figure 1: BiasScore bars by language × dimension ─────────────────────────

def fig_biasbars(df: pd.DataFrame, model_name: str, show: bool) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    cells = [
        ("en", "warmth"),  ("en", "competence"),
        ("fr", "warmth"),  ("fr", "competence"),
        ("bg", "warmth"),  ("bg", "competence"),
    ]
    labels = [f"{l.upper()}\n{d}" for l, d in cells]

    stats = []
    for lang, dim in cells:
        sub = df[(df["language"] == lang) & (df["dimension"] == dim)]
        stats.append(_bias_stats(sub["chose_stereotype"]))

    pvals    = [s["p"] for s in stats]
    fdr_sig  = _fdr_bh(pvals)
    bon_thr  = 0.05 / len(stats)

    fig, ax = plt.subplots(figsize=(9, 5))
    x       = np.arange(len(cells))
    width   = 0.55

    for i, (s, (lang, dim)) in enumerate(zip(stats, cells)):
        color  = LANG_COLORS.get(lang, "#888888")
        alpha  = 0.8 if dim == "competence" else 0.5
        bar    = ax.bar(x[i], s["bs"], width, color=color, alpha=alpha,
                        edgecolor="white", linewidth=0.8)
        ax.errorbar(x[i], s["bs"],
                    yerr=[[s["bs"] - s["ci_lo"]], [s["ci_hi"] - s["bs"]]],
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=5)

        # Significance markers
        y_mark = s["ci_hi"] + 0.02
        marker = ""
        if s["p"] < bon_thr:
            marker = "***"
        elif fdr_sig[i]:
            marker = "† "
        if marker:
            ax.text(x[i], y_mark, marker, ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0,
               label="Null (0.5)", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("BiasScore  (proportion stereotypical choices)", fontsize=10)
    ax.set_title(f"RQ1 — BiasScore by Language × Dimension\nModel: {model_name}",
                 fontsize=11, pad=10)
    ax.set_ylim(0.25, 0.80)
    ax.set_xlim(-0.5, len(cells) - 0.5)

    # Legend
    patches = [
        mpatches.Patch(color=LANG_COLORS["en"], label="English"),
        mpatches.Patch(color=LANG_COLORS["fr"], label="French"),
        mpatches.Patch(color=LANG_COLORS["bg"], label="Bulgarian"),
        mpatches.Patch(color="#bbbbbb", alpha=0.5, label="warmth (lighter)"),
        mpatches.Patch(color="#bbbbbb", alpha=0.9, label="competence (darker)"),
    ]
    ax.legend(handles=patches + [
        plt.Line2D([0], [0], color="black", linestyle="--", label="Null (0.5)"),
    ], loc="upper right", fontsize=8, framealpha=0.9)

    ax.text(0.01, 0.01,
            f"*** p<{bon_thr:.4f} (Bonferroni)  † FDR significant (BH 5%)",
            transform=ax.transAxes, fontsize=7, color="gray",
            verticalalignment="bottom")

    plt.tight_layout()
    out = FIG_DIR / "biasbars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    if show:
        plt.show()
    plt.close()


# ── Figure 2: FR vs BG logit-diff scatter for parallel groups ─────────────────

def fig_parallel_scatter(df: pd.DataFrame, fid_df: pd.DataFrame,
                         model_name: str, show: bool) -> None:
    """
    Scatter of continuous logit_diff per parallel group.
    X = logit_diff in FR (positive → model preferred stereo sentence in FR)
    Y = logit_diff in BG (positive → model preferred stereo sentence in BG)
    One point per parallel group; coloured by dimension; HF pairs highlighted.
    """
    import matplotlib.pyplot as plt

    par = df[df["parallel_group_id"].str.startswith("PG-", na=False)].copy()
    par["logit_diff"] = np.where(
        par["A_is_stereotype"],
        par["logprob_A"] - par["logprob_B"],
        par["logprob_B"] - par["logprob_A"],
    )
    par["logit_diff"] = par["logit_diff"].replace([np.inf, -np.inf], np.nan)

    fr_ld = (par[par["language"] == "fr"]
             [["parallel_group_id", "dimension", "logit_diff"]]
             .rename(columns={"logit_diff": "fr_ld"}))
    bg_ld = (par[par["language"] == "bg"]
             [["parallel_group_id", "logit_diff"]]
             .rename(columns={"logit_diff": "bg_ld"}))
    merged = fr_ld.merge(bg_ld, on="parallel_group_id")

    if fid_df is not None:
        hf_set = set(fid_df[fid_df["high_fidelity"]]["parallel_group_id"])
        merged["hf"] = merged["parallel_group_id"].isin(hf_set)
    else:
        merged["hf"] = True

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    for dim in ["warmth", "competence"]:
        sub     = merged[merged["dimension"] == dim]
        hf_sub  = sub[sub["hf"]]
        nhf_sub = sub[~sub["hf"]]
        color   = DIM_COLORS[dim]
        marker  = DIM_MARKERS[dim]
        ax.scatter(hf_sub["fr_ld"],  hf_sub["bg_ld"],
                   c=color, marker=marker, s=55, alpha=0.85,
                   label=f"{dim} (HF, n={len(hf_sub)})", zorder=4)
        ax.scatter(nhf_sub["fr_ld"], nhf_sub["bg_ld"],
                   facecolors="none", edgecolors=color, marker=marker,
                   s=55, alpha=0.6, label=f"{dim} (non-HF, n={len(nhf_sub)})", zorder=3)

    lim = max(abs(merged[["fr_ld", "bg_ld"]].values).max() * 1.1, 5)
    ax.axhline(0, color="gray",  linestyle="-",  linewidth=0.7, alpha=0.6)
    ax.axvline(0, color="gray",  linestyle="-",  linewidth=0.7, alpha=0.6)
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, alpha=0.4,
            label="FR = BG")

    # Quadrant labels
    ax.text( lim*0.85,  lim*0.85, "Both\nstereo",   ha="center", va="center",
             fontsize=7, color="#777777", style="italic")
    ax.text(-lim*0.85, -lim*0.85, "Both\nanti",      ha="center", va="center",
             fontsize=7, color="#777777", style="italic")
    ax.text(-lim*0.85,  lim*0.85, "FR anti\nBG stereo", ha="center", va="center",
             fontsize=7, color="#777777", style="italic")
    ax.text( lim*0.85, -lim*0.85, "FR stereo\nBG anti",  ha="center", va="center",
             fontsize=7, color="#777777", style="italic")

    ax.set_xlabel("logit diff — French  (stereo − anti)", fontsize=10)
    ax.set_ylabel("logit diff — Bulgarian  (stereo − anti)", fontsize=10)
    ax.set_title(
        f"Parallel Group Preference: FR vs BG\nModel: {model_name}\n"
        "Continuous signal (logit diff); positive = preferred stereotypical sentence",
        fontsize=10
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "parallel_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    if show:
        plt.show()
    plt.close()


# ── Figure 3: Native vs parallel bars ────────────────────────────────────────

def fig_origin_bars(df: pd.DataFrame, model_name: str, show: bool) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    rows = []
    for origin in ["native", "parallel"]:
        for lang in ["en", "fr", "bg"]:
            for dim in ["warmth", "competence"]:
                sub = df[
                    (df["origin"] == origin) &
                    (df["language"] == lang) &
                    (df["dimension"] == dim)
                ]
                if len(sub) == 0:
                    continue
                s = _bias_stats(sub["chose_stereotype"])
                rows.append({"origin": origin, "language": lang,
                             "dimension": dim, **s})
    tbl = pd.DataFrame(rows)

    # Compute FDR across all cells
    tbl["sig_fdr"] = _fdr_bh(tbl["p"].tolist())

    langs = ["en", "fr", "bg"]
    dims  = ["warmth", "competence"]
    origins = ["native", "parallel"]

    n_groups = len(langs) * len(dims)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax_i, orig in enumerate(origins):
        ax    = axes[ax_i]
        sub_t = tbl[tbl["origin"] == orig]
        x     = np.arange(n_groups)
        ticks = []
        for j, (lang, dim) in enumerate([(l, d) for l in langs for d in dims]):
            row = sub_t[(sub_t["language"] == lang) & (sub_t["dimension"] == dim)]
            if row.empty:
                ticks.append(f"{lang.upper()}\n{dim}")
                continue
            row = row.iloc[0]
            color = LANG_COLORS.get(lang, "#888888")
            alpha = 0.85 if dim == "competence" else 0.5
            ax.bar(j, row["bs"], width=0.6, color=color, alpha=alpha,
                   edgecolor="white")
            ax.errorbar(j, row["bs"],
                        yerr=[[row["bs"] - row["ci_lo"]], [row["ci_hi"] - row["bs"]]],
                        fmt="none", color="black", capsize=3, linewidth=1)
            if row["sig_fdr"]:
                ax.text(j, row["ci_hi"] + 0.02, "†", ha="center",
                        fontsize=10, fontweight="bold")
            ticks.append(f"{lang.upper()}\n{dim}")

        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_title(f"{orig.capitalize()} items", fontsize=11)
        ax.set_ylim(0.2, 0.85)

    axes[0].set_ylabel("BiasScore", fontsize=10)
    fig.suptitle(f"Native vs Parallel BiasScore — {model_name}", fontsize=12, y=1.02)

    patches = [
        mpatches.Patch(color=LANG_COLORS["en"], label="EN"),
        mpatches.Patch(color=LANG_COLORS["fr"], label="FR"),
        mpatches.Patch(color=LANG_COLORS["bg"], label="BG"),
    ]
    axes[1].legend(handles=patches + [
        mpatches.Patch(color="#cccccc", alpha=0.5, label="warmth"),
        mpatches.Patch(color="#cccccc", alpha=0.9, label="competence"),
        plt.Line2D([0], [0], color="black", linestyle="--", label="Null"),
    ], fontsize=8, loc="upper right")

    plt.tight_layout()
    out = FIG_DIR / "origin_bars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    if show:
        plt.show()
    plt.close()


# ── Figure 4: Logit-diff distribution ────────────────────────────────────────

def fig_logit_dist(df: pd.DataFrame, model_name: str, show: bool) -> None:
    import matplotlib.pyplot as plt

    df = df.copy()
    df["logit_diff"] = np.where(
        df["A_is_stereotype"],
        df["logprob_A"] - df["logprob_B"],
        df["logprob_B"] - df["logprob_A"],
    )
    df["logit_diff"] = df["logit_diff"].replace([np.inf, -np.inf], np.nan)

    langs = ["en", "fr", "bg"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for ax, lang in zip(axes, langs):
        sub  = df[df["language"] == lang]["logit_diff"].dropna()
        mean = sub.mean()

        ax.hist(sub, bins=30, color=LANG_COLORS.get(lang, "#888888"),
                alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0, label="Null")
        ax.axvline(mean, color="red", linestyle="-", linewidth=1.5,
                   label=f"Mean={mean:.2f}")
        ax.set_title(lang.upper(), fontsize=12)
        ax.set_xlabel("logit diff (stereo − anti)", fontsize=9)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count", fontsize=10)
    fig.suptitle(
        f"Logit-Scale Preference Distribution — {model_name}\n"
        "Positive = preferred stereotypical sentence",
        fontsize=11
    )
    plt.tight_layout()
    out = FIG_DIR / "logit_dist.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    if show:
        plt.show()
    plt.close()


# ── Figure 5: Cue-based subgroup comparison ───────────────────────────────────

def fig_cue_comparison(df: pd.DataFrame, fid_df: pd.DataFrame,
                       model_name: str, show: bool) -> None:
    import matplotlib.pyplot as plt

    if fid_df is None or fid_df.empty:
        print("  (fidelity CSV not found — skipping cue comparison figure)")
        return

    explicit = set(fid_df[
        fid_df["cue_preserved_fr"] & fid_df["cue_preserved_bg"]
    ]["parallel_group_id"])
    behavioural = set(fid_df[
        ~(fid_df["cue_preserved_fr"] & fid_df["cue_preserved_bg"])
    ]["parallel_group_id"])

    par = df[df["parallel_group_id"].str.startswith("PG-", na=False)]

    rows = []
    for sub_label, group_set in [("Explicit-cue", explicit),
                                  ("Behavioural", behavioural)]:
        sub = par[par["parallel_group_id"].isin(group_set)]
        for lang in ["fr", "bg"]:
            for dim in ["warmth", "competence"]:
                s = sub[(sub["language"] == lang) & (sub["dimension"] == dim)]
                if len(s) == 0:
                    continue
                stat = _bias_stats(s["chose_stereotype"])
                rows.append({"subgroup": sub_label, "language": lang,
                             "dimension": dim, **stat})

    tbl = pd.DataFrame(rows)

    combos = [(l, d) for l in ["fr", "bg"] for d in ["warmth", "competence"]]
    x      = np.arange(len(combos))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for k, (sub_label, offset) in enumerate([("Explicit-cue", -width/2),
                                              ("Behavioural",   width/2)]):
        sub_tbl = tbl[tbl["subgroup"] == sub_label]
        for j, (lang, dim) in enumerate(combos):
            row = sub_tbl[(sub_tbl["language"] == lang) & (sub_tbl["dimension"] == dim)]
            if row.empty:
                continue
            row   = row.iloc[0]
            color = LANG_COLORS.get(lang, "#888888")
            alpha = 0.85 if dim == "competence" else 0.5
            hatch = "/" if sub_label == "Behavioural" else ""
            ax.bar(j + offset, row["bs"], width, color=color, alpha=alpha,
                   edgecolor="gray", hatch=hatch, linewidth=0.6)
            ax.errorbar(j + offset, row["bs"],
                        yerr=[[row["bs"] - row["ci_lo"]], [row["ci_hi"] - row["bs"]]],
                        fmt="none", color="black", capsize=3, linewidth=1)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="Null (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l.upper()}\n{d}" for l, d in combos], fontsize=9)
    ax.set_ylabel("BiasScore", fontsize=10)
    ax.set_title(
        f"Cue Subgroup Analysis — {model_name}\n"
        "Parallel items: explicit SCM cue vs behavioural expression",
        fontsize=10
    )
    ax.set_ylim(0.1, 0.85)
    ax.legend(fontsize=8)

    import matplotlib.patches as mpatches
    legend_items = [
        mpatches.Patch(facecolor="white", edgecolor="gray", label="Explicit-cue (solid)"),
        mpatches.Patch(facecolor="white", edgecolor="gray", hatch="////",
                       label="Behavioural (hatched)"),
        mpatches.Patch(color=LANG_COLORS["fr"], label="FR"),
        mpatches.Patch(color=LANG_COLORS["bg"], label="BG"),
        plt.Line2D([0], [0], color="black", linestyle="--", label="Null"),
    ]
    ax.legend(handles=legend_items, fontsize=8, loc="upper right")

    plt.tight_layout()
    out = FIG_DIR / "cue_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    if show:
        plt.show()
    plt.close()


# ── Figure 6: Target-group × dimension heatmap + native vs translated ────────

def fig_target_group(df: pd.DataFrame, stimuli_df: pd.DataFrame, show: bool) -> None:
    """
    Two-panel figure:
    Left  — BiasScore heatmap: target_group (rows) × language/dimension (cols)
             SCM-predicted primary dimension marked with a border.
    Right — Native vs translated BiasScore divergence: dot plot per
             language × target_group cell, showing BS_native and BS_translated
             side by side, connected by a line coloured by direction of delta.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.stats import mannwhitneyu

    TRANSLATED_SOURCES = {
        "en_nationality_translated", "en_profession_translated",
        "fr_gender_translated", "fr_roma_translated", "eurogest_fr_translated",
    }
    SCM_PRIMARY = {
        ("profession",  "competence"): True,
        ("gender",      "warmth"):     True,
        ("nationality", "warmth"):     True,
    }

    df = df.copy()
    df["logit_diff"] = np.where(
        df["A_is_stereotype"],
        df["logprob_A"] - df["logprob_B"],
        df["logprob_B"] - df["logprob_A"],
    )
    df["logit_diff"] = df["logit_diff"].replace([np.inf, -np.inf], np.nan)

    # ── Panel 1: heatmap ──────────────────────────────────────────────────────
    tgs   = ["gender", "nationality", "profession"]
    langs = ["en", "fr", "bg"]
    dims  = ["warmth", "competence"]
    cols  = [f"{l}/{d}" for l in langs for d in dims]

    heat_bs   = np.zeros((len(tgs), len(cols)))
    heat_sig  = np.zeros((len(tgs), len(cols)), dtype=bool)
    heat_n    = np.zeros((len(tgs), len(cols)), dtype=int)

    pvals_flat = []
    for i, tg in enumerate(tgs):
        for j, col in enumerate(cols):
            lang, dim = col.split("/")
            sub = df[(df["target_group"] == tg) & (df["language"] == lang) &
                     (df["dimension"] == dim)]
            if len(sub) == 0:
                heat_bs[i, j] = float("nan")
                pvals_flat.append(1.0)
            else:
                from scipy.stats import binomtest
                n = len(sub); k = int(sub["chose_stereotype"].sum())
                heat_bs[i, j] = k / n
                heat_n[i, j]  = n
                pvals_flat.append(binomtest(k, n, p=0.5, alternative="two-sided").pvalue)

    fdr_sig = _fdr_bh(pvals_flat)
    for idx, sig in enumerate(fdr_sig):
        i, j = divmod(idx, len(cols))
        heat_sig[i, j] = sig

    fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                             gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    masked = np.ma.masked_invalid(heat_bs)
    im = ax.imshow(masked, cmap="RdBu_r", vmin=0.30, vmax=0.70, aspect="auto")

    # Cell annotations
    for i in range(len(tgs)):
        for j in range(len(cols)):
            if np.isnan(heat_bs[i, j]):
                continue
            val = heat_bs[i, j]
            txt = f"{val:.2f}"
            color = "white" if abs(val - 0.5) > 0.12 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
            if heat_sig[i, j]:
                ax.text(j + 0.35, i - 0.35, "*", ha="center", va="center",
                        fontsize=10, color="black", fontweight="bold")

    # Thick border on SCM-primary cells
    for i, tg in enumerate(tgs):
        for j, col in enumerate(cols):
            lang, dim = col.split("/")
            if SCM_PRIMARY.get((tg, dim), False):
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     fill=False, edgecolor="#222222",
                                     linewidth=2.0, zorder=5)
                ax.add_patch(rect)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(tgs)))
    ax.set_yticklabels([t.capitalize() for t in tgs], fontsize=10)
    ax.set_title("BiasScore by target group × language/dimension\n"
                 "Bold border = SCM-predicted primary dimension  |  * = FDR sig",
                 fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label="BiasScore (0.5 = null)")

    # ── Panel 2: native vs translated dot plot ────────────────────────────────
    ax2 = axes[1]

    if stimuli_df is not None:
        src_map = stimuli_df.set_index("item_id")["source"]
        df["provenance"] = df["item_id"].map(src_map).apply(
            lambda s: "translated" if s in TRANSLATED_SOURCES else "native"
        )

        cells_nt, y_labels = [], []
        y = 0
        for tg in tgs:
            for lang in langs:
                nat  = df[(df["target_group"] == tg) & (df["language"] == lang) &
                          (df["provenance"] == "native")]
                tran = df[(df["target_group"] == tg) & (df["language"] == lang) &
                          (df["provenance"] == "translated")]
                if len(nat) < 5 or len(tran) < 5:
                    continue
                nat_bs  = nat["chose_stereotype"].mean()
                tran_bs = tran["chose_stereotype"].mean()
                _, p_mw = mannwhitneyu(nat["logit_diff"].dropna(),
                                       tran["logit_diff"].dropna(),
                                       alternative="two-sided")
                cells_nt.append((y, nat_bs, tran_bs, p_mw, f"{lang}/{tg}"))
                y_labels.append(f"{lang}/{tg}")
                y += 1

        if cells_nt:
            for y_pos, nat_bs, tran_bs, p_mw, label in cells_nt:
                color = "#cc2200" if nat_bs < tran_bs else "#1a6b2e"
                ax2.plot([nat_bs, tran_bs], [y_pos, y_pos],
                         color=color, linewidth=1.5, alpha=0.7, zorder=2)
                ax2.scatter([nat_bs], [y_pos], color="#1a6b2e", s=50, zorder=3,
                            marker="o", label="native" if y_pos == 0 else "")
                ax2.scatter([tran_bs], [y_pos], color="#4C72B0", s=50, zorder=3,
                            marker="s", label="translated" if y_pos == 0 else "")
                if p_mw < 0.05:
                    ax2.text(max(nat_bs, tran_bs) + 0.01, y_pos, "*",
                             va="center", fontsize=10, fontweight="bold")

            ax2.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
            ax2.set_yticks(range(len(y_labels)))
            ax2.set_yticklabels(y_labels, fontsize=8)
            ax2.set_xlabel("BiasScore", fontsize=9)
            ax2.set_title("Native vs translated items\n(green=native, blue=translated; * p<.05 MW)",
                          fontsize=9)
            ax2.set_xlim(0.15, 0.85)
            ax2.legend(fontsize=8, loc="lower right")
        else:
            ax2.text(0.5, 0.5, "Insufficient data\nfor comparison",
                     ha="center", va="center", transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, "stimuli CSV not loaded",
                 ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    out = FIG_DIR / "target_group_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    if show:
        plt.show()
    plt.close()


# ── Figure 7: Prompt-variant robustness ──────────────────────────────────────

def fig_variant_robustness(show: bool) -> None:
    """
    Three-panel figure comparing BiasScore across prompt variants
    (natural / grammar / typical) for each language × dimension cell.

    Panel 1: BiasScore per cell per variant (grouped bars)
    Panel 2: Mean logit-diff per cell per variant (grouped bars)
    Panel 3: Variance heatmap (max delta in BiasScore across variants)

    Cells with direction flip across variants are outlined in red.
    The only FDR-significant cell (fr/warmth, natural) is starred.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    variant_files = {
        "natural": TEXT_DIR / "gpt-4o-mini_results.csv",
        "grammar": TEXT_DIR / "gpt-4o-mini_grammar_results.csv",
        "typical": TEXT_DIR / "gpt-4o-mini_typical_results.csv",
    }

    dfs = {}
    for v, path in variant_files.items():
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8")
            df["logit_diff"] = np.where(
                df["A_is_stereotype"],
                df["logprob_A"] - df["logprob_B"],
                df["logprob_B"] - df["logprob_A"],
            )
            df["logit_diff"] = df["logit_diff"].replace([np.inf, -np.inf], np.nan)
            dfs[v] = df

    if len(dfs) < 2:
        print("  (fewer than 2 variant files found — skipping variant robustness figure)")
        return

    cells      = [f"{l}/{d}" for l in ["en", "fr", "bg"]
                  for d in ["warmth", "competence"]]
    cell_langs = [c.split("/")[0] for c in cells]
    variants   = list(dfs.keys())
    n_var      = len(variants)
    n_cell     = len(cells)
    x          = np.arange(n_cell)
    width      = 0.22
    offsets    = np.linspace(-(n_var - 1) / 2, (n_var - 1) / 2, n_var) * width
    var_colors = {"natural": "#4C72B0", "grammar": "#DD8452", "typical": "#55A868"}

    # Pre-compute stats
    bs_table  = {}   # {variant: {cell: BiasScore}}
    ld_table  = {}   # {variant: {cell: mean logit_diff}}
    for v, df in dfs.items():
        bs_table[v] = {}
        ld_table[v] = {}
        for cell in cells:
            lang, dim = cell.split("/")
            sub = df[(df["language"] == lang) & (df["dimension"] == dim)]
            bs_table[v][cell] = sub["chose_stereotype"].mean() if len(sub) > 0 else float("nan")
            ld_table[v][cell] = sub["logit_diff"].mean()       if len(sub) > 0 else float("nan")

    # Delta: max - min BiasScore across variants
    deltas = {}
    for cell in cells:
        vals = [bs_table[v][cell] for v in variants if not pd.isna(bs_table[v][cell])]
        deltas[cell] = max(vals) - min(vals) if len(vals) >= 2 else 0.0

    # Direction flip: any variant above 0.5 AND any below?
    def _has_flip(cell):
        vals = [bs_table[v][cell] for v in variants if not pd.isna(bs_table[v][cell])]
        return any(v > 0.5 for v in vals) and any(v < 0.5 for v in vals)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel 1: BiasScore ────────────────────────────────────────────────────
    ax = axes[0]
    for k, v in enumerate(variants):
        scores = [bs_table[v][c] for c in cells]
        ax.bar(x + offsets[k], scores, width,
               color=var_colors[v], alpha=0.85, edgecolor="white",
               linewidth=0.5, label=v)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    # Outline cells with direction flip
    for j, cell in enumerate(cells):
        if _has_flip(cell):
            ax.axvspan(j - 0.45, j + 0.45, color="none",
                       edgecolor="#cc2200", linewidth=1.5, zorder=5,
                       fill=False)
    # Star the fr/warmth natural bar (the one significant result)
    fw_idx = cells.index("fr/warmth")
    fw_bs  = bs_table["natural"].get("fr/warmth", float("nan"))
    if not pd.isna(fw_bs):
        ax.text(fw_idx + offsets[0], fw_bs - 0.035, "*", ha="center",
                va="top", fontsize=13, color="black", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(cells, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("BiasScore", fontsize=10)
    ax.set_ylim(0.30, 0.70)
    ax.set_title("BiasScore by Variant", fontsize=10)
    ax.legend(fontsize=8)

    # ── Panel 2: Mean logit-diff ──────────────────────────────────────────────
    ax = axes[1]
    for k, v in enumerate(variants):
        lds = [ld_table[v][c] for c in cells]
        ax.bar(x + offsets[k], lds, width,
               color=var_colors[v], alpha=0.85, edgecolor="white",
               linewidth=0.5, label=v)

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
    for j, cell in enumerate(cells):
        if _has_flip(cell):
            ax.axvspan(j - 0.45, j + 0.45, color="none",
                       edgecolor="#cc2200", linewidth=1.5, zorder=5, fill=False)

    ax.set_xticks(x)
    ax.set_xticklabels(cells, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Mean logit-diff  (stereo − anti)", fontsize=10)
    ax.set_title("Mean Logit-Diff by Variant", fontsize=10)
    ax.legend(fontsize=8)

    # ── Panel 3: Delta heatmap ────────────────────────────────────────────────
    ax = axes[2]
    delta_vals = np.array([deltas[c] for c in cells]).reshape(1, -1)
    im = ax.imshow(delta_vals, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=0.20)
    ax.set_xticks(np.arange(n_cell))
    ax.set_xticklabels(cells, fontsize=8, rotation=20, ha="right")
    ax.set_yticks([])
    ax.set_title("Instability (max BiasScore delta)", fontsize=10)
    for j, cell in enumerate(cells):
        ax.text(j, 0, f"{deltas[cell]:.2f}", ha="center", va="center",
                fontsize=9, color="black" if deltas[cell] < 0.12 else "white")
    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)

    fig.suptitle(
        "RQ1 — Prompt-Variant Robustness (gpt-4o-mini)\n"
        "Red outline = direction flip across variants  |  * = FDR-significant cell",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    out = FIG_DIR / "variant_robustness.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    if show:
        plt.show()
    plt.close()


# ── Figure 7: Three-way model/prompt comparison ───────────────────────────────

def fig_model_comparison(show: bool) -> None:
    """
    Side-by-side BiasScore for each language × dimension cell across three
    conditions:
      - gpt-4o-mini, natural prompt  (primary)
      - gpt-4o-mini, grammar prompt  (prompt stability)
      - mDeBERTa-v3, PLL scoring     (second model)
    Highlights unstable cells (max_delta > 0.10) with a shaded background.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    paths = {
        "GPT-4o-mini\n(natural)": TEXT_DIR / "gpt-4o-mini_results.csv",
        "GPT-4o-mini\n(grammar)": TEXT_DIR / "gpt-4o-mini_grammar_results.csv",
        "mDeBERTa-v3\n(PLL)":     TEXT_DIR / "microsoft-mdeberta-v3-base_results.csv",
    }
    dfs = {}
    for label, path in paths.items():
        if not path.exists():
            print(f"  SKIP {label}: file not found")
            continue
        dfs[label] = pd.read_csv(path, encoding="utf-8")

    if len(dfs) < 2:
        print("  Need at least 2 result files for comparison figure.")
        return

    cells  = [f"{l}/{d}" for l in ["en", "fr", "bg"] for d in ["warmth", "competence"]]
    labels = list(dfs.keys())
    n_cond = len(labels)
    x      = np.arange(len(cells))
    width  = 0.22
    offsets = np.linspace(-(n_cond - 1) / 2, (n_cond - 1) / 2, n_cond) * width

    cond_colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(13, 5))

    # Compute BiasScores per cell per condition
    cell_scores = {}
    for label, df in dfs.items():
        scores = {}
        for cell in cells:
            lang, dim = cell.split("/")
            sub = df[(df["language"] == lang) & (df["dimension"] == dim)]
            scores[cell] = sub["chose_stereotype"].mean() if len(sub) > 0 else float("nan")
        cell_scores[label] = scores

    # Shade unstable cells (max delta across conditions > 0.10)
    for j, cell in enumerate(cells):
        vals = [cell_scores[lbl][cell] for lbl in labels
                if not pd.isna(cell_scores[lbl][cell])]
        if len(vals) >= 2 and (max(vals) - min(vals)) > 0.10:
            ax.axvspan(j - 0.4, j + 0.4, color="#ffeeee", zorder=0, alpha=0.7)

    for k, (label, color) in enumerate(zip(labels, cond_colors)):
        for j, cell in enumerate(cells):
            bs = cell_scores[label][cell]
            if pd.isna(bs):
                continue
            ax.bar(j + offsets[k], bs, width, color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.5, label=label if j == 0 else "")

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(cells, fontsize=9)
    ax.set_ylabel("BiasScore", fontsize=10)
    ax.set_ylim(0.25, 0.80)
    ax.set_title(
        "RQ1 — Three-Way Model/Prompt Comparison\n"
        "Red shading = unstable cell (max delta > 0.10 across conditions)",
        fontsize=11
    )

    legend_handles = [
        mpatches.Patch(color=c, label=l.replace("\n", " "))
        for l, c in zip(labels, cond_colors)
    ] + [
        plt.Line2D([0], [0], color="black", linestyle="--", label="Null (0.5)"),
        mpatches.Patch(color="#ffeeee", label="Unstable (delta > 0.10)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

    plt.tight_layout()
    out = FIG_DIR / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    if show:
        plt.show()
    plt.close()


# ── Figure 9a: Speech variant robustness — ΔASR across prompt phrasings ──────

def fig_speech_variant_robustness(
    text_df: pd.DataFrame,
    speech_dfs: dict,        # {"natural": df, "grammar": df, "typical": df}
    asr_model: str,
    model_name: str,
    show: bool,
) -> None:
    """
    Grouped-bar chart: ΔASR = BiasScore(speech) − BiasScore(text) per
    language × dimension cell, one bar per speech prompt variant
    (natural / grammar / typical).

    Checks robustness of the modality gap across prompt phrasings.
    A consistent pattern across all three variants supports the conclusion
    that ΔASR reflects ASR-introduced distortion rather than prompt artefacts.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    cells = [
        ("en", "warmth"), ("en", "competence"),
        ("fr", "warmth"), ("fr", "competence"),
        ("bg", "warmth"), ("bg", "competence"),
    ]
    cell_labels = [f"{l.upper()}\n{d}" for l, d in cells]

    variants   = list(speech_dfs.keys())
    var_colors = {"natural": "#4C72B0", "grammar": "#DD8452", "typical": "#55A868"}
    n_var  = len(variants)
    x      = np.arange(len(cells))
    width  = 0.22
    offsets = np.linspace(-(n_var - 1) / 2, (n_var - 1) / 2, n_var) * width

    # Pre-compute ΔASR per cell per variant
    gaps: dict[str, list] = {v: [] for v in variants}
    for vname, sp_df in speech_dfs.items():
        merged = text_df[["item_id", "language", "dimension", "chose_stereotype"]].merge(
            sp_df[["item_id", "chose_stereotype"]],
            on="item_id", suffixes=("_text", "_speech")
        )
        for lang, dim in cells:
            grp = merged[(merged["language"] == lang) & (merged["dimension"] == dim)]
            if len(grp) == 0:
                gaps[vname].append(float("nan"))
            else:
                gaps[vname].append(
                    grp["chose_stereotype_speech"].mean()
                    - grp["chose_stereotype_text"].mean()
                )

    fig, ax = plt.subplots(figsize=(11, 5))

    for k, vname in enumerate(variants):
        ax.bar(x + offsets[k], gaps[vname], width,
               color=var_colors.get(vname, "#888888"), alpha=0.85,
               edgecolor="white", linewidth=0.5, label=vname)

    ax.axhline(0, color="black", linestyle="-", linewidth=1.0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(cell_labels, fontsize=9)
    ax.set_ylabel("ΔASR  (BiasScore_speech − BiasScore_text)", fontsize=10)
    ax.set_title(
        f"Speech Prompt-Variant Robustness: ΔASR per Cell\n"
        f"ASR: {asr_model}  ·  LLM: {model_name}  ·  "
        f"{'|'.join(variants)}",
        fontsize=10, pad=8,
    )

    legend_handles = [
        mpatches.Patch(color=var_colors.get(v, "#888888"), alpha=0.85, label=v)
        for v in variants
    ] + [plt.Line2D([0], [0], color="black", linestyle="-", label="no gap (0)")]
    ax.legend(handles=legend_handles, fontsize=8.5, loc="upper right", framealpha=0.9)

    # Annotate max spread per cell
    for i, (lang, dim) in enumerate(cells):
        vals = [gaps[v][i] for v in variants if not pd.isna(gaps[v][i])]
        if len(vals) >= 2:
            spread = max(vals) - min(vals)
            if spread > 0.015:
                ax.text(x[i], ax.get_ylim()[0] + 0.002,
                        f"Δ{spread:.2f}", ha="center", va="bottom",
                        fontsize=7, color="darkred", style="italic")

    plt.tight_layout()
    out = FIG_DIR / "speech_variant_robustness.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    if show:
        plt.show()
    plt.close()


# ── Figure 9b: Speech comparison — text vs speech BiasScore + ΔASR ───────────

def fig_speech_comparison(text_df: pd.DataFrame, speech_df: pd.DataFrame,
                           asr_model: str, model_name: str, show: bool) -> None:
    """
    Two-panel RQ2 figure.

    Panel A: Grouped bar chart — BiasScore(text, solid) vs BiasScore(speech, hatched)
             per language × dimension cell, with 95% bootstrap CI error bars.

    Panel B: ΔASR = BiasScore(speech) − BiasScore(text) per cell, with 95%
             bootstrap CI.  Cells significant under McNemar's test (BH-FDR
             corrected) are marked with a dagger †.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.stats import chi2 as chi2_dist

    cells       = [
        ("en", "warmth"),  ("en", "competence"),
        ("fr", "warmth"),  ("fr", "competence"),
        ("bg", "warmth"),  ("bg", "competence"),
    ]
    cell_labels = [f"{l.upper()}\n{d}" for l, d in cells]

    merged = text_df[["item_id", "language", "dimension", "chose_stereotype"]].merge(
        speech_df[["item_id", "chose_stereotype"]],
        on="item_id", suffixes=("_text", "_speech")
    )

    _rng = np.random.default_rng(42)

    def _stats(series: pd.Series) -> dict:
        return _bias_stats(series)

    def _boot_gap(grp: pd.DataFrame, n: int = 5000) -> tuple:
        diffs = (grp["chose_stereotype_speech"].values.astype(float)
                 - grp["chose_stereotype_text"].values.astype(float))
        if len(diffs) < 5:
            return float("nan"), float("nan")
        boot = _rng.choice(diffs, size=(n, len(diffs)), replace=True).mean(axis=1)
        return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    def _mcnemar_p(grp: pd.DataFrame) -> float:
        b = int(( grp["chose_stereotype_text"] & ~grp["chose_stereotype_speech"]).sum())
        c = int((~grp["chose_stereotype_text"] &  grp["chose_stereotype_speech"]).sum())
        if b + c == 0:
            return 1.0
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        return float(1 - chi2_dist.cdf(chi2_stat, df=1))

    # Collect per-cell statistics
    text_stats, speech_stats, gaps, gap_cis, mc_pvals = [], [], [], [], []
    for lang, dim in cells:
        grp = merged[(merged["language"] == lang) & (merged["dimension"] == dim)]
        ts  = _stats(grp["chose_stereotype_text"])
        ss  = _stats(grp["chose_stereotype_speech"])
        text_stats.append(ts)
        speech_stats.append(ss)
        gaps.append(ss["bs"] - ts["bs"])
        gap_cis.append(_boot_gap(grp))
        mc_pvals.append(_mcnemar_p(grp))

    mc_fdr = _fdr_bh(mc_pvals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x     = np.arange(len(cells))
    width = 0.35

    # ── Panel A: BiasScore text vs speech ────────────────────────────────────
    for i, (ts, ss, (lang, dim)) in enumerate(zip(text_stats, speech_stats, cells)):
        color = LANG_COLORS.get(lang, "#888888")
        # Text — solid
        ax1.bar(x[i] - width / 2, ts["bs"], width,
                color=color, alpha=0.85, edgecolor="white", linewidth=0.8)
        ax1.errorbar(x[i] - width / 2, ts["bs"],
                     yerr=[[ts["bs"] - ts["ci_lo"]], [ts["ci_hi"] - ts["bs"]]],
                     fmt="none", color="black", capsize=3, linewidth=1.0, zorder=5)
        # Speech — hatched
        ax1.bar(x[i] + width / 2, ss["bs"], width,
                color=color, alpha=0.40, edgecolor=color, linewidth=0.8, hatch="///")
        ax1.errorbar(x[i] + width / 2, ss["bs"],
                     yerr=[[ss["bs"] - ss["ci_lo"]], [ss["ci_hi"] - ss["bs"]]],
                     fmt="none", color="black", capsize=3, linewidth=1.0, zorder=5)

    ax1.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cell_labels, fontsize=9)
    ax1.set_ylabel("BiasScore  (proportion stereotypical choices)", fontsize=10)
    ax1.set_title(f"(A) Text vs Speech BiasScore\nASR: {asr_model}  ·  LLM: {model_name}",
                  fontsize=10, pad=8)
    ax1.set_ylim(0.30, 0.70)
    ax1.set_xlim(-0.5, len(cells) - 0.5)

    legend_patches = [
        mpatches.Patch(color="#666666", alpha=0.85, label="text (solid)"),
        mpatches.Patch(color="#666666", alpha=0.40, hatch="///", label="speech (hatched)"),
        plt.Line2D([0], [0], color="black", linestyle="--", label="Null (0.5)"),
    ] + [mpatches.Patch(color=LANG_COLORS[l], label=l.upper())
         for l in ["en", "fr", "bg"]]
    ax1.legend(handles=legend_patches, fontsize=7.5, loc="upper right", framealpha=0.9)

    # ── Panel B: ΔASR bars ────────────────────────────────────────────────────
    for i, (gap, (ci_lo, ci_hi), (lang, dim)) in enumerate(zip(gaps, gap_cis, cells)):
        color = LANG_COLORS.get(lang, "#888888")
        alpha = 0.85 if dim == "competence" else 0.50
        ax2.bar(x[i], gap, 0.55, color=color, alpha=alpha,
                edgecolor="white", linewidth=0.8)
        if not (pd.isna(ci_lo) or pd.isna(ci_hi)):
            ax2.errorbar(x[i], gap,
                         yerr=[[gap - ci_lo], [ci_hi - gap]],
                         fmt="none", color="black", capsize=4, linewidth=1.2, zorder=5)
        if mc_fdr[i]:
            y_tip = (ci_hi if not pd.isna(ci_hi) else gap) + 0.004
            if gap < 0:
                y_tip = (ci_lo if not pd.isna(ci_lo) else gap) - 0.012
            ax2.text(x[i], y_tip, "†", ha="center", va="bottom",
                     fontsize=13, fontweight="bold", color="darkred")

    ax2.axhline(0, color="black", linestyle="-", linewidth=1.0, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cell_labels, fontsize=9)
    ax2.set_ylabel("ΔASR  (BiasScore_speech − BiasScore_text)", fontsize=10)
    ax2.set_title("(B) Modality Gap (ΔASR) per Cell\n"
                  "Error bars = 95% bootstrap CI  ·  † McNemar p < 0.05 (FDR)",
                  fontsize=10, pad=8)

    lang_patches = [mpatches.Patch(color=LANG_COLORS[l], label=l.upper())
                    for l in ["en", "fr", "bg"]]
    dim_patches  = [
        mpatches.Patch(color="#aaaaaa", alpha=0.85, label="competence (darker)"),
        mpatches.Patch(color="#aaaaaa", alpha=0.50, label="warmth (lighter)"),
    ]
    ax2.legend(handles=lang_patches + dim_patches, fontsize=7.5,
               loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out = FIG_DIR / "speech_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    if show:
        plt.show()
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RQ1 figures")
    parser.add_argument("--model",    default="gpt-4o-mini")
    parser.add_argument("--no-show", action="store_true",
                        help="Save figures without displaying them")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend (works without a display)
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("ERROR: matplotlib not installed. Run: pip install matplotlib")

    safe_model   = args.model.replace("/", "-")
    results_path = TEXT_DIR / f"{safe_model}_results.csv"

    if not results_path.exists():
        sys.exit(
            f"Results not found: {results_path}\n"
            f"Run: python src/inference_text.py --model {args.model}"
        )

    df     = pd.read_csv(results_path, encoding="utf-8")
    fid_df = pd.read_csv(FIDELITY, encoding="utf-8") if FIDELITY.exists() else None
    show   = not args.no_show

    print(f"Generating figures for model: {args.model}")
    print(f"Output directory: {FIG_DIR.relative_to(ROOT)}")
    print()

    stimuli_df = pd.read_csv(ROOT / "data" / "stimuli_seed.csv", encoding="utf-8") \
        if (ROOT / "data" / "stimuli_seed.csv").exists() else None

    fig_biasbars(df, args.model, show)
    fig_parallel_scatter(df, fid_df, args.model, show)
    fig_origin_bars(df, args.model, show)
    fig_logit_dist(df, args.model, show)
    fig_cue_comparison(df, fid_df, args.model, show)
    fig_target_group(df, stimuli_df, show)
    fig_variant_robustness(show)
    fig_model_comparison(show)

    # RQ2 speech figures (skipped if speech results not available)
    safe_model   = args.model.replace("/", "-")
    speech_path  = SPEECH_DIR / f"large-v3_{safe_model}_results.csv"
    if speech_path.exists():
        speech_df = pd.read_csv(speech_path, encoding="utf-8")
        fig_speech_comparison(df, speech_df, "large-v3", args.model, show)

        # Speech variant robustness figure — load grammar/typical if present
        speech_dfs: dict = {"natural": speech_df}
        for vname in ("grammar", "typical"):
            vpath = SPEECH_DIR / f"large-v3_{safe_model}_{vname}_results.csv"
            if vpath.exists():
                speech_dfs[vname] = pd.read_csv(vpath, encoding="utf-8")
        if len(speech_dfs) >= 2:
            fig_speech_variant_robustness(df, speech_dfs, "large-v3", args.model, show)
        else:
            print(f"  SKIP speech_variant_robustness.png: need ≥2 speech variant files")
    else:
        print(f"  SKIP speech_comparison.png / speech_variant_robustness.png: "
              f"{speech_path.name} not found")

    print(f"\nDone. All figures saved to {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
