"""
visualize.py
~~~~~~~~~~~~
Generates all RQ1 figures for the thesis.

Figures produced
─────────────────
1.  biasbars.png       — BiasScore ± 95% CI by language × dimension
                         (Bonferroni and FDR significance markers)
2.  parallel_scatter.png — FR vs BG BiasScore per parallel group,
                           coloured by dimension, HF pairs highlighted
3.  origin_bars.png    — Native vs parallel BiasScore comparison
                         split by language and dimension
4.  logit_dist.png     — Distribution of logit_diff (continuous preference)
                         by language, with zero line and mean markers
5.  cue_comparison.png — Explicit-cue vs behavioural subgroup BiasScore
                         for parallel items

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

    fig_biasbars(df, args.model, show)
    fig_parallel_scatter(df, fid_df, args.model, show)
    fig_origin_bars(df, args.model, show)
    fig_logit_dist(df, args.model, show)
    fig_cue_comparison(df, fid_df, args.model, show)

    print(f"\nDone. All figures saved to {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
