"""
Generate the four thesis result figures.
Run: python src/make_figures.py
"""

import pathlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = pathlib.Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STIM = pd.read_csv(ROOT / "data" / "stimuli_seed.csv")[["item_id", "origin"]]

def load(path):
    df = pd.read_csv(ROOT / path)
    return df.drop(columns=["origin"], errors="ignore").merge(STIM, on="item_id", how="left")

def ci95(k, n):
    """Wilson 95% CI half-width."""
    if n == 0:
        return 0
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return (hi - lo) / 2

def cohen_h(p):
    return 2 * np.arcsin(np.sqrt(p)) - 2 * np.arcsin(np.sqrt(0.5))

PALETTE = {
    "GPT-4o-mini":                  "#5B9BD5",
    "mDeBERTa":                     "#9B59B6",
    "Mistral-7B":                   "#E67E22",
    "BLOOM-7B1":                    "#27AE60",
    "Llama-3.2-3B-Instruct":        "#2980B9",
    "Llama-3.2-3B-Instruct-Logprob": "#3498DB",
    "Llama-3.2-3B-Base":            "#D35400",
}

NULL_KW = dict(color="black", linestyle="--", linewidth=1.0, alpha=0.6)


# Overall BiasScore: all model/scoring conditions 

def fig1_overall():
    rows = [
        ("GPT-4o-mini\n(RLHF instruction-tuned)",
         "data/results/text/gpt-4o-mini_results.csv", "GPT-4o-mini"),
        ("Llama-3.2-3B-Instruct\n(prompted, RLHF)",
         "data/results/text/meta-llama-Llama-3.2-3B-Instruct_instruct_results.csv",
         "Llama-3.2-3B-Instruct"),
        ("Llama-3.2-3B-Instruct\n(logprob, RLHF)",
         "data/results/text/meta-llama-Llama-3.2-3B-Instruct_results.csv",
         "Llama-3.2-3B-Instruct-Logprob"),
        ("mDeBERTa-v3-base\n(Masked LM, no RLHF)",
         "data/results/text/microsoft-mdeberta-v3-base_results.csv", "mDeBERTa"),
        ("BLOOM-7B1\n(Causal base LM, no RLHF)",
         "data/results/text/bigscience-bloom-7b1_results.csv", "BLOOM-7B1"),
        ("Llama-3.2-3B-Base\n(Causal base LM, no RLHF)",
         "data/results/text/meta-llama-Llama-3.2-3B_results.csv", "Llama-3.2-3B-Base"),
        ("Mistral-7B-v0.1\n(Causal base LM, no RLHF)",
         "data/results/text/mistralai-Mistral-7B-v0.1_results.csv", "Mistral-7B"),
    ]
    n_aligned = 3  # first n_aligned rows are RLHF-aligned models

    fig, ax = plt.subplots(figsize=(8.5, 6))
    fig.patch.set_facecolor("white")

    for i, (label, path, key) in enumerate(rows):
        df = load(path)
        cs = df["chose_stereotype"]
        n, k = len(cs), int(cs.sum())
        m = k / n
        err = ci95(k, n)
        color = PALETTE[key]
        ax.errorbar(m, i, xerr=err, fmt="o", color=color,
                    markersize=9, capsize=4, linewidth=1.5, zorder=4)
        p = binomtest(k, n, 0.5).pvalue
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        h = cohen_h(m)
        ax.text(m + err + 0.006, i,
                f"BS={m:.3f}  h={h:+.3f}  {sig}",
                va="center", fontsize=8, color="#333333")

    ax.set_xlim(0.34, 0.84)
    ax.axvline(0.5, **NULL_KW, label="Null (0.500)", zorder=2)
    ax.axvspan(0.5, ax.get_xlim()[1], alpha=0.04, color="red")
    ax.axvspan(ax.get_xlim()[0], 0.5, alpha=0.04, color="blue")

    # divider between RLHF-aligned and unaligned/base models
    ax.axhline(n_aligned - 0.5, color="#999999", linestyle="-", linewidth=0.8, zorder=2)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _, _ in rows], fontsize=8.5)
    ax.set_ylim(len(rows) - 0.5, -0.7)  
    ax.set_xlabel("BiasScore (proportion stereotypical choices)", fontsize=9)
    ax.set_title("Overall BiasScore by Model\n"
                 r"$\leftarrow$ anti-stereotypical  |  null = 0.500  |  pro-stereotypical $\rightarrow$",
                 fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="lower right")
    ax.text(0.365, -0.55, r"$\leftarrow$ RLHF debiasing / instruction-following", fontsize=7.5,
            color="#2471A3", fontstyle="italic")
    ax.text(0.60,  -0.55, r"Pretraining stereotypical associations $\rightarrow$", fontsize=7.5,
            color="#B7440A", fontstyle="italic")

    plt.tight_layout()
    out = FIG_DIR / "fig1_overall_biasscore.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# Parallel vs Native: all three main models
def fig2_origin():
    models = [
        ("GPT-4o-mini", "data/results/text/gpt-4o-mini_results.csv"),
        ("mDeBERTa",    "data/results/text/microsoft-mdeberta-v3-base_results.csv"),
        ("Mistral-7B",  "data/results/text/mistralai-Mistral-7B-v0.1_results.csv"),
        ("Llama-3.2-3B-Instruct",
         "data/results/text/meta-llama-Llama-3.2-3B-Instruct_instruct_results.csv"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    fig.patch.set_facecolor("white")

    origins = ["native", "parallel"]
    origin_labels = ["Native\n(culture-specific)", "Parallel\n(translated)"]
    bar_colors = ["#A8C8E8", "#3470A3"]   

    for ax, (name, path) in zip(axes, models):
        df = load(path)
        color = PALETTE[name]
        bs_vals, err_vals, sig_vals = [], [], []

        for orig in origins:
            grp = df[df["origin"] == orig]["chose_stereotype"]
            n, k = len(grp), int(grp.sum())
            m = k / n
            err = ci95(k, n)
            p = binomtest(k, n, 0.5).pvalue
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            bs_vals.append(m)
            err_vals.append(err)
            sig_vals.append(sig)

        bars = ax.bar(origin_labels, bs_vals,
                      color=[color + "88", color], 
                      edgecolor="white", linewidth=0.8, width=0.5, zorder=3)
        ax.errorbar(origin_labels, bs_vals, yerr=err_vals,
                    fmt="none", color="#333333", capsize=4, linewidth=1.2, zorder=4)

        for xi, (sig, bsv) in enumerate(zip(sig_vals, bs_vals)):
            ax.text(xi, bsv + err_vals[xi] + 0.012, sig,
                    ha="center", fontsize=9, color="#333333", fontweight="bold")

        ax.axhline(0.5, **NULL_KW)
        ax.set_ylim(0.35, 0.70)
        ax.set_title(name, fontsize=10, color=color, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8.5)

    axes[0].set_ylabel("BiasScore", fontsize=9)
    fig.text(0.5, -0.04, "Item origin", ha="center", fontsize=9)

    # legend
    patch_n = mpatches.Patch(facecolor="#A8C8E888", edgecolor="white",
                              label="Native (culture-specific)")
    patch_p = mpatches.Patch(facecolor="#3470A3", edgecolor="white",
                              label="Parallel (translated)")
    null_line = plt.Line2D([0], [0], **NULL_KW, label="Null (0.500)")
    fig.legend(handles=[patch_n, patch_p, null_line],
               loc="upper center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("BiasScore by Item Origin and Model", fontsize=11, y=1.07)

    plt.tight_layout()
    out = FIG_DIR / "fig2_origin_biasscore.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")



# ΔASR: all three models × two ASR conditions
def fig3_delta_asr():
    specs = [
        ("GPT-4o-mini",
         "data/results/text/gpt-4o-mini_results.csv",
         "data/results/speech/large-v3_gpt-4o-mini_results.csv",
         "data/results/speech/medium_gpt-4o-mini_results.csv"),
        ("mDeBERTa",
         "data/results/text/microsoft-mdeberta-v3-base_results.csv",
         "data/results/speech/large-v3_microsoft-mdeberta-v3-base_results.csv",
         "data/results/speech/medium_microsoft-mdeberta-v3-base_results.csv"),
        ("Mistral-7B",
         "data/results/text/mistralai-Mistral-7B-v0.1_results.csv",
         "data/results/speech/large-v3_mistralai-Mistral-7B-v0.1_results.csv",
         "data/results/speech/medium_mistralai-Mistral-7B-v0.1_results.csv"),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(specs))
    width = 0.3
    asr_labels = ["large-v3\n(WER 4.9%)", "medium\n(WER 8.4%)"]
    asr_colors = ["#5B9BD5", "#ED7D31"]

    for ci, (asr_label, asr_color) in enumerate(zip(asr_labels, asr_colors)):
        deltas, errs = [], []
        for name, text_p, lv3_p, med_p in specs:
            tx = load(text_p)["chose_stereotype"]
            sp = load([lv3_p, med_p][ci])["chose_stereotype"]
            merged = tx.reset_index(drop=True).to_frame("tx").join(
                sp.reset_index(drop=True).to_frame("sp"))
            d = merged["sp"].mean() - merged["tx"].mean()
            # bootstrap CI for delta
            diffs = []
            for _ in range(1000):
                idx = np.random.randint(0, len(merged), len(merged))
                diffs.append(merged.iloc[idx]["sp"].mean() -
                             merged.iloc[idx]["tx"].mean())
            errs.append(np.percentile(np.abs(np.array(diffs) - d), 95))
            deltas.append(d)

        bars = ax.bar(x + ci * width, deltas, width,
                      label=f"Whisper {asr_label}",
                      color=asr_color, alpha=0.85,
                      edgecolor="white", linewidth=0.8, zorder=3)
        ax.errorbar(x + ci * width, deltas, yerr=errs,
                    fmt="none", color="#333333", capsize=3,
                    linewidth=1.0, zorder=4)

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([n for n, *_ in specs], fontsize=10)
    ax.set_ylabel("ΔASR (BiasScore speech - text)", fontsize=9)
    ax.set_title("ASR-Attributable Bias Shift (ΔASR) by Model and ASR Quality\n"
                 "Positive = speech pipeline increases stereotypical preference",
                 fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.5)
    ax.set_ylim(-0.005, 0.052)
    ax.annotate("Direction reversal\n(text: anti-stereo\nspeech: pro-stereo)",
                xy=(1 + 0 * width, 0.028), xytext=(1.55, 0.042),
                fontsize=7.5, color="#C0392B",
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.0))

    plt.tight_layout()
    out = FIG_DIR / "fig3_delta_asr.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_overall()
    fig2_origin()
    fig3_delta_asr()
    print("Done.")
