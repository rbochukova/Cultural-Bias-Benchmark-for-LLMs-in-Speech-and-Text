"""
Generates appendix figures for the thesis

Figures produced:
  A1 - Flip rate by WER bin 
  A2 - ΔASR by language x all 4 ASR conditions (GPT-4o-mini)
  A3 - WER distribution by language x ASR system 
  A4 - Warmth vs. competence ΔASR across ASR conditions
  A5 - Llama decomposition: base logprob / instruct logprob / instruct prompted
  A6 - Cross-model ΔASR comparison at Whisper large-v3
  A7 - Prompt-variant robustness of ΔASR (speech pipeline)
  A8 - BiasScore by source dataset (GPT-4o-mini)
  A9 - Text-condition prompt-variant robustness (natural/grammar/typical)
"""

import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT       = pathlib.Path(__file__).resolve().parent.parent
TEXT_DIR   = ROOT / "data" / "results" / "text"
SPEECH_DIR = ROOT / "data" / "results" / "speech"
ASR_DIR    = ROOT / "data" / "results" / "asr"
FIG_DIR    = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LANG_COLORS = {"en": "#4C72B0", "fr": "#DD8452", "bg": "#55A868"}
ASR_COLORS  = {
    "large-v3": "#2196F3",
    "azure":    "#FF9800",
    "medium":   "#9C27B0",
    "small":    "#F44336",
}
ASR_LABELS  = {
    "large-v3": "Whisper large-v3\n(WER 4.9%)",
    "azure":    "Azure STT\n(WER 5.7%)",
    "medium":   "Whisper medium\n(WER 8.4%)",
    "small":    "Whisper small\n(WER 16.0%)",
}


def _bias_score(series: pd.Series) -> float:
    return series.mean() if len(series) > 0 else float("nan")


def _bootstrap_ci(series: pd.Series, n: int = 5000) -> tuple:
    if len(series) == 0:
        return (float("nan"), float("nan"))
    rng  = np.random.default_rng(42)
    vals = series.values.astype(float)
    boot = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


# Figure A1: Flip rate by WER bin

def fig_a1_flip_rate_by_wer():
    text_df   = pd.read_csv(TEXT_DIR / "gpt-4o-mini_results.csv")
    speech_df = pd.read_csv(SPEECH_DIR / "large-v3_gpt-4o-mini_results.csv")

    merged = text_df[["item_id", "chose_stereotype"]].merge(
        speech_df[["item_id", "chose_stereotype", "wer_S", "wer_A"]],
        on="item_id", suffixes=("_text", "_speech")
    )
    merged["flip"]    = merged["chose_stereotype_text"] != merged["chose_stereotype_speech"]
    merged["wer_max"] = merged[["wer_S", "wer_A"]].max(axis=1)

    bins   = [0.0, 0.0001, 0.05, 0.10, 0.20, 0.30, 2.0]
    labels = ["WER=0", "0–5%", "5–10%", "10–20%", "20–30%", ">30%"]
    merged["wer_bin"] = pd.cut(merged["wer_max"], bins=bins, labels=labels, right=True)

    rates, cis, ns = [], [], []
    for lab in labels:
        sub = merged[merged["wer_bin"] == lab]["flip"]
        rates.append(sub.mean() if len(sub) > 0 else float("nan"))
        cis.append(_bootstrap_ci(sub))
        ns.append(len(sub))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, rates, color="#4C72B0", alpha=0.75, edgecolor="white", linewidth=0.8)
    for i, (lo, hi) in enumerate(cis):
        ax.errorbar(x[i], rates[i],
                    yerr=[[rates[i] - lo], [hi - rates[i]]],
                    fmt="none", color="black", capsize=5, linewidth=1.2, zorder=5)
    for i, (r, n) in enumerate(zip(rates, ns)):
        if not np.isnan(r):
            ax.text(x[i], r + 0.015, f"N={n}", ha="center", va="bottom",
                    fontsize=8, color="#444444")

    ax.axhline(0.131, color="red", linestyle="--", linewidth=1.2,
               label="Overall flip rate (13.1%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Max WER across sentence pair", fontsize=11)
    ax.set_ylabel("Flip rate  (text vs. speech decision reversal)", fontsize=11)
    ax.set_title(
        "Flip Rate by WER Bin\n",
        fontsize=11, pad=10
    )
    ax.set_ylim(0, 0.55)
    ax.legend(fontsize=9)
    ax.text(0.01, 0.97,
            "Baseline at WER=0 reflects residual TTS→ASR formatting differences,\n"
            "not transcription errors.",
            transform=ax.transAxes, fontsize=7.5, color="gray",
            va="top")
    plt.tight_layout()
    out = FIG_DIR / "appendix_a1_flip_rate_wer.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    plt.close()


# Figure A2: ΔASR by language × all 4 ASR conditions (GPT-4o-mini)

def fig_a2_delta_asr_by_language():
    text_df = pd.read_csv(TEXT_DIR / "gpt-4o-mini_results.csv")
    text_bs = (text_df.groupby("language")["chose_stereotype"]
                      .mean().rename("text_bs"))

    asr_systems = ["large-v3", "azure", "medium", "small"]
    speech_files = {
        "large-v3": SPEECH_DIR / "large-v3_gpt-4o-mini_results.csv",
        "azure":    SPEECH_DIR / "azure_gpt-4o-mini_results.csv",
        "medium":   SPEECH_DIR / "medium_gpt-4o-mini_results.csv",
        "small":    SPEECH_DIR / "small_gpt-4o-mini_results.csv",
    }

    langs   = ["en", "fr", "bg"]
    results = {asr: {} for asr in asr_systems}
    for asr, fpath in speech_files.items():
        if not fpath.exists():
            continue
        sp = pd.read_csv(fpath)
        for lang in langs:
            sub_sp   = sp[sp["language"] == lang]["chose_stereotype"]
            sub_text = text_df[text_df["language"] == lang]["chose_stereotype"]
            sp_bs    = _bias_score(sub_sp)
            tx_bs    = _bias_score(sub_text)
            results[asr][lang] = sp_bs - tx_bs

    x      = np.arange(len(langs))
    width  = 0.18
    offsets = np.linspace(-1.5, 1.5, len(asr_systems)) * width

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, asr in enumerate(asr_systems):
        vals = [results[asr].get(l, float("nan")) for l in langs]
        ax.bar(x + offsets[i], vals, width,
               color=ASR_COLORS[asr], label=ASR_LABELS[asr].replace("\n", " "),
               alpha=0.85, edgecolor="white", linewidth=0.7)

    ax.axhline(0, color="black", linewidth=1.0, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(["English", "French", "Bulgarian"], fontsize=11)
    ax.set_ylabel("ΔASR  (BiasScore speech − text)", fontsize=11)
    ax.set_title(
        "ASR-Attributable Bias Shift (ΔASR) by Language and ASR System\n",
        fontsize=11, pad=10
    )
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.set_ylim(-0.02, 0.065)
    plt.tight_layout()
    out = FIG_DIR / "appendix_a2_delta_asr_language.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    plt.close()


# Figure A3: WER distribution by language × ASR system (violin plots)

def fig_a3_wer_distribution():
    asr_files = {
        "large-v3": ASR_DIR / "large-v3_transcripts.csv",
        "azure":    ASR_DIR / "azure_transcripts.csv",
        "medium":   ASR_DIR / "medium_transcripts.csv",
        "small":    ASR_DIR / "small_transcripts.csv",
    }
    dfs = []
    for asr, fpath in asr_files.items():
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        df["asr_system"] = asr
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df[all_df["wer"] <= 1.0]

    langs       = ["en", "fr", "bg"]
    lang_labels = ["English", "French", "Bulgarian"]
    asr_systems = ["large-v3", "azure", "medium", "small"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    for col, (lang, llabel) in enumerate(zip(langs, lang_labels)):
        ax = axes[col]
        data_by_asr = []
        pos_labels  = []
        colors      = []
        for asr in asr_systems:
            sub = all_df[(all_df["language"] == lang) & (all_df["asr_system"] == asr)]["wer"]
            data_by_asr.append(sub.dropna().values)
            pos_labels.append(ASR_LABELS[asr])
            colors.append(ASR_COLORS[asr])

        positions = np.arange(1, len(asr_systems) + 1)
        parts = ax.violinplot(data_by_asr, positions=positions,
                              showmedians=True, showextrema=False)
        for pc, col_hex in zip(parts["bodies"], colors):
            pc.set_facecolor(col_hex)
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        for pos, data in zip(positions, data_by_asr):
            mean_val = np.mean(data) if len(data) > 0 else float("nan")
            ax.scatter([pos], [mean_val], color="black", s=20, zorder=5)

        ax.set_xticks(positions)
        ax.set_xticklabels(pos_labels, fontsize=7.5)
        ax.set_title(llabel, fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Word Error Rate (WER)", fontsize=10)
        ax.set_ylim(-0.02, 0.8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    fig.suptitle(
        "WER Distribution by Language and ASR System\n",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    out = FIG_DIR / "appendix_a3_wer_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    plt.close()


# Figure A4: Warmth vs. competence ΔASR across ASR conditions

def fig_a4_dim_delta_asr():
    text_df = pd.read_csv(TEXT_DIR / "gpt-4o-mini_results.csv")
    speech_files = {
        "large-v3": SPEECH_DIR / "large-v3_gpt-4o-mini_results.csv",
        "azure":    SPEECH_DIR / "azure_gpt-4o-mini_results.csv",
        "medium":   SPEECH_DIR / "medium_gpt-4o-mini_results.csv",
        "small":    SPEECH_DIR / "small_gpt-4o-mini_results.csv",
    }
    asr_systems = ["large-v3", "azure", "medium", "small"]
    dims        = ["warmth", "competence"]
    dim_colors  = {"warmth": "#C44E52", "competence": "#4C72B0"}

    results = {dim: {} for dim in dims}
    for asr, fpath in speech_files.items():
        if not fpath.exists():
            continue
        sp = pd.read_csv(fpath)
        for dim in dims:
            sp_bs = _bias_score(sp[sp["dimension"] == dim]["chose_stereotype"])
            tx_bs = _bias_score(text_df[text_df["dimension"] == dim]["chose_stereotype"])
            results[dim][asr] = sp_bs - tx_bs

    x      = np.arange(len(asr_systems))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, dim in enumerate(dims):
        vals   = [results[dim].get(asr, float("nan")) for asr in asr_systems]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width,
               color=dim_colors[dim], label=dim.capitalize(),
               alpha=0.80, edgecolor="white", linewidth=0.8)

    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([ASR_LABELS[a].replace("\n", " ") for a in asr_systems], fontsize=9.5)
    ax.set_ylabel("ΔASR  (BiasScore speech - text)", fontsize=11)
    ax.set_title(
        "ΔASR by SCM Dimension and ASR Condition\n",
        fontsize=11, pad=10
    )
    ax.legend(fontsize=10)
    ax.set_ylim(-0.02, 0.04)
    plt.tight_layout()
    out = FIG_DIR / "appendix_a4_dim_delta_asr.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    plt.close()


# Figure A5: Llama decomposition by language

def fig_a5_llama_decomposition():
    base_df      = pd.read_csv(TEXT_DIR / "meta-llama-Llama-3.2-3B_results.csv")
    inst_lp_df   = pd.read_csv(TEXT_DIR / "meta-llama-Llama-3.2-3B-Instruct_results.csv")
    inst_pr_df   = pd.read_csv(TEXT_DIR / "meta-llama-Llama-3.2-3B-Instruct_instruct_results.csv")

    conditions = {
        "Base\n(logprob)":      base_df,
        "Instruct\n(logprob)":  inst_lp_df,
        "Instruct\n(prompted)": inst_pr_df,
    }
    langs       = ["en", "fr", "bg"]
    lang_labels = ["English", "French", "Bulgarian"]
    cond_labels = list(conditions.keys())
    cond_colors = ["#78909C", "#5C6BC0", "#26A69A"]

    x      = np.arange(len(langs))
    width  = 0.25
    offsets = np.linspace(-1, 1, 3) * width

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, df) in enumerate(conditions.items()):
        vals = []
        cis  = []
        for lang in langs:
            sub = df[df["language"] == lang]["chose_stereotype"]
            vals.append(_bias_score(sub))
            cis.append(_bootstrap_ci(sub))

        bars = ax.bar(x + offsets[i], vals, width,
                      color=cond_colors[i], label=label.replace("\n", " "),
                      alpha=0.85, edgecolor="white", linewidth=0.7)
        for j, (v, (lo, hi)) in enumerate(zip(vals, cis)):
            ax.errorbar(x[j] + offsets[i], v,
                        yerr=[[v - lo], [hi - v]],
                        fmt="none", color="black", capsize=4,
                        linewidth=1.1, zorder=5)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="Null (0.5)")

    ax.annotate("", xy=(x[0] + offsets[1], 0.51), xytext=(x[0] + offsets[0], 0.51),
                arrowprops=dict(arrowstyle="<->", color="#78909C", lw=1.2))
    ax.text(x[0] + (offsets[0] + offsets[1]) / 2, 0.515,
            "Repr.\nΔ=−0.017", ha="center", va="bottom", fontsize=7, color="#78909C")

    ax.annotate("", xy=(x[0] + offsets[2], 0.545), xytext=(x[0] + offsets[1], 0.545),
                arrowprops=dict(arrowstyle="<->", color="#5C6BC0", lw=1.2))
    ax.text(x[0] + (offsets[1] + offsets[2]) / 2, 0.55,
            "Behav.\nΔ=−0.041", ha="center", va="bottom", fontsize=7, color="#5C6BC0")

    ax.set_xticks(x)
    ax.set_xticklabels(lang_labels, fontsize=11)
    ax.set_ylabel("BiasScore  (proportion stereotypical choices)", fontsize=11)
    ax.set_title(
        "Llama 3.2-3B Decomposition: Representational vs. Behavioural Bias\n",
        fontsize=11, pad=10
    )
    ax.set_ylim(0.40, 0.70)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    out = FIG_DIR / "appendix_a5_llama_decomposition.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    plt.close()


# Figure A6: Cross-model ΔASR comparison at large-v3 ASR

def fig_a6_cross_model_delta_asr():
    models = {
        "GPT-4o-mini": ("gpt-4o-mini", "#5B9BD5"),
        "mDeBERTa":    ("microsoft-mdeberta-v3-base", "#9B59B6"),
        "Mistral-7B":  ("mistralai-Mistral-7B-v0.1", "#E67E22"),
        "BLOOM-7B1":   ("bigscience-bloom-7b1", "#27AE60"),
    }
    langs       = ["en", "fr", "bg"]
    lang_labels = ["English", "French", "Bulgarian"]

    text_files = {
        "GPT-4o-mini": TEXT_DIR / "gpt-4o-mini_results.csv",
        "mDeBERTa":    TEXT_DIR / "microsoft-mdeberta-v3-base_results.csv",
        "Mistral-7B":  TEXT_DIR / "mistralai-Mistral-7B-v0.1_results.csv",
        "BLOOM-7B1":   TEXT_DIR / "bigscience-bloom-7b1_results.csv",
    }
    speech_files = {
        "GPT-4o-mini": SPEECH_DIR / "large-v3_gpt-4o-mini_results.csv",
        "mDeBERTa":    SPEECH_DIR / "large-v3_microsoft-mdeberta-v3-base_results.csv",
        "Mistral-7B":  SPEECH_DIR / "large-v3_mistralai-Mistral-7B-v0.1_results.csv",
        "BLOOM-7B1":   SPEECH_DIR / "large-v3_bigscience-bloom-7b1_results.csv",
    }

    results = {}
    for model_label in models:
        text   = pd.read_csv(text_files[model_label])
        speech = pd.read_csv(speech_files[model_label])
        results[model_label] = {}
        for lang in langs:
            t = _bias_score(text[text["language"] == lang]["chose_stereotype"])
            s = _bias_score(speech[speech["language"] == lang]["chose_stereotype"])
            results[model_label][lang] = s - t

    x       = np.arange(len(langs))
    n_m     = len(models)
    width   = 0.18
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, (_, color)) in enumerate(models.items()):
        vals = [results[label][l] for l in langs]
        ax.bar(x + offsets[i], vals, width,
               color=color, label=label, alpha=0.85,
               edgecolor="white", linewidth=0.7)

    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(lang_labels, fontsize=12)
    ax.set_ylabel("ΔASR  (BiasScore speech − text)", fontsize=11)
    ax.set_title(
        "ΔASR by Model and Language (Whisper large-v3)\n",
        fontsize=11, pad=10
    )
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.set_ylim(-0.03, 0.07)
    plt.tight_layout()
    out = FIG_DIR / "appendix_a6_cross_model_delta_asr.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    plt.close()


# Figure A7: Prompt variant robustness in the speech pipeline

def fig_a7_prompt_variant_robustness():
    variants = {
        "Standard":         ("gpt-4o-mini_results.csv",         "large-v3_gpt-4o-mini_results.csv",         "#2196F3"),
        "Grammar-aware":    ("gpt-4o-mini_grammar_results.csv", "large-v3_gpt-4o-mini_grammar_results.csv", "#FF9800"),
        "Typical-language": ("gpt-4o-mini_typical_results.csv", "large-v3_gpt-4o-mini_typical_results.csv", "#9C27B0"),
    }
    langs       = ["en", "fr", "bg"]
    lang_labels = ["English", "French", "Bulgarian"]

    results = {}
    for variant_label, (text_f, speech_f, _) in variants.items():
        text   = pd.read_csv(TEXT_DIR   / text_f)
        speech = pd.read_csv(SPEECH_DIR / speech_f)
        results[variant_label] = {}
        for lang in langs:
            t = _bias_score(text[text["language"] == lang]["chose_stereotype"])
            s = _bias_score(speech[speech["language"] == lang]["chose_stereotype"])
            results[variant_label][lang] = s - t

    x       = np.arange(len(langs))
    width   = 0.25
    offsets = np.linspace(-1, 1, 3) * width

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, (_, _, color)) in enumerate(variants.items()):
        vals = [results[label][l] for l in langs]
        ax.bar(x + offsets[i], vals, width,
               color=color, label=label, alpha=0.85,
               edgecolor="white", linewidth=0.7)

    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(lang_labels, fontsize=12)
    ax.set_ylabel("ΔASR  (BiasScore speech - text)", fontsize=11)
    ax.set_title(
        "Prompt Variant Robustness: ΔASR by Language\n",
        fontsize=11, pad=10
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(-0.015, 0.035)
    plt.tight_layout()
    out = FIG_DIR / "appendix_a7_prompt_variant_robustness.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    plt.close()


# Figure A8: BiasScore by source dataset (GPT-4o-mini)

def fig_a8_biasscore_by_source():
    null_c, low, high = "#9a9a9a", "#2166ac", "#b2182b"

    stim = pd.read_csv(ROOT / "data" / "stimuli_seed.csv")
    stim = stim[stim["validated"] == True]  # noqa: E712
    meta = stim[["item_id", "source"]]

    d = pd.read_csv(TEXT_DIR / "gpt-4o-mini_results.csv")
    if "modality" in d:
        d = d[d["modality"] == "text"]
    d = d.merge(meta, on="item_id")
    g = (d.groupby("source")["chose_stereotype"]
          .agg(["mean", "count"]).sort_values("mean"))

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    colors = [low if m < 0.5 else high for m in g["mean"]]
    ax.barh(range(len(g)), g["mean"], color=colors, edgecolor="white", height=0.72)
    ax.axvline(0.5, color=null_c, lw=1.2, ls=(0, (5, 4)))
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([s.replace("_", " ") for s in g.index], fontsize=9)
    for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
        ax.text(m + (0.004 if m >= 0.5 else -0.004), i, f"N={int(n)}",
                va="center", ha="left" if m >= 0.5 else "right", fontsize=7.5,
                color="#444444")
    ax.set_xlabel("BiasScore (proportion stereotypical), null = 0.500", fontsize=11)
    ax.set_title(
        "BiasScore by Source Dataset\n",
        fontsize=11, pad=10
    )
    ax.set_xlim(0.36, 0.58)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "appendix_a8_biasscore_by_source.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    plt.close()


# Figure A9: Text-condition prompt-variant robustness

def fig_a9_text_prompt_variants():
    null_c, low = "#9a9a9a", "#2166ac"

    variants = [("Natural", "gpt-4o-mini_results.csv"),
                ("Grammar", "gpt-4o-mini_grammar_results.csv"),
                ("Typical", "gpt-4o-mini_typical_results.csv")]
    vals = []
    for _, f in variants:
        d = pd.read_csv(TEXT_DIR / f)
        if "modality" in d:
            d = d[d["modality"] == "text"]
        vals.append(_bias_score(d["chose_stereotype"]))

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    bars = ax.bar([v for v, _ in variants], vals, color=low, edgecolor="white",
                  width=0.6)
    ax.axhline(0.5, color=null_c, lw=1.2, ls=(0, (5, 4)))
    ax.text(2.4, 0.502, "null = 0.500", color=null_c, fontsize=8, ha="right",
            va="bottom", style="italic")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.012, f"{v:.3f}", ha="center",
                va="top", color="white", fontsize=10, fontweight="bold")
    ax.set_ylim(0.40, 0.52)
    ax.set_ylabel("BiasScore", fontsize=11)
    ax.set_title(
        "Text Prompt-Variant Robustness\n", 
        fontsize=11, pad=10
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "appendix_a9_text_prompt_variants.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out.relative_to(ROOT)}")
    plt.close()


if __name__ == "__main__":
    print("Generating appendix figures...")
    fig_a1_flip_rate_by_wer()
    fig_a2_delta_asr_by_language()
    fig_a3_wer_distribution()
    fig_a4_dim_delta_asr()
    fig_a5_llama_decomposition()
    fig_a6_cross_model_delta_asr()
    fig_a7_prompt_variant_robustness()
    fig_a8_biasscore_by_source()
    fig_a9_text_prompt_variants()
    print("Done.")
