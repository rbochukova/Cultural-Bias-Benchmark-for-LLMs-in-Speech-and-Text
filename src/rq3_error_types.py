"""
rq3_error_types.py
~~~~~~~~~~~~~~~~~~
RQ3: Which ASR error types predict SCM decision flips beyond WER alone?

For each item scored in both text and speech conditions we:
  1. Derive a binary outcome: flip = (chose_stereotype_text != chose_stereotype_speech)
  2. Extract error-type features from word-level alignment of reference vs transcript
  3. Fit logistic regression: flip ~ wer + error_type_features + lang + dim
  4. Report coefficient table and save a forest plot

Error-type features
───────────────────
  negation_flip    : negation token gained or lost in either sentence
  pronoun_altered  : gendered pronoun changed in either sentence
  insertion_heavy  : any sentence has >50 % of operations as word insertions
  deletion_heavy   : any sentence has >50 % of operations as word deletions
  trait_altered    : any SCM trait-cue word changed in either sentence
  wer_mean         : (wer_S + wer_A) / 2  [continuous]
  wer_asym         : wer_S − wer_A        [positive → stereotype sentence more corrupted]

Usage:
    python src/rq3_error_types.py
    python src/rq3_error_types.py --text-model gpt-4o-mini --asr-model large-v3
    python src/rq3_error_types.py --save-features
"""

import argparse
import difflib
import pathlib
import re
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT       = pathlib.Path(__file__).resolve().parent.parent
TEXT_DIR   = ROOT / "data" / "results" / "text"
SPEECH_DIR = ROOT / "data" / "results" / "speech"
STIMULI    = ROOT / "data" / "stimuli_seed.csv"
FIGURES    = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


# ── Negation tokens by language ───────────────────────────────────────────────

NEGATION: dict[str, set[str]] = {
    "en": {"not", "n't", "never", "no", "neither", "nor",
           "nobody", "nothing", "nowhere", "without"},
    "fr": {"pas", "jamais", "rien", "plus", "ni", "guère",
           "nullement", "aucun", "aucune", "sans"},
    "bg": {"не", "няма", "никога", "нито", "никой", "нищо", "никъде"},
}

# ── Gendered pronoun sets by language ─────────────────────────────────────────

PRONOUNS: dict[str, dict[str, set[str]]] = {
    "en": {
        "masc": {"he", "him", "his", "himself"},
        "fem":  {"she", "her", "hers", "herself"},
    },
    "fr": {
        "masc": {"il", "son", "lui-même"},
        "fem":  {"elle", "sa", "elle-même"},
    },
    "bg": {
        "masc": {"той", "него", "му", "негов", "неговата", "неговото",
                 "неговите", "самият"},
        "fem":  {"тя", "нея", "й", "нейн", "нейната", "нейното",
                 "нейните", "самата"},
    },
}

# ── SCM trait cue words by language × dimension ───────────────────────────────

TRAIT_WORDS: dict[str, dict[str, set[str]]] = {
    "en": {
        "warmth": {
            "friendly", "warm", "kind", "caring", "sincere", "honest",
            "trustworthy", "gentle", "compassionate", "loving", "generous",
            "sympathetic", "helpful", "supportive", "nurturing",
        },
        "competence": {
            "capable", "skilled", "intelligent", "competent", "efficient",
            "qualified", "expert", "professional", "smart", "talented",
            "knowledgeable", "successful", "analytical", "logical",
        },
    },
    "fr": {
        "warmth": {
            "chaleureux", "chaleureuse", "aimable", "gentil", "gentille",
            "sincère", "honnête", "bienveillant", "bienveillante",
            "généreux", "généreuse", "doux", "douce", "attentionné",
            "attentionnée",
        },
        "competence": {
            "compétent", "compétente", "qualifié", "qualifiée", "intelligent",
            "intelligente", "efficace", "expert", "experte", "professionnel",
            "professionnelle", "capable", "talentueux", "talentueuse",
        },
    },
    "bg": {
        "warmth": {
            "топъл", "топла", "добър", "добра", "мил", "мила",
            "грижовен", "грижовна", "искрен", "искрена", "честен",
            "честна", "нежен", "нежна", "щедър", "щедра",
        },
        "competence": {
            "компетентен", "компетентна", "квалифициран", "квалифицирана",
            "умен", "умна", "способен", "способна", "опитен", "опитна",
            "успешен", "успешна", "талантлив", "талантлива",
        },
    },
}


# ── Word-alignment helpers ────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace (punctuation stripped)."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _word_ops(ref: str, hyp: str) -> tuple[int, int, int]:
    """
    Word-level alignment via difflib.SequenceMatcher.
    Returns (n_sub, n_ins, n_del).
    A 'replace' block is counted as max(deleted_words, inserted_words).
    """
    ref_words = _tokenize(ref)
    hyp_words = _tokenize(hyp)
    n_sub = n_ins = n_del = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, ref_words, hyp_words, autojunk=False
    ).get_opcodes():
        if tag == "replace":
            n_sub += max(i2 - i1, j2 - j1)
        elif tag == "insert":
            n_ins += j2 - j1
        elif tag == "delete":
            n_del += i2 - i1
    return n_sub, n_ins, n_del


def _token_set(text: str, word_set: set[str]) -> set[str]:
    """Words in text that appear in word_set."""
    return set(_tokenize(text)) & word_set


def _negation_set(text: str, lang: str) -> set[str]:
    return _token_set(text, NEGATION.get(lang, set()))


def _pronoun_set(text: str, lang: str) -> set[str]:
    all_pronouns = set().union(*PRONOUNS.get(lang, {}).values())
    return _token_set(text, all_pronouns)


def _trait_set(text: str, lang: str, dimension: str) -> set[str]:
    return _token_set(text, TRAIT_WORDS.get(lang, {}).get(dimension, set()))


# ── Per-item feature extraction ───────────────────────────────────────────────

def extract_features(row: pd.Series) -> dict:
    """
    Given a merged row with reference_S, reference_A, transcript_S,
    transcript_A, wer_S, wer_A, language, dimension — return feature dict.
    """
    lang  = str(row["language"])
    dim   = str(row["dimension"])
    ref_S = str(row["reference_S"])
    ref_A = str(row["reference_A"])
    hyp_S = str(row["transcript_S"])
    hyp_A = str(row["transcript_A"])

    # Word operation counts per sentence
    sub_S, ins_S, del_S = _word_ops(ref_S, hyp_S)
    sub_A, ins_A, del_A = _word_ops(ref_A, hyp_A)
    total_S = sub_S + ins_S + del_S
    total_A = sub_A + ins_A + del_A

    # Insertion-heavy: any sentence >50 % insertions
    ins_frac_S = ins_S / total_S if total_S > 0 else 0.0
    ins_frac_A = ins_A / total_A if total_A > 0 else 0.0

    # Deletion-heavy: any sentence >50 % deletions
    del_frac_S = del_S / total_S if total_S > 0 else 0.0
    del_frac_A = del_A / total_A if total_A > 0 else 0.0

    # Negation flip: negation tokens gained or lost in either sentence
    neg_changed = (
        (_negation_set(ref_S, lang) != _negation_set(hyp_S, lang))
        or (_negation_set(ref_A, lang) != _negation_set(hyp_A, lang))
    )

    # Pronoun altered: any gendered pronoun changed in either sentence
    pro_changed = (
        (_pronoun_set(ref_S, lang) != _pronoun_set(hyp_S, lang))
        or (_pronoun_set(ref_A, lang) != _pronoun_set(hyp_A, lang))
    )

    # Trait word altered: SCM trait cue word gained or lost in either sentence
    trait_changed = (
        (_trait_set(ref_S, lang, dim) != _trait_set(hyp_S, lang, dim))
        or (_trait_set(ref_A, lang, dim) != _trait_set(hyp_A, lang, dim))
    )

    wer_S_val = float(row["wer_S"])
    wer_A_val = float(row["wer_A"])

    return {
        "wer_mean":        (wer_S_val + wer_A_val) / 2,
        "wer_max":         max(wer_S_val, wer_A_val),
        "wer_asym":        wer_S_val - wer_A_val,
        "negation_flip":   int(neg_changed),
        "pronoun_altered": int(pro_changed),
        "insertion_heavy": int(ins_frac_S > 0.5 or ins_frac_A > 0.5),
        "deletion_heavy":  int(del_frac_S > 0.5 or del_frac_A > 0.5),
        "trait_altered":   int(trait_changed),
        "n_ops_S":         total_S,
        "n_ops_A":         total_A,
    }


# ── Logistic regression ───────────────────────────────────────────────────────

def run_logit(df: pd.DataFrame, formula: str, label: str):
    import statsmodels.formula.api as smf

    model = smf.logit(formula, data=df).fit(disp=False, maxiter=300)
    n_events = int(df["flip"].sum())
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"  N = {len(df)},  events = {n_events}  ({100*n_events/len(df):.1f}%)")
    print(f"  Log-likelihood = {model.llf:.2f},  AIC = {model.aic:.1f}")
    print(f"{'─' * 60}")

    tbl = model.summary2().tables[1].copy()
    # Rename columns for readability
    tbl.columns = ["coef", "std_err", "z", "p", "CI_lo", "CI_hi"]
    tbl["OR"]     = np.exp(tbl["coef"])
    tbl["OR_lo"]  = np.exp(tbl["CI_lo"])
    tbl["OR_hi"]  = np.exp(tbl["CI_hi"])
    tbl["sig"]    = tbl["p"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )
    print(tbl[["coef", "OR", "OR_lo", "OR_hi", "p", "sig"]].to_string(float_format="{:.4f}".format))
    return model


def likelihood_ratio_test(m0, m1) -> None:
    from scipy.stats import chi2 as chi2_dist
    lr_stat = 2 * (m1.llf - m0.llf)
    df_diff = int(m1.df_model - m0.df_model)
    lr_p    = float(1 - chi2_dist.cdf(lr_stat, df=df_diff))
    print(f"\nLikelihood ratio test (extended vs base):")
    print(f"  χ²({df_diff}) = {lr_stat:.3f},  p = {lr_p:.4f}")
    delta_aic = m1.aic - m0.aic
    print(f"  ΔAIC = {delta_aic:+.1f}  ({'extended better' if delta_aic < 0 else 'base better'})")


# ── Forest plot ───────────────────────────────────────────────────────────────

PREDICTOR_LABELS = {
    "wer_mean":        "WER mean (S+A)/2",
    "wer_asym":        "WER asymmetry (S−A)",
    "negation_flip":   "Negation flip",
    "pronoun_altered": "Pronoun altered",
    "insertion_heavy": "Insertion-heavy",
    "deletion_heavy":  "Deletion-heavy",
    "trait_altered":   "Trait word altered",
}


def fig_logreg(model, save_path: pathlib.Path) -> None:
    """Forest plot of odds ratios (95 % CI) for core predictors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    params = model.params
    conf   = model.conf_int()
    pvals  = model.pvalues

    keep = [n for n in params.index if n in PREDICTOR_LABELS]
    if not keep:
        print("WARNING: no recognised predictor names in model; skipping figure.")
        return

    ors = np.exp(params[keep])
    lo  = np.exp(conf.loc[keep, 0])
    hi  = np.exp(conf.loc[keep, 1])
    pv  = pvals[keep]

    # Drop predictors with infinite/NaN OR (e.g. complete separation)
    finite_mask = np.isfinite(ors.values) & np.isfinite(lo.values) & np.isfinite(hi.values)
    dropped = [k for k, f in zip(keep, finite_mask) if not f]
    if dropped:
        print(f"NOTE: excluded from forest plot (non-finite OR): {dropped}")
    keep = [k for k, f in zip(keep, finite_mask) if f]
    ors  = ors[keep]
    lo   = lo[keep]
    hi   = hi[keep]
    pv   = pv[keep]

    if not keep:
        print("WARNING: no plottable predictors; skipping figure.")
        return

    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(keep) + 1.4))
    y = np.arange(len(keep))

    colors = ["#d62728" if p < 0.05 else "#aec7e8" for p in pv]
    ax.barh(y, ors.values - 1, left=1, height=0.5, color=colors, alpha=0.85)
    ax.errorbar(
        ors.values, y,
        xerr=[ors.values - lo.values, hi.values - ors.values],
        fmt="none", color="black", capsize=3, linewidth=1.2,
    )
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)

    x_max = max(hi.values) * 1.18
    for i, (o, h, p) in enumerate(zip(ors.values, hi.values, pv)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(h + (x_max - 1) * 0.03, i, f"{o:.2f}{sig}", va="center", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels([PREDICTOR_LABELS.get(k, k) for k in keep])
    ax.set_xlabel("Odds ratio (95 % CI)", fontsize=9)
    ax.set_title("RQ3 — Predictors of speech-vs-text decision flip", fontsize=10, pad=8)
    ax.set_xlim(left=max(0, min(lo.values) * 0.8), right=x_max)

    sig_patch = mpatches.Patch(color="#d62728", alpha=0.85, label="p < .05")
    ns_patch  = mpatches.Patch(color="#aec7e8", alpha=0.85, label="n.s.")
    ax.legend(handles=[sig_patch, ns_patch], loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {save_path.relative_to(ROOT)}")


# ── Descriptive tables ────────────────────────────────────────────────────────

def print_flip_by_cell(df: pd.DataFrame) -> None:
    print("\nFlip rate by language × dimension:")
    print(f"  {'lang':<5}  {'dim':<12}  {'N':>6}  {'flips':>6}  {'flip_rate':>10}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*10}")
    for (lang, dim), grp in df.groupby(["language", "dimension"]):
        flips = int(grp["flip"].sum())
        print(f"  {lang:<5}  {dim:<12}  {len(grp):>6}  {flips:>6}  {grp['flip'].mean():>10.3f}")


def print_flip_by_wer_bin(df: pd.DataFrame) -> None:
    bins   = [-0.0001, 0.0001, 0.10, 0.30, 1.01]
    labels = ["WER = 0", "0 < WER ≤ 0.10", "0.10 < WER ≤ 0.30", "WER > 0.30"]
    df = df.copy()
    df["wer_bin"] = pd.cut(df["wer_mean"], bins=bins, labels=labels)
    print("\nFlip rate by WER bin (wer_mean):")
    print(f"  {'bin':<22}  {'N':>6}  {'flips':>6}  {'flip_rate':>10}")
    print(f"  {'─'*22}  {'─'*6}  {'─'*6}  {'─'*10}")
    for lab, grp in df.groupby("wer_bin", observed=True):
        if len(grp) == 0:
            continue
        print(f"  {lab:<22}  {len(grp):>6}  {int(grp['flip'].sum()):>6}  {grp['flip'].mean():>10.3f}")


def print_feature_prevalence(df_err: pd.DataFrame) -> None:
    features = ["negation_flip", "pronoun_altered", "insertion_heavy",
                "deletion_heavy", "trait_altered"]
    print("\nError-type prevalence among items with wer_max > 0:")
    print(f"  {'feature':<22}  {'N':>5}  {'%':>6}  {'flip_rate':>10}")
    print(f"  {'─'*22}  {'─'*5}  {'─'*6}  {'─'*10}")
    for feat in features:
        n   = int(df_err[feat].sum())
        pct = 100 * n / len(df_err) if len(df_err) > 0 else 0.0
        sub = df_err[df_err[feat] == 1]
        fr  = sub["flip"].mean() if len(sub) > 0 else float("nan")
        fr_str = f"{fr:>10.3f}" if not pd.isna(fr) else f"{'—':>10}"
        print(f"  {feat:<22}  {n:>5}  {pct:>5.1f}%  {fr_str}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3 error-type logistic regression")
    parser.add_argument("--text-model",   default="gpt-4o-mini")
    parser.add_argument("--asr-model",    default="large-v3")
    parser.add_argument("--save-features", action="store_true",
                        help="Save per-item feature table to data/results/rq3_features.csv")
    args = parser.parse_args()

    safe_llm = args.text_model.replace("/", "-")
    safe_asr = args.asr_model.replace("/", "-")

    # ── Load and filter text results ──────────────────────────────────────────
    text_path = TEXT_DIR / f"{safe_llm}_results.csv"
    if not text_path.exists():
        sys.exit(f"ERROR: {text_path} not found.")
    text_df = pd.read_csv(text_path, encoding="utf-8")

    # Keep natural prompt variant only
    if "prompt_variant" in text_df.columns:
        text_df = text_df[
            text_df["prompt_variant"].isna()
            | (text_df["prompt_variant"] == "natural")
        ]

    # ── Load and filter speech results ────────────────────────────────────────
    speech_path = SPEECH_DIR / f"{safe_asr}_{safe_llm}_results.csv"
    if not speech_path.exists():
        sys.exit(f"ERROR: {speech_path} not found.")
    speech_df = pd.read_csv(speech_path, encoding="utf-8")

    if "prompt_variant" in speech_df.columns:
        speech_df = speech_df[
            speech_df["prompt_variant"].isna()
            | (speech_df["prompt_variant"] == "natural")
        ]

    # ── Load reference texts from stimuli ─────────────────────────────────────
    stim = pd.read_csv(STIMULI, encoding="utf-8").set_index("item_id")
    ref  = stim[["sent_stereotype", "sent_anti_stereotype"]].rename(
        columns={"sent_stereotype": "reference_S", "sent_anti_stereotype": "reference_A"}
    )

    # ── Merge: one row per item_id, both conditions ───────────────────────────
    text_sub = (
        text_df.set_index("item_id")
        [["language", "dimension", "chose_stereotype"]]
        .rename(columns={"chose_stereotype": "chose_stereotype_text"})
    )
    speech_sub = (
        speech_df.set_index("item_id")
        [["chose_stereotype", "wer_S", "wer_A", "transcript_S", "transcript_A"]]
        .rename(columns={"chose_stereotype": "chose_stereotype_speech"})
    )

    merged = text_sub.join(speech_sub, how="inner")
    merged = merged.join(ref, how="left")

    missing_ref = merged["reference_S"].isna().sum()
    if missing_ref:
        print(f"WARNING: {missing_ref} items have no reference text — dropped.")
        merged = merged.dropna(subset=["reference_S", "reference_A"])

    merged["flip"] = (
        merged["chose_stereotype_text"] != merged["chose_stereotype_speech"]
    ).astype(int)

    print(f"\n{'=' * 60}")
    print(f"  RQ3 — ASR Error-Type Mechanism Analysis")
    print(f"{'=' * 60}")
    print(f"Items with text + speech scores : {len(merged)}")
    print(f"Decision flips                  : {int(merged['flip'].sum())}  "
          f"({100 * merged['flip'].mean():.1f} %)")

    # ── Extract error-type features ───────────────────────────────────────────
    print("\nExtracting error-type features (word-level alignment)...")
    features = merged.apply(extract_features, axis=1, result_type="expand")
    df = pd.concat([merged, features], axis=1)

    df_err = df[df["wer_max"] > 0].copy()
    print(f"Items with any ASR error (wer_max > 0) : {len(df_err)}")

    # ── Descriptive summaries ─────────────────────────────────────────────────
    print_flip_by_cell(df)
    print_flip_by_wer_bin(df)
    print_feature_prevalence(df_err)

    # ── Logistic regression ───────────────────────────────────────────────────
    # Reference levels: language = "en", dimension = "warmth"
    reg_df = df.copy()
    reg_df["language"]  = reg_df["language"].astype(str)
    reg_df["dimension"] = reg_df["dimension"].astype(str)

    ctrl = "C(language, Treatment('en')) + C(dimension, Treatment('warmth'))"
    base_formula = f"flip ~ wer_mean + wer_asym + {ctrl}"
    ext_formula  = (
        f"flip ~ wer_mean + wer_asym + negation_flip + pronoun_altered + "
        f"insertion_heavy + deletion_heavy + trait_altered + {ctrl}"
    )

    m0 = run_logit(reg_df, base_formula,  "Base model   : WER + controls")
    m1 = run_logit(reg_df, ext_formula,   "Extended model: WER + error types + controls")

    likelihood_ratio_test(m0, m1)

    # ── Forest plot ───────────────────────────────────────────────────────────
    fig_logreg(m1, FIGURES / "rq3_logreg.png")

    # ── Save feature table ────────────────────────────────────────────────────
    if args.save_features:
        save_cols = [
            "language", "dimension", "flip",
            "wer_mean", "wer_max", "wer_asym",
            "negation_flip", "pronoun_altered",
            "insertion_heavy", "deletion_heavy", "trait_altered",
            "n_ops_S", "n_ops_A",
        ]
        out = ROOT / "data" / "results" / "rq3_features.csv"
        df[save_cols].to_csv(out, encoding="utf-8")
        print(f"\nFeature table saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
