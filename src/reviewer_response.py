"""Robustness analyses. Reproduces three checks cited in the thesis:
  Profession dominance; Cluster-robust (GEE) logistic - re-estimate of the parallel-vs-native contrast, grouped by parallel triplet; Tokenisation / low-resource - whether the BG native-nationality items are atypically fragmented.

"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RES = "data/results/text/"
MODELS = {
    "GPT-4o-mini": "gpt-4o-mini_results.csv",
    "mDeBERTa-v3": "microsoft-mdeberta-v3-base_results.csv",
    "BLOOM-7B1": "bigscience-bloom-7b1_results.csv",
    "Mistral-7B-v0.1": "mistralai-Mistral-7B-v0.1_results.csv",
    "Llama-Base": "meta-llama-Llama-3.2-3B_results.csv",
    "Llama-Instruct(prompted)": "meta-llama-Llama-3.2-3B-Instruct_instruct_results.csv",
}


def _load(fname):
    d = pd.read_csv(RES + fname)
    return d[d["modality"] == "text"] if "modality" in d else d


def profession_dominance():
    print(f"{'model':26s}{'all':>8}{'no-prof':>9}{'gender':>8}{'nat':>8}")
    for name, f in MODELS.items():
        d = _load(f)
        row = (
            d["chose_stereotype"].mean(),
            d[d.target_group != "profession"]["chose_stereotype"].mean(),
            d[d.target_group == "gender"]["chose_stereotype"].mean(),
            d[d.target_group == "nationality"]["chose_stereotype"].mean(),
        )
        print(f"{name:26s}" + "".join(f"{v:8.3f}" for v in row))


def mixed_effects():
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    for name, f in [("GPT-4o-mini", MODELS["GPT-4o-mini"]),
                    ("Mistral-7B-v0.1", MODELS["Mistral-7B-v0.1"])]:
        d = _load(f).dropna(subset=["origin"]).copy()
        d["y"] = d["chose_stereotype"].astype(int)
        d["cluster"] = d["parallel_group_id"].fillna(d["item_id"])
        d["origin"] = pd.Categorical(d["origin"], ["parallel", "native"])
        m = smf.gee("y ~ C(origin)", groups="cluster", data=d,
                    family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
        b = m.params["C(origin)[T.native]"]
        p = m.pvalues["C(origin)[T.native]"]
        print(f"{name:18s} native-vs-parallel OR = {np.exp(b):.3f}  p = {p:.4f}")


def tokenisation():
    from transformers import AutoTokenizer

    tok = None
    for cand in ("mistralai/Mistral-7B-v0.1", "xlm-roberta-base",
                 "bert-base-multilingual-cased"):
        try:
            tok = AutoTokenizer.from_pretrained(cand)
            print("tokeniser:", cand)
            break
        except Exception:
            continue
    if tok is None:
        print("no tokeniser available (offline); skipping.\n")
        return
    d = pd.read_csv("data/stimuli_seed.csv")
    d = d[d["validated"] == True].copy()  # noqa: E712

    def fert(s):
        s = str(s)
        w = len(s.split())
        return np.nan if w == 0 else len(tok.encode(s, add_special_tokens=False)) / w

    d["fert"] = d["sent_stereotype"].map(fert)
    by_lang = d.groupby("language")["fert"].mean()
    print({k: round(v, 3) for k, v in by_lang.items()})
    print(f"BG / EN ratio: {by_lang['bg'] / by_lang['en']:.2f}x")
    bg = d[d.language == "bg"]
    natl = bg[(bg.origin == "native") & (bg.target_group == "nationality")]["fert"].mean()
    rest = bg[~((bg.origin == "native") & (bg.target_group == "nationality"))]["fert"].mean()
    print(f"BG native-nationality fertility {natl:.3f} vs other BG {rest:.3f} "
          "-> not atypically fragmented.\n")


if __name__ == "__main__":
    profession_dominance()
    mixed_effects()
    tokenisation()
