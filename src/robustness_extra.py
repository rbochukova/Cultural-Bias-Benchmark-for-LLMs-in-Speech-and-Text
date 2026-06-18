"""Additional robustness analyses
Surface-controlled origin effect: does native-vs-parallel survive controlling for sentence length and subword fertility? (closes the cultural-vs-linguistic point)
BiasScore by source dataset (is the effect driven by one source?)
Text-condition prompt-variant robustness (natural/grammar/typical).
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import statsmodels.api as sm
import statsmodels.formula.api as smf
from transformers import AutoTokenizer

RES = "data/results/text/"


def load(f):
    d = pd.read_csv(RES + f)
    return d[d["modality"] == "text"] if "modality" in d else d


# per-item surface features (length, subword fertility) from Mistral tokeniser
stim = pd.read_csv("data/stimuli_seed.csv")
stim = stim[stim["validated"] == True].copy()  # noqa: E712
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")


def feats(s):
    s = str(s)
    nw = len(s.split())
    if nw == 0:
        return (np.nan, np.nan)
    return (nw, len(tok.encode(s, add_special_tokens=False)) / nw)


f1 = stim["sent_stereotype"].map(feats)
f2 = stim["sent_anti_stereotype"].map(feats)
stim["length"] = [(a[0] + b[0]) / 2 for a, b in zip(f1, f2)]
stim["fertility"] = [(a[1] + b[1]) / 2 for a, b in zip(f1, f2)]
meta = stim[["item_id", "source", "length", "fertility"]]

print("Surface-controlled origin effect (GEE logistic, clustered)")
for name, f in [("GPT-4o-mini", "gpt-4o-mini_results.csv"),
                ("Mistral-7B-v0.1", "mistralai-Mistral-7B-v0.1_results.csv")]:
    d = load(f).merge(meta, on="item_id").dropna(subset=["origin", "length", "fertility"])
    d["y"] = d["chose_stereotype"].astype(int)
    d["cluster"] = d["parallel_group_id"].fillna(d["item_id"])
    d["origin"] = pd.Categorical(d["origin"], ["parallel", "native"])
    d["zlen"] = (d["length"] - d["length"].mean()) / d["length"].std()
    d["zfert"] = (d["fertility"] - d["fertility"].mean()) / d["fertility"].std()
    m = smf.gee("y ~ C(origin) + zlen + zfert + C(language) + C(dimension)",
                groups="cluster", data=d, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    b, p = m.params["C(origin)[T.native]"], m.pvalues["C(origin)[T.native]"]
    print(f"{name}: native-vs-parallel OR={np.exp(b):.3f}, p={p:.4f} "
          "(controlling length + fertility)")
    print(f"length OR={np.exp(m.params['zlen']):.3f} (p={m.pvalues['zlen']:.3f}); "
          f"fertility OR={np.exp(m.params['zfert']):.3f} (p={m.pvalues['zfert']:.3f})")


print("BiasScore by source dataset (GPT-4o-mini)")
d = load("gpt-4o-mini_results.csv").merge(meta, on="item_id")
print(d.groupby("source")["chose_stereotype"].agg(["mean", "count"]).round(3))

print("Text-condition prompt-variant robustness (GPT-4o-mini)")
for label, f in [("natural", "gpt-4o-mini_results.csv"),
                 ("grammar", "gpt-4o-mini_grammar_results.csv"),
                 ("typical", "gpt-4o-mini_typical_results.csv")]:
    d = load(f)
    print(f"  {label:8s} BiasScore={d['chose_stereotype'].mean():.3f}  (N={len(d)})")
