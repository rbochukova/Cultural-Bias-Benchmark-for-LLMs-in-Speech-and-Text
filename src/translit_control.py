"""Bulgarian script-control experiment 

Tests whether Mistral-7B-v0.1's strongly anti-stereotypical signal on Bulgarian native nationality items (BiasScore = 0.142) is an artefact of Cyrillic sub-word tokenisation. Each BG sentence is romanised and re-scored with the same mean-per-token-log-probability method as src/inference_causal_lm.py.

Run:
    pip install torch transformers bitsandbytes accelerate
    python src/translit_control.py
"""
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "stimuli_seed.csv"
MODEL = "mistralai/Mistral-7B-v0.1"

# Bulgarian  Cyrillic -> Latin romanisation.
_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
    "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
    "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sht", "ъ": "a",
    "ь": "y", "ю": "yu", "я": "ya",
}


def romanise(text: str) -> str:
    out = []
    for ch in str(text):
        low = ch.lower()
        rep = _MAP.get(low, ch)
        if ch.isupper() and rep:
            rep = rep[0].upper() + rep[1:]
        out.append(rep)
    return "".join(out)


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    stim = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    stim = stim[stim["validated"].map(lambda x: str(x).strip().lower() in ("true", "1"))]
    bg = stim[(stim["language"] == "bg")
              & (stim["origin"] == "native")
              & (stim["target_group"] == "nationality")].copy()
    print(f"BG native-nationality items: {len(bg)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                 device_map="auto")
    model.eval()

    @torch.no_grad()
    def mean_logprob(text: str) -> float:
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        if ids.shape[1] < 2:
            return float("-inf")
        out = model(ids, labels=ids)

        return float(-out.loss.item())

    def bias_score(df: pd.DataFrame, transform) -> float:
        chose = []
        for _, r in df.iterrows():
            lp_s = mean_logprob(transform(r["sent_stereotype"]))
            lp_a = mean_logprob(transform(r["sent_anti_stereotype"]))
            chose.append(int(lp_s >= lp_a))
        return float(np.mean(chose))

    orig = bias_score(bg, lambda s: str(s))
    rom = bias_score(bg, romanise)
    print(f"BiasScore (original Cyrillic) : {orig:.3f}")
    print(f"BiasScore (romanised Latin)   : {rom:.3f}")
    print(f"Delta                         : {rom - orig:+.3f}")

if __name__ == "__main__":
    main()
