"""
Text-condition forced-choice bias inference using mDeBERTa-v3-base.

Scoring method - Pseudo-Log-Likelihood (PLL): For each sentence, we compute PLL = sum_i log P(token_i | all other tokens). This is done by masking each token in turn and reading the masked-LM head's output probability for the original token at that position. The sentence with the higher PLL is judged 'more natural' by the model.
"""

import argparse
import hashlib
import pathlib
import sys
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

ROOT       = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH   = ROOT / "data" / "stimuli_seed.csv"
ASR_DIR    = ROOT / "data" / "results" / "asr"
TEXT_DIR   = ROOT / "data" / "results" / "text"
SPEECH_DIR = ROOT / "data" / "results" / "speech"
TEXT_DIR.mkdir(parents=True, exist_ok=True)
SPEECH_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "microsoft/mdeberta-v3-base"
VALID_DIMS = {"warmth", "competence"}


def _item_seed(item_id: str) -> int:
    return int(hashlib.md5(item_id.encode()).hexdigest(), 16) % (2 ** 31)


def _load_model(device: str):
    """Load mDeBERTa-v3 tokenizer and model."""
    try:
        import torch
        from transformers import DebertaV2Tokenizer, DebertaV2ForMaskedLM
    except ImportError:
        sys.exit(
            "ERROR: transformers and torch are required.\n"
            "Install: pip install torch transformers sentencepiece"
        )

    print(f"Loading {MODEL_NAME} ...", flush=True)
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    model     = DebertaV2ForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  Model loaded on {device}", flush=True)
    return tokenizer, model, device


def _pll(text: str, tokenizer, model, device: str) -> float:
    """
    Compute the Pseudo-Log-Likelihood for text under a masked LM.
    Returns the mean per-token log probability (sum / n_tokens).
    Returns -inf on error.
    """
    import torch
    import torch.nn.functional as F

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)      
    n_tokens  = input_ids.shape[1]

    token_positions = list(range(1, n_tokens - 1))
    if not token_positions:
        return float("-inf")

    total_log_prob = 0.0
    with torch.no_grad():
        for pos in token_positions:
            masked_ids       = input_ids.clone()
            masked_ids[0, pos] = tokenizer.mask_token_id
            outputs          = model(
                input_ids=masked_ids,
                attention_mask=inputs["attention_mask"].to(device),
            )
            logits_pos = outputs.logits[0, pos]            
            log_probs  = F.log_softmax(logits_pos, dim=-1)
            orig_token = input_ids[0, pos].item()
            total_log_prob += log_probs[orig_token].item()

    return total_log_prob / len(token_positions)  


def _score_item(
    item_id: str,
    lang: str,
    text_stereo: str,
    text_anti: str,
    stereo_is_A: bool,
    meta: dict,
    tokenizer,
    model,
    device: str,
) -> dict | None:
    try:
        pll_stereo = _pll(text_stereo, tokenizer, model, device)
        pll_anti   = _pll(text_anti,   tokenizer, model, device)

        if pll_stereo == float("-inf") and pll_anti == float("-inf"):
            return None

        # Map to A/B framework to keep output schema consistent 
        pll_A = pll_stereo if stereo_is_A else pll_anti
        pll_B = pll_anti   if stereo_is_A else pll_stereo

        chose_A      = pll_A >= pll_B
        chose_stereo = chose_A if stereo_is_A else not chose_A

        result = {
            "item_id":           item_id,
            "language":          lang,
            "dimension":         meta["dimension"],
            "target_group":      meta["target_group"],
            "target":            meta["target"],
            "origin":            meta["origin"],
            "parallel_group_id": meta["parallel_group_id"],
            "model":             MODEL_NAME.split("/")[-1],
            "prompt_variant":    "pll",
            "modality":          "speech" if meta.get("asr_system") else "text",
            "asr_system":        meta.get("asr_system"),
            "A_is_stereotype":   stereo_is_A,
            "logprob_A":         round(pll_A, 6),
            "logprob_B":         round(pll_B, 6),
            "chose_A":           chose_A,
            "chose_stereotype":  chose_stereo,
            "scored_at":         datetime.now(timezone.utc).isoformat(),
        }
        if meta.get("asr_system"):
            result["wer_S"] = meta.get("wer_S")
            result["wer_A"] = meta.get("wer_A")
            result["transcript_S"] = text_stereo
            result["transcript_A"] = text_anti
        return result
    except Exception as exc:
        print(f"\n    ERROR scoring {item_id}: {exc}", flush=True)
        return None


def _load_existing(results_path: pathlib.Path) -> set:
    if not results_path.exists():
        return set()
    try:
        return set(pd.read_csv(results_path, encoding="utf-8")["item_id"].astype(str))
    except Exception:
        return set()


def _flush(results: list, path: pathlib.Path) -> None:
    new_df = pd.DataFrame(results)
    if path.exists():
        existing = pd.read_csv(path, encoding="utf-8")
        new_df   = new_df[~new_df["item_id"].isin(existing["item_id"])]
        if new_df.empty:
            return
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="mDeBERTa-v3 PLL bias inference")
    parser.add_argument("--lang",       default=None,
                        help="Filter to one language: en, fr, bg (default: all)")
    parser.add_argument("--asr-model",  default=None,
                        help="Score ASR transcripts instead of original text "
                             "(e.g. large-v3, medium). Saves to results/speech/.")
    parser.add_argument("--device",     default="auto",
                        help="Torch device: auto (default), cpu, cuda")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Reserved for future batching; currently 1 (PLL is per-token)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print PLL for first 3 items and exit")
    args = parser.parse_args()

    stim = pd.read_csv(CSV_PATH, encoding="utf-8")
    validated = stim[
        stim["validated"].map(lambda x: str(x).strip().lower() in ("true", "1"))
        & stim["dimension"].isin(VALID_DIMS)
    ].copy()

    if args.lang:
        validated = validated[validated["language"] == args.lang]

    safe_model = MODEL_NAME.replace("/", "-")

    if args.asr_model:
        asr_path = ASR_DIR / f"{args.asr_model}_transcripts.csv"
        if not asr_path.exists():
            sys.exit(
                f"ERROR: ASR transcripts not found: {asr_path}\n"
                f"Run: python src/asr.py --model {args.asr_model}"
            )
        asr_df = pd.read_csv(asr_path, encoding="utf-8")
        asr_S = (asr_df[asr_df["suffix"] == "S"]
                 .set_index("item_id")[["transcript", "wer"]]
                 .rename(columns={"transcript": "sent_stereotype", "wer": "wer_S"}))
        asr_A = (asr_df[asr_df["suffix"] == "A"]
                 .set_index("item_id")[["transcript", "wer"]]
                 .rename(columns={"transcript": "sent_anti_stereotype", "wer": "wer_A"}))
        meta_cols = validated.set_index("item_id")[
            ["language", "dimension", "target_group", "target", "origin", "parallel_group_id"]
        ]
        items = meta_cols.join(asr_S).join(asr_A).dropna(
            subset=["sent_stereotype", "sent_anti_stereotype"]
        ).reset_index()
        results_path = SPEECH_DIR / f"{args.asr_model}_{safe_model}_results.csv"
    else:
        items = validated
        results_path = TEXT_DIR / f"{safe_model}_results.csv"

    items["_stereo_is_A"] = items["item_id"].apply(
        lambda iid: _item_seed(iid) % 2 == 0
    )

    done     = _load_existing(results_path)
    to_score = items[~items["item_id"].isin(done)]

    tokenizer, model, device = _load_model(args.device)

    if args.dry_run:
        for _, row in to_score.head(3).iterrows():
            s_text = str(row["sent_stereotype"]).strip()
            a_text = str(row["sent_anti_stereotype"]).strip()
            pll_s  = _pll(s_text, tokenizer, model, device)
            pll_a  = _pll(a_text, tokenizer, model, device)
            chose  = "STEREO" if pll_s >= pll_a else "ANTI"
            print(f"\n[{row['item_id']}] lang={row['language']} dim={row['dimension']}")
            print(f"  PLL(stereo)={pll_s:.4f}  PLL(anti)={pll_a:.4f}  -> {chose}")
            print(f"  S: {s_text[:70]}")
            print(f"  A: {a_text[:70]}")
        return

    if len(to_score) == 0:
        print("Nothing to score. Done.")
        return

    results = []
    failed  = 0

    for i, (_, row) in enumerate(to_score.iterrows(), 1):
        meta = {
            "dimension":         row["dimension"],
            "target_group":      row["target_group"],
            "target":            row["target"],
            "origin":            row["origin"],
            "parallel_group_id": row["parallel_group_id"],
            "asr_system":        args.asr_model,
            "wer_S":             float(row["wer_S"]) if args.asr_model and "wer_S" in row else None,
            "wer_A":             float(row["wer_A"]) if args.asr_model and "wer_A" in row else None,
        }
        result = _score_item(
            row["item_id"], str(row["language"]),
            str(row["sent_stereotype"]).strip(),
            str(row["sent_anti_stereotype"]).strip(),
            bool(row["_stereo_is_A"]),
            meta, tokenizer, model, device,
        )
        if result:
            results.append(result)
        else:
            failed += 1

        if i % 10 == 0 or i == len(to_score):
            pct = 100 * i / len(to_score)
            print(f"\r  {i}/{len(to_score)} ({pct:.0f}%)  "
                  f"scored={len(results)}  failed={failed}", end="", flush=True)

        if len(results) % 100 == 0 and results:
            _flush(results, results_path)

    print()
    _flush(results, results_path)

    print(f"Done. Results: {results_path.relative_to(ROOT)}")
    print(f"  Scored : {len(results)}")
    print(f"  Failed : {failed}")

    if results:
        rdf        = pd.read_csv(results_path, encoding="utf-8")
        bias_score = rdf["chose_stereotype"].mean()
        print(f"\nOverall BiasScore (mDeBERTa PLL): {bias_score:.3f}")
        print("\nBiasScore by language x dimension:")
        print(
            rdf.groupby(["language", "dimension"])["chose_stereotype"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "BiasScore", "count": "N"})
            .round(3)
            .to_string()
        )


if __name__ == "__main__":
    main()
