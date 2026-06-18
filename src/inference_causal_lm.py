"""
Forced-choice bias scoring using a causal base LM (no instruction tuning).
Scores each sentence by its mean per-token log-probability; the sentence
with higher log-probability is treated as the model's preference.

Usage (text condition):
    python src/inference_causal_lm.py --model mistralai/Mistral-7B-v0.1

Usage (speech condition - score ASR transcripts):
    python src/inference_causal_lm.py --model mistralai/Mistral-7B-v0.1 --asr-model large-v3
    python src/inference_causal_lm.py --model mistralai/Mistral-7B-v0.1 --asr-model medium
"""

import argparse
import os
import pathlib
import sys
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

ROOT       = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH   = ROOT / "data" / "stimuli_seed.csv"
ASR_DIR    = ROOT / "data" / "results" / "asr"
TEXT_DIR   = ROOT / "data" / "results" / "text"
SPEECH_DIR = ROOT / "data" / "results" / "speech"
TEXT_DIR.mkdir(parents=True, exist_ok=True)
SPEECH_DIR.mkdir(parents=True, exist_ok=True)

VALID_DIMS = {"warmth", "competence"}


def _sentence_logprob(model, tokenizer, text: str, device: str) -> float:
    """Mean per-token log-probability of text under the causal LM."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)
    return -outputs.loss.item()


def _load_model(model_name: str, use_4bit: bool):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    hf_token = os.environ.get("HF_TOKEN", "") or None
    print(f"Loading {model_name} ...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token,
        )

    model.eval()
    device = next(model.parameters()).device
    print(f"  Loaded on {device}", flush=True)
    return model, tokenizer, str(device)


def _score_item(
    model, tokenizer, row: pd.Series, stereo_is_A: bool, device: str,
    model_name: str, asr_model: str | None = None,
    wer_S: float | None = None, wer_A: float | None = None,
) -> dict | None:
    sent_S = str(row["sent_stereotype"]).strip()
    sent_A = str(row["sent_anti_stereotype"]).strip()

    try:
        lp_S = _sentence_logprob(model, tokenizer, sent_S, device)
        lp_A = _sentence_logprob(model, tokenizer, sent_A, device)

        chose_stereo = lp_S >= lp_A

        result = {
            "item_id":           row["item_id"],
            "language":          row["language"],
            "dimension":         row["dimension"],
            "target_group":      row["target_group"],
            "target":            row["target"],
            "origin":            row["origin"],
            "parallel_group_id": row["parallel_group_id"],
            "model":             model_name,
            "prompt_variant":    "sentence_logprob",
            "modality":          "speech" if asr_model else "text",
            "asr_system":        asr_model,
            "A_is_stereotype":   stereo_is_A,
            "logprob_S":         lp_S,
            "logprob_A":         lp_A,
            "chose_stereotype":  int(chose_stereo),
            "scored_at":         datetime.now(timezone.utc).isoformat(),
        }
        if asr_model:
            result["wer_S"] = wer_S
            result["wer_A"] = wer_A
            result["transcript_S"] = sent_S
            result["transcript_A"] = sent_A
        return result
    except Exception as exc:
        print(f"\n  FAILED {row['item_id']}: {exc}")
        return None


def _flush(results: list[dict], path: pathlib.Path) -> None:
    new_df = pd.DataFrame(results)
    if path.exists():
        existing = pd.read_csv(path, encoding="utf-8")
        new_df = new_df[~new_df["item_id"].isin(existing["item_id"])]
        if new_df.empty:
            return
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False, encoding="utf-8")


def _load_items(args) -> tuple[pd.DataFrame, pathlib.Path]:
    """Return (items_df, results_path) for text or speech condition."""
    stim = pd.read_csv(CSV_PATH, encoding="utf-8")
    validated = stim[
        stim["validated"].map(lambda x: str(x).strip().lower() in ("true", "1"))
        & stim["dimension"].isin(VALID_DIMS)
    ].copy()

    if args.lang:
        validated = validated[validated["language"] == args.lang]

    safe_model = args.model.replace("/", "-")

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
        meta = validated.set_index("item_id")[
            ["language", "dimension", "target_group", "target", "origin", "parallel_group_id"]
        ]
        items = meta.join(asr_S).join(asr_A).dropna(
            subset=["sent_stereotype", "sent_anti_stereotype"]
        ).reset_index()
        results_path = SPEECH_DIR / f"{args.asr_model}_{safe_model}_results.csv"
    else:
        items = validated
        results_path = TEXT_DIR / f"{safe_model}_results.csv"

    return items, results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal LM forced-choice bias scoring")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1",
                        help="HuggingFace model ID")
    parser.add_argument("--asr-model", default=None,
                        help="Score ASR transcripts instead of original text "
                             "(e.g. large-v3, medium). Saves to results/speech/.")
    parser.add_argument("--lang", default=None,
                        help="Filter to one language: en, fr, bg (default: all)")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (requires more GPU RAM)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first 3 items without scoring")
    args = parser.parse_args()

    items, results_path = _load_items(args)
    modality = "speech" if args.asr_model else "text"
    print(f"Stimuli: {len(items)} items  "
          f"(lang={args.lang or 'all'}, modality={modality})")

    if results_path.exists():
        done = set(pd.read_csv(results_path, encoding="utf-8")["item_id"].astype(str))
        items = items[~items["item_id"].isin(done)]
        print(f"Already scored: {len(done)}, remaining: {len(items)}")

    if args.dry_run:
        for _, row in items.head(3).iterrows():
            print(f"\n  {row['item_id']}  S: {str(row['sent_stereotype'])[:70]}")
            print(f"             A: {str(row['sent_anti_stereotype'])[:70]}")
        return

    if len(items) == 0:
        print("Nothing to score.")
        return

    use_4bit = not args.no_4bit
    model, tokenizer, device = _load_model(args.model, use_4bit)

    results    = []
    failed     = 0
    write_every = 100

    for i, (_, row) in enumerate(items.iterrows(), 1):
        wer_S = float(row["wer_S"]) if args.asr_model and "wer_S" in row else None
        wer_A = float(row["wer_A"]) if args.asr_model and "wer_A" in row else None
        result = _score_item(
            model, tokenizer, row, stereo_is_A=True, device=device,
            model_name=args.model, asr_model=args.asr_model,
            wer_S=wer_S, wer_A=wer_A,
        )
        if result:
            results.append(result)
        else:
            failed += 1

        if i % 50 == 0 or i == len(items):
            pct = 100 * i / len(items)
            print(f"\r  {i}/{len(items)} ({pct:.0f}%)  scored={len(results)}  failed={failed}",
                  end="", flush=True)

        if len(results) % write_every == 0 and results:
            _flush(results, results_path)

    print()
    _flush(results, results_path)

    print(f"Done. Results: {results_path.relative_to(ROOT)}")
    print(f"  Scored: {len(results)}  Failed: {failed}")

    if results:
        rdf = pd.read_csv(results_path, encoding="utf-8")
        bs  = rdf["chose_stereotype"].mean()
        print(f"\nOverall BiasScore: {bs:.3f}  (null=0.500)")
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
