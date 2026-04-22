"""
Transcribes audio stimuli with Whisper and records per-item WER/CER. Processes all .wav files in data/audio/ that match the stimuli in stimuli_seed.csv. 
WER and CER are computed per sentence using `jiwer` comparing the ASR transcript against the original stimulus text. These per-item error rates are used in the attribution analysis to explain text-speech BiasScore differences.
"""

import argparse
import pathlib
import sys
import time
from datetime import datetime, timezone

import pandas as pd

ROOT         = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH     = ROOT / "data" / "stimuli_seed.csv"
AUDIO_DIR    = ROOT / "data" / "audio"
RESULTS_DIR  = ROOT / "data" / "results" / "asr"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VALID_DIMS = {"warmth", "competence"}

WHISPER_LANG = {"en": "english", "fr": "french", "bg": "bulgarian"}


def _load_whisper(model_name: str):
    try:
        import whisper
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        return whisper.load_model(model_name, device=device)
    except ImportError:
        sys.exit(
            "ERROR: openai-whisper not installed.\n"
            "Run: pip install openai-whisper"
        )


def _transcribe(model, audio_path: pathlib.Path, lang_code: str) -> str:
    """Return transcript string for one audio file."""
    import whisper
    import torch
    result = model.transcribe(
        str(audio_path),
        language=WHISPER_LANG.get(lang_code, None),
        fp16=torch.cuda.is_available(),
    )
    return result["text"].strip()


def _compute_wer(reference: str, hypothesis: str) -> tuple[float, float]:
    """Return (WER, CER) for a reference/hypothesis pair"""
    try:
        import jiwer
        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)
        return round(wer, 4), round(cer, 4)
    except ImportError:
        return float("nan"), float("nan")


def _load_existing(results_path: pathlib.Path) -> set[tuple[str, str]]:
    """Return set of (item_id, suffix) already transcribed"""
    if not results_path.exists():
        return set()
    try:
        existing = pd.read_csv(results_path, encoding="utf-8")
        return set(zip(existing["item_id"].astype(str), existing["suffix"].astype(str)))
    except Exception:
        return set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper ASR transcription")
    parser.add_argument("--model", default="large-v3",
                        help="Whisper model name (default: large-v3)")
    parser.add_argument("--lang",  default=None,
                        help="Filter to one language: en, fr, bg (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check file availability only, no inference")
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    items = df[
        df["validated"].map(lambda x: str(x).strip().lower() in ("true", "1"))
        & df["dimension"].isin(VALID_DIMS)
    ].copy()
    if args.lang:
        items = items[items["language"] == args.lang]

    tasks = []
    for _, row in items.iterrows():
        iid  = str(row["item_id"])
        lang = str(row["language"])
        for sfx, col in [("S", "sent_stereotype"), ("A", "sent_anti_stereotype")]:
            wav = AUDIO_DIR / f"{iid}_{sfx}.wav"
            tasks.append({
                "item_id":   iid,
                "suffix":    sfx,
                "language":  lang,
                "reference": str(row[col]).strip(),
                "wav":       wav,
                "exists":    wav.exists(),
            })

    missing = [t for t in tasks if not t["exists"]]
    if missing:
        print(f"WARNING: {len(missing)} audio files not found - run tts.py first.")
        if args.dry_run:
            for t in missing[:10]:
                print(f"  missing: {t['wav'].name}")
            return

    tasks = [t for t in tasks if t["exists"]]

    safe_model   = args.model.replace("/", "-")
    results_path = RESULTS_DIR / f"{safe_model}_transcripts.csv"
    done         = _load_existing(results_path)
    pending      = [t for t in tasks if (t["item_id"], t["suffix"]) not in done]

    if args.dry_run:
        for t in pending[:5]:
            print(f"  {t['wav'].name}  ref: {t['reference'][:60]}")
        return

    if not pending:
        print("Nothing to transcribe. Done.")
        return

    print(f"\nLoading Whisper {args.model}")
    model = _load_whisper(args.model)
    print("Model loaded.\n")

    results  = []
    failed   = 0
    write_every = 50

    for i, t in enumerate(pending, 1):
        try:
            transcript = _transcribe(model, t["wav"], t["language"])
            wer, cer   = _compute_wer(t["reference"], transcript)
            results.append({
                "item_id":        t["item_id"],
                "suffix":         t["suffix"],
                "language":       t["language"],
                "whisper_model":  args.model,
                "reference_text": t["reference"],
                "transcript":     transcript,
                "wer":            wer,
                "cer":            cer,
                "transcribed_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            failed += 1
            print(f"\n    FAILED {t['wav'].name}: {exc}", end=" ")

        if i % 10 == 0 or i == len(pending):
            pct = 100 * i / len(pending)
            print(f"\r  {i}/{len(pending)} ({pct:.0f}%)  "
                  f"done={len(results)}  failed={failed}", end="", flush=True)

        if len(results) % write_every == 0 and results:
            _flush(results, results_path)

    print()
    _flush(results, results_path)

    print(f"\n{'=' * 55}")
    print(f"Done. Results: {results_path.relative_to(ROOT)}")
    if results:
        rdf = pd.read_csv(results_path, encoding="utf-8")
        print(f"  Transcribed : {len(rdf)}")
        print(f"  Mean WER    : {rdf['wer'].mean():.3f}")
        print(f"  Mean CER    : {rdf['cer'].mean():.3f}")
        print(f"  WER=0 items : {(rdf['wer']==0).sum()}  "
              f"({100*(rdf['wer']==0).mean():.1f}% perfect)")


def _flush(results: list[dict], path: pathlib.Path) -> None:
    if not results:
        return
    new_df = pd.DataFrame(results)
    if path.exists():
        try:
            existing = pd.read_csv(path, encoding="utf-8")
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        if existing.empty:
            combined = new_df
            combined.to_csv(path, index=False, encoding="utf-8")
            return
        done_keys = set(zip(existing["item_id"].astype(str), existing["suffix"].astype(str)))
        new_df = new_df[
            ~new_df.apply(lambda r: (str(r["item_id"]), str(r["suffix"])) in done_keys, axis=1)
        ]
        if new_df.empty:
            return
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
