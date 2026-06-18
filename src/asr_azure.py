"""
Azure Cognitive Services Speech-to-Text transcription.
Transcribes all stimulus audio files and saves WER/CER metrics,
matching the output format of asr.py (Whisper).

Saves to: data/results/asr/azure_transcripts.csv
"""

import argparse
import pathlib
import sys
import time
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from dotenv import load_dotenv
import jiwer
import os

load_dotenv()

ROOT      = pathlib.Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "audio"
STIM_CSV  = ROOT / "data" / "stimuli_seed.csv"
ASR_DIR   = ROOT / "data" / "results" / "asr"
ASR_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH  = ASR_DIR / "azure_transcripts.csv"

LANG_MAP = {
    "en": "en-US",
    "fr": "fr-FR",
    "bg": "bg-BG",
}


def _transcribe_azure(wav_path: pathlib.Path, lang_code: str,
                       speech_key: str, speech_region: str) -> str:
    import azure.cognitiveservices.speech as speechsdk

    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key, region=speech_region
    )
    speech_config.speech_recognition_language = lang_code
    audio_config = speechsdk.AudioConfig(filename=str(wav_path))
    recognizer  = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )
    result = recognizer.recognize_once()

    if result.reason.name == "RecognizedSpeech":
        return result.text.strip()
    return ""


def _compute_metrics(reference: str, hypothesis: str) -> tuple[float, float]:
    if not reference.strip():
        return 0.0, 0.0
    try:
        wer = jiwer.wer(reference, hypothesis)
    except Exception:
        wer = 1.0
    try:
        cer = jiwer.cer(reference, hypothesis)
    except Exception:
        cer = 1.0
    return min(wer, 1.0), min(cer, 1.0)


def _load_existing() -> set[tuple[str, str]]:
    if not OUT_PATH.exists():
        return set()
    df = pd.read_csv(OUT_PATH, encoding="utf-8")
    return set(zip(df["item_id"].astype(str), df["suffix"].astype(str)))


def _flush(rows: list[dict]) -> None:
    new_df = pd.DataFrame(rows)
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH, encoding="utf-8")
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(OUT_PATH, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Azure STT transcription")
    parser.add_argument("--lang",    default=None, help="Filter: en, fr, bg")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    speech_key    = os.environ.get("AZURE_SPEECH_KEY", "")
    speech_region = os.environ.get("AZURE_SPEECH_REGION", "westeurope")

    if not speech_key and not args.dry_run:
        sys.exit("ERROR: AZURE_SPEECH_KEY not set in .env")

    stim = pd.read_csv(STIM_CSV, encoding="utf-8")
    stim = stim[stim["validated"].map(
        lambda x: str(x).strip().lower() in ("true", "1")
    )].copy()

    if args.lang:
        stim = stim[stim["language"] == args.lang]

    done = _load_existing()
    rows = []
    failed = 0
    total_tasks = len(stim) * 2

    print(f"Items: {len(stim)}  Tasks (S+A): {total_tasks}  "
          f"Already done: {len(done)}", flush=True)

    if args.dry_run:
        for _, row in stim.head(2).iterrows():
            for suffix in ["S", "A"]:
                wav = AUDIO_DIR / f"{row['item_id']}_{suffix}.wav"
                ref = row["sent_stereotype"] if suffix == "S" else row["sent_anti_stereotype"]
                lang_code = LANG_MAP.get(row["language"], "en-US")
                print(f"  [{row['item_id']}_{suffix}] {lang_code}  "
                      f"wav={'exists' if wav.exists() else 'MISSING'}")
                print(f"    REF: {str(ref)[:70]}")
        return

    i = 0
    for _, row in stim.iterrows():
        for suffix in ["S", "A"]:
            key = (str(row["item_id"]), suffix)
            if key in done:
                continue

            wav_path = AUDIO_DIR / f"{row['item_id']}_{suffix}.wav"
            if not wav_path.exists():
                failed += 1
                continue

            reference = str(
                row["sent_stereotype"] if suffix == "S"
                else row["sent_anti_stereotype"]
            ).strip()

            lang_code = LANG_MAP.get(str(row["language"]), "en-US")

            for attempt in range(3):
                try:
                    transcript = _transcribe_azure(
                        wav_path, lang_code, speech_key, speech_region
                    )
                    wer, cer = _compute_metrics(reference, transcript)
                    rows.append({
                        "item_id":       row["item_id"],
                        "suffix":        suffix,
                        "language":      row["language"],
                        "whisper_model": "azure",
                        "reference_text": reference,
                        "transcript":    transcript,
                        "wer":           round(wer, 6),
                        "cer":           round(cer, 6),
                        "transcribed_at": datetime.now(timezone.utc).isoformat(),
                    })
                    break
                except Exception as exc:
                    if attempt == 2:
                        print(f"\n  FAILED {row['item_id']}_{suffix}: {exc}")
                        failed += 1
                    else:
                        time.sleep(2 ** attempt)

            i += 1
            if i % 20 == 0 or i == total_tasks:
                pct = 100 * i / total_tasks
                print(f"\r  {i}/{total_tasks} ({pct:.0f}%)  "
                      f"rows={len(rows)}  failed={failed}",
                      end="", flush=True)
                time.sleep(0.1)

            if len(rows) % 200 == 0 and rows:
                _flush(rows)
                rows = []

    if rows:
        _flush(rows)

    print()
    print(f"\nDone. Results: {OUT_PATH.relative_to(ROOT)}")
    print(f"  Transcribed: {len(rows)}  Failed: {failed}")

    if OUT_PATH.exists():
        df = pd.read_csv(OUT_PATH)
        print(f"\nOverall WER: {df['wer'].mean():.4f}")
        print(f"Overall CER: {df['cer'].mean():.4f}")
        for lang, grp in df.groupby("language"):
            print(f"  {lang}: WER={grp['wer'].mean():.4f}  CER={grp['cer'].mean():.4f}")


if __name__ == "__main__":
    main()
