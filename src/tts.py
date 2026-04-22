"""
Generates audio files for all validated stimuli using Azure Cognitive Services
Text-to-Speech.
"""

import argparse
import os
import pathlib
import sys
import time
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT      = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH  = ROOT / "data" / "stimuli_seed.csv"
AUDIO_DIR = ROOT / "data" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

VALID_DIMS = {"warmth", "competence"}

DEFAULT_VOICES = {
    "en": "en-US-AndrewNeural",
    "fr": "fr-FR-HenriNeural",
    "bg": "bg-BG-BorislavNeural",
}

LANG_LOCALE = {
    "en": "en-US",
    "fr": "fr-FR",
    "bg": "bg-BG",
}


def _build_ssml(text: str, voice: str, locale: str) -> str:
    """Wrap text in SSML for Azure TTS."""
    speak = ET.Element("speak")
    speak.set("version", "1.0")
    speak.set("xmlns", "http://www.w3.org/2001/10/synthesis")
    speak.set("xml:lang", locale)
    voice_el = ET.SubElement(speak, "voice")
    voice_el.set("name", voice)
    voice_el.text = text
    return ET.tostring(speak, encoding="unicode")


def _synthesise(
    text: str,
    voice: str,
    locale: str,
    out_path: pathlib.Path,
    key: str,
    region: str,
) -> bool:
    """
    Call Azure TTS REST API and write WAV to out_path.
    Returns True on success.
    """
    try:
        import requests
    except ImportError:
        sys.exit("ERROR: 'requests' package not installed. Run: pip install requests")

    url     = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
        "User-Agent": "cultural-bias-benchmark/1.0",
    }
    ssml = _build_ssml(text, voice, locale)

    backoff = 2
    for attempt in range(4):
        try:
            resp = requests.post(url, headers=headers, data=ssml.encode("utf-8"),
                                 timeout=30)
            if resp.status_code == 200:
                out_path.write_bytes(resp.content)
                return True
            elif resp.status_code == 429:
                print(f"\n    rate limited — waiting {backoff}s ...", end=" ")
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
            else:
                print(f"\n    HTTP {resp.status_code}: {resp.text[:120]}", end=" ")
                return False
        except Exception as exc:
            print(f"\n    attempt {attempt + 1}/4 failed: {exc}", end=" ")
            time.sleep(3)

    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="TTS audio generation")
    parser.add_argument("--lang",       default=None,
                        help="Filter to one language: en, fr, bg (default: all)")
    parser.add_argument("--voice-en",   default=DEFAULT_VOICES["en"])
    parser.add_argument("--voice-fr",   default=DEFAULT_VOICES["fr"])
    parser.add_argument("--voice-bg",   default=DEFAULT_VOICES["bg"])
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print first 5 requests without calling API")
    args = parser.parse_args()

    voices = {
        "en": args.voice_en,
        "fr": args.voice_fr,
        "bg": args.voice_bg,
    }

    key    = os.environ.get("AZURE_SPEECH_KEY", "")
    region = os.environ.get("AZURE_SPEECH_REGION", "westeurope")
    if not key and not args.dry_run:
        sys.exit(
            "ERROR: AZURE_SPEECH_KEY not set.\n"
            "Get a key from Azure Portal → Cognitive Services → Speech.\n"
            "Set AZURE_SPEECH_REGION too if not in westeurope."
        )

    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    items = df[
        df["validated"].map(lambda x: str(x).strip().lower() in ("true", "1"))
        & df["dimension"].isin(VALID_DIMS)
    ].copy()
    if args.lang:
        items = items[items["language"] == args.lang]

    tasks = []
    for _, row in items.iterrows():
        lang = str(row["language"])
        tasks.append((row["item_id"], "S", str(row["sent_stereotype"]).strip(), lang))
        tasks.append((row["item_id"], "A", str(row["sent_anti_stereotype"]).strip(), lang))

    pending = [
        (iid, sfx, text, lang)
        for iid, sfx, text, lang in tasks
        if not (AUDIO_DIR / f"{iid}_{sfx}.wav").exists()
    ]

    if args.dry_run:
        for iid, sfx, text, lang in pending[:5]:
            voice  = voices[lang]
            locale = LANG_LOCALE[lang]
            ssml   = _build_ssml(text, voice, locale)
            print(f"\n[{iid}_{sfx}] voice={voice}")
            print(f"  text : {text[:80]}")
            print(f"  ssml : {ssml[:120]}...")
        return

    if not pending:
        print("Nothing to generate. Done.")
        return

    succeeded = failed = 0
    for i, (iid, sfx, text, lang) in enumerate(pending, 1):
        out_path = AUDIO_DIR / f"{iid}_{sfx}.wav"
        voice    = voices[lang]
        locale   = LANG_LOCALE[lang]
        ok       = _synthesise(text, voice, locale, out_path, key, region)
        if ok:
            succeeded += 1
        else:
            failed += 1
            print(f"\n    FAILED: {iid}_{sfx}")

        if i % 20 == 0 or i == len(pending):
            pct = 100 * i / len(pending)
            print(f"\r  {i}/{len(pending)} ({pct:.0f}%)  "
                  f"ok={succeeded}  failed={failed}", end="", flush=True)

        time.sleep(0.5) 

    print()
    print(f"\n{'=' * 55}")
    print(f"Done.  Generated={succeeded}  Failed={failed}")
    print(f"Audio files in: {AUDIO_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
