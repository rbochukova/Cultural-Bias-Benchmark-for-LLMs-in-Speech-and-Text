"""
add_winobias_items.py
~~~~~~~~~~~~~~~~~~~~~
Ingests English gender-profession items from WinoBias (Zhao et al., 2018)
into stimuli_seed.csv as new EN source items.

WinoBias structure:
  - type1_pro / type1_anti: sentences where a pronoun (he/she) is coreferent
    with a stereotypically male or female profession. Type1 requires world
    knowledge of gender-profession stereotypes to resolve.
  - type2_pro / type2_anti: syntactically disambiguated (pronoun resolution
    follows syntax, not stereotypes).

This script uses type1 only, as it directly encodes gender-profession
stereotypes in the pronoun choice.

Minimal pair construction:
  - type1_pro row  → sent_stereotype    (pronoun matches professional stereotype)
  - type1_anti row → sent_anti_stereotype (pronoun contradicts stereotype)
  Pairs are matched by their numeric document ID.

Dimension is always 'competence' (profession bias = capability/status
stereotypes per SCM). Target is extracted as "profession/gender" from the
sentence (e.g. "developer/male" → target="developer/he–she").

Items are deduplicated against existing CSV by sent_stereotype text.
After ingestion, run add_translated_items.py --dim profession for FR+BG.

Usage:
    OPENAI_API_KEY=sk-... python src/add_winobias_items.py
    OPENAI_API_KEY=sk-... python src/add_winobias_items.py --dry-run
"""

import argparse
import json
import os
import pathlib
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT     = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "stimuli_seed.csv"

# SCM-informed target extraction prompt
TARGET_SYSTEM = (
    "You are a social-psychology researcher annotating gender-profession bias sentences.\n\n"
    "Given a stereotypical sentence and its anti-stereotypical counterpart, output JSON:\n"
    '{"dimension": "competence", "target": "<profession>/<stereotyped_gender>"}\n\n'
    "Rules:\n"
    "- dimension is always 'competence' for profession-based gender stereotypes\n"
    "- target format: 'engineer/male' if the stereotype is that engineers are male,\n"
    "  or 'nurse/female' if the stereotype is that nurses are female\n"
    "- use lowercase, singular form\n"
    "Return ONLY valid JSON."
)


def _load_winobias() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load type1_pro and type1_anti splits. Returns (pro_df, anti_df)."""
    print("Loading WinoBias via HuggingFace datasets library ...", flush=True)
    try:
        from datasets import load_dataset
        pro  = load_dataset("uclanlp/wino_bias", "type1_pro",
                            trust_remote_code=True)
        anti = load_dataset("uclanlp/wino_bias", "type1_anti",
                            trust_remote_code=True)

        # Combine validation + test splits
        pro_df  = pd.concat([pro["validation"].to_pandas(),
                             pro["test"].to_pandas()], ignore_index=True)
        anti_df = pd.concat([anti["validation"].to_pandas(),
                             anti["test"].to_pandas()], ignore_index=True)

        print(f"  type1_pro : {len(pro_df)} rows")
        print(f"  type1_anti: {len(anti_df)} rows")
        return pro_df, anti_df
    except Exception as e:
        sys.exit(f"Could not load WinoBias: {e}\n"
                 f"Install: pip install datasets")


def _tokens_to_sentence(tokens: list) -> str:
    """Join tokens into a sentence, handling punctuation spacing."""
    if not tokens:
        return ""
    result = tokens[0]
    for tok in tokens[1:]:
        if tok in {",", ".", "!", "?", ";", ":", "'s", "n't", "'re", "'ve",
                   "'ll", "'m", "'d"}:
            result += tok
        else:
            result += " " + tok
    return result.strip()


def _extract_doc_num(doc_id: str) -> str:
    """Extract numeric ID from document_id for matching pro/anti pairs."""
    m = re.search(r"//(\d+)$", str(doc_id))
    return m.group(1) if m else str(doc_id)


def _build_pairs(pro_df: pd.DataFrame, anti_df: pd.DataFrame) -> list[dict]:
    """
    Match pro/anti rows that share the same base sentence with only the
    gendered pronoun swapped (he ↔ she / him ↔ her / his ↔ her).

    WinoBias pro and anti splits have the same sentences in the same order,
    with pronouns swapped. Match by positional index within each split.
    """
    PRONOUNS = {"he", "she", "him", "her", "his", "hers", "himself", "herself"}

    pro_rows  = pro_df.reset_index(drop=True)
    anti_rows = anti_df.reset_index(drop=True)
    n = min(len(pro_rows), len(anti_rows))

    pairs = []
    for i in range(n):
        pro_tokens  = list(pro_rows.loc[i, "tokens"])
        anti_tokens = list(anti_rows.loc[i, "tokens"])

        pro_sent  = _tokens_to_sentence(pro_tokens)
        anti_sent = _tokens_to_sentence(anti_tokens)

        if pro_sent == anti_sent:
            continue

        # Verify the only difference is a pronoun swap
        if len(pro_tokens) != len(anti_tokens):
            continue
        diffs = [(p, a) for p, a in zip(pro_tokens, anti_tokens) if p.lower() != a.lower()]
        if not diffs:
            continue
        # All differing tokens must be pronouns
        if not all(p.lower() in PRONOUNS and a.lower() in PRONOUNS
                   for p, a in diffs):
            continue

        pairs.append({
            "pro_sent":  pro_sent,
            "anti_sent": anti_sent,
        })

    # Deduplicate by pro_sent
    seen, unique = set(), []
    for p in pairs:
        if p["pro_sent"] not in seen:
            seen.add(p["pro_sent"])
            unique.append(p)

    return unique


def _get_target(client, pro_sent: str, anti_sent: str) -> dict | None:
    """Ask GPT-4o-mini for dimension + target label."""
    prompt = f"Stereotypical: {pro_sent}\nAnti-stereotypical: {anti_sent}"
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": TARGET_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )
            data = json.loads(resp.choices[0].message.content)
            dim  = data.get("dimension", "competence").strip().lower()
            tgt  = data.get("target", "").strip()
            if tgt:
                return {"dimension": dim, "target": tgt}
            return None
        except Exception as exc:
            if attempt == 2:
                print(f"\n    WARN: target extraction failed: {exc}")
                return None
            time.sleep(2)
    return None


def _max_num(df: pd.DataFrame, prefix: str) -> int:
    ids  = df[df["item_id"].str.startswith(prefix, na=False)]["item_id"]
    nums = ids.str.extract(re.escape(prefix) + r"(\d+)")[0].dropna().astype(int)
    return int(nums.max()) if not nums.empty else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest WinoBias type1 gender-profession items as EN source items"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit("ERROR: OPENAI_API_KEY not set.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key) if api_key else None

    pro_df, anti_df = _load_winobias()
    pairs = _build_pairs(pro_df, anti_df)
    print(f"Unique sentence pairs built: {len(pairs)}")

    # Deduplicate against existing CSV
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    existing_stereo = set(df["sent_stereotype"].astype(str).str.strip())
    pairs = [p for p in pairs if p["pro_sent"].strip() not in existing_stereo]
    print(f"After deduplication vs existing: {len(pairs)}")

    if args.dry_run:
        print(f"\n=== DRY RUN — first 5 pairs ===")
        for p in pairs[:5]:
            print(f"\n  Stereo: {p['pro_sent']}")
            print(f"  Anti  : {p['anti_sent']}")
        return

    if len(pairs) == 0:
        print("Nothing to add.")
        return

    next_num = _max_num(df, "EN-P-") + 1
    new_rows = []
    skipped  = 0

    print(f"\nAnnotating {len(pairs)} WinoBias pairs (dimension + target) ...")
    for i, p in enumerate(pairs, 1):
        meta = _get_target(client, p["pro_sent"], p["anti_sent"])
        if meta is None:
            skipped += 1
        else:
            new_rows.append({
                "item_id":              f"EN-P-{next_num:03d}",
                "parallel_group_id":    None,
                "language":             "en",
                "origin":               "native",
                "dimension":            meta["dimension"],
                "target_group":         "profession",
                "target":               meta["target"],
                "sent_stereotype":      p["pro_sent"],
                "sent_anti_stereotype": p["anti_sent"],
                "source":               "winobias_en",
                "validated":            True,
                "notes":                "WinoBias type1 gender-profession pair",
            })
            next_num += 1

        if i % 10 == 0 or i == len(pairs):
            print(f"\r  {i}/{len(pairs)}  added={len(new_rows)}  skipped={skipped}",
                  end="", flush=True)
        time.sleep(0.05)

    print()

    if not new_rows:
        print("No valid items produced.")
        return

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"Added {len(new_rows)} EN profession items from WinoBias (skipped {skipped})")
    print(f"Total items in CSV: {len(df)}")
    print(f"\nNext: python src/add_translated_items.py --dim profession")


if __name__ == "__main__":
    main()
