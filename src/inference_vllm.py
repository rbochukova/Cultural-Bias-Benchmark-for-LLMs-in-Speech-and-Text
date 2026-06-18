"""
Forced-choice bias scoring via a vLLM server 
Connects to a running vLLM OpenAI-compatible server and scores each stimulus pair by mean per-token log-probability of the prompt (echo=True, max_tokens=0).
The sentence with higher mean log-prob is the model's preference.

Usage (after starting vLLM server separately):
    python src/inference_vllm.py \
        --model mistralai/Mistral-7B-v0.1 \
        --base-url http://localhost:8000/v1 \
        --lang en
"""

import argparse
import asyncio
import pathlib
import sys
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT        = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH    = ROOT / "data" / "stimuli_seed.csv"
RESULTS_DIR = ROOT / "data" / "results" / "text"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VALID_DIMS = {"warmth", "competence"}


async def _sentence_logprob(client, model: str, text: str) -> float:
    """Mean per-token log-probability via vLLM completions endpoint."""
    from openai import AsyncOpenAI  

    response = await client.completions.create(
        model=model,
        prompt=text,
        max_tokens=0,
        echo=True,
        logprobs=1,
    )
    token_logprobs = response.choices[0].logprobs.token_logprobs
    valid = [lp for lp in token_logprobs if lp is not None]
    if not valid:
        raise ValueError(f"No valid logprobs returned for: {text[:60]}")
    return sum(valid) / len(valid)


async def _score_item(client, model: str, row: pd.Series, semaphore: asyncio.Semaphore) -> dict | None:
    sent_s = str(row["sent_stereotype"]).strip()
    sent_a = str(row["sent_anti_stereotype"]).strip()

    try:
        async with semaphore:
            lp_s, lp_a = await asyncio.gather(
                _sentence_logprob(client, model, sent_s),
                _sentence_logprob(client, model, sent_a),
            )

        return {
            "item_id":           row["item_id"],
            "language":          row["language"],
            "dimension":         row["dimension"],
            "target_group":      row["target_group"],
            "target":            row["target"],
            "origin":            row["origin"],
            "parallel_group_id": row["parallel_group_id"],
            "model":             model,
            "prompt_variant":    "sentence_logprob",
            "modality":          "text",
            "asr_system":        None,
            "A_is_stereotype":   True,
            "logprob_S":         lp_s,
            "logprob_A":         lp_a,
            "chose_stereotype":  int(lp_s >= lp_a),
            "scored_at":         datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        print(f"\n  FAILED {row['item_id']}: {exc}", flush=True)
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


async def _run(args) -> None:
    from openai import AsyncOpenAI

    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    items = df[
        df["validated"].map(lambda x: str(x).strip().lower() in ("true", "1"))
        & df["dimension"].isin(VALID_DIMS)
    ].copy()

    if args.lang:
        items = items[items["language"] == args.lang]

    print(f"Stimuli: {len(items)} items  (lang={args.lang or 'all'})", flush=True)

    safe_model   = args.model.replace("/", "-")
    results_path = RESULTS_DIR / f"{safe_model}_results.csv"

    if results_path.exists():
        done = set(pd.read_csv(results_path, encoding="utf-8")["item_id"].astype(str))
        items = items[~items["item_id"].isin(done)]
        print(f"Already scored: {len(done)}, remaining: {len(items)}", flush=True)

    if args.dry_run:
        for _, row in items.head(3).iterrows():
            print(f"\n  {row['item_id']}  S: {row['sent_stereotype'][:70]}")
            print(f"             A: {row['sent_anti_stereotype'][:70]}")
        return

    if len(items) == 0:
        print("Nothing to score.", flush=True)
        return

    client    = AsyncOpenAI(base_url=args.base_url, api_key="dummy")
    semaphore = asyncio.Semaphore(args.max_concurrent)
    rows      = list(items.iterrows())
    results   = []
    failed    = 0
    total     = len(rows)

    tasks = [_score_item(client, args.model, row, semaphore) for _, row in rows]

    write_every = 200
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        if result:
            results.append(result)
        else:
            failed += 1

        if i % 50 == 0 or i == total:
            pct = 100 * i / total
            print(f"\r  {i}/{total} ({pct:.0f}%)  scored={len(results)}  failed={failed}",
                  end="", flush=True)

        if len(results) % write_every == 0 and results:
            _flush(results, results_path)

    print(flush=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM forced-choice bias scoring")
    parser.add_argument("--model",          default="mistralai/Mistral-7B-v0.1",
                        help="Model name as served by vLLM (must match --served-model-name)")
    parser.add_argument("--base-url",       default="http://localhost:8000/v1",
                        help="Base URL of the running vLLM server")
    parser.add_argument("--lang",           default=None,
                        help="Filter to one language: en, fr, bg (default: all)")
    parser.add_argument("--max-concurrent", type=int, default=64,
                        help="Max simultaneous requests to vLLM server")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Print first 3 items without scoring")
    args = parser.parse_args()

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
