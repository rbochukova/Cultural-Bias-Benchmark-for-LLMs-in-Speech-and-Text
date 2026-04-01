"""
validate_csv.py
~~~~~~~~~~~~~~~
Schema validation for data/stimuli_seed.csv.

Run directly to validate:
    python src/validate_csv.py

Or import and call validate() from other scripts before writing.
Raises ValueError with a full list of violations if any are found.
"""

import pathlib
import sys

import pandas as pd

ROOT     = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "stimuli_seed.csv"

VALID_LANGUAGES   = {"en", "fr", "bg"}
VALID_ORIGINS     = {"native", "parallel"}
VALID_DIMENSIONS  = {"warmth", "competence", "needs_review", "exclude"}
VALID_GROUPS      = {"gender", "nationality", "profession"}
REQUIRED_COLS     = [
    "item_id", "parallel_group_id", "language", "origin",
    "dimension", "target_group", "target",
    "sent_stereotype", "sent_anti_stereotype",
    "source", "validated", "notes",
]
ID_GROUP_MAP      = {"G": "gender", "N": "nationality", "P": "profession"}


def validate(df: pd.DataFrame, path: str = "") -> None:
    """
    Validate df against the stimulus CSV schema.
    Raises ValueError listing all violations found.
    Call this before every CSV write.
    """
    violations: list[str] = []
    label = path or "dataframe"

    # ── Column presence ────────────────────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        violations.append(f"Missing columns: {missing_cols}")
        raise ValueError(f"Schema violations in {label}:\n" + "\n".join(violations))

    # ── Uniqueness ────────────────────────────────────────────────────────────
    dups = df[df.duplicated("item_id", keep=False)]["item_id"].unique()
    if len(dups):
        violations.append(f"Duplicate item_ids ({len(dups)}): {list(dups)[:10]}")

    # ── Allowed values ────────────────────────────────────────────────────────
    bad_lang = df[~df["language"].isin(VALID_LANGUAGES)]["item_id"].tolist()
    if bad_lang:
        violations.append(f"Invalid language values in {len(bad_lang)} rows: {bad_lang[:5]}")

    bad_origin = df[~df["origin"].isin(VALID_ORIGINS)]["item_id"].tolist()
    if bad_origin:
        violations.append(f"Invalid origin values in {len(bad_origin)} rows: {bad_origin[:5]}")

    bad_dim = df[~df["dimension"].isin(VALID_DIMENSIONS)]["item_id"].tolist()
    if bad_dim:
        violations.append(f"Invalid dimension values in {len(bad_dim)} rows: {bad_dim[:5]}")

    bad_group = df[~df["target_group"].isin(VALID_GROUPS)]["item_id"].tolist()
    if bad_group:
        violations.append(f"Invalid target_group values in {len(bad_group)} rows: {bad_group[:5]}")

    # ── Boolean validated column ───────────────────────────────────────────────
    df["_val_coerced"] = pd.to_numeric(df["validated"], errors="coerce")
    non_bool = df[~df["validated"].isin([True, False, 0, 1])]["item_id"].tolist()
    if non_bool:
        violations.append(f"Non-boolean validated values in {len(non_bool)} rows: {non_bool[:5]}")
    df.drop(columns=["_val_coerced"], inplace=True)

    # ── Non-empty sentence fields ─────────────────────────────────────────────
    for col in ("sent_stereotype", "sent_anti_stereotype", "item_id", "target_group"):
        empty = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]["item_id"].tolist()
        if empty:
            violations.append(f"Empty {col} in {len(empty)} rows: {empty[:5]}")

    # ── Identical sentence pairs ───────────────────────────────────────────────
    identical = df[
        df["sent_stereotype"].str.strip() == df["sent_anti_stereotype"].str.strip()
    ]["item_id"].tolist()
    if identical:
        violations.append(f"Identical sent_stereotype/anti in {len(identical)} rows: {identical[:5]}")

    # ── item_id format: XX-G-001 ───────────────────────────────────────────────
    bad_format = []
    for iid in df["item_id"]:
        parts = str(iid).split("-")
        if len(parts) != 3 or parts[1] not in ID_GROUP_MAP:
            bad_format.append(iid)
            continue
        try:
            int(parts[2])
        except ValueError:
            bad_format.append(iid)
    if bad_format:
        violations.append(f"Malformed item_ids ({len(bad_format)}): {bad_format[:5]}")

    # ── target_group consistent with item_id group letter ─────────────────────
    mismatch = []
    for _, row in df.iterrows():
        parts = str(row["item_id"]).split("-")
        if len(parts) == 3 and parts[1] in ID_GROUP_MAP:
            expected = ID_GROUP_MAP[parts[1]]
            if row["target_group"] != expected:
                mismatch.append(row["item_id"])
    if mismatch:
        violations.append(
            f"target_group/item_id letter mismatch in {len(mismatch)} rows: {mismatch[:5]}"
        )

    if violations:
        raise ValueError(
            f"Schema violations in {label} ({len(violations)} issues):\n"
            + "\n".join(f"  • {v}" for v in violations)
        )


def load_validated(path: pathlib.Path = CSV_PATH) -> pd.DataFrame:
    """Load CSV, coerce types, validate schema. Returns clean DataFrame."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["notes"]     = df["notes"].fillna("")
    df["validated"] = df["validated"].map(
        lambda x: True if str(x).strip().lower() in ("true", "1") else False
    )
    validate(df, str(path))
    return df


if __name__ == "__main__":
    try:
        df = load_validated()
        print(f"OK — {len(df)} rows passed schema validation.")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
