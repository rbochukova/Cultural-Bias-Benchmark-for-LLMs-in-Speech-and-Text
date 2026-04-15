"""
Tests for stimulus construction, validation logic, and data integrity.
Run with: pytest tests/test_stimulus.py
"""

import sys
import pathlib
import pandas as pd
import pytest

# Make src importable
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stimulus_expander import infer_dimension, _Expander
from validate_csv import validate, VALID_DIMENSIONS, VALID_GROUPS, VALID_LANGUAGES


# ── infer_dimension ───────────────────────────────────────────────────────────

class TestInferDimension:
    def test_competence_keyword(self):
        dim, gender = infer_dimension("He is a skilled engineer who leads the team.")
        assert dim == "competence"
        assert gender == "masculine"

    def test_warmth_keyword(self):
        dim, gender = infer_dimension("She is very caring and kind to her family.")
        assert dim == "warmth"
        assert gender == "feminine"

    def test_tie_returns_needs_review(self):
        dim, _ = infer_dimension("Hello world.")
        assert dim == "needs_review"

    def test_punctuation_stripped(self):
        # "skilled," should match "skilled" after punctuation stripping
        dim, _ = infer_dimension("She is skilled, efficient, and smart.")
        assert dim == "competence"

    def test_empty_string(self):
        dim, _ = infer_dimension("")
        assert dim == "needs_review"

    def test_case_insensitive(self):
        dim, _ = infer_dimension("HE IS A LEADER AND A MANAGER.")
        assert dim == "competence"


# ── _Expander ─────────────────────────────────────────────────────────────────

class TestExpander:
    def _make_existing(self, item_ids: list[str]) -> pd.DataFrame:
        rows = []
        for iid in item_ids:
            lang, grp, _ = iid.split("-")
            tg = {"G": "gender", "N": "nationality", "P": "profession"}[grp]
            rows.append({
                "item_id": iid, "parallel_group_id": iid[3:],
                "language": lang.lower(), "origin": "native",
                "dimension": "warmth", "target_group": tg, "target": "test",
                "sent_stereotype": "A", "sent_anti_stereotype": "B",
                "source": "test", "validated": False, "notes": "",
            })
        return pd.DataFrame(rows)

    def test_counter_continues_from_existing(self):
        existing = self._make_existing(["EN-G-005", "EN-G-010"])
        exp = _Expander(existing)
        exp.add("en", "native", "warmth", "gender", "woman/man", "S", "A", "test")
        assert exp.new_rows[0]["item_id"] == "EN-G-011"

    def test_counter_independent_per_language(self):
        existing = self._make_existing(["EN-G-005", "FR-G-003"])
        exp = _Expander(existing)
        exp.add("fr", "native", "warmth", "gender", "femme/homme", "S", "A", "test")
        assert exp.new_rows[0]["item_id"] == "FR-G-004"

    def test_counter_independent_per_group(self):
        existing = self._make_existing(["EN-G-010", "EN-N-002"])
        exp = _Expander(existing)
        exp.add("en", "native", "warmth", "nationality", "French", "S", "A", "test")
        assert exp.new_rows[0]["item_id"] == "EN-N-003"

    def test_malformed_id_skipped_gracefully(self):
        existing = self._make_existing(["EN-G-005"])
        existing.loc[0, "item_id"] = "BADFORMAT"
        # Should not crash
        exp = _Expander(existing)
        exp.add("en", "native", "warmth", "gender", "woman/man", "S", "A", "test")
        assert exp.new_rows[0]["item_id"] == "EN-G-001"

    def test_new_rows_not_validated(self):
        existing = self._make_existing(["EN-G-001"])
        exp = _Expander(existing)
        exp.add("en", "native", "warmth", "gender", "woman/man", "S", "A", "test")
        assert exp.new_rows[0]["validated"] is False


# ── validate_csv ──────────────────────────────────────────────────────────────

def _minimal_df(**overrides) -> pd.DataFrame:
    row = {
        "item_id": "EN-G-001", "parallel_group_id": "G-001",
        "language": "en", "origin": "native", "dimension": "warmth",
        "target_group": "gender", "target": "woman/man",
        "sent_stereotype": "She is warm.", "sent_anti_stereotype": "He is warm.",
        "source": "test", "validated": False, "notes": "",
    }
    row.update(overrides)
    return pd.DataFrame([row])


class TestValidateCsv:
    def test_valid_row_passes(self):
        validate(_minimal_df())  # should not raise

    def test_duplicate_item_id_raises(self):
        df = pd.concat([_minimal_df(), _minimal_df()], ignore_index=True)
        with pytest.raises(ValueError, match="Duplicate item_ids"):
            validate(df)

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError, match="Invalid dimension"):
            validate(_minimal_df(dimension="unknown"))

    def test_invalid_language_raises(self):
        with pytest.raises(ValueError, match="Invalid language"):
            validate(_minimal_df(language="de"))

    def test_invalid_target_group_raises(self):
        with pytest.raises(ValueError, match="Invalid target_group"):
            validate(_minimal_df(target_group="age"))

    def test_identical_sentences_raises(self):
        with pytest.raises(ValueError, match="Identical sent_stereotype"):
            validate(_minimal_df(sent_stereotype="Same.", sent_anti_stereotype="Same."))

    def test_missing_column_raises(self):
        df = _minimal_df().drop(columns=["notes"])
        with pytest.raises(ValueError, match="Missing columns"):
            validate(df)

    def test_malformed_item_id_raises(self):
        with pytest.raises(ValueError, match="Malformed item_ids"):
            validate(_minimal_df(item_id="BADFORMAT"))

    def test_group_letter_mismatch_raises(self):
        # EN-N-001 but target_group=gender -- inconsistent
        with pytest.raises(ValueError, match="target_group/item_id letter mismatch"):
            validate(_minimal_df(item_id="EN-N-001", parallel_group_id="N-001"))

    def test_all_valid_dimensions_accepted(self):
        for dim in VALID_DIMENSIONS:
            validate(_minimal_df(dimension=dim))

    def test_all_valid_languages_accepted(self):
        for lang in VALID_LANGUAGES:
            iid = f"{lang.upper()}-G-001"
            validate(_minimal_df(language=lang, item_id=iid, parallel_group_id="G-001"))


# ── Structural data-integrity tests ──────────────────────────────────────────

STIMULI_PATH    = ROOT / "data" / "stimuli_seed.csv"
TEXT_RESULTS    = ROOT / "data" / "results" / "text" / "gpt-4o-mini_results.csv"
SPEECH_RESULTS  = ROOT / "data" / "results" / "speech" / "large-v3_gpt-4o-mini_results.csv"

REQUIRED_STIMULI_COLS = {
    "item_id", "language", "dimension", "target_group", "target",
    "origin", "sent_stereotype", "sent_anti_stereotype", "validated",
}
REQUIRED_TEXT_COLS = {
    "item_id", "language", "dimension", "chose_stereotype",
    "logprob_A", "logprob_B", "A_is_stereotype",
}
REQUIRED_SPEECH_COLS = REQUIRED_TEXT_COLS | {
    "wer_S", "wer_A", "transcript_S", "transcript_A",
    "prompt_variant", "asr_system",
}
VALID_LANGS_SET = {"en", "fr", "bg"}
VALID_DIMS_SET  = {"warmth", "competence"}


@pytest.fixture(scope="module")
def stimuli_df():
    if not STIMULI_PATH.exists():
        pytest.skip(f"Stimuli file not found: {STIMULI_PATH}")
    return pd.read_csv(STIMULI_PATH, encoding="utf-8")


@pytest.fixture(scope="module")
def text_results_df():
    if not TEXT_RESULTS.exists():
        pytest.skip(f"Text results not found: {TEXT_RESULTS}")
    return pd.read_csv(TEXT_RESULTS, encoding="utf-8")


@pytest.fixture(scope="module")
def speech_results_df():
    if not SPEECH_RESULTS.exists():
        pytest.skip(f"Speech results not found: {SPEECH_RESULTS}")
    return pd.read_csv(SPEECH_RESULTS, encoding="utf-8")


class TestStimuliStructure:
    def test_required_columns_present(self, stimuli_df):
        missing = REQUIRED_STIMULI_COLS - set(stimuli_df.columns)
        assert not missing, f"Missing columns in stimuli_seed.csv: {missing}"

    def test_no_duplicate_item_ids(self, stimuli_df):
        dupes = stimuli_df["item_id"].duplicated()
        assert not dupes.any(), (
            f"Duplicate item_ids: {stimuli_df.loc[dupes, 'item_id'].tolist()[:5]}"
        )

    def test_languages_valid(self, stimuli_df):
        invalid = set(stimuli_df["language"].unique()) - VALID_LANGS_SET
        assert not invalid, f"Unexpected languages in stimuli: {invalid}"

    def test_dimensions_valid(self, stimuli_df):
        invalid = set(stimuli_df["dimension"].unique()) - VALID_DIMS_SET
        assert not invalid, f"Unexpected dimensions in stimuli: {invalid}"

    def test_no_identical_sentence_pairs_in_validated(self, stimuli_df):
        # Only validated items must have distinct sentence pairs;
        # known-invalid items (FR-G-020, FR-G-091, FR-G-119) are allowed to be identical.
        validated = stimuli_df[stimuli_df["validated"] == True]
        same = validated["sent_stereotype"] == validated["sent_anti_stereotype"]
        assert not same.any(), (
            f"{same.sum()} validated items have identical stereotype/anti-stereotype sentences"
        )

    def test_validated_items_dominate(self, stimuli_df):
        pct = stimuli_df["validated"].mean()
        assert pct >= 0.99, f"Only {100*pct:.1f}% of items are validated (expected ≥99%)"

    def test_all_three_languages_present(self, stimuli_df):
        langs = set(stimuli_df["language"].unique())
        assert langs == VALID_LANGS_SET, f"Expected languages {VALID_LANGS_SET}, got {langs}"


class TestTextResultsStructure:
    def test_required_columns_present(self, text_results_df):
        missing = REQUIRED_TEXT_COLS - set(text_results_df.columns)
        assert not missing, f"Missing columns in text results: {missing}"

    def test_chose_stereotype_is_binary(self, text_results_df):
        vals = set(text_results_df["chose_stereotype"].dropna().unique())
        assert vals <= {0, 1, True, False}, f"Non-binary chose_stereotype values: {vals}"

    def test_all_item_ids_in_stimuli(self, text_results_df, stimuli_df):
        # Allow a small number of orphaned items (artifacts from removed stimuli,
        # e.g. the 33 SHADES nationality items removed after initial inference runs).
        # Threshold: < 1% of result rows may be orphaned.
        result_ids  = set(text_results_df["item_id"].astype(str))
        stimuli_ids = set(stimuli_df["item_id"].astype(str))
        orphans = result_ids - stimuli_ids
        pct = len(orphans) / len(result_ids)
        assert pct < 0.01, (
            f"{len(orphans)} result item_ids ({100*pct:.1f}%) not found in stimuli_seed.csv "
            f"(threshold 1%): {sorted(orphans)[:5]}"
        )

    def test_no_duplicate_item_ids(self, text_results_df):
        dupes = text_results_df["item_id"].duplicated()
        assert not dupes.any(), (
            f"{dupes.sum()} duplicate item_ids in text results"
        )


class TestSpeechResultsStructure:
    def test_required_columns_present(self, speech_results_df):
        missing = REQUIRED_SPEECH_COLS - set(speech_results_df.columns)
        assert not missing, f"Missing columns in speech results: {missing}"

    def test_prompt_variant_is_natural(self, speech_results_df):
        vals = set(speech_results_df["prompt_variant"].dropna().unique())
        assert vals == {"natural"}, (
            f"Expected prompt_variant={{'natural'}} in natural speech file, got {vals}"
        )

    def test_wer_values_non_negative(self, speech_results_df):
        # WER can exceed 1 when insertions outnumber reference words (jiwer behaviour).
        # Only sanity-check that values are non-negative and below a loose ceiling.
        for col in ("wer_S", "wer_A"):
            vals = speech_results_df[col].dropna()
            assert (vals >= 0).all(), f"{col} has negative WER values"
            assert (vals <= 5).all(), f"{col} has implausibly large WER values (>5)"

    def test_all_item_ids_in_stimuli(self, speech_results_df, stimuli_df):
        result_ids  = set(speech_results_df["item_id"].astype(str))
        stimuli_ids = set(stimuli_df["item_id"].astype(str))
        orphans = result_ids - stimuli_ids
        assert not orphans, (
            f"{len(orphans)} speech result item_ids not found in stimuli_seed.csv: "
            f"{sorted(orphans)[:5]}"
        )

    def test_speech_item_ids_subset_of_text(self, speech_results_df, text_results_df):
        speech_ids = set(speech_results_df["item_id"].astype(str))
        text_ids   = set(text_results_df["item_id"].astype(str))
        only_speech = speech_ids - text_ids
        assert not only_speech, (
            f"{len(only_speech)} speech item_ids have no text counterpart: "
            f"{sorted(only_speech)[:5]}"
        )
