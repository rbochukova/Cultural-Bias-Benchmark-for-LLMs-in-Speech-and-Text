"""
Tests for stimulus construction and validation logic.
Run with: pytest tests/test_stimulus.py
"""

import sys
import pathlib
import pandas as pd
import pytest

# Make src importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

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
