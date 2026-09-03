"""Source mapping tests for the quarantined Hugging Face candidates."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from autovalue_ml.acquisition.huggingface_dataset import ApprovalStatus
from autovalue_ml.acquisition.sources.huggingface_candidates import (
    CARSON_SHIVELY_SPEC,
    YOAD22_CRAIGSLIST_SPEC,
    normalize_accident_status,
    normalize_candidate_records,
    normalize_carson_record,
    normalize_clean_title,
    normalize_yoad_record,
    parse_mileage_text,
)

_NOW = datetime(2026, 9, 1, tzinfo=UTC)


def _carson_row() -> dict[str, object]:
    return {
        "brand": "Ford",
        "model": "Utility Police Interceptor Base",
        "model_year": 2013,
        "milage": "51,000 mi.",
        "fuel_type": "E85 Flex Fuel",
        "engine": "3.7L V6",
        "transmission": "6-Speed A/T",
        "ext_col": "Black",
        "int_col": "Black",
        "accident": "At least 1 accident or damage reported",
        "clean_title": "Yes",
        "price": "$10,300",
    }


def _yoad_row() -> dict[str, object]:
    return {
        "price": 15_500,
        "year": 2018.0,
        "manufacturer": "toyota",
        "condition": "good",
        "cylinders": 4.0,
        "fuel": "gas",
        "odometer": 68_000.0,
        "title_status": "clean",
        "transmission": "automatic",
        "drive": "fwd",
        "type": "sedan",
        "paint_color": "blue",
        "state": "ny",
        "car_age": 8,
    }


def test_carson_mapping_parses_price_mileage_accident_and_title() -> None:
    candidate = normalize_carson_record(_carson_row(), observed_at=_NOW)

    assert candidate.make == "Ford"
    assert candidate.model == "Utility Police Interceptor Base"
    assert candidate.year == 2013
    assert candidate.mileage == 51_000
    assert candidate.price_cents == 1_030_000
    assert candidate.accident_status == "accident_or_damage_reported"
    assert candidate.title_status == "clean"
    assert candidate.market_country is None
    assert candidate.raw_values["milage"] == "51,000 mi."
    assert "source_id" not in candidate.feature_values()
    assert "raw_values" not in candidate.feature_values()


@pytest.mark.parametrize("value", ["51000", "51,000 km", "about 51,000 mi.", -1])
def test_mileage_parser_rejects_unreviewed_formats(value: object) -> None:
    with pytest.raises(ValueError, match="milage"):
        parse_mileage_text(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("None reported", "none_reported"),
        ("At least 1 accident or damage reported", "accident_or_damage_reported"),
        (None, None),
        ("other", "unknown"),
    ],
)
def test_accident_mapping(value: object, expected: str | None) -> None:
    assert normalize_accident_status(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("Yes", "clean"), ("No", "not_clean"), (None, None), ("?", "unknown")],
)
def test_title_mapping(value: object, expected: str | None) -> None:
    assert normalize_clean_title(value) == expected


def test_carson_rejects_malformed_price_and_missing_model() -> None:
    malformed = _carson_row()
    malformed["price"] = "$399/mo"
    with pytest.raises(ValueError, match="monthly payment"):
        normalize_carson_record(malformed, observed_at=_NOW)

    missing_model = _carson_row()
    missing_model["model"] = None
    with pytest.raises(ValueError, match="model is missing"):
        normalize_carson_record(missing_model, observed_at=_NOW)


def test_yoad_mapping_preserves_absent_model_without_fabrication() -> None:
    candidate = normalize_yoad_record(_yoad_row(), observed_at=_NOW)

    assert candidate.market_country == "US"
    assert candidate.currency == "USD"
    assert candidate.model is None
    assert "model" not in candidate.feature_values()
    assert candidate.state == "NY"
    assert candidate.price_cents == 1_550_000


def test_yoad_rejects_non_us_geography() -> None:
    record = _yoad_row()
    record["state"] = "on"

    with pytest.raises(ValueError, match="50 U.S. states"):
        normalize_yoad_record(record, observed_at=_NOW)


def test_batch_quarantines_bad_rows_and_deduplicates_exact_records() -> None:
    valid = _carson_row()
    invalid = _carson_row()
    invalid["milage"] = "unknown"
    batch = normalize_candidate_records(
        [valid, valid.copy(), invalid],
        source_id=CARSON_SHIVELY_SPEC.source_id,
        observed_at=_NOW,
    )

    assert batch.rows_seen == 3
    assert len(batch.records) == 1
    assert batch.duplicates_skipped == 1
    assert len(batch.rejections) == 1
    assert batch.rejections[0].reason_code == "invalid_milage"


def test_source_specs_enforce_independent_batch_and_online_decisions() -> None:
    assert YOAD22_CRAIGSLIST_SPEC.approvals.acquisition is ApprovalStatus.APPROVED
    assert YOAD22_CRAIGSLIST_SPEC.approvals.batch_training is ApprovalStatus.APPROVED
    assert YOAD22_CRAIGSLIST_SPEC.approvals.online_learning is ApprovalStatus.BLOCKED
    assert CARSON_SHIVELY_SPEC.approvals.batch_training is ApprovalStatus.BLOCKED
    assert CARSON_SHIVELY_SPEC.approvals.online_learning is ApprovalStatus.BLOCKED
