"""Tests for source-independent vehicle record normalization."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from autovalue_ml.acquisition.normalization import (
    NormalizationContext,
    VehicleFieldMapping,
    normalize_vehicle_records,
)


def _context() -> NormalizationContext:
    return NormalizationContext(
        source_id="licensed-fixture",
        source_record_url_prefix="https://data.example.test/vehicles/",
        observed_at=datetime(2026, 8, 27, tzinfo=UTC),
        ingestion_run_id="run-001",
        authorization_policy_id="manifest-sha256-001",
        parser_version="licensed-csv/1.0.0",
    )


def test_normalizes_valid_records_skips_exact_duplicate_and_preserves_missing_values() -> None:
    first = {
        "source_listing_id": "public-001",
        "year": "2021",
        "make": " Toyota ",
        "model": "Camry",
        "mileage": "34,120 miles",
        "price": "$23,900",
        "owner_count": "1 owner",
    }
    missing_optional = {
        "source_listing_id": "public-002",
        "year": 2019,
        "make": "Subaru",
        "model": "Outback",
        "price": 19_450,
    }

    batch = normalize_vehicle_records(
        (first, first.copy(), missing_optional),
        mapping=VehicleFieldMapping(),
        context=_context(),
    )

    assert batch.records_seen == 3
    assert batch.duplicates_skipped == 1
    assert len(batch.listings) == 2
    assert batch.rejected_listings == ()
    assert batch.listings[0].make == "Toyota"
    assert batch.listings[0].mileage == 34_120
    assert batch.listings[0].price_cents == 2_390_000
    assert batch.listings[0].owner_count == 1
    assert batch.listings[1].mileage is None
    assert batch.listings[1].condition is None
    assert batch.listings[1].canonical_url.endswith("/public-002")


def test_quarantines_bad_rows_without_losing_valid_records() -> None:
    valid = {
        "source_listing_id": "valid-001",
        "year": 2022,
        "make": "Ford",
        "model": "F-150",
        "price": 38_700,
    }
    malformed_price = {
        "source_listing_id": "bad-price",
        "year": 2020,
        "make": "Honda",
        "model": "Accord",
        "price": "$399/mo",
    }
    missing_year = {
        "source_listing_id": "missing-year",
        "make": "Honda",
        "model": "Civic",
        "price": 20_000,
    }

    batch = normalize_vehicle_records(
        (valid, malformed_price, missing_year),
        mapping=VehicleFieldMapping(),
        context=_context(),
    )

    assert [listing.source_listing_id for listing in batch.listings] == ["valid-001"]
    assert len(batch.rejected_listings) == 2
    assert {item.source_listing_id for item in batch.rejected_listings} == {
        "bad-price",
        "missing-year",
    }
    assert all(item.reason_code == "normalization_failed" for item in batch.rejected_listings)
    assert all(len(item.raw_content_sha256) == 64 for item in batch.rejected_listings)


def test_custom_mapping_supports_a_licensed_dataset_schema() -> None:
    mapping = VehicleFieldMapping(
        year="Year",
        make="Brand",
        model="Model",
        price="Price",
        source_listing_id=None,
        canonical_url=None,
        mileage="Mileage",
    )
    row = {
        "Year": "2020",
        "Brand": "Ford",
        "Model": "Escape",
        "Price": "14500 USD",
        "Mileage": "52,000 mi",
    }

    batch = normalize_vehicle_records((row,), mapping=mapping, context=_context())

    assert len(batch.listings) == 1
    assert batch.listings[0].source_listing_id.startswith("derived-")
    assert batch.listings[0].mileage == 52_000
    assert batch.listings[0].currency == "USD"


def test_vehicle_status_is_normalized_without_overloading_condition() -> None:
    rows = (
        {
            "source_listing_id": "new-001",
            "year": 2024,
            "make": "Ford",
            "model": "Escape",
            "price": 31_500,
            "vehicle_status": "New",
        },
        {
            "source_listing_id": "unknown-001",
            "year": 2024,
            "make": "Ford",
            "model": "Escape",
            "price": 31_500,
            "vehicle_status": "fleet return",
        },
    )

    batch = normalize_vehicle_records(rows, mapping=VehicleFieldMapping(), context=_context())

    assert len(batch.listings) == 1
    assert batch.listings[0].vehicle_status == "new"
    assert batch.listings[0].condition is None
    assert len(batch.rejected_listings) == 1
    assert batch.rejected_listings[0].source_listing_id == "unknown-001"


def test_structured_values_are_quarantined_instead_of_stringified_or_concatenated() -> None:
    rows = (
        {
            "source_listing_id": "structured-make",
            "year": 2021,
            "make": {"brand": "Toyota"},
            "model": "Camry",
            "price": 23_900,
        },
        {
            "source_listing_id": "structured-model",
            "year": 2021,
            "make": "Toyota",
            "model": ["Camry"],
            "price": 23_900,
        },
        {
            "source_listing_id": "structured-price",
            "year": 2021,
            "make": "Toyota",
            "model": "Camry",
            "price": {"asking": 23_900, "monthly": 399},
        },
    )

    batch = normalize_vehicle_records(rows, mapping=VehicleFieldMapping(), context=_context())

    assert batch.listings == ()
    assert {item.source_listing_id for item in batch.rejected_listings} == {
        "structured-make",
        "structured-model",
        "structured-price",
    }


def test_price_parser_rejects_negative_and_mismatched_currency_values() -> None:
    rows = (
        {
            "source_listing_id": "negative",
            "year": 2021,
            "make": "Toyota",
            "model": "Camry",
            "price": "-$23,900",
        },
        {
            "source_listing_id": "wrong-currency",
            "year": 2021,
            "make": "Toyota",
            "model": "Camry",
            "price": "23,900 CAD",
        },
    )

    batch = normalize_vehicle_records(rows, mapping=VehicleFieldMapping(), context=_context())

    assert batch.listings == ()
    assert len(batch.rejected_listings) == 2


def test_nonfinite_source_record_is_quarantined_before_hashing() -> None:
    row = {
        "source_listing_id": "nonfinite-extra",
        "year": 2021,
        "make": "Toyota",
        "model": "Camry",
        "price": 23_900,
        "unmapped_quality_score": float("nan"),
    }

    batch = normalize_vehicle_records((row,), mapping=VehicleFieldMapping(), context=_context())

    assert batch.listings == ()
    assert len(batch.rejected_listings) == 1
    assert batch.rejected_listings[0].source_listing_id == "row-1"
    assert "JSON-serializable" in batch.rejected_listings[0].message


def test_non_us_normalization_context_cannot_emit_a_listing() -> None:
    row = {
        "source_listing_id": "canadian-record",
        "year": 2021,
        "make": "Toyota",
        "model": "Camry",
        "price": 23_900,
    }

    batch = normalize_vehicle_records(
        (row,),
        mapping=VehicleFieldMapping(),
        context=replace(_context(), market_country="CA"),
    )

    assert batch.listings == ()
    assert "market_country must be US" in batch.rejected_listings[0].message


def test_currency_symbol_must_match_the_reviewed_source_currency() -> None:
    row = {
        "source_listing_id": "euro-source-with-dollar",
        "year": 2021,
        "make": "Toyota",
        "model": "Camry",
        "price": "$23,900",
    }

    batch = normalize_vehicle_records(
        (row,),
        mapping=VehicleFieldMapping(),
        context=replace(_context(), currency="EUR"),
    )

    assert batch.listings == ()
    assert "price symbol does not match" in batch.rejected_listings[0].message
