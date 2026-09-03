"""Contract checks for the published common listing JSON schema."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from autovalue_ml.acquisition.contracts import PriceKind, VehicleListingSnapshot
from jsonschema import Draft202012Validator, FormatChecker


def _valid_listing() -> VehicleListingSnapshot:
    return VehicleListingSnapshot(
        source_id="reviewed-source",
        source_listing_id="listing-001",
        canonical_url="https://example.test/vehicles/listing-001",
        observed_at=datetime(2026, 8, 27, tzinfo=UTC),
        market_country="US",
        year=2021,
        make="Toyota",
        model="Camry",
        trim=None,
        mileage=34_120,
        mileage_unit="miles",
        condition=None,
        engine="2.5L I4",
        drivetrain="FWD",
        accident_status=None,
        accident_count=None,
        owner_count=1,
        vehicle_type="Sedan",
        price_cents=2_390_000,
        currency="USD",
        price_kind=PriceKind.ASKING,
        sale_status="active",
        raw_content_sha256="a" * 64,
        parser_version="fixture/1",
        normalization_version="1",
        ingestion_run_id="run-001",
        authorization_policy_id="reviewed-policy-v1",
        vehicle_status="used",
    )


def test_json_schema_matches_the_normalized_dataclass_contract() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    schema_path = repository_root / "ml" / "schemas" / "vehicle-listing-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    dataclass_fields = set(VehicleListingSnapshot.__dataclass_fields__)
    assert set(schema["required"]) == dataclass_fields
    assert set(schema["properties"]) == dataclass_fields
    assert set(schema["properties"]["price_kind"]["enum"]) == {
        price_kind.value for price_kind in PriceKind
    }
    assert schema["additionalProperties"] is False

    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    validator.validate(_valid_listing().to_dict())

    invalid = _valid_listing().to_dict()
    invalid["make"] = ""
    assert any(error.json_path == "$.make" for error in validator.iter_errors(invalid))


def test_runtime_contract_rejects_values_the_published_schema_forbids() -> None:
    listing = _valid_listing()

    with pytest.raises(ValueError, match="required listing text"):
        replace(listing, make="")
    with pytest.raises(ValueError, match="mileage_unit"):
        replace(listing, mileage_unit="kilometers")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(listing, raw_content_sha256="z" * 64)
    with pytest.raises(ValueError, match=r"HTTP\(S\)"):
        replace(listing, canonical_url="file:///tmp/listing")
    with pytest.raises(ValueError, match="optional listing text"):
        replace(listing, trim=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer cents"):
        replace(listing, price_cents=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="market_country must be US"):
        replace(listing, market_country="CA")
    with pytest.raises(ValueError, match="currency must be USD"):
        replace(listing, currency="CAD")
    with pytest.raises(ValueError, match="vehicle_status"):
        replace(listing, vehicle_status="preowned")
