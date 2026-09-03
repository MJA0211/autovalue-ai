"""Security and contract tests for the reviewed Kaggle vehicle-sales adapter."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import cast

import pytest
from autovalue_ml.acquisition.sources.kaggle_vehicle_sales import (
    KAGGLE_VEHICLE_SALES_HEADER,
    KaggleVehicleSalesError,
    load_kaggle_vehicle_sales_review,
    prepare_kaggle_training_rows,
    process_kaggle_vehicle_sales_csv,
    require_kaggle_ml_training_approval,
    verify_kaggle_candidate_artifact_set,
)

_TODAY = date(2026, 8, 28)
_COMMITTED_REVIEW = Path("docs/data-reviews/kaggle-vehicle-sales-data-v1.review.json")
_SALE_DATE = "Tue Dec 16 2014 12:30:00 GMT-0800 (PST)"
_NEXT_SALE_DATE = "Wed Dec 17 2014 12:30:00 GMT-0800 (PST)"


def _row(
    *,
    vin: str = "1HGCM82633A004352",
    state: str = "ca",
    make: str = "Toyota",
    model: str = "Camry",
    trim: str = "SE",
    body: str = "Sedan",
    condition: str = "45",
    odometer: str = "999999",
    seller: str = "PRIVATE-SELLER-MARKER",
    mmr: str = "11000",
    sellingprice: str = "12000",
    saledate: str = _SALE_DATE,
) -> list[str]:
    return [
        "2014",
        make,
        model,
        trim,
        body,
        "automatic",
        vin,
        state,
        condition,
        odometer,
        "black",
        "gray",
        seller,
        mmr,
        sellingprice,
        saledate,
    ]


def _csv_bytes(
    rows: Sequence[Sequence[str]],
    *,
    header: Sequence[str] = KAGGLE_VEHICLE_SALES_HEADER,
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _review_dict() -> dict[str, object]:
    return cast(dict[str, object], json.loads(_COMMITTED_REVIEW.read_text(encoding="utf-8")))


def _section(value: dict[str, object], key: str) -> dict[str, object]:
    return cast(dict[str, object], value[key])


def _fixture(
    tmp_path: Path,
    rows: Sequence[Sequence[str]],
    *,
    header: Sequence[str] = KAGGLE_VEHICLE_SALES_HEADER,
    review_mutation: Callable[[dict[str, object]], object] | None = None,
) -> tuple[Path, Path, Path, bytes]:
    payload = _csv_bytes(rows, header=header)
    source_path = tmp_path / "data" / "raw" / "kaggle_vehicle_sales_v1" / "car_prices.csv"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(payload)

    review = _review_dict()
    retrieval = _section(review, "retrieval")
    csv_pin = _section(retrieval, "csv")
    csv_pin["size_bytes"] = len(payload)
    csv_pin["sha256"] = hashlib.sha256(payload).hexdigest()
    csv_pin["row_count"] = len(rows)
    market = _section(review, "market_scope")
    market["raw_row_count"] = len(rows)
    market["us_50_plus_dc_row_count"] = 0
    market["us_50_plus_dc_valid_target_and_date_row_count"] = 0
    market["us_50_plus_dc_core_complete_row_count"] = 0
    if review_mutation is not None:
        review_mutation(review)
    review_path = tmp_path / "source.review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    output_path = tmp_path / "private" / "kaggle-candidate.csv"
    return source_path, review_path, output_path, payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_committed_review_smoke_test_matches_the_real_artifact_pin() -> None:
    review = load_kaggle_vehicle_sales_review(_COMMITTED_REVIEW, today=_TODAY)

    assert review.review_id == "kaggle-vehicle-sales-data-v1-2026-08-28"
    assert review.expected_row_count == 558_837
    assert review.expected_size_bytes == 88_047_552
    assert review.expected_sha256 == (
        "32ba3ce51664e6a12c0c927ed193b41e3c4743fdf18bc0317389892aed27f556"
    )
    assert review.expected_csv_path.as_posix() == (
        "data/raw/kaggle_vehicle_sales_v1/car_prices.csv"
    )
    assert review.approved_for_acquisition is True
    assert review.approved_for_ml_training is True


def test_processes_only_safe_us_rows_and_records_aggregate_metrics(tmp_path: Path) -> None:
    first = _row()
    rows = [
        first,
        _row(
            sellingprice="12500",
            saledate=_NEXT_SALE_DATE,
            odometer="42000",
            condition="4",
        ),
        _row(vin="1FAHP2F80DG100001", state="on"),
        _row(vin="1FAHP2F80DG100002", state="pr"),
        _row(vin="1FAHP2F80DG100003", make=""),
        _row(vin="1FAHP2F80DG100004", sellingprice="$12,000"),
        _row(vin=""),
        _row(vin="1FAHP2F80DG100005", condition="NaN"),
        _row(vin="1FAHP2F80DG100006", model='=HYPERLINK("https://bad")'),
        ["malformed", "row", "width"],
        first,
    ]
    source, review_path, output, _ = _fixture(tmp_path, rows)

    result = process_kaggle_vehicle_sales_csv(source, review_path, output, today=_TODAY)

    assert result.metrics.rows_seen == 11
    assert result.metrics.rows_accepted == 2
    assert result.metrics.non_us_rows == 2
    assert result.metrics.quarantined_rows == 8
    assert result.metrics.exact_duplicate_rows == 1
    assert result.metrics.repeated_vin_rows == 1
    assert result.metrics.distinct_repeated_vins == 1
    assert result.metrics.missing_or_invalid_vin_rows == 1
    assert dict(result.metrics.quarantine_reason_counts) == {
        "condition_invalid": 1,
        "csv_formula_injection": 1,
        "make_missing": 1,
        "market_not_us": 2,
        "row_width_invalid": 1,
        "sellingprice_invalid": 1,
        "vin_missing_or_invalid": 1,
    }

    with result.candidate_path.open(encoding="utf-8", newline="") as source_file:
        candidate_rows = list(csv.DictReader(source_file))
    assert len(candidate_rows) == 2
    assert candidate_rows[0]["source_listing_id"] == "row-000000002"
    assert candidate_rows[0]["market_country"] == "US"
    assert candidate_rows[0]["currency"] == "USD"
    assert candidate_rows[0]["price_kind"] == "completed_sale"
    assert candidate_rows[0]["price_cents"] == "1200000"
    assert candidate_rows[0]["condition"] == "4.5"
    assert candidate_rows[0]["vehicle_status"] == ""
    assert candidate_rows[0]["mileage"] == ""
    assert candidate_rows[0]["drivetrain"] == ""
    assert candidate_rows[1]["condition"] == "4.0"
    assert candidate_rows[1]["mileage"] == "42000"

    candidate_text = result.candidate_path.read_text(encoding="utf-8")
    quarantine_text = result.quarantine_path.read_text(encoding="utf-8")
    for forbidden_value in (
        "1HGCM82633A004352",
        "PRIVATE-SELLER-MARKER",
        "11000",
        "automatic",
    ):
        assert forbidden_value not in candidate_text
        assert forbidden_value not in quarantine_text
    assert '"safe_record_sha256"' in quarantine_text
    assert '"record_sha256"' not in quarantine_text

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["review_sha256"] == _sha256(review_path)
    assert manifest["raw_source_sha256"] == _sha256(source)
    assert manifest["training_readiness"] == (
        "blocked_pending_reviewed_chronological_vin_isolated_split"
    )
    assert manifest["feature_allowlist"] == [
        "year",
        "make",
        "model",
        "trim",
        "mileage",
        "condition",
        "vehicle_type",
    ]
    verify_kaggle_candidate_artifact_set(result.manifest_path, review_path, today=_TODAY)


def test_unsplit_candidate_cannot_enter_training_even_with_ml_permission(
    tmp_path: Path,
) -> None:
    source, review_path, output, _ = _fixture(tmp_path, [_row()])
    result = process_kaggle_vehicle_sales_csv(source, review_path, output, today=_TODAY)

    with pytest.raises(KaggleVehicleSalesError, match="chronological.*VIN"):
        prepare_kaggle_training_rows(
            result.candidate_path,
            result.manifest_path,
            review_path,
            today=_TODAY,
        )


def test_ml_permission_is_independent_from_private_acquisition(tmp_path: Path) -> None:
    def make_ml_pending(review: dict[str, object]) -> None:
        _section(review, "permissions")["ml_training_and_evaluation"] = "pending"

    source, review_path, output, _ = _fixture(
        tmp_path,
        [_row()],
        review_mutation=make_ml_pending,
    )
    review = load_kaggle_vehicle_sales_review(review_path, today=_TODAY)

    assert review.approved_for_acquisition is True
    assert review.approved_for_ml_training is False
    with pytest.raises(KaggleVehicleSalesError, match="does not approve ML training"):
        require_kaggle_ml_training_approval(review)

    result = process_kaggle_vehicle_sales_csv(source, review_path, output, today=_TODAY)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["approved_for_ml_training"] is False
    with pytest.raises(KaggleVehicleSalesError, match="does not approve ML training"):
        prepare_kaggle_training_rows(
            result.candidate_path,
            result.manifest_path,
            review_path,
            today=_TODAY,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda review: _section(review, "retrieval").pop("retrieved_on"),
            "retrieval fields",
        ),
        (
            lambda review: _section(review, "retrieval").__setitem__("retrieved_on", "2026-08-29"),
            "retrieval date cannot follow",
        ),
        (
            lambda review: _section(review, "source").__setitem__(
                "dataset_url", "https://example.com/vehicle-sales-data"
            ),
            "official Kaggle host",
        ),
        (
            lambda review: _section(review, "market_scope").__setitem__(
                "us_50_plus_dc_row_count", 99
            ),
            "market row counts are inconsistent",
        ),
        (
            lambda review: review.__setitem__("training_status", "trained"),
            "training status is not_started",
        ),
    ],
)
def test_review_contract_fails_closed(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    _, review_path, _, _ = _fixture(tmp_path, [_row()], review_mutation=mutation)

    with pytest.raises(KaggleVehicleSalesError, match=message):
        load_kaggle_vehicle_sales_review(review_path, today=_TODAY)


def test_rejects_raw_artifact_tampering_before_creating_output(tmp_path: Path) -> None:
    source, review_path, output, _ = _fixture(tmp_path, [_row()])
    source.write_bytes(source.read_bytes() + b"\n")

    with pytest.raises(KaggleVehicleSalesError, match="byte size does not match"):
        process_kaggle_vehicle_sales_csv(source, review_path, output, today=_TODAY)
    assert not output.exists()
    assert not output.with_suffix(".ready.json").exists()


def test_rejects_wrong_exact_header_after_successful_hash_verification(tmp_path: Path) -> None:
    wrong_header = list(KAGGLE_VEHICLE_SALES_HEADER)
    wrong_header[13] = "market_estimate"
    source, review_path, output, _ = _fixture(
        tmp_path,
        [_row()],
        header=wrong_header,
    )

    with pytest.raises(KaggleVehicleSalesError, match="exact 16-column schema"):
        process_kaggle_vehicle_sales_csv(source, review_path, output, today=_TODAY)
    assert not output.with_suffix(".ready.json").exists()


def test_rejects_output_collision_without_modifying_the_raw_source(tmp_path: Path) -> None:
    source, review_path, _, payload = _fixture(tmp_path, [_row()])

    with pytest.raises(KaggleVehicleSalesError, match="must not overwrite an input"):
        process_kaggle_vehicle_sales_csv(source, review_path, source, today=_TODAY)
    assert source.read_bytes() == payload


def test_requires_csv_candidate_suffix(tmp_path: Path) -> None:
    source, review_path, _, _ = _fixture(tmp_path, [_row()])

    with pytest.raises(KaggleVehicleSalesError, match="must use the .csv suffix"):
        process_kaggle_vehicle_sales_csv(
            source,
            review_path,
            tmp_path / "candidate.jsonl",
            today=_TODAY,
        )


def test_verifier_rejects_candidate_tampering(tmp_path: Path) -> None:
    source, review_path, output, _ = _fixture(tmp_path, [_row()])
    result = process_kaggle_vehicle_sales_csv(source, review_path, output, today=_TODAY)
    result.candidate_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(KaggleVehicleSalesError, match="candidate_file hash does not match"):
        verify_kaggle_candidate_artifact_set(
            result.manifest_path,
            review_path,
            today=_TODAY,
        )


def test_verifier_rejects_manifest_semantic_tampering_before_hash_marker(
    tmp_path: Path,
) -> None:
    source, review_path, output, _ = _fixture(tmp_path, [_row()])
    result = process_kaggle_vehicle_sales_csv(source, review_path, output, today=_TODAY)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["market_country"] = "CA"
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(KaggleVehicleSalesError, match="market_country is invalid"):
        verify_kaggle_candidate_artifact_set(
            result.manifest_path,
            review_path,
            today=_TODAY,
        )


def test_quarantine_hash_is_not_derived_from_private_row_values(tmp_path: Path) -> None:
    rejected = _row(state="on")
    source, review_path, output, _ = _fixture(tmp_path, [rejected])
    result = process_kaggle_vehicle_sales_csv(source, review_path, output, today=_TODAY)
    quarantine = json.loads(result.quarantine_path.read_text(encoding="utf-8"))
    expected = hashlib.sha256(f"{_sha256(source)}:row:2".encode("ascii")).hexdigest()
    raw_row_hash = hashlib.sha256(
        json.dumps(rejected, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert quarantine["safe_record_sha256"] == expected
    assert quarantine["safe_record_sha256"] != raw_row_hash


def test_condition_nan_and_infinity_are_quarantined_not_fatal(tmp_path: Path) -> None:
    source, review_path, output, _ = _fixture(
        tmp_path,
        [
            _row(vin="1FAHP2F80DG100001", condition="NaN"),
            _row(vin="1FAHP2F80DG100002", condition="Infinity"),
        ],
    )

    result = process_kaggle_vehicle_sales_csv(source, review_path, output, today=_TODAY)

    assert result.metrics.rows_accepted == 0
    assert dict(result.metrics.quarantine_reason_counts) == {"condition_invalid": 2}
    verify_kaggle_candidate_artifact_set(result.manifest_path, review_path, today=_TODAY)
