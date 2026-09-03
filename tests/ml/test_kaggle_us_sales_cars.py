from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import pytest
from autovalue_ml.acquisition.sources.kaggle_us_sales_cars import (
    KAGGLE_US_SALES_CARS_HEADER,
    KaggleUSSalesCarsError,
    load_kaggle_us_sales_cars_review,
    prepare_kaggle_us_sales_cars_training_rows,
    process_kaggle_us_sales_cars_csv,
    require_kaggle_us_sales_cars_ml_training_approval,
    verify_kaggle_us_sales_cars_artifact_set,
)

_TODAY = date(2026, 8, 28)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REVIEW_TEMPLATE = _PROJECT_ROOT / "docs" / "data-reviews" / "kaggle-us-sales-cars-v2.review.json"


@dataclass(frozen=True)
class _Fixture:
    source: Path
    review: Path
    output: Path
    dealer_secrets: tuple[str, ...]


def _base_rows() -> list[list[str]]:
    certified = [
        "Toyota",
        "Camry",
        "2021",
        "Certified",
        "5000.0",
        "Private Dealer Three",
        "25000.0",
    ]
    return [
        [
            "Tesla",
            "Model Y",
            "2024",
            "New",
            "",
            "Private Dealer One",
            "50000.0",
        ],
        [
            "Ford",
            "F-150",
            "2020",
            "Used",
            "10000.0",
            "Private Dealer Two",
            "30000.0",
        ],
        certified,
        certified.copy(),
        [
            "Honda",
            "Civic",
            "2019",
            "Used",
            "20000.0",
            "Private Dealer Four",
            "",
        ],
    ]


def _make_fixture(
    tmp_path: Path,
    rows: Sequence[Sequence[str]],
    *,
    status_counts: dict[str, int],
    target_valid_rows: int,
    duplicate_rows: int,
    rows_after_deduplication: int,
    invalid_price_rows: int,
    missing_mileage_rows: int,
    year_min: int,
    year_max: int,
    price_min: int,
    price_max: int,
    ml_approved: bool = True,
    encoding: str = "utf-16",
    header: Sequence[str] = KAGGLE_US_SALES_CARS_HEADER,
) -> _Fixture:
    source = tmp_path / "data" / "raw" / "kaggle_us_sales_cars_v2" / "cars.csv"
    source.parent.mkdir(parents=True)
    with source.open("w", encoding=encoding, newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    payload = source.read_bytes()

    review_value = cast(dict[str, Any], json.loads(_REVIEW_TEMPLATE.read_text(encoding="utf-8")))
    csv_pin = cast(dict[str, Any], cast(dict[str, Any], review_value["retrieval"])["csv"])
    csv_pin["size_bytes"] = len(payload)
    csv_pin["sha256"] = hashlib.sha256(payload).hexdigest()
    csv_pin["row_count"] = len(rows)
    quality = cast(dict[str, Any], review_value["quality_profile"])
    quality.update(
        {
            "raw_rows": len(rows),
            "status_counts": status_counts,
            "target_valid_rows_before_deduplication": target_valid_rows,
            "target_valid_exact_duplicate_rows": duplicate_rows,
            "target_valid_rows_after_exact_deduplication": rows_after_deduplication,
            "rows_missing_or_invalid_price": invalid_price_rows,
            "target_valid_rows_missing_mileage": missing_mileage_rows,
            "year_min_for_target_valid_rows": year_min,
            "year_max_for_target_valid_rows": year_max,
            "price_min_usd_for_target_valid_rows": price_min,
            "price_max_usd_for_target_valid_rows": price_max,
        }
    )
    if not ml_approved:
        permissions = cast(dict[str, Any], review_value["permissions"])
        permissions["ml_training_and_evaluation"] = "pending"
        source_review = cast(dict[str, Any], review_value["source"])
        evidence = cast(dict[str, Any], source_review["permission_evidence"])
        evidence["ml_training_permission"] = "pending_no_ml_training_permission"
    review = tmp_path / "source.review.json"
    review.write_text(
        json.dumps(review_value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dealer_secrets = tuple(row[5] for row in rows if len(row) == 7)
    return _Fixture(
        source=source,
        review=review,
        output=tmp_path / "processed" / "asking-candidate.csv",
        dealer_secrets=dealer_secrets,
    )


def _base_fixture(tmp_path: Path, *, ml_approved: bool = True) -> _Fixture:
    return _make_fixture(
        tmp_path,
        _base_rows(),
        status_counts={"New": 1, "Used": 2, "Certified": 2},
        target_valid_rows=4,
        duplicate_rows=1,
        rows_after_deduplication=3,
        invalid_price_rows=1,
        missing_mileage_rows=1,
        year_min=2020,
        year_max=2024,
        price_min=25_000,
        price_max=50_000,
        ml_approved=ml_approved,
    )


def _read_candidate(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, strict=True))


def _set_nested(value: dict[str, Any], path: tuple[str, ...], replacement: object) -> None:
    cursor = value
    for key in path[:-1]:
        cursor = cast(dict[str, Any], cursor[key])
    cursor[path[-1]] = replacement


def test_processes_all_statuses_deduplicates_and_preserves_privacy(tmp_path: Path) -> None:
    fixture = _base_fixture(tmp_path)

    artifacts = process_kaggle_us_sales_cars_csv(
        fixture.source, fixture.review, fixture.output, today=_TODAY
    )

    assert artifacts.metrics.rows_seen == 5
    assert artifacts.metrics.rows_accepted == 3
    assert artifacts.metrics.exact_duplicate_rows == 1
    assert artifacts.metrics.quarantined_rows == 1
    assert artifacts.metrics.target_valid_missing_mileage_rows == 1
    assert dict(artifacts.metrics.source_status_counts) == {
        "New": 1,
        "Used": 2,
        "Certified": 2,
    }
    rows = _read_candidate(artifacts.candidate_path)
    assert {row["vehicle_status"] for row in rows} == {"new", "used", "certified"}
    assert all(row["condition"] == "" for row in rows)
    assert next(row for row in rows if row["vehicle_status"] == "new")["mileage"] == ""
    assert all(row["price_kind"] == "asking" for row in rows)
    assert all(row["sale_status"] == "active" for row in rows)
    assert all(row["market_country"] == "US" and row["currency"] == "USD" for row in rows)
    assert all(row["source_listing_id"].startswith("row-") for row in rows)
    assert all("select=cars.csv" in row["canonical_url"] for row in rows)

    derived_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            artifacts.candidate_path,
            artifacts.quarantine_path,
            artifacts.manifest_path,
            artifacts.readiness_path,
        )
    )
    assert all(secret not in derived_text for secret in fixture.dealer_secrets)
    quarantine = artifacts.quarantine_path.read_text(encoding="utf-8")
    assert "Honda" not in quarantine
    assert "Civic" not in quarantine
    assert "price_invalid" in quarantine
    verify_kaggle_us_sales_cars_artifact_set(artifacts.manifest_path, fixture.review, today=_TODAY)


def test_training_rows_are_separately_approved_and_river_shaped(tmp_path: Path) -> None:
    fixture = _base_fixture(tmp_path)
    artifacts = process_kaggle_us_sales_cars_csv(
        fixture.source, fixture.review, fixture.output, today=_TODAY
    )

    training = prepare_kaggle_us_sales_cars_training_rows(
        artifacts.candidate_path,
        artifacts.manifest_path,
        fixture.review,
        today=_TODAY,
    )
    rows = list(training)

    assert len(rows) == 3
    assert all("vehicle_status" in features for features, _ in rows)
    assert all("condition" not in features for features, _ in rows)
    new_features = next(features for features, _ in rows if features["vehicle_status"] == "new")
    assert "mileage" not in new_features
    assert sorted(target for _, target in rows) == [25_000.0, 30_000.0, 50_000.0]


def test_training_stream_rejects_changed_or_wrong_location_candidate(tmp_path: Path) -> None:
    fixture = _base_fixture(tmp_path)
    artifacts = process_kaggle_us_sales_cars_csv(
        fixture.source, fixture.review, fixture.output, today=_TODAY
    )
    training = prepare_kaggle_us_sales_cars_training_rows(
        artifacts.candidate_path,
        artifacts.manifest_path,
        fixture.review,
        today=_TODAY,
    )
    other_candidate = tmp_path / "other" / artifacts.candidate_path.name
    other_candidate.parent.mkdir()
    other_candidate.write_bytes(artifacts.candidate_path.read_bytes())
    with pytest.raises(KaggleUSSalesCarsError, match="differs"):
        prepare_kaggle_us_sales_cars_training_rows(
            other_candidate,
            artifacts.manifest_path,
            fixture.review,
            today=_TODAY,
        )

    artifacts.candidate_path.write_text("changed", encoding="utf-8")
    with pytest.raises(KaggleUSSalesCarsError, match="changed after training approval"):
        list(training)


def test_acquisition_can_run_while_ml_reuse_remains_pending(tmp_path: Path) -> None:
    fixture = _base_fixture(tmp_path, ml_approved=False)
    review = load_kaggle_us_sales_cars_review(fixture.review, today=_TODAY)
    assert review.approved_for_acquisition is True
    assert review.approved_for_ml_training is False
    with pytest.raises(KaggleUSSalesCarsError, match="does not approve ML training"):
        require_kaggle_us_sales_cars_ml_training_approval(review)

    artifacts = process_kaggle_us_sales_cars_csv(
        fixture.source, fixture.review, fixture.output, today=_TODAY
    )
    with pytest.raises(KaggleUSSalesCarsError, match="does not approve ML training"):
        prepare_kaggle_us_sales_cars_training_rows(
            artifacts.candidate_path,
            artifacts.manifest_path,
            fixture.review,
            today=_TODAY,
        )


def test_unknown_status_is_quarantined_without_source_values(tmp_path: Path) -> None:
    rows = [
        ["Ford", "Escape", "2020", "Used", "10000.0", "Allowed Secret", "20000"],
        [
            "SensitiveBrand",
            "SensitiveModel",
            "2021",
            "LeaseOnly",
            "2000.0",
            "Rejected Secret",
            "25000",
        ],
    ]
    fixture = _make_fixture(
        tmp_path,
        rows,
        status_counts={"New": 0, "Used": 1, "Certified": 0},
        target_valid_rows=1,
        duplicate_rows=0,
        rows_after_deduplication=1,
        invalid_price_rows=0,
        missing_mileage_rows=0,
        year_min=2020,
        year_max=2020,
        price_min=20_000,
        price_max=20_000,
    )

    artifacts = process_kaggle_us_sales_cars_csv(
        fixture.source, fixture.review, fixture.output, today=_TODAY
    )

    assert artifacts.metrics.unknown_status_rows == 1
    assert artifacts.metrics.quarantine_reason_counts == {"status_invalid": 1}
    quarantine = artifacts.quarantine_path.read_text(encoding="utf-8")
    assert "SensitiveBrand" not in quarantine
    assert "SensitiveModel" not in quarantine
    assert "LeaseOnly" not in quarantine
    assert "Rejected Secret" not in quarantine


def test_exact_dedup_uses_dealer_transiently_but_never_emits_it(tmp_path: Path) -> None:
    rows = [
        ["Ford", "Escape", "2020", "Used", "10000.0", "Secret A", "20000"],
        ["Ford", "Escape", "2020", "Used", "10000.0", "Secret B", "20000"],
    ]
    fixture = _make_fixture(
        tmp_path,
        rows,
        status_counts={"New": 0, "Used": 2, "Certified": 0},
        target_valid_rows=2,
        duplicate_rows=0,
        rows_after_deduplication=2,
        invalid_price_rows=0,
        missing_mileage_rows=0,
        year_min=2020,
        year_max=2020,
        price_min=20_000,
        price_max=20_000,
    )

    artifacts = process_kaggle_us_sales_cars_csv(
        fixture.source, fixture.review, fixture.output, today=_TODAY
    )
    candidate = artifacts.candidate_path.read_text(encoding="utf-8")

    assert artifacts.metrics.rows_accepted == 2
    assert artifacts.metrics.exact_duplicate_rows == 0
    assert "Secret A" not in candidate and "Secret B" not in candidate


def test_invalid_core_values_are_quarantined_with_codes_only(tmp_path: Path) -> None:
    rows = [
        ["Ford", "Escape", "2020", "Used", "10000", "Secret 1", "20000"],
        ["", "Escape", "2020", "Used", "10000", "Secret 2", "20000"],
        ["Ford", "Escape", "year", "Used", "10000", "Secret 3", "20000"],
        ["Ford", "Escape", "2020", "Used", "1.5", "Secret 4", "20000"],
        ["Ford", "Escape", "2020", "Used", "10000", "Secret 5", "NaN"],
    ]
    fixture = _make_fixture(
        tmp_path,
        rows,
        status_counts={"New": 0, "Used": 5, "Certified": 0},
        target_valid_rows=1,
        duplicate_rows=0,
        rows_after_deduplication=1,
        invalid_price_rows=1,
        missing_mileage_rows=0,
        year_min=2020,
        year_max=2020,
        price_min=20_000,
        price_max=20_000,
    )

    artifacts = process_kaggle_us_sales_cars_csv(
        fixture.source, fixture.review, fixture.output, today=_TODAY
    )

    assert dict(artifacts.metrics.quarantine_reason_counts) == {
        "brand_missing": 1,
        "mileage_invalid": 1,
        "price_invalid": 1,
        "year_invalid": 1,
    }
    quarantine = artifacts.quarantine_path.read_text(encoding="utf-8")
    assert all(secret not in quarantine for secret in fixture.dealer_secrets)
    assert "Escape" not in quarantine


def test_raw_hash_header_encoding_and_path_fail_closed(tmp_path: Path) -> None:
    fixture = _base_fixture(tmp_path)
    fixture.source.write_bytes(fixture.source.read_bytes() + b"tamper")
    with pytest.raises(KaggleUSSalesCarsError, match="byte size"):
        process_kaggle_us_sales_cars_csv(
            fixture.source, fixture.review, fixture.output, today=_TODAY
        )

    wrong_header = _make_fixture(
        tmp_path / "header",
        [["Ford", "Escape", "2020", "Used", "100", "Secret", "20000"]],
        status_counts={"New": 0, "Used": 1, "Certified": 0},
        target_valid_rows=1,
        duplicate_rows=0,
        rows_after_deduplication=1,
        invalid_price_rows=0,
        missing_mileage_rows=0,
        year_min=2020,
        year_max=2020,
        price_min=20_000,
        price_max=20_000,
        header=("Maker", "Model", "Year", "Status", "Mileage", "Dealer", "Price"),
    )
    with pytest.raises(KaggleUSSalesCarsError, match="header"):
        process_kaggle_us_sales_cars_csv(
            wrong_header.source, wrong_header.review, wrong_header.output, today=_TODAY
        )

    wrong_encoding = _make_fixture(
        tmp_path / "encoding",
        [["Ford", "Escape", "2020", "Used", "100", "Secret", "20000"]],
        status_counts={"New": 0, "Used": 1, "Certified": 0},
        target_valid_rows=1,
        duplicate_rows=0,
        rows_after_deduplication=1,
        invalid_price_rows=0,
        missing_mileage_rows=0,
        year_min=2020,
        year_max=2020,
        price_min=20_000,
        price_max=20_000,
        encoding="utf-8-sig",
    )
    with pytest.raises(KaggleUSSalesCarsError, match="UTF-16"):
        process_kaggle_us_sales_cars_csv(
            wrong_encoding.source, wrong_encoding.review, wrong_encoding.output, today=_TODAY
        )

    moved_source = tmp_path / "moved.csv"
    moved_source.write_bytes(wrong_encoding.source.read_bytes())
    with pytest.raises(KaggleUSSalesCarsError, match="reviewed path"):
        process_kaggle_us_sales_cars_csv(
            moved_source, wrong_encoding.review, wrong_encoding.output, today=_TODAY
        )


def test_derived_tampering_and_stale_review_lineage_fail_closed(tmp_path: Path) -> None:
    fixture = _base_fixture(tmp_path)
    artifacts = process_kaggle_us_sales_cars_csv(
        fixture.source, fixture.review, fixture.output, today=_TODAY
    )
    artifacts.candidate_path.write_text(
        artifacts.candidate_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(KaggleUSSalesCarsError, match="hash does not match"):
        verify_kaggle_us_sales_cars_artifact_set(
            artifacts.manifest_path, fixture.review, today=_TODAY
        )

    artifacts = process_kaggle_us_sales_cars_csv(
        fixture.source, fixture.review, fixture.output, today=_TODAY
    )
    review_value = json.loads(fixture.review.read_text(encoding="utf-8"))
    review_value["notes"].append("A later review edit changes the lineage digest.")
    fixture.review.write_text(json.dumps(review_value), encoding="utf-8")
    with pytest.raises(KaggleUSSalesCarsError, match="lineage"):
        verify_kaggle_us_sales_cars_artifact_set(
            artifacts.manifest_path, fixture.review, today=_TODAY
        )


def test_review_schema_and_permission_evidence_fail_closed(tmp_path: Path) -> None:
    fixture = _base_fixture(tmp_path)
    value = json.loads(fixture.review.read_text(encoding="utf-8"))
    value["source"]["permission_evidence"]["ml_training_permission"] = "unverified"
    fixture.review.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(KaggleUSSalesCarsError, match="permission evidence"):
        load_kaggle_us_sales_cars_review(fixture.review, today=_TODAY)

    fixture = _base_fixture(tmp_path / "extra")
    value = json.loads(fixture.review.read_text(encoding="utf-8"))
    value["unexpected"] = True
    fixture.review.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(KaggleUSSalesCarsError, match="fields"):
        load_kaggle_us_sales_cars_review(fixture.review, today=_TODAY)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("review_schema_version",), 2, "schema"),
        (("decision",), "pending", "does not approve"),
        (("project_role",), "mixed_target", "modeling role"),
        (("reviewed_on",), "2099-01-01", "future"),
        (("training_status",), "", "training_status"),
        (("source", "uploader"), "someone-else", "source identity"),
        (("retrieval", "reviewed_on"), "2026-08-27", "dates differ"),
        (("retrieval", "retrieved_on"), "2026-08-29", "retrieval date"),
        (("retrieval", "required_method"), "manual copy", "download method"),
        (("retrieval", "csv_path"), "../cars.csv", "safe relative"),
        (("retrieval", "csv", "file_name"), "other.csv", "filename or encoding"),
        (("retrieval", "csv", "columns"), ["Brand"], "columns"),
        (("permissions", "local_storage"), "pending", "private acquisition"),
        (("permissions", "ml_training_and_evaluation"), "pending", "inconsistent"),
        (("permissions", "autovalue_direct_scraping_of_cars_com"), "approved", "prohibited"),
        (("market_scope", "currency"), "EUR", "market scope"),
        (("target", "meaning"), "sale price", "target semantics"),
        (("quality_profile", "raw_rows"), 99, "row counts"),
        (("quality_profile", "status_counts", "Used"), 99, "status counts"),
        (
            ("quality_profile", "target_valid_rows_before_deduplication"),
            99,
            "quality counts",
        ),
        (("quality_profile", "year_min_for_target_valid_rows"), 2025, "ranges"),
        (("required_processing_gates",), ["skip integrity"], "processing gates"),
        (("publication_policy", "blocked_pending_review"), ["raw dataset files"], "publication"),
    ],
)
def test_review_semantic_tampering_fails_closed(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: object,
    message: str,
) -> None:
    fixture = _base_fixture(tmp_path)
    value = cast(dict[str, Any], json.loads(fixture.review.read_text(encoding="utf-8")))
    _set_nested(value, path, replacement)
    fixture.review.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(KaggleUSSalesCarsError, match=message):
        load_kaggle_us_sales_cars_review(fixture.review, today=_TODAY)


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '{"duplicate":1,"duplicate":2}',
        '{"nonfinite":NaN}',
        "not-json",
    ],
)
def test_review_requires_strict_json(tmp_path: Path, payload: str) -> None:
    review = tmp_path / "review.json"
    review.write_text(payload, encoding="utf-8")

    with pytest.raises(KaggleUSSalesCarsError, match="JSON"):
        load_kaggle_us_sales_cars_review(review, today=_TODAY)
