from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import cast

import pytest
from autovalue_ml.acquisition.sources import kaggle_vehicle_sales_split as split_module
from autovalue_ml.acquisition.sources.kaggle_vehicle_sales import (
    KAGGLE_VEHICLE_SALES_HEADER,
    KaggleVehicleSalesError,
    process_kaggle_vehicle_sales_csv,
)
from autovalue_ml.acquisition.sources.kaggle_vehicle_sales_split import (
    KaggleVehicleSalesSplitArtifactSet,
    KaggleVehicleSalesSplitError,
    KaggleVehicleSalesTrainingRows,
    VerifiedKaggleVehicleSalesSplit,
    load_kaggle_vehicle_sales_split_policy,
    prepare_kaggle_vehicle_sales_training_rows,
    process_kaggle_vehicle_sales_split,
    verify_kaggle_vehicle_sales_split_artifact_set,
)

_ROOT = Path(__file__).parents[2]
_COMMITTED_REVIEW = _ROOT / "docs/data-reviews/kaggle-vehicle-sales-data-v1.review.json"
_COMMITTED_POLICY = _ROOT / "docs/data-reviews/kaggle-vehicle-sales-v1.split.json"
_TODAY = date(2026, 8, 28)


def _source_row(
    vin: str,
    sale_date: str,
    *,
    make: str = "Ford",
    model: str = "Fusion",
    trim: str = "SE",
    body: str = "Sedan",
    mileage: str = "50000",
    condition: str = "45",
    price: str = "12000",
) -> list[str]:
    return [
        "2013",
        make,
        model,
        trim,
        body,
        "automatic",
        vin,
        "ca",
        condition,
        mileage,
        "black",
        "gray",
        "PRIVATE-SELLER-MARKER",
        "11000",
        price,
        sale_date,
    ]


def _sale_date(day: date) -> str:
    return f"{day:%a %b %d %Y} 12:00:00 GMT-0800 (PST)"


def _fixture_rows() -> list[list[str]]:
    return [
        _source_row("1FAHP2F80DG100001", _sale_date(date(2015, 5, 20))),
        _source_row("1FAHP2F80DG100001", _sale_date(date(2015, 6, 5)), price="12100"),
        _source_row("1FAHP2F80DG100002", _sale_date(date(2014, 12, 15))),
        _source_row("1FAHP2F80DG100002", _sale_date(date(2015, 1, 15)), price="12200"),
        _source_row("1FAHP2F80DG100003", _sale_date(date(2015, 2, 15))),
        _source_row("1FAHP2F80DG100004", _sale_date(date(2015, 4, 15))),
        _source_row("1FAHP2F80DG100005", _sale_date(date(2015, 5, 15))),
        _source_row("1FAHP2F80DG100006", _sale_date(date(2015, 6, 10))),
        _source_row("1FAHP2F80DG100007", _sale_date(date(2014, 12, 10))),
    ]


def _csv_payload(rows: Sequence[Sequence[str]]) -> bytes:
    import io

    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(KAGGLE_VEHICLE_SALES_HEADER)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_fixture_review(
    path: Path,
    source_payload: bytes,
    row_count: int,
    *,
    mutate: Callable[[dict[str, object]], None] | None = None,
) -> None:
    value = json.loads(_COMMITTED_REVIEW.read_text(encoding="utf-8"))
    retrieval = cast(dict[str, object], value["retrieval"])
    csv_pin = cast(dict[str, object], retrieval["csv"])
    csv_pin["size_bytes"] = len(source_payload)
    csv_pin["sha256"] = hashlib.sha256(source_payload).hexdigest()
    csv_pin["row_count"] = row_count
    market = cast(dict[str, object], value["market_scope"])
    market["raw_row_count"] = row_count
    market["us_50_plus_dc_row_count"] = row_count
    market["us_50_plus_dc_valid_target_and_date_row_count"] = row_count
    market["us_50_plus_dc_core_complete_row_count"] = row_count
    if mutate is not None:
        mutate(cast(dict[str, object], value))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(
    tmp_path: Path,
    *,
    rows: Sequence[Sequence[str]] | None = None,
    review_mutation: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, Path]:
    fixture_rows = list(_fixture_rows() if rows is None else rows)
    source_payload = _csv_payload(fixture_rows)
    raw = tmp_path / "data/raw/kaggle_vehicle_sales_v1/car_prices.csv"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(source_payload)
    review = tmp_path / "docs/data-reviews/kaggle-vehicle-sales-data-v1.review.json"
    _write_fixture_review(
        review,
        source_payload,
        len(fixture_rows),
        mutate=review_mutation,
    )
    policy = tmp_path / "docs/data-reviews/kaggle-vehicle-sales-v1.split.json"
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_bytes(_COMMITTED_POLICY.read_bytes())
    candidate = tmp_path / "data/interim/kaggle_vehicle_sales_v1.csv"
    candidate.parent.mkdir(parents=True)
    candidate_result = process_kaggle_vehicle_sales_csv(
        raw,
        review,
        candidate,
        today=_TODAY,
    )
    assignment = tmp_path / "data/processed/kaggle_vehicle_sales_v1/split_assignments.csv"
    return {
        "raw": raw,
        "review": review,
        "policy": policy,
        "candidate": candidate_result.candidate_path,
        "candidate_manifest": candidate_result.manifest_path,
        "assignment": assignment,
        "split_manifest": assignment.with_suffix(".manifest.json"),
    }


def _process(paths: dict[str, Path]) -> KaggleVehicleSalesSplitArtifactSet:
    return process_kaggle_vehicle_sales_split(
        paths["raw"],
        paths["candidate"],
        paths["candidate_manifest"],
        paths["review"],
        paths["policy"],
        paths["assignment"],
        today=_TODAY,
    )


def _verify(paths: dict[str, Path]) -> VerifiedKaggleVehicleSalesSplit:
    return verify_kaggle_vehicle_sales_split_artifact_set(
        paths["split_manifest"],
        paths["raw"],
        paths["candidate"],
        paths["candidate_manifest"],
        paths["review"],
        paths["policy"],
        today=_TODAY,
    )


def _reseal_split(paths: dict[str, Path]) -> None:
    assignment = paths["assignment"]
    assignment_payload = assignment.read_bytes()
    assignment_sha256 = hashlib.sha256(assignment_payload).hexdigest()
    manifest_path = paths["split_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assignment_sha256"] = assignment_sha256
    manifest["assignment_size_bytes"] = len(assignment_payload)
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_payload)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    ready_path = assignment.with_suffix(".ready.json")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    ready["assignment_sha256"] = assignment_sha256
    ready["manifest_sha256"] = manifest_sha256
    ready["artifact_set_id"] = hashlib.sha256(
        f"{manifest_sha256}|{assignment_sha256}".encode("ascii")
    ).hexdigest()
    ready_path.write_text(
        json.dumps(ready, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_committed_policy_is_byte_pinned_and_semantically_reviewed() -> None:
    policy = load_kaggle_vehicle_sales_split_policy(_COMMITTED_POLICY, today=_TODAY)

    assert policy.policy_id == "kaggle-vehicle-sales-v1-chronological-vin-isolated-v1"
    assert policy.policy_sha256 == (
        "4c0d4b68d2ad1b8bcbbfc89d1936e0b8ba77287f3f1bc3f97b9c3301224e6833"
    )
    assert policy.cutoff_date == date(2015, 6, 1)


def test_split_promotes_cross_cutoff_groups_and_builds_ordered_cv_buckets(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    result = _process(paths)

    with result.assignment_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {"source_listing_id": "row-000000002", "split": "test", "cv_bucket": ""},
        {"source_listing_id": "row-000000003", "split": "test", "cv_bucket": ""},
        {"source_listing_id": "row-000000004", "split": "train", "cv_bucket": "2015_01"},
        {"source_listing_id": "row-000000005", "split": "train", "cv_bucket": "2015_01"},
        {"source_listing_id": "row-000000006", "split": "train", "cv_bucket": "2015_02"},
        {
            "source_listing_id": "row-000000007",
            "split": "train",
            "cv_bucket": "2015_03_04",
        },
        {"source_listing_id": "row-000000008", "split": "train", "cv_bucket": "2015_05"},
        {"source_listing_id": "row-000000009", "split": "test", "cv_bucket": ""},
        {"source_listing_id": "row-000000010", "split": "train", "cv_bucket": "warmup"},
    ]
    assert result.metrics.candidate_rows == 9
    assert result.metrics.train_rows == 6
    assert result.metrics.test_rows == 3
    assert result.metrics.initial_date_holdout_rows == 2
    assert result.metrics.initial_date_holdout_percent == "22.2222"
    assert result.metrics.promoted_earlier_rows == 1
    assert result.metrics.vin_groups_total == 7
    assert result.metrics.vin_groups_train == 5
    assert result.metrics.vin_groups_test == 2
    assert result.metrics.vin_groups_promoted == 1
    assert dict(result.metrics.train_cv_bucket_rows) == {
        "warmup": 1,
        "2015_01": 2,
        "2015_02": 1,
        "2015_03_04": 1,
        "2015_05": 1,
    }
    assert result.metrics.train_rows_on_or_after_cutoff == 0
    assert result.metrics.vin_overlap_between_partitions == 0
    verified = _verify(paths)
    assert verified.train_rows == 6
    assert verified.test_rows == 3

    durable_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (result.assignment_path, result.manifest_path, result.readiness_path)
    )
    for source_value in (
        "1FAHP2F80DG100001",
        "PRIVATE-SELLER-MARKER",
        "automatic",
        "11000",
    ):
        assert source_value not in durable_text


def test_verified_training_stream_exposes_only_partition_bucket_features_and_target(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    _process(paths)
    training = prepare_kaggle_vehicle_sales_training_rows(
        paths["split_manifest"],
        paths["raw"],
        paths["candidate"],
        paths["candidate_manifest"],
        paths["review"],
        paths["policy"],
        today=_TODAY,
    )

    rows = list(training)
    assert len(rows) == 9
    split, bucket, features, target = rows[0]
    assert split == "test"
    assert bucket is None
    assert tuple(features) == (
        "year",
        "make",
        "model",
        "trim",
        "mileage",
        "condition",
        "vehicle_type",
    )
    assert features == {
        "year": 2013,
        "make": "Ford",
        "model": "Fusion",
        "trim": "SE",
        "mileage": 50000,
        "condition": 4.5,
        "vehicle_type": "Sedan",
    }
    assert target == 12000.0
    assert rows[-1][0:2] == ("train", "warmup")
    assert training.train_rows == 6
    assert training.test_rows == 3


def test_training_stream_cannot_be_constructed_without_gate(tmp_path: Path) -> None:
    dummy = VerifiedKaggleVehicleSalesSplit(
        assignment_path=tmp_path / "assignment.csv",
        assignment_sha256="0" * 64,
        candidate_path=tmp_path / "candidate.csv",
        candidate_sha256="0" * 64,
        manifest_path=tmp_path / "manifest.json",
        manifest_sha256="0" * 64,
        train_rows=1,
        test_rows=1,
    )

    with pytest.raises(KaggleVehicleSalesSplitError, match="verified preparation gate"):
        KaggleVehicleSalesTrainingRows(dummy, _token=object())


def test_verifier_rejects_semantic_partition_tampering_even_if_hashes_are_resealed(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    _process(paths)
    rows = list(csv.DictReader(paths["assignment"].open(encoding="utf-8", newline="")))
    rows[0]["split"] = "train"
    rows[0]["cv_bucket"] = "2015_05"
    with paths["assignment"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["source_listing_id", "split", "cv_bucket"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    _reseal_split(paths)

    with pytest.raises(KaggleVehicleSalesSplitError, match="partition violates"):
        _verify(paths)


def test_verifier_rejects_cv_bucket_tampering_even_if_hashes_are_resealed(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    _process(paths)
    rows = list(csv.DictReader(paths["assignment"].open(encoding="utf-8", newline="")))
    rows[2]["cv_bucket"] = "2015_02"
    with paths["assignment"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["source_listing_id", "split", "cv_bucket"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    _reseal_split(paths)

    with pytest.raises(KaggleVehicleSalesSplitError, match="CV bucket violates"):
        _verify(paths)


def test_training_stream_rejects_assignment_changed_after_preparation(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _process(paths)
    training = prepare_kaggle_vehicle_sales_training_rows(
        paths["split_manifest"],
        paths["raw"],
        paths["candidate"],
        paths["candidate_manifest"],
        paths["review"],
        paths["policy"],
        today=_TODAY,
    )
    paths["assignment"].write_bytes(paths["assignment"].read_bytes() + b"\n")

    with pytest.raises(KaggleVehicleSalesSplitError, match="changed after verification"):
        list(training)


def test_policy_tampering_and_wrong_filename_fail_closed(tmp_path: Path) -> None:
    altered = tmp_path / "kaggle-vehicle-sales-v1.split.json"
    altered.write_bytes(_COMMITTED_POLICY.read_bytes() + b" ")
    with pytest.raises(KaggleVehicleSalesSplitError, match="SHA-256"):
        load_kaggle_vehicle_sales_split_policy(altered, today=_TODAY)

    wrong_name = tmp_path / "split.json"
    wrong_name.write_bytes(_COMMITTED_POLICY.read_bytes())
    with pytest.raises(KaggleVehicleSalesSplitError, match="filename"):
        load_kaggle_vehicle_sales_split_policy(wrong_name, today=_TODAY)


def test_independent_ml_permission_gate_blocks_split_publication(tmp_path: Path) -> None:
    def revoke_ml(value: dict[str, object]) -> None:
        permissions = cast(dict[str, object], value["permissions"])
        permissions["ml_training_and_evaluation"] = "pending"

    paths = _fixture(tmp_path, review_mutation=revoke_ml)

    with pytest.raises(KaggleVehicleSalesError, match="does not approve ML training"):
        _process(paths)
    assert not paths["assignment"].exists()


def test_split_rejects_candidate_tampering_and_invalid_output_target(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["candidate"].write_bytes(paths["candidate"].read_bytes() + b"\n")
    with pytest.raises(KaggleVehicleSalesError, match="candidate_file hash does not match"):
        _process(paths)

    clean_paths = _fixture(tmp_path / "clean")
    clean_paths["assignment"] = clean_paths["assignment"].with_suffix(".jsonl")
    with pytest.raises(KaggleVehicleSalesSplitError, match=".csv suffix"):
        _process(clean_paths)


def test_split_output_cannot_overwrite_a_lineage_input(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["assignment"] = paths["raw"]

    with pytest.raises(KaggleVehicleSalesSplitError, match="must not overwrite"):
        _process(paths)


def test_zero_row_metric_percentage_is_well_defined() -> None:
    metrics = split_module.KaggleVehicleSalesSplitMetrics(
        candidate_rows=0,
        train_rows=0,
        test_rows=0,
        initial_date_holdout_rows=0,
        promoted_earlier_rows=0,
        vin_groups_total=0,
        vin_groups_train=0,
        vin_groups_test=0,
        vin_groups_promoted=0,
        train_cv_bucket_rows=dict.fromkeys(split_module._CV_BUCKET_NAMES, 0),
        train_cv_bucket_vin_groups=dict.fromkeys(split_module._CV_BUCKET_NAMES, 0),
    )

    assert metrics.initial_date_holdout_percent == "0.0000"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "NaN",
        "1FAHP2F80DG10000I",
    ],
)
def test_private_identifier_normalization_rejects_invalid_values(value: str) -> None:
    assert split_module._normalize_identifier(value) is None


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not-a-date", "invalid sale date"),
        ("Mon Jan 01 2015 25:00:00 GMT-0800 (PST)", "invalid sale date"),
        ("Thu Feb 30 2015 12:00:00 GMT-0800 (PST)", "invalid sale date"),
        ("Mon Jan 01 2015 12:00:00 GMT-0800 (PST)", "invalid sale weekday"),
    ],
)
def test_private_date_parser_fails_closed(value: str, message: str) -> None:
    with pytest.raises(KaggleVehicleSalesSplitError, match=message):
        split_module._parse_local_sale_date(value)


def test_partition_and_bucket_validators_fail_closed() -> None:
    with pytest.raises(KaggleVehicleSalesSplitError, match="train or test"):
        split_module._require_split("validation")
    with pytest.raises(KaggleVehicleSalesSplitError, match="must be text"):
        split_module._require_cv_bucket(None, split="train")
    with pytest.raises(KaggleVehicleSalesSplitError, match="must not have"):
        split_module._require_cv_bucket("warmup", split="test")
    with pytest.raises(KaggleVehicleSalesSplitError, match="is invalid"):
        split_module._require_cv_bucket("future", split="train")


def test_training_scalar_validators_cover_missing_and_invalid_values() -> None:
    assert split_module._optional_training_int("", label="mileage") is None
    assert split_module._optional_training_float("", label="condition") is None
    with pytest.raises(KaggleVehicleSalesSplitError, match="not an integer"):
        split_module._training_int("1.5", label="year")
    with pytest.raises(KaggleVehicleSalesSplitError, match="is invalid"):
        split_module._optional_training_float(object(), label="condition")
    with pytest.raises(KaggleVehicleSalesSplitError, match="is invalid"):
        split_module._optional_training_float("bad", label="condition")
    with pytest.raises(KaggleVehicleSalesSplitError, match="out of range"):
        split_module._optional_training_float("9.0", label="condition")
    with pytest.raises(KaggleVehicleSalesSplitError, match="is empty"):
        split_module._training_text("", label="make")


@pytest.mark.parametrize(
    "payload",
    [
        b'{"value": NaN}',
        b'{"value": 1, "value": 2}',
        b"[]",
    ],
)
def test_strict_json_rejects_non_finite_duplicates_and_non_objects(payload: bytes) -> None:
    with pytest.raises(KaggleVehicleSalesSplitError, match="strict UTF-8 JSON|JSON object"):
        split_module._strict_json_object(payload, label="fixture")


def test_aggregate_bucket_validators_reject_invalid_shapes() -> None:
    with pytest.raises(KaggleVehicleSalesSplitError, match="query is invalid"):
        split_module._bucket_counts({}, label="bucket")
    with pytest.raises(KaggleVehicleSalesSplitError, match="returned invalid"):
        split_module._bucket_counts([("future", 1)], label="bucket")
    with pytest.raises(KaggleVehicleSalesSplitError, match="keys are invalid"):
        split_module._require_bucket_mapping({}, label="bucket")
    with pytest.raises(KaggleVehicleSalesSplitError, match="nonnegative"):
        split_module._require_bucket_mapping(
            dict.fromkeys(split_module._CV_BUCKET_NAMES, -1), label="bucket"
        )


def test_local_file_guards_reject_missing_directory_and_empty_file(tmp_path: Path) -> None:
    with pytest.raises(KaggleVehicleSalesSplitError, match="missing or inaccessible"):
        split_module._require_regular_file(tmp_path / "missing", label="fixture")
    with pytest.raises(KaggleVehicleSalesSplitError, match="regular file"):
        split_module._require_regular_file(tmp_path, label="fixture")
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    with pytest.raises(KaggleVehicleSalesSplitError, match="must not be empty"):
        split_module._hash_regular_file(empty)
