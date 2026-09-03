"""Reviewed chronological split for the Kaggle wholesale-sale candidate.

The source's vehicle identifier is needed to prevent leakage, but it is never
published.  This module reads it only from the pinned raw artifact, converts it
to an ephemeral keyed digest inside a temporary SQLite index, and deletes that
index before returning.  The durable assignment contains only the adapter's
safe row identifier, ``train``/``test``, and a non-identifying ordered CV
bucket for train rows.

Rows dated on or after 2015-06-01 seed the test partition.  If an identifier
group contains any seeded row, every accepted row in that group is assigned to
test.  Consequently the train partition is strictly pre-cutoff and no vehicle
group can cross partitions.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, cast

from autovalue_ml.acquisition.sources.kaggle_vehicle_sales import (
    KAGGLE_VEHICLE_SALES_HEADER,
    KaggleVehicleSalesError,
    KaggleVehicleSalesReview,
    load_kaggle_vehicle_sales_review,
    require_kaggle_ml_training_approval,
    verify_kaggle_candidate_artifact_set,
)

SplitName = Literal["train", "test"]
TrainingValue = str | int | float | None
TrainingRow = tuple[SplitName, str | None, dict[str, TrainingValue], float]

_SOURCE_ID: Final = "kaggle_vehicle_sales_v1"
_SOURCE_REVIEW_ID: Final = "kaggle-vehicle-sales-data-v1-2026-08-28"
_POLICY_FILE_NAME: Final = "kaggle-vehicle-sales-v1.split.json"
_POLICY_ID: Final = "kaggle-vehicle-sales-v1-chronological-vin-isolated-v1"
_POLICY_SHA256: Final = "4c0d4b68d2ad1b8bcbbfc89d1936e0b8ba77287f3f1bc3f97b9c3301224e6833"
_ALGORITHM_VERSION: Final = "chronological-vin-isolated-v1.0.0"
_CUTOFF: Final = date(2015, 6, 1)
_DATE_RULE: Final = "Rows dated on or after the cutoff begin in the test partition."
_GROUP_RULE: Final = (
    "If any accepted row in an identifier group is on or after the cutoff, every accepted "
    "row in that group is assigned to test; groups entirely before the cutoff are assigned "
    "to train."
)
_CV_BUCKET_RULE: Final = (
    "Each train identifier group is assigned wholly to one ordered bucket using its latest "
    "accepted sale date; test rows have no CV bucket."
)
_CV_BUCKETS: Final[tuple[tuple[str, str, str], ...]] = (
    ("warmup", "2014-01-01", "2015-01-01"),
    ("2015_01", "2015-01-01", "2015-02-01"),
    ("2015_02", "2015-02-01", "2015-03-01"),
    ("2015_03_04", "2015-03-01", "2015-05-01"),
    ("2015_05", "2015-05-01", "2015-06-01"),
)
_CV_BUCKET_NAMES: Final = tuple(bucket[0] for bucket in _CV_BUCKETS)
_PUBLICATION_STATUS: Final = "private_local_only"
_TRAINING_READINESS: Final = "approved_private_training_and_evaluation"
_ASSIGNMENT_HEADER: Final[tuple[str, ...]] = ("source_listing_id", "split", "cv_bucket")
_FEATURE_ALLOWLIST: Final[tuple[str, ...]] = (
    "year",
    "make",
    "model",
    "trim",
    "mileage",
    "condition",
    "vehicle_type",
)
_TARGET_COLUMN: Final = "price_cents"
_CANDIDATE_HEADER: Final[tuple[str, ...]] = (
    "source_id",
    "source_listing_id",
    "canonical_url",
    "observed_at",
    "market_country",
    "year",
    "make",
    "model",
    "trim",
    "mileage",
    "mileage_unit",
    "condition",
    "vehicle_status",
    "engine",
    "drivetrain",
    "accident_status",
    "accident_count",
    "owner_count",
    "vehicle_type",
    "price_cents",
    "currency",
    "price_kind",
    "sale_status",
    "raw_content_sha256",
    "parser_version",
    "normalization_version",
    "ingestion_run_id",
    "authorization_policy_id",
)
_POLICY_KEYS: Final = {
    "split_policy_schema_version",
    "policy_id",
    "reviewed_on",
    "decision",
    "source_id",
    "source_review_id",
    "algorithm_version",
    "cutoff_date",
    "date_rule",
    "group_rule",
    "cv_bucket_rule",
    "cv_buckets",
    "assignment_columns",
    "feature_allowlist",
    "target_column",
    "publication_status",
}
_MANIFEST_KEYS: Final = {
    "schema_version",
    "artifact_type",
    "source_id",
    "source_review_id",
    "source_review_sha256",
    "split_policy_id",
    "split_policy_file",
    "split_policy_sha256",
    "algorithm_version",
    "cutoff_date",
    "date_rule",
    "group_rule",
    "cv_bucket_rule",
    "cv_buckets",
    "raw_source_file",
    "raw_source_sha256",
    "raw_source_size_bytes",
    "candidate_file",
    "candidate_sha256",
    "candidate_size_bytes",
    "candidate_manifest_file",
    "candidate_manifest_sha256",
    "assignment_file",
    "assignment_sha256",
    "assignment_size_bytes",
    "assignment_columns",
    "feature_allowlist",
    "target_column",
    "publication_status",
    "training_readiness",
    "metrics",
    "readiness_file",
}
_METRICS_KEYS: Final = {
    "candidate_rows",
    "train_rows",
    "test_rows",
    "initial_date_holdout_rows",
    "initial_date_holdout_percent",
    "promoted_earlier_rows",
    "vin_groups_total",
    "vin_groups_train",
    "vin_groups_test",
    "vin_groups_promoted",
    "train_rows_on_or_after_cutoff",
    "vin_overlap_between_partitions",
    "train_cv_bucket_rows",
    "train_cv_bucket_vin_groups",
    "train_rows_without_cv_bucket",
    "train_vin_groups_crossing_cv_buckets",
}
_ROW_ID_PATTERN = re.compile(r"^row-([0-9]{9})$")
_VIN_PATTERN = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")
_SALE_DATE_PATTERN = re.compile(
    r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun) "
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
    r"([0-9]{2}) ([0-9]{4}) ([0-9]{2}):([0-9]{2}):([0-9]{2}) "
    r"GMT([+-])([0-9]{2})([0-9]{2}) \(([A-Z]{3})\)$"
)
_MONTHS: Final = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
_WEEKDAYS: Final = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
_TZ_OFFSETS: Final = {
    "PST": -8 * 60,
    "PDT": -7 * 60,
    "MST": -7 * 60,
    "MDT": -6 * 60,
    "CST": -6 * 60,
    "CDT": -5 * 60,
    "EST": -5 * 60,
    "EDT": -4 * 60,
}
_MAX_JSON_BYTES: Final = 5_000_000
_CONSTRUCTOR_TOKEN = object()


class KaggleVehicleSalesSplitError(KaggleVehicleSalesError):
    """The reviewed split or one of its lineage artifacts failed closed."""


@dataclass(frozen=True, slots=True)
class KaggleVehicleSalesSplitPolicy:
    """A byte-pinned, semantically validated split decision."""

    policy_id: str
    policy_sha256: str
    policy_path: Path
    source_review_id: str
    algorithm_version: str
    cutoff_date: date


@dataclass(frozen=True, slots=True)
class KaggleVehicleSalesSplitMetrics:
    """Aggregate split metrics; no source identifier value is retained."""

    candidate_rows: int
    train_rows: int
    test_rows: int
    initial_date_holdout_rows: int
    promoted_earlier_rows: int
    vin_groups_total: int
    vin_groups_train: int
    vin_groups_test: int
    vin_groups_promoted: int
    train_cv_bucket_rows: Mapping[str, int]
    train_cv_bucket_vin_groups: Mapping[str, int]
    train_rows_on_or_after_cutoff: int = 0
    vin_overlap_between_partitions: int = 0
    train_rows_without_cv_bucket: int = 0
    train_vin_groups_crossing_cv_buckets: int = 0

    @property
    def initial_date_holdout_percent(self) -> str:
        if self.candidate_rows == 0:
            return "0.0000"
        return f"{self.initial_date_holdout_rows * 100 / self.candidate_rows:.4f}"

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_rows": self.candidate_rows,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "initial_date_holdout_rows": self.initial_date_holdout_rows,
            "initial_date_holdout_percent": self.initial_date_holdout_percent,
            "promoted_earlier_rows": self.promoted_earlier_rows,
            "vin_groups_total": self.vin_groups_total,
            "vin_groups_train": self.vin_groups_train,
            "vin_groups_test": self.vin_groups_test,
            "vin_groups_promoted": self.vin_groups_promoted,
            "train_rows_on_or_after_cutoff": self.train_rows_on_or_after_cutoff,
            "vin_overlap_between_partitions": self.vin_overlap_between_partitions,
            "train_cv_bucket_rows": dict(self.train_cv_bucket_rows),
            "train_cv_bucket_vin_groups": dict(self.train_cv_bucket_vin_groups),
            "train_rows_without_cv_bucket": self.train_rows_without_cv_bucket,
            "train_vin_groups_crossing_cv_buckets": (self.train_vin_groups_crossing_cv_buckets),
        }


@dataclass(frozen=True, slots=True)
class KaggleVehicleSalesSplitArtifactSet:
    assignment_path: Path
    manifest_path: Path
    readiness_path: Path
    metrics: KaggleVehicleSalesSplitMetrics


@dataclass(frozen=True, slots=True)
class VerifiedKaggleVehicleSalesSplit:
    """A complete verification result suitable for opening a training stream."""

    assignment_path: Path
    assignment_sha256: str
    candidate_path: Path
    candidate_sha256: str
    manifest_path: Path
    manifest_sha256: str
    train_rows: int
    test_rows: int


@dataclass(frozen=True, slots=True, init=False)
class KaggleVehicleSalesTrainingRows:
    """Lazily yield only partition, approved features, and target price."""

    _assignment_path: Path
    _assignment_sha256: str
    _candidate_path: Path
    _candidate_sha256: str
    _manifest_path: Path
    _manifest_sha256: str
    train_rows: int
    test_rows: int

    def __init__(
        self,
        verified: VerifiedKaggleVehicleSalesSplit,
        *,
        _token: object,
    ) -> None:
        if _token is not _CONSTRUCTOR_TOKEN:
            raise KaggleVehicleSalesSplitError(
                "training rows can only be created by the verified preparation gate"
            )
        object.__setattr__(self, "_assignment_path", verified.assignment_path)
        object.__setattr__(self, "_assignment_sha256", verified.assignment_sha256)
        object.__setattr__(self, "_candidate_path", verified.candidate_path)
        object.__setattr__(self, "_candidate_sha256", verified.candidate_sha256)
        object.__setattr__(self, "_manifest_path", verified.manifest_path)
        object.__setattr__(self, "_manifest_sha256", verified.manifest_sha256)
        object.__setattr__(self, "train_rows", verified.train_rows)
        object.__setattr__(self, "test_rows", verified.test_rows)

    def __iter__(self) -> Iterator[TrainingRow]:
        _require_hash(self._manifest_path, self._manifest_sha256, label="split manifest")
        _require_hash(self._assignment_path, self._assignment_sha256, label="split assignment")
        _require_hash(self._candidate_path, self._candidate_sha256, label="training candidate")
        yielded = 0
        try:
            with (
                self._candidate_path.open("r", encoding="utf-8", newline="") as candidate_file,
                self._assignment_path.open("r", encoding="utf-8", newline="") as assignment_file,
            ):
                candidates = csv.DictReader(candidate_file, strict=True)
                assignments = csv.DictReader(assignment_file, strict=True)
                if tuple(candidates.fieldnames or ()) != _CANDIDATE_HEADER:
                    raise KaggleVehicleSalesSplitError("training candidate header is invalid")
                if tuple(assignments.fieldnames or ()) != _ASSIGNMENT_HEADER:
                    raise KaggleVehicleSalesSplitError("split assignment header is invalid")
                for candidate, assignment in zip(candidates, assignments, strict=True):
                    if candidate["source_listing_id"] != assignment["source_listing_id"]:
                        raise KaggleVehicleSalesSplitError(
                            "training candidate and split assignment are not aligned"
                        )
                    split = _require_split(assignment["split"])
                    cv_bucket = _require_cv_bucket(assignment["cv_bucket"], split=split)
                    features: dict[str, TrainingValue] = {
                        "year": _training_int(candidate["year"], label="year"),
                        "make": _training_text(candidate["make"], label="make"),
                        "model": _training_text(candidate["model"], label="model"),
                        "trim": candidate["trim"] or None,
                        "mileage": _optional_training_int(candidate["mileage"], label="mileage"),
                        "condition": _optional_training_float(
                            candidate["condition"], label="condition"
                        ),
                        "vehicle_type": candidate["vehicle_type"] or None,
                    }
                    if tuple(features) != _FEATURE_ALLOWLIST:
                        raise KaggleVehicleSalesSplitError("training feature allowlist drifted")
                    target_cents = _training_int(candidate[_TARGET_COLUMN], label="target")
                    if target_cents <= 0:
                        raise KaggleVehicleSalesSplitError("training target must be positive")
                    yielded += 1
                    yield split, cv_bucket, features, target_cents / 100.0
        except (csv.Error, UnicodeError, ValueError) as error:
            if isinstance(error, KaggleVehicleSalesSplitError):
                raise
            raise KaggleVehicleSalesSplitError("training artifacts are malformed") from error
        if yielded != self.train_rows + self.test_rows:
            raise KaggleVehicleSalesSplitError("training row count differs from the verified split")
        _require_hash(self._assignment_path, self._assignment_sha256, label="split assignment")
        _require_hash(self._candidate_path, self._candidate_sha256, label="training candidate")


@dataclass(frozen=True, slots=True)
class _CandidateLineage:
    path: Path
    sha256: str
    size_bytes: int
    row_count: int
    manifest_path: Path
    manifest_sha256: str


def load_kaggle_vehicle_sales_split_policy(
    policy_path: Path,
    *,
    today: date | None = None,
) -> KaggleVehicleSalesSplitPolicy:
    """Load the exact committed policy; altered bytes or semantics fail closed."""

    path = _require_regular_file(policy_path, label="split policy")
    if path.name != _POLICY_FILE_NAME:
        raise KaggleVehicleSalesSplitError("split policy filename is not the reviewed filename")
    payload = _read_bounded_bytes(path, max_bytes=256_000, label="split policy")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != _POLICY_SHA256:
        raise KaggleVehicleSalesSplitError("split policy SHA-256 is not the reviewed pin")
    value = _strict_json_object(payload, label="split policy")
    _require_exact_keys(value, _POLICY_KEYS, label="split policy")
    exact: dict[str, object] = {
        "split_policy_schema_version": 1,
        "policy_id": _POLICY_ID,
        "reviewed_on": "2026-08-28",
        "decision": "approved",
        "source_id": _SOURCE_ID,
        "source_review_id": _SOURCE_REVIEW_ID,
        "algorithm_version": _ALGORITHM_VERSION,
        "cutoff_date": _CUTOFF.isoformat(),
        "date_rule": _DATE_RULE,
        "group_rule": _GROUP_RULE,
        "cv_bucket_rule": _CV_BUCKET_RULE,
        "cv_buckets": _cv_bucket_policy_value(),
        "assignment_columns": list(_ASSIGNMENT_HEADER),
        "feature_allowlist": list(_FEATURE_ALLOWLIST),
        "target_column": _TARGET_COLUMN,
        "publication_status": _PUBLICATION_STATUS,
    }
    for key, expected in exact.items():
        if value[key] != expected:
            raise KaggleVehicleSalesSplitError(f"split policy {key} is invalid")
    reviewed_on = date.fromisoformat(cast(str, value["reviewed_on"]))
    if reviewed_on > (date.today() if today is None else today):
        raise KaggleVehicleSalesSplitError("split policy review date cannot be in the future")
    return KaggleVehicleSalesSplitPolicy(
        policy_id=_POLICY_ID,
        policy_sha256=digest,
        policy_path=path,
        source_review_id=_SOURCE_REVIEW_ID,
        algorithm_version=_ALGORITHM_VERSION,
        cutoff_date=_CUTOFF,
    )


def process_kaggle_vehicle_sales_split(
    raw_source_path: Path,
    candidate_path: Path,
    candidate_manifest_path: Path,
    source_review_path: Path,
    split_policy_path: Path,
    assignment_output_path: Path,
    *,
    today: date | None = None,
) -> KaggleVehicleSalesSplitArtifactSet:
    """Build and atomically publish the reviewed private split assignment."""

    if assignment_output_path.suffix.lower() != ".csv":
        raise KaggleVehicleSalesSplitError("split assignment output must use the .csv suffix")
    policy = load_kaggle_vehicle_sales_split_policy(split_policy_path, today=today)
    review = require_kaggle_ml_training_approval(
        load_kaggle_vehicle_sales_review(source_review_path, today=today)
    )
    _require_policy_review_match(policy, review)
    raw_source = _verify_raw_source(raw_source_path, review)
    lineage = _load_candidate_lineage(
        candidate_path,
        candidate_manifest_path,
        source_review_path,
        review=review,
        today=today,
    )

    assignment_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_parent = assignment_output_path.parent.resolve(strict=True)
    assignment_path = output_parent / assignment_output_path.name
    manifest_path = assignment_path.with_suffix(".manifest.json")
    readiness_path = assignment_path.with_suffix(".ready.json")
    _validate_output_targets(
        (assignment_path, manifest_path, readiness_path),
        protected=(
            raw_source,
            lineage.path,
            lineage.manifest_path,
            review.review_path,
            policy.policy_path,
        ),
    )
    readiness_path.unlink(missing_ok=True)

    staging = Path(tempfile.mkdtemp(prefix=f".{assignment_path.stem}.", dir=output_parent))
    try:
        index_path = staging / "ephemeral-group-index.sqlite3"
        metrics = _build_split_index(
            raw_source,
            lineage.path,
            review=review,
            cutoff=policy.cutoff_date,
            index_path=index_path,
        )
        staged_assignment = staging / assignment_path.name
        _write_assignments(index_path, staged_assignment)
        assignment_sha256, assignment_size = _hash_regular_file(staged_assignment)

        # Recheck all immutable lineage immediately before publishing the marker.
        _verify_raw_source(raw_source, review)
        _require_hash(lineage.path, lineage.sha256, label="candidate")
        _require_hash(lineage.manifest_path, lineage.manifest_sha256, label="candidate manifest")
        _require_hash(review.review_path, review.review_sha256, label="source review")
        _require_hash(policy.policy_path, policy.policy_sha256, label="split policy")

        manifest = _manifest_value(
            assignment_path=assignment_path,
            manifest_path=manifest_path,
            readiness_path=readiness_path,
            assignment_sha256=assignment_sha256,
            assignment_size=assignment_size,
            raw_source=raw_source,
            review=review,
            policy=policy,
            lineage=lineage,
            metrics=metrics,
        )
        staged_manifest = staging / manifest_path.name
        manifest_payload = _json_payload(manifest)
        _write_fsynced(staged_manifest, manifest_payload)
        manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        artifact_set_id = hashlib.sha256(
            f"{manifest_sha256}|{assignment_sha256}".encode("ascii")
        ).hexdigest()
        readiness = {
            "schema_version": 1,
            "artifact_set_id": artifact_set_id,
            "manifest_file": manifest_path.name,
            "manifest_sha256": manifest_sha256,
            "assignment_file": assignment_path.name,
            "assignment_sha256": assignment_sha256,
        }
        staged_readiness = staging / readiness_path.name
        _write_fsynced(staged_readiness, _json_payload(readiness))

        os.replace(staged_assignment, assignment_path)
        os.replace(staged_manifest, manifest_path)
        os.replace(staged_readiness, readiness_path)
        return KaggleVehicleSalesSplitArtifactSet(
            assignment_path=assignment_path,
            manifest_path=manifest_path,
            readiness_path=readiness_path,
            metrics=metrics,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def verify_kaggle_vehicle_sales_split_artifact_set(
    split_manifest_path: Path,
    raw_source_path: Path,
    candidate_path: Path,
    candidate_manifest_path: Path,
    source_review_path: Path,
    split_policy_path: Path,
    *,
    today: date | None = None,
) -> VerifiedKaggleVehicleSalesSplit:
    """Verify hashes, lineage, row accounting, cutoff semantics, and group isolation."""

    policy = load_kaggle_vehicle_sales_split_policy(split_policy_path, today=today)
    review = require_kaggle_ml_training_approval(
        load_kaggle_vehicle_sales_review(source_review_path, today=today)
    )
    _require_policy_review_match(policy, review)
    raw_source = _verify_raw_source(raw_source_path, review)
    lineage = _load_candidate_lineage(
        candidate_path,
        candidate_manifest_path,
        source_review_path,
        review=review,
        today=today,
    )
    manifest_path = _require_regular_file(split_manifest_path, label="split manifest")
    manifest_payload = _read_bounded_bytes(
        manifest_path, max_bytes=_MAX_JSON_BYTES, label="split manifest"
    )
    manifest = _strict_json_object(manifest_payload, label="split manifest")
    _validate_manifest_lineage(
        manifest,
        manifest_path=manifest_path,
        raw_source=raw_source,
        review=review,
        policy=policy,
        lineage=lineage,
    )
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()

    assignment_name = _safe_file_name(manifest["assignment_file"], label="assignment_file")
    readiness_name = _safe_file_name(manifest["readiness_file"], label="readiness_file")
    assignment_path = _require_regular_file(
        manifest_path.parent / assignment_name, label="split assignment"
    )
    readiness_path = _require_regular_file(
        manifest_path.parent / readiness_name, label="split readiness marker"
    )
    readiness = _strict_json_object(
        _read_bounded_bytes(readiness_path, max_bytes=1_000_000, label="split readiness marker"),
        label="split readiness marker",
    )
    expected_ready_keys = {
        "schema_version",
        "artifact_set_id",
        "manifest_file",
        "manifest_sha256",
        "assignment_file",
        "assignment_sha256",
    }
    _require_exact_keys(readiness, expected_ready_keys, label="split readiness marker")
    if readiness["schema_version"] != 1 or readiness["manifest_file"] != manifest_path.name:
        raise KaggleVehicleSalesSplitError("split readiness marker lineage is invalid")
    if readiness["assignment_file"] != assignment_name:
        raise KaggleVehicleSalesSplitError("split readiness assignment lineage is invalid")
    _require_digest_value(readiness["manifest_sha256"], label="manifest_sha256")
    if readiness["manifest_sha256"] != manifest_sha256:
        raise KaggleVehicleSalesSplitError("split manifest hash does not match readiness")
    assignment_sha256, assignment_size = _hash_regular_file(assignment_path)
    if (
        manifest["assignment_sha256"] != assignment_sha256
        or readiness["assignment_sha256"] != assignment_sha256
    ):
        raise KaggleVehicleSalesSplitError("split assignment hash does not match lineage")
    if manifest["assignment_size_bytes"] != assignment_size:
        raise KaggleVehicleSalesSplitError("split assignment size does not match lineage")
    expected_set_id = hashlib.sha256(
        f"{manifest_sha256}|{assignment_sha256}".encode("ascii")
    ).hexdigest()
    if readiness["artifact_set_id"] != expected_set_id:
        raise KaggleVehicleSalesSplitError("split artifact-set identifier is invalid")

    with tempfile.TemporaryDirectory(prefix="autovalue-split-verify-") as temp_dir:
        index_path = Path(temp_dir) / "ephemeral-group-index.sqlite3"
        expected_metrics = _build_split_index(
            raw_source,
            lineage.path,
            review=review,
            cutoff=policy.cutoff_date,
            index_path=index_path,
        )
        _verify_assignments(assignment_path, index_path, expected_metrics)
    if manifest["metrics"] != expected_metrics.to_dict():
        raise KaggleVehicleSalesSplitError("split metrics differ from recomputed semantics")

    # Close read-to-return races for every training-critical artifact.
    _verify_raw_source(raw_source, review)
    _require_hash(lineage.path, lineage.sha256, label="candidate")
    _require_hash(lineage.manifest_path, lineage.manifest_sha256, label="candidate manifest")
    _require_hash(review.review_path, review.review_sha256, label="source review")
    _require_hash(policy.policy_path, policy.policy_sha256, label="split policy")
    _require_hash(manifest_path, manifest_sha256, label="split manifest")
    _require_hash(assignment_path, assignment_sha256, label="split assignment")
    return VerifiedKaggleVehicleSalesSplit(
        assignment_path=assignment_path,
        assignment_sha256=assignment_sha256,
        candidate_path=lineage.path,
        candidate_sha256=lineage.sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        train_rows=expected_metrics.train_rows,
        test_rows=expected_metrics.test_rows,
    )


def prepare_kaggle_vehicle_sales_training_rows(
    split_manifest_path: Path,
    raw_source_path: Path,
    candidate_path: Path,
    candidate_manifest_path: Path,
    source_review_path: Path,
    split_policy_path: Path,
    *,
    today: date | None = None,
) -> KaggleVehicleSalesTrainingRows:
    """Open the training boundary only after a complete split verification."""

    verified = verify_kaggle_vehicle_sales_split_artifact_set(
        split_manifest_path,
        raw_source_path,
        candidate_path,
        candidate_manifest_path,
        source_review_path,
        split_policy_path,
        today=today,
    )
    return KaggleVehicleSalesTrainingRows(verified, _token=_CONSTRUCTOR_TOKEN)


def _load_candidate_lineage(
    candidate_path: Path,
    candidate_manifest_path: Path,
    source_review_path: Path,
    *,
    review: KaggleVehicleSalesReview,
    today: date | None,
) -> _CandidateLineage:
    verify_kaggle_candidate_artifact_set(
        candidate_manifest_path,
        source_review_path,
        today=today,
    )
    path = _require_regular_file(candidate_path, label="candidate CSV")
    manifest_path = _require_regular_file(candidate_manifest_path, label="candidate manifest")
    manifest_payload = _read_bounded_bytes(
        manifest_path, max_bytes=_MAX_JSON_BYTES, label="candidate manifest"
    )
    manifest = _strict_json_object(manifest_payload, label="candidate manifest")
    if path.parent != manifest_path.parent or path.name != manifest.get("candidate_file"):
        raise KaggleVehicleSalesSplitError(
            "candidate path differs from the verified candidate manifest"
        )
    if manifest.get("review_sha256") != review.review_sha256:
        raise KaggleVehicleSalesSplitError("candidate review lineage is stale")
    if manifest.get("raw_source_sha256") != review.expected_sha256:
        raise KaggleVehicleSalesSplitError("candidate raw-source lineage is stale")
    if manifest.get("approved_for_ml_training") is not True:
        raise KaggleVehicleSalesSplitError("candidate is not approved for ML training")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict):
        raise KaggleVehicleSalesSplitError("candidate metrics are invalid")
    row_count = _require_nonnegative_int(metrics.get("rows_accepted"), label="candidate rows")
    if row_count <= 0:
        raise KaggleVehicleSalesSplitError("candidate must contain at least one accepted row")
    sha256, size_bytes = _hash_regular_file(path)
    if sha256 != manifest.get("candidate_sha256") or size_bytes != manifest.get(
        "candidate_size_bytes"
    ):
        raise KaggleVehicleSalesSplitError("candidate bytes differ from verified lineage")
    return _CandidateLineage(
        path=path,
        sha256=sha256,
        size_bytes=size_bytes,
        row_count=row_count,
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
    )


def _build_split_index(
    raw_source: Path,
    candidate_path: Path,
    *,
    review: KaggleVehicleSalesReview,
    cutoff: date,
    index_path: Path,
) -> KaggleVehicleSalesSplitMetrics:
    secret = secrets.token_bytes(32)
    connection = sqlite3.connect(index_path)
    try:
        connection.execute("PRAGMA journal_mode=MEMORY")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE accepted_rows ("
            "row_number INTEGER PRIMARY KEY, group_key BLOB NOT NULL, "
            "initial_test INTEGER NOT NULL "
            "CHECK (initial_test IN (0, 1)))"
        )
        connection.execute(
            "CREATE TABLE groups ("
            "group_key BLOB PRIMARY KEY, has_test INTEGER NOT NULL, has_earlier INTEGER NOT NULL, "
            "row_count INTEGER NOT NULL, latest_date TEXT NOT NULL, cv_bucket TEXT) WITHOUT ROWID"
        )
        with (
            raw_source.open("r", encoding="utf-8-sig", newline="") as raw_file,
            candidate_path.open("r", encoding="utf-8", newline="") as candidate_file,
        ):
            raw_reader = csv.reader(raw_file, strict=True)
            try:
                raw_header = next(raw_reader)
            except (StopIteration, csv.Error) as error:
                raise KaggleVehicleSalesSplitError("raw CSV has no header") from error
            if tuple(raw_header) != KAGGLE_VEHICLE_SALES_HEADER:
                raise KaggleVehicleSalesSplitError("raw CSV header is invalid for splitting")
            candidate_reader = csv.DictReader(candidate_file, strict=True)
            if tuple(candidate_reader.fieldnames or ()) != _CANDIDATE_HEADER:
                raise KaggleVehicleSalesSplitError("candidate header is invalid for splitting")
            candidate_iterator = iter(candidate_reader)
            candidate_row = next(candidate_iterator, None)
            candidate_row_number = _candidate_row_number(candidate_row)
            raw_rows = 0
            accepted_rows = 0
            try:
                for row_number, values in enumerate(raw_reader, start=2):
                    raw_rows += 1
                    if candidate_row_number is not None and candidate_row_number < row_number:
                        raise KaggleVehicleSalesSplitError(
                            "candidate row identifiers are not aligned to the raw CSV"
                        )
                    if candidate_row_number != row_number:
                        continue
                    if len(values) != len(KAGGLE_VEHICLE_SALES_HEADER):
                        raise KaggleVehicleSalesSplitError(
                            "accepted candidate maps to a malformed raw row"
                        )
                    normalized_identifier = _normalize_identifier(values[6])
                    if normalized_identifier is None:
                        raise KaggleVehicleSalesSplitError(
                            "accepted candidate maps to an invalid raw identifier"
                        )
                    sale_date = _parse_local_sale_date(values[15])
                    initial_test = int(sale_date >= cutoff)
                    group_key = hmac.digest(secret, normalized_identifier.encode("ascii"), "sha256")
                    connection.execute(
                        "INSERT INTO accepted_rows (row_number, group_key, initial_test) "
                        "VALUES (?, ?, ?)",
                        (row_number, group_key, initial_test),
                    )
                    connection.execute(
                        "INSERT INTO groups ("
                        "group_key, has_test, has_earlier, row_count, latest_date, cv_bucket"
                        ") VALUES (?, ?, ?, 1, ?, NULL) ON CONFLICT(group_key) DO UPDATE SET "
                        "has_test = MAX(has_test, excluded.has_test), "
                        "has_earlier = MAX(has_earlier, excluded.has_earlier), "
                        "row_count = row_count + 1, "
                        "latest_date = MAX(latest_date, excluded.latest_date)",
                        (group_key, initial_test, 1 - initial_test, sale_date.isoformat()),
                    )
                    accepted_rows += 1
                    candidate_row = next(candidate_iterator, None)
                    candidate_row_number = _candidate_row_number(candidate_row)
                    if accepted_rows % 10_000 == 0:
                        connection.commit()
            except csv.Error as error:
                raise KaggleVehicleSalesSplitError("split inputs contain malformed CSV") from error
            if candidate_row is not None:
                raise KaggleVehicleSalesSplitError(
                    "candidate contains a row identifier outside the raw CSV"
                )
        if raw_rows != review.expected_row_count:
            raise KaggleVehicleSalesSplitError("raw row count differs from the source review")
        connection.commit()
        connection.execute("CREATE INDEX accepted_group_idx ON accepted_rows(group_key)")
        connection.execute(
            "UPDATE groups SET cv_bucket = CASE "
            "WHEN has_test = 1 THEN NULL "
            "WHEN latest_date < '2015-01-01' THEN 'warmup' "
            "WHEN latest_date < '2015-02-01' THEN '2015_01' "
            "WHEN latest_date < '2015-03-01' THEN '2015_02' "
            "WHEN latest_date < '2015-05-01' THEN '2015_03_04' "
            "WHEN latest_date < '2015-06-01' THEN '2015_05' "
            "ELSE NULL END"
        )
        connection.commit()
        metrics = _query_metrics(connection)
        if accepted_rows != metrics.candidate_rows:
            raise KaggleVehicleSalesSplitError("split index row accounting failed")
        return metrics
    except (UnicodeError, sqlite3.Error) as error:
        if isinstance(error, KaggleVehicleSalesSplitError):
            raise
        raise KaggleVehicleSalesSplitError("temporary split index failed") from error
    finally:
        connection.close()


def _query_metrics(connection: sqlite3.Connection) -> KaggleVehicleSalesSplitMetrics:
    row = connection.execute(
        "SELECT COUNT(*), "
        "COALESCE(SUM(CASE WHEN g.has_test = 0 THEN 1 ELSE 0 END), 0), "
        "COALESCE(SUM(CASE WHEN g.has_test = 1 THEN 1 ELSE 0 END), 0), "
        "COALESCE(SUM(r.initial_test), 0), "
        "COALESCE(SUM(CASE WHEN g.has_test = 1 AND r.initial_test = 0 THEN 1 ELSE 0 END), 0), "
        "COALESCE(SUM(CASE WHEN g.has_test = 0 AND r.initial_test = 1 THEN 1 ELSE 0 END), 0), "
        "COALESCE(SUM(CASE WHEN g.has_test = 0 AND g.cv_bucket IS NULL THEN 1 ELSE 0 END), 0) "
        "FROM accepted_rows r JOIN groups g ON g.group_key = r.group_key"
    ).fetchone()
    group_row = connection.execute(
        "SELECT COUNT(*), "
        "COALESCE(SUM(CASE WHEN has_test = 0 THEN 1 ELSE 0 END), 0), "
        "COALESCE(SUM(CASE WHEN has_test = 1 THEN 1 ELSE 0 END), 0), "
        "COALESCE(SUM(CASE WHEN has_test = 1 AND has_earlier = 1 THEN 1 ELSE 0 END), 0) "
        "FROM groups"
    ).fetchone()
    values = _sqlite_nonnegative_row(row, width=7, label="split row metrics")
    groups = _sqlite_nonnegative_row(group_row, width=4, label="split group metrics")
    bucket_rows = _bucket_counts(
        connection.execute(
            "SELECT g.cv_bucket, COUNT(*) FROM accepted_rows r "
            "JOIN groups g ON g.group_key = r.group_key "
            "WHERE g.has_test = 0 GROUP BY g.cv_bucket"
        ).fetchall(),
        label="CV bucket rows",
    )
    bucket_groups = _bucket_counts(
        connection.execute(
            "SELECT cv_bucket, COUNT(*) FROM groups WHERE has_test = 0 GROUP BY cv_bucket"
        ).fetchall(),
        label="CV bucket groups",
    )
    metrics = KaggleVehicleSalesSplitMetrics(
        candidate_rows=values[0],
        train_rows=values[1],
        test_rows=values[2],
        initial_date_holdout_rows=values[3],
        promoted_earlier_rows=values[4],
        vin_groups_total=groups[0],
        vin_groups_train=groups[1],
        vin_groups_test=groups[2],
        vin_groups_promoted=groups[3],
        train_cv_bucket_rows=MappingProxyType(bucket_rows),
        train_cv_bucket_vin_groups=MappingProxyType(bucket_groups),
        train_rows_on_or_after_cutoff=values[5],
        vin_overlap_between_partitions=0,
        train_rows_without_cv_bucket=values[6],
        train_vin_groups_crossing_cv_buckets=0,
    )
    if (
        metrics.candidate_rows != metrics.train_rows + metrics.test_rows
        or metrics.test_rows != metrics.initial_date_holdout_rows + metrics.promoted_earlier_rows
        or metrics.vin_groups_total != metrics.vin_groups_train + metrics.vin_groups_test
        or metrics.train_rows_on_or_after_cutoff != 0
        or metrics.train_rows_without_cv_bucket != 0
        or sum(metrics.train_cv_bucket_rows.values()) != metrics.train_rows
        or sum(metrics.train_cv_bucket_vin_groups.values()) != metrics.vin_groups_train
    ):
        raise KaggleVehicleSalesSplitError("split metrics violate required invariants")
    return metrics


def _write_assignments(index_path: Path, output_path: Path) -> None:
    connection = sqlite3.connect(f"file:{index_path.as_posix()}?mode=ro", uri=True)
    try:
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=list(_ASSIGNMENT_HEADER),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            cursor = connection.execute(
                "SELECT r.row_number, CASE WHEN g.has_test = 1 THEN 'test' ELSE 'train' END, "
                "COALESCE(g.cv_bucket, '') "
                "FROM accepted_rows r JOIN groups g ON g.group_key = r.group_key "
                "ORDER BY r.row_number"
            )
            for row_number, split, cv_bucket in cursor:
                writer.writerow(
                    {
                        "source_listing_id": _safe_row_id(cast(int, row_number)),
                        "split": split,
                        "cv_bucket": cv_bucket,
                    }
                )
            output_file.flush()
            os.fsync(output_file.fileno())
    except sqlite3.Error as error:
        raise KaggleVehicleSalesSplitError("could not publish split assignments") from error
    finally:
        connection.close()


def _verify_assignments(
    assignment_path: Path,
    index_path: Path,
    metrics: KaggleVehicleSalesSplitMetrics,
) -> None:
    connection = sqlite3.connect(f"file:{index_path.as_posix()}?mode=ro", uri=True)
    expected: sqlite3.Cursor | None = None
    try:
        expected = connection.execute(
            "SELECT r.row_number, CASE WHEN g.has_test = 1 THEN 'test' ELSE 'train' END, "
            "COALESCE(g.cv_bucket, '') "
            "FROM accepted_rows r JOIN groups g ON g.group_key = r.group_key "
            "ORDER BY r.row_number"
        )
        count = 0
        with assignment_path.open("r", encoding="utf-8", newline="") as assignment_file:
            reader = csv.DictReader(assignment_file, strict=True)
            if tuple(reader.fieldnames or ()) != _ASSIGNMENT_HEADER:
                raise KaggleVehicleSalesSplitError("split assignment header is invalid")
            for actual, expected_row in zip(reader, expected, strict=True):
                row_number, expected_split, expected_bucket = expected_row
                if set(actual) != set(_ASSIGNMENT_HEADER):
                    raise KaggleVehicleSalesSplitError("split assignment row width is invalid")
                if actual["source_listing_id"] != _safe_row_id(cast(int, row_number)):
                    raise KaggleVehicleSalesSplitError(
                        "split assignment row identifier violates recomputed semantics"
                    )
                if _require_split(actual["split"]) != expected_split:
                    raise KaggleVehicleSalesSplitError(
                        "split assignment partition violates cutoff or group isolation"
                    )
                actual_bucket = _require_cv_bucket(
                    actual["cv_bucket"], split=cast(SplitName, expected_split)
                )
                if (actual_bucket or "") != expected_bucket:
                    raise KaggleVehicleSalesSplitError(
                        "split assignment CV bucket violates chronological group isolation"
                    )
                count += 1
        if count != metrics.candidate_rows:
            raise KaggleVehicleSalesSplitError("split assignment count is invalid")
    except (csv.Error, UnicodeError, sqlite3.Error, ValueError) as error:
        if isinstance(error, KaggleVehicleSalesSplitError):
            raise
        raise KaggleVehicleSalesSplitError("split assignment is malformed") from error
    finally:
        if expected is not None:
            expected.close()
        connection.close()


def _manifest_value(
    *,
    assignment_path: Path,
    manifest_path: Path,
    readiness_path: Path,
    assignment_sha256: str,
    assignment_size: int,
    raw_source: Path,
    review: KaggleVehicleSalesReview,
    policy: KaggleVehicleSalesSplitPolicy,
    lineage: _CandidateLineage,
    metrics: KaggleVehicleSalesSplitMetrics,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "private_split_assignment",
        "source_id": _SOURCE_ID,
        "source_review_id": review.review_id,
        "source_review_sha256": review.review_sha256,
        "split_policy_id": policy.policy_id,
        "split_policy_file": policy.policy_path.name,
        "split_policy_sha256": policy.policy_sha256,
        "algorithm_version": policy.algorithm_version,
        "cutoff_date": policy.cutoff_date.isoformat(),
        "date_rule": _DATE_RULE,
        "group_rule": _GROUP_RULE,
        "cv_bucket_rule": _CV_BUCKET_RULE,
        "cv_buckets": _cv_bucket_policy_value(),
        "raw_source_file": raw_source.name,
        "raw_source_sha256": review.expected_sha256,
        "raw_source_size_bytes": review.expected_size_bytes,
        "candidate_file": lineage.path.name,
        "candidate_sha256": lineage.sha256,
        "candidate_size_bytes": lineage.size_bytes,
        "candidate_manifest_file": lineage.manifest_path.name,
        "candidate_manifest_sha256": lineage.manifest_sha256,
        "assignment_file": assignment_path.name,
        "assignment_sha256": assignment_sha256,
        "assignment_size_bytes": assignment_size,
        "assignment_columns": list(_ASSIGNMENT_HEADER),
        "feature_allowlist": list(_FEATURE_ALLOWLIST),
        "target_column": _TARGET_COLUMN,
        "publication_status": _PUBLICATION_STATUS,
        "training_readiness": _TRAINING_READINESS,
        "metrics": metrics.to_dict(),
        "readiness_file": readiness_path.name,
    }


def _validate_manifest_lineage(
    manifest: dict[str, object],
    *,
    manifest_path: Path,
    raw_source: Path,
    review: KaggleVehicleSalesReview,
    policy: KaggleVehicleSalesSplitPolicy,
    lineage: _CandidateLineage,
) -> None:
    _require_exact_keys(manifest, _MANIFEST_KEYS, label="split manifest")
    exact: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "private_split_assignment",
        "source_id": _SOURCE_ID,
        "source_review_id": review.review_id,
        "source_review_sha256": review.review_sha256,
        "split_policy_id": policy.policy_id,
        "split_policy_file": policy.policy_path.name,
        "split_policy_sha256": policy.policy_sha256,
        "algorithm_version": policy.algorithm_version,
        "cutoff_date": policy.cutoff_date.isoformat(),
        "date_rule": _DATE_RULE,
        "group_rule": _GROUP_RULE,
        "cv_bucket_rule": _CV_BUCKET_RULE,
        "cv_buckets": _cv_bucket_policy_value(),
        "raw_source_file": raw_source.name,
        "raw_source_sha256": review.expected_sha256,
        "raw_source_size_bytes": review.expected_size_bytes,
        "candidate_file": lineage.path.name,
        "candidate_sha256": lineage.sha256,
        "candidate_size_bytes": lineage.size_bytes,
        "candidate_manifest_file": lineage.manifest_path.name,
        "candidate_manifest_sha256": lineage.manifest_sha256,
        "assignment_columns": list(_ASSIGNMENT_HEADER),
        "feature_allowlist": list(_FEATURE_ALLOWLIST),
        "target_column": _TARGET_COLUMN,
        "publication_status": _PUBLICATION_STATUS,
        "training_readiness": _TRAINING_READINESS,
    }
    for key, expected in exact.items():
        if manifest[key] != expected:
            raise KaggleVehicleSalesSplitError(f"split manifest {key} is invalid")
    assignment_name = _safe_file_name(manifest["assignment_file"], label="assignment_file")
    readiness_name = _safe_file_name(manifest["readiness_file"], label="readiness_file")
    assignment = Path(assignment_name)
    if (
        assignment.suffix.lower() != ".csv"
        or assignment.with_suffix(".manifest.json").name != manifest_path.name
        or assignment.with_suffix(".ready.json").name != readiness_name
    ):
        raise KaggleVehicleSalesSplitError("split artifact filenames are inconsistent")
    _require_digest_value(manifest["assignment_sha256"], label="assignment_sha256")
    _require_positive_int(manifest["assignment_size_bytes"], label="assignment_size_bytes")
    _validate_manifest_metrics(manifest["metrics"], candidate_rows=lineage.row_count)


def _validate_manifest_metrics(value: object, *, candidate_rows: int) -> None:
    if not isinstance(value, dict):
        raise KaggleVehicleSalesSplitError("split metrics must be an object")
    metrics = cast(dict[str, object], value)
    _require_exact_keys(metrics, _METRICS_KEYS, label="split metrics")
    mapping_keys = {"train_cv_bucket_rows", "train_cv_bucket_vin_groups"}
    integer_values = {
        key: _require_nonnegative_int(metrics[key], label=f"split metrics {key}")
        for key in _METRICS_KEYS - {"initial_date_holdout_percent"} - mapping_keys
    }
    bucket_rows = _require_bucket_mapping(metrics["train_cv_bucket_rows"], label="CV bucket rows")
    bucket_groups = _require_bucket_mapping(
        metrics["train_cv_bucket_vin_groups"], label="CV bucket groups"
    )
    percent = metrics["initial_date_holdout_percent"]
    if not isinstance(percent, str) or not re.fullmatch(r"[0-9]+\.[0-9]{4}", percent):
        raise KaggleVehicleSalesSplitError("split holdout percentage is invalid")
    expected_percent = (
        "0.0000"
        if candidate_rows == 0
        else f"{integer_values['initial_date_holdout_rows'] * 100 / candidate_rows:.4f}"
    )
    if percent != expected_percent:
        raise KaggleVehicleSalesSplitError("split holdout percentage does not match counts")
    if (
        integer_values["candidate_rows"] != candidate_rows
        or candidate_rows != integer_values["train_rows"] + integer_values["test_rows"]
        or integer_values["test_rows"]
        != integer_values["initial_date_holdout_rows"] + integer_values["promoted_earlier_rows"]
        or integer_values["vin_groups_total"]
        != integer_values["vin_groups_train"] + integer_values["vin_groups_test"]
        or integer_values["train_rows_on_or_after_cutoff"] != 0
        or integer_values["vin_overlap_between_partitions"] != 0
        or integer_values["train_rows_without_cv_bucket"] != 0
        or integer_values["train_vin_groups_crossing_cv_buckets"] != 0
        or sum(bucket_rows.values()) != integer_values["train_rows"]
        or sum(bucket_groups.values()) != integer_values["vin_groups_train"]
    ):
        raise KaggleVehicleSalesSplitError("split manifest metrics violate invariants")


def _verify_raw_source(path: Path, review: KaggleVehicleSalesReview) -> Path:
    raw_source = _require_regular_file(path, label="raw source CSV")
    expected_parts = tuple(part.casefold() for part in review.expected_csv_path.parts)
    actual_parts = tuple(part.casefold() for part in raw_source.parts)
    if (
        len(actual_parts) < len(expected_parts)
        or actual_parts[-len(expected_parts) :] != expected_parts
    ):
        raise KaggleVehicleSalesSplitError("raw source is not stored at the reviewed path")
    sha256, size_bytes = _hash_regular_file(raw_source)
    if sha256 != review.expected_sha256 or size_bytes != review.expected_size_bytes:
        raise KaggleVehicleSalesSplitError("raw source differs from the reviewed artifact pin")
    return raw_source


def _require_policy_review_match(
    policy: KaggleVehicleSalesSplitPolicy,
    review: KaggleVehicleSalesReview,
) -> None:
    if policy.source_review_id != review.review_id or review.review_id != _SOURCE_REVIEW_ID:
        raise KaggleVehicleSalesSplitError("split policy and source review do not match")


def _candidate_row_number(row: Mapping[str, str] | None) -> int | None:
    if row is None:
        return None
    if set(row) != set(_CANDIDATE_HEADER):
        raise KaggleVehicleSalesSplitError("candidate row width is invalid for splitting")
    value = row.get("source_listing_id", "")
    match = _ROW_ID_PATTERN.fullmatch(value)
    if match is None:
        raise KaggleVehicleSalesSplitError("candidate row identifier is invalid for splitting")
    row_number = int(match.group(1))
    if row_number < 2:
        raise KaggleVehicleSalesSplitError("candidate row identifier is out of range")
    return row_number


def _normalize_identifier(value: str) -> str | None:
    normalized = value.strip().upper()
    if not normalized or normalized == "NAN" or _VIN_PATTERN.fullmatch(normalized) is None:
        return None
    return normalized


def _parse_local_sale_date(value: str) -> date:
    match = _SALE_DATE_PATTERN.fullmatch(value)
    if match is None:
        raise KaggleVehicleSalesSplitError("accepted row has an invalid sale date")
    (
        weekday,
        month,
        day_text,
        year_text,
        hour_text,
        minute_text,
        second_text,
        sign,
        offset_hour_text,
        offset_minute_text,
        zone,
    ) = match.groups()
    offset_minutes = int(offset_hour_text) * 60 + int(offset_minute_text)
    if sign == "-":
        offset_minutes = -offset_minutes
    if (
        int(hour_text) >= 24
        or int(minute_text) >= 60
        or int(second_text) >= 60
        or int(offset_minute_text) >= 60
        or abs(offset_minutes) > 14 * 60
        or _TZ_OFFSETS.get(zone) != offset_minutes
    ):
        raise KaggleVehicleSalesSplitError("accepted row has an invalid sale date")
    try:
        result = date(int(year_text), _MONTHS[month], int(day_text))
    except ValueError as error:
        raise KaggleVehicleSalesSplitError("accepted row has an invalid sale date") from error
    if result.weekday() != _WEEKDAYS[weekday]:
        raise KaggleVehicleSalesSplitError("accepted row has an invalid sale weekday")
    return result


def _require_split(value: object) -> SplitName:
    if value == "train":
        return "train"
    if value == "test":
        return "test"
    raise KaggleVehicleSalesSplitError("split value must be train or test")


def _require_cv_bucket(value: object, *, split: SplitName) -> str | None:
    if not isinstance(value, str):
        raise KaggleVehicleSalesSplitError("CV bucket must be text")
    if split == "test":
        if value:
            raise KaggleVehicleSalesSplitError("test rows must not have a CV bucket")
        return None
    if value not in _CV_BUCKET_NAMES:
        raise KaggleVehicleSalesSplitError("train row CV bucket is invalid")
    return value


def _cv_bucket_policy_value() -> list[dict[str, str]]:
    return [
        {"name": name, "start_inclusive": start, "end_exclusive": end}
        for name, start, end in _CV_BUCKETS
    ]


def _bucket_counts(rows: object, *, label: str) -> dict[str, int]:
    if not isinstance(rows, list):
        raise KaggleVehicleSalesSplitError(f"{label} query is invalid")
    result = dict.fromkeys(_CV_BUCKET_NAMES, 0)
    for row in rows:
        if (
            not isinstance(row, tuple)
            or len(row) != 2
            or row[0] not in _CV_BUCKET_NAMES
            or type(row[1]) is not int
            or row[1] < 0
        ):
            raise KaggleVehicleSalesSplitError(f"{label} query returned invalid data")
        result[cast(str, row[0])] = row[1]
    return result


def _require_bucket_mapping(value: object, *, label: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(_CV_BUCKET_NAMES):
        raise KaggleVehicleSalesSplitError(f"{label} keys are invalid")
    result: dict[str, int] = {}
    for bucket in _CV_BUCKET_NAMES:
        result[bucket] = _require_nonnegative_int(value[bucket], label=f"{label} {bucket}")
    return result


def _safe_row_id(row_number: int) -> str:
    return f"row-{row_number:09d}"


def _training_int(value: object, *, label: str) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise KaggleVehicleSalesSplitError(f"training {label} is not an integer")
    return int(value)


def _optional_training_int(value: object, *, label: str) -> int | None:
    if value == "":
        return None
    return _training_int(value, label=label)


def _optional_training_float(value: object, *, label: str) -> float | None:
    if value == "":
        return None
    if not isinstance(value, str):
        raise KaggleVehicleSalesSplitError(f"training {label} is invalid")
    try:
        result = float(value)
    except ValueError as error:
        raise KaggleVehicleSalesSplitError(f"training {label} is invalid") from error
    if not 1.0 <= result <= 5.0:
        raise KaggleVehicleSalesSplitError(f"training {label} is out of range")
    return result


def _training_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise KaggleVehicleSalesSplitError(f"training {label} is empty")
    return value


def _sqlite_nonnegative_row(value: object, *, width: int, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or len(value) != width
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise KaggleVehicleSalesSplitError(f"{label} are invalid")
    return cast(tuple[int, ...], value)


def _validate_output_targets(paths: Sequence[Path], *, protected: Sequence[Path]) -> None:
    normalized = [os.path.normcase(str(path.resolve(strict=False))) for path in paths]
    if len(normalized) != len(set(normalized)):
        raise KaggleVehicleSalesSplitError("split output paths must be distinct")
    protected_values = {os.path.normcase(str(path.resolve(strict=True))) for path in protected}
    if any(value in protected_values for value in normalized):
        raise KaggleVehicleSalesSplitError("split output must not overwrite a lineage input")
    for path in paths:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise KaggleVehicleSalesSplitError("split output targets must be regular files")


def _require_regular_file(path: Path, *, label: str) -> Path:
    if not isinstance(path, Path) or path.is_symlink():
        raise KaggleVehicleSalesSplitError(f"{label} must be a non-symlink local file")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise KaggleVehicleSalesSplitError(f"{label} is missing or inaccessible") from error
    if not resolved.is_file():
        raise KaggleVehicleSalesSplitError(f"{label} must be a regular file")
    return resolved


def _hash_regular_file(path: Path) -> tuple[str, int]:
    resolved = _require_regular_file(path, label="artifact")
    before = resolved.stat()
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    after = resolved.stat()
    before_signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_signature != after_signature or size != after.st_size:
        raise KaggleVehicleSalesSplitError("artifact changed while being hashed")
    if size == 0:
        raise KaggleVehicleSalesSplitError("artifact must not be empty")
    return digest.hexdigest(), size


def _require_hash(path: Path, expected: str, *, label: str) -> None:
    actual, _ = _hash_regular_file(path)
    if actual != expected:
        raise KaggleVehicleSalesSplitError(f"{label} changed after verification")


def _read_bounded_bytes(path: Path, *, max_bytes: int, label: str) -> bytes:
    resolved = _require_regular_file(path, label=label)
    before = resolved.stat()
    if before.st_size <= 0 or before.st_size > max_bytes:
        raise KaggleVehicleSalesSplitError(f"{label} is empty or exceeds its byte limit")
    payload = resolved.read_bytes()
    after = resolved.stat()
    before_signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_signature != after_signature or len(payload) != after.st_size:
        raise KaggleVehicleSalesSplitError(f"{label} changed while being read")
    return payload


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise KaggleVehicleSalesSplitError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise KaggleVehicleSalesSplitError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _require_exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise KaggleVehicleSalesSplitError(f"{label} fields are invalid")


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise KaggleVehicleSalesSplitError(f"{label} must be a nonnegative integer")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise KaggleVehicleSalesSplitError(f"{label} must be a positive integer")
    return value


def _require_digest_value(value: object, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise KaggleVehicleSalesSplitError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_file_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise KaggleVehicleSalesSplitError(f"{label} must be a safe filename")
    return value


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("wb") as file_handle:
        file_handle.write(payload)
        file_handle.flush()
        os.fsync(file_handle.fileno())


def _json_payload(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


__all__ = [
    "KaggleVehicleSalesSplitArtifactSet",
    "KaggleVehicleSalesSplitError",
    "KaggleVehicleSalesSplitMetrics",
    "KaggleVehicleSalesSplitPolicy",
    "KaggleVehicleSalesTrainingRows",
    "TrainingRow",
    "VerifiedKaggleVehicleSalesSplit",
    "load_kaggle_vehicle_sales_split_policy",
    "prepare_kaggle_vehicle_sales_training_rows",
    "process_kaggle_vehicle_sales_split",
    "verify_kaggle_vehicle_sales_split_artifact_set",
]
