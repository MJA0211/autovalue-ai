"""Deterministic, group-aware holdout split for the retail asking-price track.

The reviewed US Sales Cars artifact has neither row-level observation dates nor
stable upstream listing identifiers.  A temporal split is therefore impossible
without inventing chronology.  This module uses a reproducible SHA-256 group
holdout instead.  Rows with the same predictor tuple always receive the same
partition, while ``price_cents`` never participates in grouping or assignment.

Published assignments are deliberately narrow and private: they contain only
the adapter's opaque ``source_listing_id`` and ``train``/``test``.  Source row
values, including Dealer, never leave the verified candidate during splitting.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, Literal, TextIO, cast

from autovalue_ml.acquisition.sources.kaggle_us_sales_cars import (
    KaggleUSSalesCarsError,
    load_kaggle_us_sales_cars_review,
    require_kaggle_us_sales_cars_ml_training_approval,
    verify_kaggle_us_sales_cars_artifact_set,
)

SplitPartition = Literal["train", "test"]

SPLIT_SCHEMA_VERSION: Final = 1
SPLIT_ASSIGNMENT_COLUMNS: Final[tuple[str, str]] = ("source_listing_id", "split")
SPLIT_GROUP_FIELDS: Final[tuple[str, ...]] = (
    "year",
    "make",
    "model",
    "mileage",
    "vehicle_status",
)
SPLIT_SEED: Final = "autovalue-ai:kaggle-us-sales-cars-v2:group-holdout:20260828:v1"
SPLIT_ALGORITHM_VERSION: Final = "sha256-status-row-balanced-threshold-v1"

_SOURCE_ID: Final = "kaggle_us_sales_cars_v2"
_TARGET_TRACK: Final = "historical_us_retail_asking_price"
_ASSIGNMENTS_NAME: Final = "split_assignments.csv"
_MANIFEST_NAME: Final = "split_assignments.manifest.json"
_READINESS_NAME: Final = "split_assignments.ready.json"
_TEST_NUMERATOR: Final = 1
_TEST_DENOMINATOR: Final = 5
_HASH_CHUNK_BYTES: Final = 1024 * 1024
_MAX_JSON_BYTES: Final = 5_000_000
_SAFE_ROW_ID_PATTERN = re.compile(r"^row-[a-f0-9]{24}$")
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_NON_TEMPORAL_LIMITATION: Final = (
    "The source has no row-level observation dates or stable upstream listing IDs; this "
    "holdout measures reproducible grouped generalization, not forward-in-time performance."
)
_GROUP_PAYLOAD_FORMAT: Final = (
    "UTF-8(seed) || 0x00 || UTF-8(compact JSON array "
    "[year,make,model,mileage_or_null,vehicle_status])"
)
_ALLOCATION_POLICY: Final = (
    "Within each vehicle_status, sort indivisible predictor groups by seeded SHA-256 and select "
    "the prefix whose cumulative row count is closest to one fifth of that status. Ties choose "
    "the smaller prefix. Targets and model metrics are excluded."
)

_CANDIDATE_COLUMNS: Final[tuple[str, ...]] = (
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
_STATUSES: Final[tuple[str, ...]] = ("certified", "new", "used")


class KaggleUSSalesCarsSplitError(KaggleUSSalesCarsError):
    """The split, its lineage, or its semantics failed closed."""


@dataclass(frozen=True, slots=True)
class KaggleUSSalesCarsSplitMetrics:
    """Aggregate split counts; no source row values are retained."""

    total_rows: int
    train_rows: int
    test_rows: int
    total_groups: int
    train_groups: int
    test_groups: int
    status_slices: Mapping[str, Mapping[str, int]]

    @property
    def realized_test_fraction(self) -> float:
        return self.test_rows / self.total_rows

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": {
                "total": self.total_rows,
                "train": self.train_rows,
                "test": self.test_rows,
            },
            "groups": {
                "total": self.total_groups,
                "train": self.train_groups,
                "test": self.test_groups,
            },
            "status_slices": {status: dict(self.status_slices[status]) for status in _STATUSES},
            "realized_test_fraction": self.realized_test_fraction,
        }


@dataclass(frozen=True, slots=True)
class KaggleUSSalesCarsSplitArtifactSet:
    assignments_path: Path
    manifest_path: Path
    readiness_path: Path
    metrics: KaggleUSSalesCarsSplitMetrics


@dataclass(frozen=True, slots=True)
class VerifiedKaggleUSSalesCarsSplit:
    assignments_path: Path
    assignments_sha256: str
    manifest_path: Path
    manifest_sha256: str
    candidate_path: Path
    candidate_sha256: str
    artifact_set_id: str
    metrics: KaggleUSSalesCarsSplitMetrics


@dataclass(frozen=True, slots=True)
class KaggleUSSalesCarsSplitTrainingRows:
    """A lazy, hash-pinned stream for one verified holdout partition."""

    candidate_path: Path
    candidate_sha256: str
    assignments_path: Path
    assignments_sha256: str
    partition: SplitPartition
    expected_rows: int

    def __iter__(self) -> Iterator[tuple[dict[str, str | int], float]]:
        _require_file_hash(
            self.assignments_path,
            self.assignments_sha256,
            label="split assignments",
        )
        selected_ids = _read_selected_assignment_ids(self.assignments_path, self.partition)
        if len(selected_ids) != self.expected_rows:
            raise KaggleUSSalesCarsSplitError("split assignment count changed after approval")
        _require_file_hash(self.candidate_path, self.candidate_sha256, label="candidate")

        emitted = 0
        try:
            with self.candidate_path.open("r", encoding="utf-8", newline="") as stream:
                for row in _read_csv_rows(
                    stream,
                    expected_header=_CANDIDATE_COLUMNS,
                    label="candidate",
                ):
                    _validate_candidate_row(row)
                    if row["source_listing_id"] not in selected_ids:
                        continue
                    features: dict[str, str | int] = {
                        "year": int(row["year"]),
                        "make": row["make"],
                        "model": row["model"],
                        "vehicle_status": row["vehicle_status"],
                    }
                    if row["mileage"]:
                        features["mileage"] = int(row["mileage"])
                    emitted += 1
                    yield features, int(row["price_cents"]) / 100
        except (OSError, UnicodeError, csv.Error) as error:
            raise KaggleUSSalesCarsSplitError("candidate cannot be streamed safely") from error
        if emitted != self.expected_rows:
            raise KaggleUSSalesCarsSplitError("candidate and split assignments no longer account")
        _require_file_hash(self.candidate_path, self.candidate_sha256, label="candidate")
        _require_file_hash(
            self.assignments_path,
            self.assignments_sha256,
            label="split assignments",
        )


@dataclass(frozen=True, slots=True)
class _VerifiedSourceContext:
    candidate_path: Path
    candidate_sha256: str
    candidate_size_bytes: int
    candidate_manifest_path: Path
    candidate_manifest_sha256: str
    candidate_artifact_set_id: str
    review_id: str
    review_sha256: str
    approved_for_ml_training: bool


@dataclass(frozen=True, slots=True)
class _SplitAllocation:
    partitions: Mapping[tuple[int, str, str, int | None, str], SplitPartition]
    status_thresholds: Mapping[str, Mapping[str, str | int | None]]


def build_kaggle_us_sales_cars_group_split(
    candidate_path: Path,
    candidate_manifest_path: Path,
    review_path: Path,
    output_dir: Path,
    *,
    today: date | None = None,
) -> KaggleUSSalesCarsSplitArtifactSet:
    """Build and atomically publish the private deterministic holdout assignments."""

    context = _verify_source_context(
        candidate_path,
        candidate_manifest_path,
        review_path,
        today=today,
        require_ml=True,
    )
    if not context.approved_for_ml_training:
        raise KaggleUSSalesCarsSplitError("source review does not approve ML training")
    output_parent = _prepare_output_directory(output_dir)
    assignments_path = output_parent / _ASSIGNMENTS_NAME
    manifest_path = output_parent / _MANIFEST_NAME
    readiness_path = output_parent / _READINESS_NAME
    for target in (assignments_path, manifest_path, readiness_path):
        if target.is_symlink():
            raise KaggleUSSalesCarsSplitError("split output targets cannot be symlinks")
        if target.resolve(strict=False) in {
            context.candidate_path,
            context.candidate_manifest_path,
        }:
            raise KaggleUSSalesCarsSplitError("split output cannot overwrite source artifacts")

    staging_path = Path(tempfile.mkdtemp(prefix=".group-split.", dir=output_parent))
    try:
        staged_assignments = staging_path / _ASSIGNMENTS_NAME
        metrics, allocation = _write_assignments(context, staged_assignments)
        assignments_sha256, assignments_size = _hash_regular_file(staged_assignments)

        manifest = _build_manifest(
            context=context,
            assignments_sha256=assignments_sha256,
            assignments_size=assignments_size,
            metrics=metrics,
            allocation=allocation,
        )
        staged_manifest = staging_path / _MANIFEST_NAME
        _write_fsynced(staged_manifest, _json_payload(manifest))
        manifest_sha256, _ = _hash_regular_file(staged_manifest)

        artifact_set_id = hashlib.sha256(
            "|".join((manifest_sha256, assignments_sha256, context.candidate_sha256)).encode(
                "ascii"
            )
        ).hexdigest()
        readiness = {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "artifact_set_id": artifact_set_id,
            "manifest_file": _MANIFEST_NAME,
            "manifest_sha256": manifest_sha256,
            "assignments_file": _ASSIGNMENTS_NAME,
            "assignments_sha256": assignments_sha256,
            "candidate_sha256": context.candidate_sha256,
        }
        staged_readiness = staging_path / _READINESS_NAME
        _write_fsynced(staged_readiness, _json_payload(readiness))

        # The readiness marker is the commit point and is always published last.
        os.replace(staged_assignments, assignments_path)
        os.replace(staged_manifest, manifest_path)
        os.replace(staged_readiness, readiness_path)
        return KaggleUSSalesCarsSplitArtifactSet(
            assignments_path=assignments_path,
            manifest_path=manifest_path,
            readiness_path=readiness_path,
            metrics=metrics,
        )
    except KaggleUSSalesCarsSplitError:
        raise
    except OSError as error:
        raise KaggleUSSalesCarsSplitError("could not publish split artifact set") from error
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)


def verify_kaggle_us_sales_cars_group_split(
    split_manifest_path: Path,
    candidate_path: Path,
    candidate_manifest_path: Path,
    review_path: Path,
    *,
    today: date | None = None,
) -> VerifiedKaggleUSSalesCarsSplit:
    """Verify hashes, lineage, policy, accounting, and group isolation."""

    context = _verify_source_context(
        candidate_path,
        candidate_manifest_path,
        review_path,
        today=today,
        require_ml=False,
    )
    manifest_file = _require_regular_file(split_manifest_path, label="split manifest")
    manifest_payload = _read_bounded_bytes(
        manifest_file,
        max_bytes=_MAX_JSON_BYTES,
        label="split manifest",
    )
    manifest = _strict_json_object(manifest_payload, label="split manifest")
    _validate_manifest(manifest, context=context)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()

    readiness_path = _require_regular_file(
        manifest_file.parent / _READINESS_NAME,
        label="split readiness marker",
    )
    readiness = _strict_json_object(
        _read_bounded_bytes(
            readiness_path,
            max_bytes=1_000_000,
            label="split readiness marker",
        ),
        label="split readiness marker",
    )
    _require_exact_keys(
        readiness,
        {
            "schema_version",
            "artifact_set_id",
            "manifest_file",
            "manifest_sha256",
            "assignments_file",
            "assignments_sha256",
            "candidate_sha256",
        },
        label="split readiness marker",
    )
    if (
        readiness["schema_version"] != SPLIT_SCHEMA_VERSION
        or readiness["manifest_file"] != manifest_file.name
        or readiness["assignments_file"] != _ASSIGNMENTS_NAME
        or readiness["candidate_sha256"] != context.candidate_sha256
    ):
        raise KaggleUSSalesCarsSplitError("split readiness semantics or lineage are invalid")
    _require_hash_match(readiness["manifest_sha256"], manifest_sha256, label="split manifest")

    assignments_path = _require_regular_file(
        manifest_file.parent / _ASSIGNMENTS_NAME,
        label="split assignments",
    )
    assignments_sha256, assignments_size = _hash_regular_file(assignments_path)
    _require_hash_match(
        readiness["assignments_sha256"],
        assignments_sha256,
        label="split assignments",
    )
    assignments_manifest = _require_object(manifest["assignments"], label="assignments")
    if (
        assignments_manifest["sha256"] != assignments_sha256
        or assignments_manifest["size_bytes"] != assignments_size
    ):
        raise KaggleUSSalesCarsSplitError("split assignment metadata is invalid")

    expected_artifact_set_id = hashlib.sha256(
        "|".join((manifest_sha256, assignments_sha256, context.candidate_sha256)).encode("ascii")
    ).hexdigest()
    _require_hash_match(
        readiness["artifact_set_id"],
        expected_artifact_set_id,
        label="split artifact set",
    )

    assignments = _read_assignments(assignments_path)
    metrics, allocation = _verify_candidate_assignments(context, assignments)
    expected_counts = _require_object(manifest["counts"], label="split counts")
    if expected_counts != metrics.to_dict():
        raise KaggleUSSalesCarsSplitError("split counts do not match verified assignments")
    if manifest["allocation"] != _allocation_to_dict(allocation):
        raise KaggleUSSalesCarsSplitError("split allocation thresholds do not match candidate")

    return VerifiedKaggleUSSalesCarsSplit(
        assignments_path=assignments_path,
        assignments_sha256=assignments_sha256,
        manifest_path=manifest_file,
        manifest_sha256=manifest_sha256,
        candidate_path=context.candidate_path,
        candidate_sha256=context.candidate_sha256,
        artifact_set_id=expected_artifact_set_id,
        metrics=metrics,
    )


def prepare_kaggle_us_sales_cars_split_training_rows(
    candidate_path: Path,
    candidate_manifest_path: Path,
    split_manifest_path: Path,
    review_path: Path,
    *,
    partition: SplitPartition,
    today: date | None = None,
) -> KaggleUSSalesCarsSplitTrainingRows:
    """Return a lazy partition stream only after ML approval and split verification."""

    if partition not in {"train", "test"}:
        raise KaggleUSSalesCarsSplitError("partition must be train or test")
    context = _verify_source_context(
        candidate_path,
        candidate_manifest_path,
        review_path,
        today=today,
        require_ml=True,
    )
    if not context.approved_for_ml_training:
        raise KaggleUSSalesCarsSplitError("source review does not approve ML training")
    verified = verify_kaggle_us_sales_cars_group_split(
        split_manifest_path,
        candidate_path,
        candidate_manifest_path,
        review_path,
        today=today,
    )
    expected_rows = (
        verified.metrics.train_rows if partition == "train" else verified.metrics.test_rows
    )
    return KaggleUSSalesCarsSplitTrainingRows(
        candidate_path=verified.candidate_path,
        candidate_sha256=verified.candidate_sha256,
        assignments_path=verified.assignments_path,
        assignments_sha256=verified.assignments_sha256,
        partition=partition,
        expected_rows=expected_rows,
    )


def _verify_source_context(
    candidate_path: Path,
    candidate_manifest_path: Path,
    review_path: Path,
    *,
    today: date | None,
    require_ml: bool,
) -> _VerifiedSourceContext:
    review = load_kaggle_us_sales_cars_review(review_path, today=today)
    if require_ml:
        review = require_kaggle_us_sales_cars_ml_training_approval(review)
    readiness = verify_kaggle_us_sales_cars_artifact_set(
        candidate_manifest_path,
        review_path,
        today=today,
    )
    candidate = _require_regular_file(candidate_path, label="candidate")
    candidate_manifest = _require_regular_file(
        candidate_manifest_path,
        label="candidate manifest",
    )
    source_manifest_payload = _read_bounded_bytes(
        candidate_manifest,
        max_bytes=_MAX_JSON_BYTES,
        label="candidate manifest",
    )
    source_manifest = _strict_json_object(source_manifest_payload, label="candidate manifest")
    candidate_name = _safe_filename(source_manifest.get("candidate_file"), label="candidate_file")
    if candidate != (candidate_manifest.parent / candidate_name).resolve(strict=True):
        raise KaggleUSSalesCarsSplitError("candidate differs from its verified manifest")
    candidate_sha256, candidate_size = _hash_regular_file(candidate)
    if (
        readiness.get("candidate_sha256") != candidate_sha256
        or source_manifest.get("candidate_sha256") != candidate_sha256
        or source_manifest.get("candidate_size_bytes") != candidate_size
        or source_manifest.get("source_id") != _SOURCE_ID
        or source_manifest.get("target_track") != _TARGET_TRACK
        or source_manifest.get("review_id") != review.review_id
        or source_manifest.get("review_sha256") != review.review_sha256
    ):
        raise KaggleUSSalesCarsSplitError("candidate lineage is invalid")
    artifact_set_id = _require_sha256(
        readiness.get("artifact_set_id"),
        label="candidate artifact_set_id",
    )
    return _VerifiedSourceContext(
        candidate_path=candidate,
        candidate_sha256=candidate_sha256,
        candidate_size_bytes=candidate_size,
        candidate_manifest_path=candidate_manifest,
        candidate_manifest_sha256=hashlib.sha256(source_manifest_payload).hexdigest(),
        candidate_artifact_set_id=artifact_set_id,
        review_id=review.review_id,
        review_sha256=review.review_sha256,
        approved_for_ml_training=review.approved_for_ml_training,
    )


def _write_assignments(
    context: _VerifiedSourceContext,
    output_path: Path,
) -> tuple[KaggleUSSalesCarsSplitMetrics, _SplitAllocation]:
    before = _file_signature(context.candidate_path.stat())
    group_counts = _collect_candidate_group_counts(context)
    allocation = _derive_split_allocation(group_counts)
    seen_ids: set[str] = set()
    row_counts = {"train": 0, "test": 0}
    status_counts = {status: {"total": 0, "train": 0, "test": 0} for status in _STATUSES}
    try:
        with (
            context.candidate_path.open("r", encoding="utf-8", newline="") as source,
            output_path.open("w", encoding="utf-8", newline="") as destination,
        ):
            writer = csv.writer(destination, lineterminator="\r\n")
            writer.writerow(SPLIT_ASSIGNMENT_COLUMNS)
            for row in _read_csv_rows(
                source,
                expected_header=_CANDIDATE_COLUMNS,
                label="candidate",
            ):
                _validate_candidate_row(row)
                listing_id = row["source_listing_id"]
                if listing_id in seen_ids:
                    raise KaggleUSSalesCarsSplitError("candidate has duplicate opaque listing IDs")
                seen_ids.add(listing_id)
                group = _candidate_group(row)
                partition = allocation.partitions[group]
                writer.writerow((listing_id, partition))
                row_counts[partition] += 1
                status = row["vehicle_status"]
                status_counts[status]["total"] += 1
                status_counts[status][partition] += 1
            destination.flush()
            os.fsync(destination.fileno())
    except KaggleUSSalesCarsSplitError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise KaggleUSSalesCarsSplitError("candidate could not be split safely") from error
    after = _file_signature(context.candidate_path.stat())
    if before != after:
        raise KaggleUSSalesCarsSplitError("candidate changed while creating split")
    _require_file_hash(context.candidate_path, context.candidate_sha256, label="candidate")
    if not seen_ids or not row_counts["train"] or not row_counts["test"]:
        raise KaggleUSSalesCarsSplitError("split must contain non-empty train and test partitions")

    train_groups = sum(partition == "train" for partition in allocation.partitions.values())
    test_groups = sum(partition == "test" for partition in allocation.partitions.values())
    metrics = KaggleUSSalesCarsSplitMetrics(
        total_rows=len(seen_ids),
        train_rows=row_counts["train"],
        test_rows=row_counts["test"],
        total_groups=len(allocation.partitions),
        train_groups=train_groups,
        test_groups=test_groups,
        status_slices=status_counts,
    )
    return metrics, allocation


def _build_manifest(
    *,
    context: _VerifiedSourceContext,
    assignments_sha256: str,
    assignments_size: int,
    metrics: KaggleUSSalesCarsSplitMetrics,
    allocation: _SplitAllocation,
) -> dict[str, object]:
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "source_id": _SOURCE_ID,
        "target_track": _TARGET_TRACK,
        "split_name": "deterministic_group_holdout_v1",
        "publication_status": "private_local_only",
        "non_temporal_split": True,
        "non_temporal_limitation": _NON_TEMPORAL_LIMITATION,
        "algorithm": {
            "name": "SHA-256 status-stratified row-balanced group threshold",
            "version": SPLIT_ALGORITHM_VERSION,
            "seed": SPLIT_SEED,
            "allocation_policy": _ALLOCATION_POLICY,
            "payload_format": _GROUP_PAYLOAD_FORMAT,
            "comparison": "group_sha256 <= status_cutoff_sha256",
            "test_fraction_numerator": _TEST_NUMERATOR,
            "test_fraction_denominator": _TEST_DENOMINATOR,
        },
        "grouping": {
            "fields": list(SPLIT_GROUP_FIELDS),
            "mileage_null_encoding": "JSON null",
            "target_fields_excluded": ["price_cents"],
            "identical_predictor_groups_may_not_cross_partitions": True,
        },
        "privacy": {
            "assignment_columns": list(SPLIT_ASSIGNMENT_COLUMNS),
            "forbidden_source_columns": ["Dealer"],
            "raw_source_values_in_assignments": False,
            "target_in_assignments": False,
        },
        "source_lineage": {
            "candidate_file": context.candidate_path.name,
            "candidate_sha256": context.candidate_sha256,
            "candidate_size_bytes": context.candidate_size_bytes,
            "candidate_manifest_file": context.candidate_manifest_path.name,
            "candidate_manifest_sha256": context.candidate_manifest_sha256,
            "candidate_artifact_set_id": context.candidate_artifact_set_id,
            "review_id": context.review_id,
            "review_sha256": context.review_sha256,
            "approved_for_ml_training": context.approved_for_ml_training,
        },
        "assignments": {
            "file": _ASSIGNMENTS_NAME,
            "sha256": assignments_sha256,
            "size_bytes": assignments_size,
            "columns": list(SPLIT_ASSIGNMENT_COLUMNS),
        },
        "allocation": _allocation_to_dict(allocation),
        "counts": metrics.to_dict(),
    }


_MANIFEST_KEYS: Final = {
    "schema_version",
    "source_id",
    "target_track",
    "split_name",
    "publication_status",
    "non_temporal_split",
    "non_temporal_limitation",
    "algorithm",
    "grouping",
    "privacy",
    "source_lineage",
    "assignments",
    "allocation",
    "counts",
}


def _validate_manifest(
    manifest: dict[str, object],
    *,
    context: _VerifiedSourceContext,
) -> None:
    _require_exact_keys(manifest, _MANIFEST_KEYS, label="split manifest")
    exact_top_level: dict[str, object] = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "source_id": _SOURCE_ID,
        "target_track": _TARGET_TRACK,
        "split_name": "deterministic_group_holdout_v1",
        "publication_status": "private_local_only",
        "non_temporal_split": True,
        "non_temporal_limitation": _NON_TEMPORAL_LIMITATION,
    }
    if any(manifest[key] != value for key, value in exact_top_level.items()):
        raise KaggleUSSalesCarsSplitError("split manifest semantics are invalid")

    algorithm = _require_object(manifest["algorithm"], label="split algorithm")
    expected_algorithm: dict[str, object] = {
        "name": "SHA-256 status-stratified row-balanced group threshold",
        "version": SPLIT_ALGORITHM_VERSION,
        "seed": SPLIT_SEED,
        "allocation_policy": _ALLOCATION_POLICY,
        "payload_format": _GROUP_PAYLOAD_FORMAT,
        "comparison": "group_sha256 <= status_cutoff_sha256",
        "test_fraction_numerator": _TEST_NUMERATOR,
        "test_fraction_denominator": _TEST_DENOMINATOR,
    }
    if algorithm != expected_algorithm:
        raise KaggleUSSalesCarsSplitError("split algorithm policy is invalid")
    grouping = _require_object(manifest["grouping"], label="split grouping")
    if grouping != {
        "fields": list(SPLIT_GROUP_FIELDS),
        "mileage_null_encoding": "JSON null",
        "target_fields_excluded": ["price_cents"],
        "identical_predictor_groups_may_not_cross_partitions": True,
    }:
        raise KaggleUSSalesCarsSplitError("split grouping policy is invalid")
    privacy = _require_object(manifest["privacy"], label="split privacy")
    if privacy != {
        "assignment_columns": list(SPLIT_ASSIGNMENT_COLUMNS),
        "forbidden_source_columns": ["Dealer"],
        "raw_source_values_in_assignments": False,
        "target_in_assignments": False,
    }:
        raise KaggleUSSalesCarsSplitError("split privacy policy is invalid")

    lineage = _require_object(manifest["source_lineage"], label="source_lineage")
    if lineage != {
        "candidate_file": context.candidate_path.name,
        "candidate_sha256": context.candidate_sha256,
        "candidate_size_bytes": context.candidate_size_bytes,
        "candidate_manifest_file": context.candidate_manifest_path.name,
        "candidate_manifest_sha256": context.candidate_manifest_sha256,
        "candidate_artifact_set_id": context.candidate_artifact_set_id,
        "review_id": context.review_id,
        "review_sha256": context.review_sha256,
        "approved_for_ml_training": context.approved_for_ml_training,
    }:
        raise KaggleUSSalesCarsSplitError("split source lineage is invalid")
    assignments = _require_object(manifest["assignments"], label="assignments")
    _require_exact_keys(
        assignments,
        {"file", "sha256", "size_bytes", "columns"},
        label="assignments",
    )
    if assignments["file"] != _ASSIGNMENTS_NAME or assignments["columns"] != list(
        SPLIT_ASSIGNMENT_COLUMNS
    ):
        raise KaggleUSSalesCarsSplitError("split assignment schema is invalid")
    _require_sha256(assignments["sha256"], label="assignments sha256")
    _require_positive_int(assignments["size_bytes"], label="assignments size_bytes")
    _validate_allocation_shape(manifest["allocation"])
    _validate_counts_shape(manifest["counts"])


def _read_assignments(path: Path) -> dict[str, SplitPartition]:
    result: dict[str, SplitPartition] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in _read_csv_rows(
                stream,
                expected_header=SPLIT_ASSIGNMENT_COLUMNS,
                label="split assignments",
            ):
                listing_id = row["source_listing_id"]
                partition = row["split"]
                if not _SAFE_ROW_ID_PATTERN.fullmatch(listing_id):
                    raise KaggleUSSalesCarsSplitError(
                        "split assignments contain an invalid opaque listing ID"
                    )
                if partition not in {"train", "test"}:
                    raise KaggleUSSalesCarsSplitError(
                        "split assignments contain an invalid partition"
                    )
                if listing_id in result:
                    raise KaggleUSSalesCarsSplitError(
                        "split assignments contain a duplicate opaque listing ID"
                    )
                result[listing_id] = cast(SplitPartition, partition)
    except KaggleUSSalesCarsSplitError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise KaggleUSSalesCarsSplitError("split assignments cannot be read safely") from error
    if not result:
        raise KaggleUSSalesCarsSplitError("split assignments must not be empty")
    return result


def _read_selected_assignment_ids(path: Path, partition: SplitPartition) -> set[str]:
    return {
        listing_id
        for listing_id, assigned_partition in _read_assignments(path).items()
        if assigned_partition == partition
    }


def _collect_candidate_group_counts(
    context: _VerifiedSourceContext,
) -> dict[tuple[int, str, str, int | None, str], int]:
    before = _file_signature(context.candidate_path.stat())
    group_counts: dict[tuple[int, str, str, int | None, str], int] = {}
    seen_ids: set[str] = set()
    try:
        with context.candidate_path.open("r", encoding="utf-8", newline="") as stream:
            for row in _read_csv_rows(
                stream,
                expected_header=_CANDIDATE_COLUMNS,
                label="candidate",
            ):
                _validate_candidate_row(row)
                listing_id = row["source_listing_id"]
                if listing_id in seen_ids:
                    raise KaggleUSSalesCarsSplitError("candidate has duplicate opaque listing IDs")
                seen_ids.add(listing_id)
                group = _candidate_group(row)
                group_counts[group] = group_counts.get(group, 0) + 1
    except KaggleUSSalesCarsSplitError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise KaggleUSSalesCarsSplitError("candidate groups cannot be counted safely") from error
    after = _file_signature(context.candidate_path.stat())
    if before != after:
        raise KaggleUSSalesCarsSplitError("candidate changed while counting predictor groups")
    _require_file_hash(context.candidate_path, context.candidate_sha256, label="candidate")
    if not group_counts:
        raise KaggleUSSalesCarsSplitError("candidate must contain predictor groups")
    return group_counts


def _derive_split_allocation(
    group_counts: Mapping[tuple[int, str, str, int | None, str], int],
) -> _SplitAllocation:
    partitions: dict[tuple[int, str, str, int | None, str], SplitPartition] = {
        group: "train" for group in group_counts
    }
    thresholds: dict[str, Mapping[str, str | int | None]] = {}
    seen_digests: dict[str, tuple[int, str, str, int | None, str]] = {}

    for status in _STATUSES:
        ranked: list[tuple[str, tuple[int, str, str, int | None, str], int]] = []
        total_rows = 0
        for group, row_count in group_counts.items():
            if group[4] != status:
                continue
            digest = _group_sha256(group)
            previous = seen_digests.setdefault(digest, group)
            if previous != group:
                raise KaggleUSSalesCarsSplitError("predictor-group SHA-256 collision")
            ranked.append((digest, group, row_count))
            total_rows += row_count
        ranked.sort(key=lambda value: value[0])

        best_group_count = 0
        best_test_rows = 0
        best_scaled_deviation = total_rows * _TEST_NUMERATOR
        cumulative_rows = 0
        for index, (_, _, row_count) in enumerate(ranked, start=1):
            cumulative_rows += row_count
            scaled_deviation = abs(
                cumulative_rows * _TEST_DENOMINATOR - total_rows * _TEST_NUMERATOR
            )
            if scaled_deviation < best_scaled_deviation:
                best_scaled_deviation = scaled_deviation
                best_group_count = index
                best_test_rows = cumulative_rows

        for _, group, _ in ranked[:best_group_count]:
            partitions[group] = "test"
        cutoff = ranked[best_group_count - 1][0] if best_group_count else None
        thresholds[status] = {
            "total_rows": total_rows,
            "train_rows": total_rows - best_test_rows,
            "test_rows": best_test_rows,
            "total_groups": len(ranked),
            "train_groups": len(ranked) - best_group_count,
            "test_groups": best_group_count,
            "cutoff_sha256": cutoff,
            "target_test_rows_floor": total_rows * _TEST_NUMERATOR // _TEST_DENOMINATOR,
            "target_test_rows_remainder_numerator": (
                total_rows * _TEST_NUMERATOR % _TEST_DENOMINATOR
            ),
        }
    return _SplitAllocation(partitions=partitions, status_thresholds=thresholds)


def _allocation_to_dict(allocation: _SplitAllocation) -> dict[str, object]:
    return {
        "stratification_field": "vehicle_status",
        "target_unit": "rows",
        "zero_group_overlap_by_construction": True,
        "status_thresholds": {
            status: dict(allocation.status_thresholds[status]) for status in _STATUSES
        },
    }


def _verify_candidate_assignments(
    context: _VerifiedSourceContext,
    assignments: Mapping[str, SplitPartition],
) -> tuple[KaggleUSSalesCarsSplitMetrics, _SplitAllocation]:
    before = _file_signature(context.candidate_path.stat())
    group_counts = _collect_candidate_group_counts(context)
    allocation = _derive_split_allocation(group_counts)
    remaining_ids = set(assignments)
    candidate_ids: set[str] = set()
    group_partitions: dict[tuple[int, str, str, int | None, str], SplitPartition] = {}
    row_counts = {"train": 0, "test": 0}
    status_counts = {status: {"total": 0, "train": 0, "test": 0} for status in _STATUSES}
    try:
        with context.candidate_path.open("r", encoding="utf-8", newline="") as stream:
            for row in _read_csv_rows(
                stream,
                expected_header=_CANDIDATE_COLUMNS,
                label="candidate",
            ):
                _validate_candidate_row(row)
                listing_id = row["source_listing_id"]
                if listing_id in candidate_ids:
                    raise KaggleUSSalesCarsSplitError("candidate has duplicate opaque listing IDs")
                candidate_ids.add(listing_id)
                partition = assignments.get(listing_id)
                if partition is None:
                    raise KaggleUSSalesCarsSplitError("candidate row is missing a split assignment")
                remaining_ids.discard(listing_id)
                group = _candidate_group(row)
                expected_partition = allocation.partitions[group]
                if partition != expected_partition:
                    raise KaggleUSSalesCarsSplitError(
                        "split assignment differs from deterministic group policy"
                    )
                previous = group_partitions.setdefault(group, partition)
                if previous != partition:
                    raise KaggleUSSalesCarsSplitError("predictor group crossed split partitions")
                row_counts[partition] += 1
                status = row["vehicle_status"]
                status_counts[status]["total"] += 1
                status_counts[status][partition] += 1
    except KaggleUSSalesCarsSplitError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise KaggleUSSalesCarsSplitError("candidate cannot be verified safely") from error
    if remaining_ids:
        raise KaggleUSSalesCarsSplitError("split assignments contain IDs absent from candidate")
    after = _file_signature(context.candidate_path.stat())
    if before != after:
        raise KaggleUSSalesCarsSplitError("candidate changed during split verification")
    _require_file_hash(context.candidate_path, context.candidate_sha256, label="candidate")
    train_groups = sum(partition == "train" for partition in allocation.partitions.values())
    test_groups = sum(partition == "test" for partition in allocation.partitions.values())
    if not row_counts["train"] or not row_counts["test"]:
        raise KaggleUSSalesCarsSplitError("split must contain non-empty train and test partitions")
    metrics = KaggleUSSalesCarsSplitMetrics(
        total_rows=len(candidate_ids),
        train_rows=row_counts["train"],
        test_rows=row_counts["test"],
        total_groups=len(group_partitions),
        train_groups=train_groups,
        test_groups=test_groups,
        status_slices=status_counts,
    )
    return metrics, allocation


def _candidate_group(row: Mapping[str, str]) -> tuple[int, str, str, int | None, str]:
    mileage = int(row["mileage"]) if row["mileage"] else None
    return (int(row["year"]), row["make"], row["model"], mileage, row["vehicle_status"])


def _group_sha256(
    group: tuple[int, str, str, int | None, str],
) -> str:
    payload = json.dumps(
        group,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(SPLIT_SEED.encode("utf-8") + b"\x00" + payload).hexdigest()


def _validate_candidate_row(row: Mapping[str, str]) -> None:
    listing_id = row["source_listing_id"]
    if (
        row["source_id"] != _SOURCE_ID
        or row["market_country"] != "US"
        or row["currency"] != "USD"
        or row["mileage_unit"] != "miles"
        or row["price_kind"] != "asking"
        or row["sale_status"] != "active"
        or row["vehicle_status"] not in set(_STATUSES)
        or not _SAFE_ROW_ID_PATTERN.fullmatch(listing_id)
        or not row["make"].strip()
        or not row["model"].strip()
        or any(character in row["make"] + row["model"] for character in "\x00\r\n")
    ):
        raise KaggleUSSalesCarsSplitError("candidate row violates retail split semantics")
    year = _parse_nonnegative_integer(row["year"], label="year")
    if not 1886 <= year <= 2025:
        raise KaggleUSSalesCarsSplitError("candidate row has an invalid year")
    if row["mileage"]:
        _parse_nonnegative_integer(row["mileage"], label="mileage")
    if _parse_nonnegative_integer(row["price_cents"], label="price_cents") <= 0:
        raise KaggleUSSalesCarsSplitError("candidate row has an invalid target")


def _parse_nonnegative_integer(value: str, *, label: str) -> int:
    if not value or not value.isascii() or not value.isdigit():
        raise KaggleUSSalesCarsSplitError(f"candidate row has invalid {label}")
    return int(value)


def _read_csv_rows(
    stream: TextIO,
    *,
    expected_header: Sequence[str],
    label: str,
) -> Iterator[dict[str, str]]:
    reader = csv.reader(stream, strict=True)
    try:
        header = next(reader)
    except StopIteration as error:
        raise KaggleUSSalesCarsSplitError(f"{label} must not be empty") from error
    if tuple(header) != tuple(expected_header):
        raise KaggleUSSalesCarsSplitError(f"{label} header is invalid")
    for values in reader:
        if len(values) != len(expected_header):
            raise KaggleUSSalesCarsSplitError(f"{label} row width is invalid")
        yield dict(zip(expected_header, values, strict=True))


def _validate_allocation_shape(value: object) -> None:
    allocation = _require_object(value, label="split allocation")
    _require_exact_keys(
        allocation,
        {
            "stratification_field",
            "target_unit",
            "zero_group_overlap_by_construction",
            "status_thresholds",
        },
        label="split allocation",
    )
    if (
        allocation["stratification_field"] != "vehicle_status"
        or allocation["target_unit"] != "rows"
        or allocation["zero_group_overlap_by_construction"] is not True
    ):
        raise KaggleUSSalesCarsSplitError("split allocation policy is invalid")
    thresholds = _require_object(allocation["status_thresholds"], label="status thresholds")
    _require_exact_keys(thresholds, set(_STATUSES), label="status thresholds")
    threshold_keys = {
        "total_rows",
        "train_rows",
        "test_rows",
        "total_groups",
        "train_groups",
        "test_groups",
        "cutoff_sha256",
        "target_test_rows_floor",
        "target_test_rows_remainder_numerator",
    }
    for status in _STATUSES:
        status_threshold = _require_object(
            thresholds[status],
            label=f"{status} status threshold",
        )
        _require_exact_keys(
            status_threshold,
            threshold_keys,
            label=f"{status} status threshold",
        )
        validated_counts: dict[str, int] = {}
        for key in threshold_keys - {"cutoff_sha256"}:
            raw = status_threshold[key]
            if type(raw) is not int or raw < 0:
                raise KaggleUSSalesCarsSplitError(
                    "status threshold counts must be nonnegative integers"
                )
            validated_counts[key] = raw
        if (
            validated_counts["total_rows"]
            != validated_counts["train_rows"] + validated_counts["test_rows"]
            or validated_counts["total_groups"]
            != validated_counts["train_groups"] + validated_counts["test_groups"]
        ):
            raise KaggleUSSalesCarsSplitError("status threshold counts do not account")
        remainder = validated_counts["target_test_rows_remainder_numerator"]
        if not 0 <= remainder < _TEST_DENOMINATOR:
            raise KaggleUSSalesCarsSplitError("status threshold target remainder is invalid")
        cutoff = status_threshold["cutoff_sha256"]
        test_groups = validated_counts["test_groups"]
        if test_groups == 0:
            if cutoff is not None:
                raise KaggleUSSalesCarsSplitError("empty status threshold must use null cutoff")
        else:
            _require_sha256(cutoff, label="status cutoff_sha256")


def _validate_counts_shape(value: object) -> None:
    counts = _require_object(value, label="split counts")
    _require_exact_keys(
        counts,
        {"rows", "groups", "status_slices", "realized_test_fraction"},
        label="split counts",
    )
    rows = _validate_partition_counts(counts["rows"], label="row counts")
    groups = _validate_partition_counts(counts["groups"], label="group counts")
    if rows["total"] != rows["train"] + rows["test"]:
        raise KaggleUSSalesCarsSplitError("split row counts do not account")
    if groups["total"] != groups["train"] + groups["test"]:
        raise KaggleUSSalesCarsSplitError("split group counts do not account")
    status_slices = _require_object(counts["status_slices"], label="status_slices")
    _require_exact_keys(status_slices, set(_STATUSES), label="status_slices")
    status_total = 0
    status_train = 0
    status_test = 0
    for status in _STATUSES:
        status_counts = _validate_partition_counts(
            status_slices[status],
            label=f"{status} status counts",
        )
        if status_counts["total"] != status_counts["train"] + status_counts["test"]:
            raise KaggleUSSalesCarsSplitError("split status counts do not account")
        status_total += status_counts["total"]
        status_train += status_counts["train"]
        status_test += status_counts["test"]
    if (status_total, status_train, status_test) != (
        rows["total"],
        rows["train"],
        rows["test"],
    ):
        raise KaggleUSSalesCarsSplitError("split status slices do not account for all rows")
    fraction = counts["realized_test_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise KaggleUSSalesCarsSplitError("realized test fraction is invalid")
    realized_fraction = float(fraction)
    if not 0 < realized_fraction < 1:
        raise KaggleUSSalesCarsSplitError("realized test fraction is invalid")
    if not hmac.compare_digest(
        f"{realized_fraction:.17g}",
        f"{rows['test'] / rows['total']:.17g}",
    ):
        raise KaggleUSSalesCarsSplitError("realized test fraction does not match row counts")


def _validate_partition_counts(value: object, *, label: str) -> dict[str, int]:
    counts = _require_object(value, label=label)
    _require_exact_keys(counts, {"total", "train", "test"}, label=label)
    result: dict[str, int] = {}
    for key in ("total", "train", "test"):
        raw = counts[key]
        if type(raw) is not int or raw < 0:
            raise KaggleUSSalesCarsSplitError(f"{label} must contain nonnegative integers")
        result[key] = raw
    return result


def _prepare_output_directory(output_dir: Path) -> Path:
    if not isinstance(output_dir, Path) or output_dir.is_symlink():
        raise KaggleUSSalesCarsSplitError("split output directory cannot be a symlink")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved = output_dir.resolve(strict=True)
    except OSError as error:
        raise KaggleUSSalesCarsSplitError("split output directory is inaccessible") from error
    if not resolved.is_dir():
        raise KaggleUSSalesCarsSplitError("split output path must be a directory")
    return resolved


def _require_regular_file(path: Path, *, label: str) -> Path:
    if not isinstance(path, Path) or path.is_symlink():
        raise KaggleUSSalesCarsSplitError(f"{label} must be a non-symlink local file")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise KaggleUSSalesCarsSplitError(f"{label} is missing or inaccessible") from error
    if not resolved.is_file():
        raise KaggleUSSalesCarsSplitError(f"{label} is not a regular file")
    return resolved


def _hash_regular_file(path: Path) -> tuple[str, int]:
    resolved = _require_regular_file(path, label="artifact")
    try:
        path_before = resolved.stat()
        digest = hashlib.sha256()
        bytes_read = 0
        with resolved.open("rb") as stream:
            descriptor_before = os.fstat(stream.fileno())
            if not stat.S_ISREG(descriptor_before.st_mode):
                raise KaggleUSSalesCarsSplitError("artifact is not a regular file")
            if _file_signature(descriptor_before) != _file_signature(path_before):
                raise KaggleUSSalesCarsSplitError("artifact changed during verification")
            while chunk := stream.read(_HASH_CHUNK_BYTES):
                bytes_read += len(chunk)
                digest.update(chunk)
            descriptor_after = os.fstat(stream.fileno())
        path_after = resolved.stat()
    except KaggleUSSalesCarsSplitError:
        raise
    except OSError as error:
        raise KaggleUSSalesCarsSplitError("artifact could not be hashed") from error
    if (
        _file_signature(path_before) != _file_signature(path_after)
        or _file_signature(descriptor_before) != _file_signature(descriptor_after)
        or bytes_read != path_after.st_size
    ):
        raise KaggleUSSalesCarsSplitError("artifact changed during verification")
    if bytes_read <= 0:
        raise KaggleUSSalesCarsSplitError("artifact must not be empty")
    return digest.hexdigest(), bytes_read


def _require_file_hash(path: Path, expected: str, *, label: str) -> None:
    actual, _ = _hash_regular_file(path)
    _require_hash_match(expected, actual, label=label)


def _file_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_bounded_bytes(path: Path, *, max_bytes: int, label: str) -> bytes:
    try:
        size = path.stat().st_size
        if size <= 0 or size > max_bytes:
            raise KaggleUSSalesCarsSplitError(f"{label} is empty or exceeds its byte limit")
        payload = path.read_bytes()
    except KaggleUSSalesCarsSplitError:
        raise
    except OSError as error:
        raise KaggleUSSalesCarsSplitError(f"{label} could not be read") from error
    if len(payload) != size:
        raise KaggleUSSalesCarsSplitError(f"{label} changed during read")
    return payload


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON value is forbidden: {value}")

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
        raise KaggleUSSalesCarsSplitError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise KaggleUSSalesCarsSplitError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _require_object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise KaggleUSSalesCarsSplitError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _require_exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise KaggleUSSalesCarsSplitError(f"{label} fields are invalid")


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise KaggleUSSalesCarsSplitError(f"{label} must be a positive integer")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise KaggleUSSalesCarsSplitError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_hash_match(expected: object, actual: str, *, label: str) -> None:
    expected_hash = _require_sha256(expected, label=f"{label} hash")
    if not hmac.compare_digest(expected_hash, actual):
        raise KaggleUSSalesCarsSplitError(f"{label} hash does not match")


def _safe_filename(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise KaggleUSSalesCarsSplitError(f"{label} must be a safe filename")
    return value


def _write_fsynced(path: Path, payload: bytes) -> None:
    try:
        with path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise KaggleUSSalesCarsSplitError("could not persist staged split artifact") from error


def _json_payload(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


__all__ = [
    "KaggleUSSalesCarsSplitArtifactSet",
    "KaggleUSSalesCarsSplitError",
    "KaggleUSSalesCarsSplitMetrics",
    "KaggleUSSalesCarsSplitTrainingRows",
    "SPLIT_ALGORITHM_VERSION",
    "SPLIT_ASSIGNMENT_COLUMNS",
    "SPLIT_GROUP_FIELDS",
    "SPLIT_SCHEMA_VERSION",
    "SPLIT_SEED",
    "VerifiedKaggleUSSalesCarsSplit",
    "build_kaggle_us_sales_cars_group_split",
    "prepare_kaggle_us_sales_cars_split_training_rows",
    "verify_kaggle_us_sales_cars_group_split",
]
