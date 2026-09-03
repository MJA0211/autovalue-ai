"""Fail-closed adapter for the reviewed Kaggle US Sales Cars v2 artifact.

This source is a historical 2023 U.S. *asking-price* snapshot.  It is kept
separate from AutoValue AI's completed wholesale-auction source because the
two labels answer different questions.  AutoValue does not scrape the
upstream marketplace: this adapter accepts only the exact, version-pinned
Kaggle CSV authorized by the project-owned source review.

The source has neither row identifiers nor observation timestamps.  Stable
opaque identifiers are derived from the reviewed source digest and CSV row
number.  A documented 2023 period-end sentinel is used for ``observed_at``.
Dealer content participates transiently in exact-row deduplication only; it is
never emitted, used in identifiers, or made available to training.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, TextIO, cast
from urllib.parse import parse_qs, urlencode, urlsplit

from autovalue_ml.acquisition.contracts import PriceKind, VehicleListingSnapshot

REVIEW_SCHEMA_VERSION: Final = 1
KAGGLE_US_SALES_CARS_HEADER: Final[tuple[str, ...]] = (
    "Brand",
    "Model",
    "Year",
    "Status",
    "Mileage",
    "Dealer",
    "Price",
)

_SOURCE_ID: Final = "kaggle_us_sales_cars_v2"
_DATASET_URL: Final = "https://www.kaggle.com/datasets/juanmerinobermejo/us-sales-cars-dataset"
_UPSTREAM_REPOSITORY: Final = "https://github.com/juanmerino89/cars-data-cleaning"
_EXPECTED_CSV_PATH: Final = PurePosixPath("data/raw/kaggle_us_sales_cars_v2/cars.csv")
_PARSER_VERSION: Final = "kaggle-us-sales-cars-v2.0.0"
_NORMALIZATION_VERSION: Final = "vehicle-listing-v1+kaggle-us-asking-v1"
_OBSERVED_AT: Final = datetime(2023, 12, 31, 23, 59, 59, tzinfo=UTC)
_OBSERVED_AT_RULE: Final = (
    "2023-12-31T23:59:59+00:00 period-end sentinel; row-level observation timestamps "
    "are unavailable"
)
_MAX_REVIEW_BYTES: Final = 256_000
_HASH_CHUNK_BYTES: Final = 1024 * 1024

_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_INTEGER_PATTERN = re.compile(r"^[0-9]+(?:\.0+)?$")
_DECIMAL_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
_SAFE_ROW_ID_PATTERN = re.compile(r"^row-[a-f0-9]{24}$")

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
_TRAINING_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "year",
    "make",
    "model",
    "mileage",
    "vehicle_status",
)
_REQUIRED_PROCESSING_GATES: Final[tuple[str, ...]] = (
    "Verify the exact CSV byte size and SHA-256 before and after streaming parse.",
    "Require the exact seven-column schema and UTF-16 BOM encoding.",
    "Retain New, Used, and Certified as explicit vehicle_status categories; quarantine unknown "
    "status values.",
    "Require non-empty brand and model, a valid vehicle year, optional nonnegative integral "
    "mileage, and a positive finite Price.",
    "Remove Dealer from every normalized record, feature table, rejection message, identifier, "
    "and published artifact.",
    "Deduplicate exact source rows without exposing source values in identifiers.",
    "Represent Price as USD asking-price cents and label every result as a historical U.S. "
    "advertised asking-price estimate.",
    "Keep this asking-price target separate from the completed wholesale-auction target unless "
    "an explicitly designed domain feature or multi-task experiment is reviewed.",
    "Do not scrape Cars.com from AutoValue AI; this approval covers only the already-created, "
    "version-pinned Kaggle artifact.",
    "Fit imputers, encoders, category grouping, outlier rules, and all learned transforms on "
    "training folds only.",
    "Do not commit or redistribute raw rows, processed rows, or a downloadable model without a "
    "new permission decision.",
)
_TOP_LEVEL_KEYS: Final = {
    "review_schema_version",
    "review_id",
    "reviewed_on",
    "decision",
    "project_role",
    "source",
    "retrieval",
    "permissions",
    "market_scope",
    "target",
    "quality_profile",
    "required_processing_gates",
    "publication_policy",
    "training_status",
    "notes",
}


class KaggleUSSalesCarsError(RuntimeError):
    """The review, raw source, or derived artifact failed closed."""


class _RowRejected(ValueError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True)
class KaggleUSSalesCarsReview:
    """Validated permission decision and immutable raw artifact pin."""

    review_id: str
    reviewed_on: date
    review_sha256: str
    review_path: Path
    dataset_url: str
    source_version: str
    expected_csv_path: PurePosixPath
    expected_file_name: str
    expected_size_bytes: int
    expected_sha256: str
    expected_row_count: int
    expected_status_counts: Mapping[str, int]
    expected_target_valid_rows: int
    expected_duplicate_rows: int
    expected_rows_after_deduplication: int
    expected_invalid_price_rows: int
    expected_target_valid_missing_mileage_rows: int
    expected_year_min: int
    expected_year_max: int
    expected_price_min_cents: int
    expected_price_max_cents: int
    approved_for_acquisition: bool
    acquisition_evidence: str
    approved_for_ml_training: bool
    ml_training_evidence: str


@dataclass(frozen=True, slots=True)
class KaggleUSSalesCarsMetrics:
    """Aggregate ingestion metrics that never contain source row values."""

    rows_seen: int
    rows_accepted: int
    source_status_counts: Mapping[str, int]
    accepted_status_counts: Mapping[str, int]
    unknown_status_rows: int
    quarantined_rows: int
    exact_duplicate_rows: int
    target_valid_missing_mileage_rows: int
    year_min: int
    year_max: int
    price_min_cents: int
    price_max_cents: int
    quarantine_reason_counts: Mapping[str, int]

    @property
    def core_valid_rows_before_deduplication(self) -> int:
        return self.rows_accepted + self.exact_duplicate_rows

    def to_dict(self) -> dict[str, object]:
        return {
            "rows_seen": self.rows_seen,
            "rows_accepted": self.rows_accepted,
            "source_status_counts": dict(self.source_status_counts),
            "accepted_status_counts": dict(self.accepted_status_counts),
            "unknown_status_rows": self.unknown_status_rows,
            "quarantined_rows": self.quarantined_rows,
            "exact_duplicate_rows": self.exact_duplicate_rows,
            "target_valid_missing_mileage_rows": self.target_valid_missing_mileage_rows,
            "core_valid_rows_before_deduplication": (self.core_valid_rows_before_deduplication),
            "year_min": self.year_min,
            "year_max": self.year_max,
            "price_min_cents": self.price_min_cents,
            "price_max_cents": self.price_max_cents,
            "quarantine_reason_counts": dict(self.quarantine_reason_counts),
        }


@dataclass(frozen=True, slots=True)
class KaggleUSSalesCarsArtifactSet:
    candidate_path: Path
    quarantine_path: Path
    manifest_path: Path
    readiness_path: Path
    metrics: KaggleUSSalesCarsMetrics


@dataclass(frozen=True, slots=True)
class KaggleUSSalesCarsTrainingRows:
    """Verified, lazily streamed River-style ``(features, target)`` rows."""

    candidate_path: Path
    candidate_sha256: str

    def __iter__(self) -> Iterator[tuple[dict[str, str | int], float]]:
        actual_sha256, _ = _hash_regular_file(self.candidate_path)
        if actual_sha256 != self.candidate_sha256:
            raise KaggleUSSalesCarsError("candidate changed after training approval")
        try:
            with self.candidate_path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream, strict=True)
                if tuple(reader.fieldnames or ()) != _CANDIDATE_HEADER:
                    raise KaggleUSSalesCarsError("candidate header is invalid")
                for row in reader:
                    _validate_training_candidate_row(row)
                    features: dict[str, str | int] = {
                        "year": int(row["year"]),
                        "make": row["make"],
                        "model": row["model"],
                        "vehicle_status": row["vehicle_status"],
                    }
                    if row["mileage"]:
                        features["mileage"] = int(row["mileage"])
                    yield features, int(row["price_cents"]) / 100
        except (OSError, UnicodeError, csv.Error) as error:
            raise KaggleUSSalesCarsError("candidate cannot be streamed safely") from error


def load_kaggle_us_sales_cars_review(
    review_path: Path,
    *,
    today: date | None = None,
) -> KaggleUSSalesCarsReview:
    """Strictly validate the committed source-specific authorization review."""

    resolved = _require_regular_file(review_path, label="source review")
    payload = _read_bounded_bytes(resolved, max_bytes=_MAX_REVIEW_BYTES, label="source review")
    review_sha256 = hashlib.sha256(payload).hexdigest()
    root = _strict_json_object(payload, label="source review")
    _require_exact_keys(root, _TOP_LEVEL_KEYS, label="source review")

    if root["review_schema_version"] != REVIEW_SCHEMA_VERSION:
        raise KaggleUSSalesCarsError("unsupported source review schema")
    if root["decision"] != "approved_with_conditions":
        raise KaggleUSSalesCarsError("source review does not approve processing")
    if root["project_role"] != "historical_us_retail_asking_price_candidate":
        raise KaggleUSSalesCarsError("source review has the wrong modeling role")
    review_id = _require_text(root["review_id"], label="review_id")
    reviewed_on = _parse_iso_date(root["reviewed_on"], label="reviewed_on")
    if reviewed_on > (date.today() if today is None else today):
        raise KaggleUSSalesCarsError("source review date cannot be in the future")
    _require_text(root["training_status"], label="training_status")
    _require_text_list(root["notes"], label="notes")

    source = _require_object(root["source"], label="source")
    _require_exact_keys(
        source,
        {
            "platform",
            "uploader",
            "dataset_slug",
            "dataset_url",
            "version",
            "version_release_date",
            "license_label",
            "license_label_source",
            "upstream_repository",
            "upstream_origin",
            "underlying_ownership_independently_verified",
            "permission_evidence",
        },
        label="source",
    )
    if (
        source["platform"] != "Kaggle"
        or source["uploader"] != "juanmerinobermejo"
        or source["dataset_slug"] != "us-sales-cars-dataset"
        or source["dataset_url"] != _DATASET_URL
        or source["version"] != "v2"
        or source["version_release_date"] != "2024-03-31"
        or source["license_label"] != "Apache-2.0"
        or source["upstream_repository"] != _UPSTREAM_REPOSITORY
        or source["underlying_ownership_independently_verified"] is not False
    ):
        raise KaggleUSSalesCarsError("source identity differs from the reviewed artifact")
    _require_text(source["license_label_source"], label="license_label_source")
    _require_text(source["upstream_origin"], label="upstream_origin")
    _require_public_url(source["dataset_url"], label="dataset_url")
    _require_public_url(source["upstream_repository"], label="upstream_repository")
    evidence = _require_object(source["permission_evidence"], label="permission_evidence")
    _require_exact_keys(
        evidence,
        {
            "kind",
            "summary",
            "scraping_permission",
            "ml_training_permission",
            "external_correspondence_committed",
        },
        label="permission_evidence",
    )
    evidence_kind = _require_text(evidence["kind"], label="permission evidence kind")
    evidence_summary = _require_text(evidence["summary"], label="permission evidence summary")
    ml_permission_evidence = _require_text(
        evidence["ml_training_permission"], label="ML training permission evidence"
    )
    ml_evidence_approved = (
        ml_permission_evidence
        == "approved_by_project_owner_attestation_for_this_noncommercial_portfolio_project"
    )
    if (
        evidence["scraping_permission"]
        != "approved_by_project_owner_attestation_for_the_historical_source_collection_only"
        or ml_permission_evidence
        not in {
            "approved_by_project_owner_attestation_for_this_noncommercial_portfolio_project",
            "pending_no_ml_training_permission",
        }
        or type(evidence["external_correspondence_committed"]) is not bool
    ):
        raise KaggleUSSalesCarsError("permission evidence does not match the scoped approval")

    retrieval = _require_object(root["retrieval"], label="retrieval")
    _require_exact_keys(
        retrieval,
        {"retrieved_on", "reviewed_on", "required_method", "csv_path", "csv"},
        label="retrieval",
    )
    if _parse_iso_date(retrieval["reviewed_on"], label="retrieval reviewed_on") != reviewed_on:
        raise KaggleUSSalesCarsError("retrieval and review dates differ")
    retrieved_on = _parse_iso_date(retrieval["retrieved_on"], label="retrieved_on")
    if retrieved_on > reviewed_on:
        raise KaggleUSSalesCarsError("retrieval date cannot follow its review date")
    if retrieval["required_method"] != "official KaggleHub v1.0.2 download of version 2":
        raise KaggleUSSalesCarsError("only the reviewed official download method is approved")
    csv_path = _require_safe_relative_path(retrieval["csv_path"], label="csv_path")
    if csv_path != _EXPECTED_CSV_PATH:
        raise KaggleUSSalesCarsError("reviewed CSV path is invalid")
    csv_pin = _require_object(retrieval["csv"], label="csv")
    _require_exact_keys(
        csv_pin,
        {"file_name", "size_bytes", "sha256", "row_count", "encoding", "columns"},
        label="csv",
    )
    if csv_pin["file_name"] != "cars.csv" or csv_pin["encoding"] != "UTF-16 with BOM":
        raise KaggleUSSalesCarsError("reviewed CSV filename or encoding is invalid")
    columns = _require_text_list(csv_pin["columns"], label="CSV columns")
    if columns != KAGGLE_US_SALES_CARS_HEADER:
        raise KaggleUSSalesCarsError("reviewed CSV columns are invalid")
    expected_size = _require_positive_int(csv_pin["size_bytes"], label="CSV size_bytes")
    expected_sha256 = _require_sha256(csv_pin["sha256"], label="CSV sha256")
    expected_rows = _require_positive_int(csv_pin["row_count"], label="CSV row_count")

    permissions = _validate_permissions(root["permissions"])
    approved_for_acquisition = all(
        permissions[key] == "approved"
        for key in ("official_download", "local_storage", "private_transformation")
    )
    if not approved_for_acquisition:
        raise KaggleUSSalesCarsError("review does not approve private acquisition")
    approved_for_ml = permissions["ml_training_and_evaluation"] == "approved"
    if approved_for_ml != ml_evidence_approved:
        raise KaggleUSSalesCarsError("ML permission decision and evidence are inconsistent")

    _validate_market_scope(root["market_scope"])
    _validate_target(root["target"])
    quality = _validate_quality_profile(root["quality_profile"], expected_rows=expected_rows)

    gates = _require_text_list(root["required_processing_gates"], label="required_processing_gates")
    if gates != _REQUIRED_PROCESSING_GATES:
        raise KaggleUSSalesCarsError("required processing gates do not match adapter v2")
    _validate_publication_policy(root["publication_policy"])

    return KaggleUSSalesCarsReview(
        review_id=review_id,
        reviewed_on=reviewed_on,
        review_sha256=review_sha256,
        review_path=resolved,
        dataset_url=_DATASET_URL,
        source_version="v2",
        expected_csv_path=csv_path,
        expected_file_name="cars.csv",
        expected_size_bytes=expected_size,
        expected_sha256=expected_sha256,
        expected_row_count=expected_rows,
        expected_status_counts=MappingProxyType(
            {
                "New": quality["status_new"],
                "Used": quality["status_used"],
                "Certified": quality["status_certified"],
            }
        ),
        expected_target_valid_rows=quality["target_valid_rows"],
        expected_duplicate_rows=quality["duplicate_rows"],
        expected_rows_after_deduplication=quality["rows_after_deduplication"],
        expected_invalid_price_rows=quality["invalid_price_rows"],
        expected_target_valid_missing_mileage_rows=quality["missing_mileage_rows"],
        expected_year_min=quality["year_min"],
        expected_year_max=quality["year_max"],
        expected_price_min_cents=quality["price_min_cents"],
        expected_price_max_cents=quality["price_max_cents"],
        approved_for_acquisition=True,
        acquisition_evidence=f"{evidence_kind}: {evidence_summary}",
        approved_for_ml_training=approved_for_ml,
        ml_training_evidence=(
            "project review permission and owner-attested scoped ML authorization"
            if approved_for_ml
            else ""
        ),
    )


def require_kaggle_us_sales_cars_ml_training_approval(
    review: KaggleUSSalesCarsReview,
) -> KaggleUSSalesCarsReview:
    """Enforce ML reuse independently from acquisition permission."""

    if not review.approved_for_ml_training or not review.ml_training_evidence:
        raise KaggleUSSalesCarsError("source review does not approve ML training")
    return review


def process_kaggle_us_sales_cars_csv(
    source_path: Path,
    review_path: Path,
    output_path: Path,
    *,
    today: date | None = None,
) -> KaggleUSSalesCarsArtifactSet:
    """Stream the pinned UTF-16 source into a private asking-price candidate."""

    if not isinstance(output_path, Path) or output_path.suffix.lower() != ".csv":
        raise KaggleUSSalesCarsError("candidate output must use a pathlib .csv path")
    review = load_kaggle_us_sales_cars_review(review_path, today=today)
    source = _require_regular_file(source_path, label="raw CSV")
    _require_reviewed_source_path(source, review.expected_csv_path)
    _verify_raw_artifact(source, review)
    _require_utf16_bom(source)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_parent = output_path.parent.resolve(strict=True)
    except OSError as error:
        raise KaggleUSSalesCarsError("candidate output directory is inaccessible") from error
    candidate_path = output_parent / output_path.name
    quarantine_path = candidate_path.with_suffix(".quarantine.jsonl")
    manifest_path = candidate_path.with_suffix(".manifest.json")
    readiness_path = candidate_path.with_suffix(".ready.json")
    output_targets = (candidate_path, quarantine_path, manifest_path, readiness_path)
    if source in {target.resolve(strict=False) for target in output_targets}:
        raise KaggleUSSalesCarsError("derived artifacts cannot overwrite the raw CSV")
    if any(target.is_symlink() for target in output_targets):
        raise KaggleUSSalesCarsError("derived artifact targets must not be symbolic links")
    try:
        readiness_path.unlink(missing_ok=True)
        staging_path = Path(tempfile.mkdtemp(prefix=f".{candidate_path.stem}.", dir=output_parent))
    except OSError as error:
        raise KaggleUSSalesCarsError("could not initialize atomic output staging") from error

    try:
        staged_candidate = staging_path / candidate_path.name
        staged_quarantine = staging_path / quarantine_path.name
        metrics = _stream_transform(
            source,
            review,
            staged_candidate,
            staged_quarantine,
            staging_path,
        )
        _validate_metrics_against_review(metrics, review)

        # Close the parse-to-publish TOCTOU window with a second full verification.
        _verify_raw_artifact(source, review)
        _require_utf16_bom(source)
        candidate_sha256, candidate_size = _hash_regular_file(staged_candidate)
        quarantine_sha256, quarantine_size = _hash_regular_file(staged_quarantine, allow_empty=True)
        manifest = _build_manifest(
            review=review,
            metrics=metrics,
            candidate_path=candidate_path,
            candidate_sha256=candidate_sha256,
            candidate_size=candidate_size,
            quarantine_path=quarantine_path,
            quarantine_sha256=quarantine_sha256,
            quarantine_size=quarantine_size,
            readiness_path=readiness_path,
        )
        staged_manifest = staging_path / manifest_path.name
        manifest_payload = _json_payload(manifest)
        _write_fsynced(staged_manifest, manifest_payload)
        manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        artifact_set_id = hashlib.sha256(
            "|".join((manifest_sha256, candidate_sha256, quarantine_sha256)).encode()
        ).hexdigest()
        readiness = {
            "schema_version": 1,
            "artifact_set_id": artifact_set_id,
            "manifest_file": manifest_path.name,
            "manifest_sha256": manifest_sha256,
            "candidate_file": candidate_path.name,
            "candidate_sha256": candidate_sha256,
            "quarantine_file": quarantine_path.name,
            "quarantine_sha256": quarantine_sha256,
        }
        staged_readiness = staging_path / readiness_path.name
        _write_fsynced(staged_readiness, _json_payload(readiness))

        os.replace(staged_candidate, candidate_path)
        os.replace(staged_quarantine, quarantine_path)
        os.replace(staged_manifest, manifest_path)
        # This final marker is the sole signal that the set is complete.
        os.replace(staged_readiness, readiness_path)
        return KaggleUSSalesCarsArtifactSet(
            candidate_path=candidate_path,
            quarantine_path=quarantine_path,
            manifest_path=manifest_path,
            readiness_path=readiness_path,
            metrics=metrics,
        )
    except KaggleUSSalesCarsError:
        raise
    except OSError as error:
        raise KaggleUSSalesCarsError("could not publish the derived artifact set") from error
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)


def verify_kaggle_us_sales_cars_artifact_set(
    manifest_path: Path,
    review_path: Path,
    *,
    today: date | None = None,
) -> dict[str, object]:
    """Verify readiness, every derived hash, and the committed review lineage."""

    review = load_kaggle_us_sales_cars_review(review_path, today=today)
    manifest_file = _require_regular_file(manifest_path, label="candidate manifest")
    manifest_payload = _read_bounded_bytes(
        manifest_file, max_bytes=5_000_000, label="candidate manifest"
    )
    manifest = _strict_json_object(manifest_payload, label="candidate manifest")
    _validate_manifest(manifest, review=review)

    readiness_name = _safe_artifact_name(manifest["readiness_file"], label="readiness_file")
    readiness_file = _require_regular_file(
        manifest_file.parent / readiness_name, label="readiness marker"
    )
    readiness = _strict_json_object(
        _read_bounded_bytes(readiness_file, max_bytes=1_000_000, label="readiness marker"),
        label="readiness marker",
    )
    ready_keys = {
        "schema_version",
        "artifact_set_id",
        "manifest_file",
        "manifest_sha256",
        "candidate_file",
        "candidate_sha256",
        "quarantine_file",
        "quarantine_sha256",
    }
    _require_exact_keys(readiness, ready_keys, label="readiness marker")
    if readiness["schema_version"] != 1 or readiness["manifest_file"] != manifest_file.name:
        raise KaggleUSSalesCarsError("readiness marker does not identify this manifest")
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    _require_hash_match(readiness["manifest_sha256"], manifest_sha256, label="manifest")

    verified_hashes: list[str] = []
    for name_key, hash_key, allow_empty in (
        ("candidate_file", "candidate_sha256", False),
        ("quarantine_file", "quarantine_sha256", True),
    ):
        file_name = _safe_artifact_name(readiness[name_key], label=name_key)
        if manifest[name_key] != file_name or manifest[hash_key] != readiness[hash_key]:
            raise KaggleUSSalesCarsError("manifest and readiness lineage differ")
        artifact = _require_regular_file(manifest_file.parent / file_name, label=name_key)
        actual_sha256, _ = _hash_regular_file(artifact, allow_empty=allow_empty)
        _require_hash_match(readiness[hash_key], actual_sha256, label=name_key)
        verified_hashes.append(actual_sha256)
    expected_set_id = hashlib.sha256(
        "|".join((manifest_sha256, *verified_hashes)).encode()
    ).hexdigest()
    _require_hash_match(readiness["artifact_set_id"], expected_set_id, label="artifact set")
    return readiness


def prepare_kaggle_us_sales_cars_training_rows(
    candidate_path: Path,
    manifest_path: Path,
    review_path: Path,
    *,
    today: date | None = None,
) -> KaggleUSSalesCarsTrainingRows:
    """Return a stream only after both artifact and ML-authorization checks."""

    review = require_kaggle_us_sales_cars_ml_training_approval(
        load_kaggle_us_sales_cars_review(review_path, today=today)
    )
    readiness = verify_kaggle_us_sales_cars_artifact_set(manifest_path, review_path, today=today)
    candidate = _require_regular_file(candidate_path, label="candidate")
    manifest_file = _require_regular_file(manifest_path, label="candidate manifest")
    manifest_payload = _read_bounded_bytes(
        manifest_file,
        max_bytes=5_000_000,
        label="candidate manifest",
    )
    _require_hash_match(
        readiness["manifest_sha256"],
        hashlib.sha256(manifest_payload).hexdigest(),
        label="training manifest",
    )
    manifest = _strict_json_object(manifest_payload, label="candidate manifest")
    _validate_manifest(manifest, review=review)
    expected_candidate = (manifest_file.parent / candidate.name).resolve(strict=True)
    if candidate != expected_candidate or candidate.name != manifest["candidate_file"]:
        raise KaggleUSSalesCarsError("training candidate differs from the verified manifest")
    if manifest["review_sha256"] != review.review_sha256:
        raise KaggleUSSalesCarsError("training candidate has stale review lineage")
    candidate_sha256 = _require_sha256(manifest["candidate_sha256"], label="candidate_sha256")
    return KaggleUSSalesCarsTrainingRows(
        candidate_path=candidate,
        candidate_sha256=candidate_sha256,
    )


def _stream_transform(
    source: Path,
    review: KaggleUSSalesCarsReview,
    candidate_path: Path,
    quarantine_path: Path,
    staging_path: Path,
) -> KaggleUSSalesCarsMetrics:
    rows_seen = rows_accepted = unknown_status_rows = 0
    quarantined_rows = duplicate_rows = 0
    reason_counts: dict[str, int] = {}
    source_status_counts = {"New": 0, "Used": 0, "Certified": 0}
    accepted_status_counts = {"New": 0, "Used": 0, "Certified": 0}
    target_valid_missing_mileage_rows = 0
    year_min: int | None = None
    year_max: int | None = None
    price_min: int | None = None
    price_max: int | None = None
    index_path = staging_path / "exact-row-index.sqlite3"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(index_path)
        connection.execute("CREATE TABLE exact_rows (digest BLOB PRIMARY KEY) WITHOUT ROWID")
    except sqlite3.Error as error:
        if connection is not None:
            connection.close()
        raise KaggleUSSalesCarsError(
            "could not initialize the bounded deduplication index"
        ) from error
    assert connection is not None
    try:
        try:
            source_stream: TextIO = source.open("r", encoding="utf-16", newline="")
            candidate_stream: TextIO = candidate_path.open("w", encoding="utf-8", newline="")
            quarantine_stream: TextIO = quarantine_path.open("w", encoding="utf-8", newline="")
        except (OSError, UnicodeError) as error:
            raise KaggleUSSalesCarsError("could not open ingestion streams") from error
        try:
            with source_stream, candidate_stream, quarantine_stream:
                reader = csv.reader(source_stream, strict=True)
                try:
                    header = next(reader)
                except StopIteration as error:
                    raise KaggleUSSalesCarsError("raw CSV is missing its header") from error
                if tuple(header) != KAGGLE_US_SALES_CARS_HEADER:
                    raise KaggleUSSalesCarsError(
                        "raw CSV header does not match the reviewed schema"
                    )
                writer = csv.DictWriter(
                    candidate_stream,
                    fieldnames=list(_CANDIDATE_HEADER),
                    extrasaction="raise",
                    lineterminator="\n",
                )
                writer.writeheader()
                for row_number, values in enumerate(reader, start=2):
                    rows_seen += 1
                    safe_id = _safe_row_id(row_number, review.expected_sha256)
                    try:
                        status = _parse_status(values)
                        source_status_counts[_source_status_label(status)] += 1
                        listing = _normalize_row(
                            values,
                            row_number=row_number,
                            review=review,
                            normalized_status=status,
                        )
                    except _RowRejected as rejection:
                        if rejection.reason_code == "status_invalid":
                            unknown_status_rows += 1
                        quarantined_rows += 1
                        reason_counts[rejection.reason_code] = (
                            reason_counts.get(rejection.reason_code, 0) + 1
                        )
                        quarantine_stream.write(
                            json.dumps(
                                {
                                    "source_id": _SOURCE_ID,
                                    "source_listing_id": safe_id,
                                    "reason_code": rejection.reason_code,
                                    "record_sha256": _safe_rejection_digest(
                                        row_number, review.expected_sha256
                                    ),
                                    "parser_version": _PARSER_VERSION,
                                    "authorization_policy_id": review.review_id,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        continue

                    if listing.mileage is None:
                        target_valid_missing_mileage_rows += 1
                    year_min = listing.year if year_min is None else min(year_min, listing.year)
                    year_max = listing.year if year_max is None else max(year_max, listing.year)
                    price_min = (
                        listing.price_cents
                        if price_min is None
                        else min(price_min, listing.price_cents)
                    )
                    price_max = (
                        listing.price_cents
                        if price_max is None
                        else max(price_max, listing.price_cents)
                    )
                    exact_digest = hashlib.sha256(_canonical_row_bytes(values)).digest()
                    try:
                        connection.execute(
                            "INSERT INTO exact_rows (digest) VALUES (?)", (exact_digest,)
                        )
                    except sqlite3.IntegrityError:
                        duplicate_rows += 1
                        continue
                    writer.writerow(_snapshot_to_csv_row(listing))
                    rows_accepted += 1
                    accepted_status_counts[_source_status_label(status)] += 1
                    if rows_seen % 10_000 == 0:
                        connection.commit()
                candidate_stream.flush()
                os.fsync(candidate_stream.fileno())
                quarantine_stream.flush()
                os.fsync(quarantine_stream.fileno())
        except (UnicodeError, csv.Error) as error:
            raise KaggleUSSalesCarsError("raw CSV is not valid strict UTF-16 CSV") from error
        except sqlite3.Error as error:
            raise KaggleUSSalesCarsError("deduplication index failed during ingestion") from error
        connection.commit()
    finally:
        connection.close()
        for suffix in ("", "-wal", "-shm", "-journal"):
            (Path(f"{index_path}{suffix}")).unlink(missing_ok=True)

    if rows_seen != rows_accepted + quarantined_rows + duplicate_rows:
        raise KaggleUSSalesCarsError("ingestion accounting invariant failed")
    if any(value is None for value in (year_min, year_max, price_min, price_max)):
        raise KaggleUSSalesCarsError("candidate must contain at least one valid record")
    return KaggleUSSalesCarsMetrics(
        rows_seen=rows_seen,
        rows_accepted=rows_accepted,
        source_status_counts=MappingProxyType(dict(source_status_counts)),
        accepted_status_counts=MappingProxyType(dict(accepted_status_counts)),
        unknown_status_rows=unknown_status_rows,
        quarantined_rows=quarantined_rows,
        exact_duplicate_rows=duplicate_rows,
        target_valid_missing_mileage_rows=target_valid_missing_mileage_rows,
        year_min=cast(int, year_min),
        year_max=cast(int, year_max),
        price_min_cents=cast(int, price_min),
        price_max_cents=cast(int, price_max),
        quarantine_reason_counts=MappingProxyType(dict(sorted(reason_counts.items()))),
    )


def _parse_status(values: Sequence[str]) -> str:
    if len(values) != len(KAGGLE_US_SALES_CARS_HEADER):
        raise _RowRejected("row_width_invalid")
    status = values[3].strip()
    if status == "New":
        return "new"
    if status == "Used":
        return "used"
    if status == "Certified":
        return "certified"
    raise _RowRejected("status_invalid")


def _source_status_label(normalized_status: str) -> str:
    return {
        "new": "New",
        "used": "Used",
        "certified": "Certified",
    }[normalized_status]


def _normalize_row(
    values: Sequence[str],
    *,
    row_number: int,
    review: KaggleUSSalesCarsReview,
    normalized_status: str,
) -> VehicleListingSnapshot:
    if len(values) != len(KAGGLE_US_SALES_CARS_HEADER):
        raise _RowRejected("row_width_invalid")
    make = _required_source_text(values[0], reason_code="brand_missing", max_length=100)
    model = _required_source_text(values[1], reason_code="model_missing", max_length=200)
    year = _parse_integral(values[2], reason_code="year_invalid")
    if not 1886 <= year <= _OBSERVED_AT.year + 2:
        raise _RowRejected("year_invalid")
    mileage = _parse_optional_nonnegative_integral(values[4], reason_code="mileage_invalid")
    price_cents = _parse_price_cents(values[6])
    source_listing_id = _safe_row_id(row_number, review.expected_sha256)
    safe_content = {
        "brand": make,
        "model": model,
        "year": year,
        "status": normalized_status,
        "mileage": mileage,
        "price_cents": price_cents,
    }
    safe_content_sha256 = hashlib.sha256(
        json.dumps(safe_content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    query = urlencode({"select": "cars.csv", "record": source_listing_id})
    canonical_url = f"{review.dataset_url}?{query}"
    return VehicleListingSnapshot(
        source_id=_SOURCE_ID,
        source_listing_id=source_listing_id,
        canonical_url=canonical_url,
        observed_at=_OBSERVED_AT,
        market_country="US",
        year=year,
        make=make,
        model=model,
        trim=None,
        mileage=mileage,
        mileage_unit="miles",
        condition=None,
        engine=None,
        drivetrain=None,
        accident_status=None,
        accident_count=None,
        owner_count=None,
        vehicle_type=None,
        price_cents=price_cents,
        currency="USD",
        price_kind=PriceKind.ASKING,
        sale_status="active",
        raw_content_sha256=safe_content_sha256,
        parser_version=_PARSER_VERSION,
        normalization_version=_NORMALIZATION_VERSION,
        ingestion_run_id=(f"kusc-{review.expected_sha256[:16]}-{review.review_sha256[:8]}"),
        authorization_policy_id=review.review_id,
        vehicle_status=normalized_status,
    )


def _snapshot_to_csv_row(listing: VehicleListingSnapshot) -> dict[str, object]:
    row = listing.to_dict()
    return {column: "" if row[column] is None else row[column] for column in _CANDIDATE_HEADER}


def _required_source_text(value: str, *, reason_code: str, max_length: int) -> str:
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise _RowRejected(reason_code)
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > max_length:
        raise _RowRejected(reason_code)
    return normalized


def _parse_integral(value: str, *, reason_code: str) -> int:
    text = value.strip()
    if not _INTEGER_PATTERN.fullmatch(text):
        raise _RowRejected(reason_code)
    try:
        return int(Decimal(text))
    except (InvalidOperation, ValueError) as error:
        raise _RowRejected(reason_code) from error


def _parse_optional_nonnegative_integral(
    value: str,
    *,
    reason_code: str,
) -> int | None:
    text = value.strip()
    if not text:
        return None
    result = _parse_integral(text, reason_code=reason_code)
    if result < 0:
        raise _RowRejected(reason_code)
    return result


def _parse_price_cents(value: str) -> int:
    text = value.strip()
    if not _DECIMAL_PATTERN.fullmatch(text):
        raise _RowRejected("price_invalid")
    try:
        amount = Decimal(text)
        cents = amount * 100
    except InvalidOperation as error:
        raise _RowRejected("price_invalid") from error
    if not amount.is_finite() or amount <= 0 or cents != cents.to_integral_value():
        raise _RowRejected("price_invalid")
    return int(cents)


def _safe_row_id(row_number: int, source_sha256: str) -> str:
    digest = hashlib.sha256(f"{source_sha256}:{row_number}".encode()).hexdigest()
    return f"row-{digest[:24]}"


def _safe_rejection_digest(row_number: int, source_sha256: str) -> str:
    return hashlib.sha256(f"{source_sha256}:{row_number}:rejected".encode()).hexdigest()


def _canonical_row_bytes(values: Sequence[str]) -> bytes:
    """Canonical raw bytes used transiently for exact-row deduplication only."""

    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode()


def _validate_metrics_against_review(
    metrics: KaggleUSSalesCarsMetrics,
    review: KaggleUSSalesCarsReview,
) -> None:
    checks = {
        "raw row count": metrics.rows_seen == review.expected_row_count,
        "status counts": dict(metrics.source_status_counts) == dict(review.expected_status_counts),
        "unknown status count": metrics.unknown_status_rows
        == review.expected_row_count - sum(review.expected_status_counts.values()),
        "target-valid row count": metrics.core_valid_rows_before_deduplication
        == review.expected_target_valid_rows,
        "exact duplicate count": metrics.exact_duplicate_rows == review.expected_duplicate_rows,
        "post-deduplication row count": metrics.rows_accepted
        == review.expected_rows_after_deduplication,
        "invalid price count": metrics.quarantine_reason_counts.get("price_invalid", 0)
        == review.expected_invalid_price_rows,
        "target-valid missing-mileage count": metrics.target_valid_missing_mileage_rows
        == review.expected_target_valid_missing_mileage_rows,
        "minimum year": metrics.year_min == review.expected_year_min,
        "maximum year": metrics.year_max == review.expected_year_max,
        "minimum price": metrics.price_min_cents == review.expected_price_min_cents,
        "maximum price": metrics.price_max_cents == review.expected_price_max_cents,
    }
    failed = next((label for label, passed in checks.items() if not passed), None)
    if failed is not None:
        raise KaggleUSSalesCarsError(f"ingestion metrics differ from reviewed {failed}")


def _build_manifest(
    *,
    review: KaggleUSSalesCarsReview,
    metrics: KaggleUSSalesCarsMetrics,
    candidate_path: Path,
    candidate_sha256: str,
    candidate_size: int,
    quarantine_path: Path,
    quarantine_sha256: str,
    quarantine_size: int,
    readiness_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_id": _SOURCE_ID,
        "target_track": "historical_us_retail_asking_price",
        "review_id": review.review_id,
        "review_sha256": review.review_sha256,
        "reviewed_source_version": review.source_version,
        "raw_source_file": review.expected_file_name,
        "raw_source_sha256": review.expected_sha256,
        "raw_source_size_bytes": review.expected_size_bytes,
        "raw_source_row_count": review.expected_row_count,
        "market_country": "US",
        "geography_basis": "source-level attestation; row-level geography unavailable",
        "currency": "USD",
        "mileage_unit": "miles",
        "price_kind": PriceKind.ASKING.value,
        "sale_status": "active",
        "approved_claim": (
            "historical U.S. advertised asking-price estimate for 2023 source snapshots"
        ),
        "forbidden_claims": [
            "completed sale price",
            "wholesale auction value",
            "current live market value",
        ],
        "label_mixing_policy": (
            "must remain separate from completed wholesale-auction targets unless a reviewed "
            "domain-feature or multi-task experiment explicitly permits combination"
        ),
        "observed_at_rule": _OBSERVED_AT_RULE,
        "publication_status": "private_local_only",
        "approved_for_acquisition": review.approved_for_acquisition,
        "acquisition_evidence": review.acquisition_evidence,
        "approved_for_ml_training": review.approved_for_ml_training,
        "ml_training_evidence": review.ml_training_evidence,
        "direct_marketplace_scraping": "prohibited",
        "parser_version": _PARSER_VERSION,
        "normalization_version": _NORMALIZATION_VERSION,
        "status_mapping": {
            "New": "vehicle_status=new",
            "Used": "vehicle_status=used",
            "Certified": "vehicle_status=certified",
        },
        "candidate_file": candidate_path.name,
        "candidate_sha256": candidate_sha256,
        "candidate_size_bytes": candidate_size,
        "candidate_columns": list(_CANDIDATE_HEADER),
        "feature_allowlist": list(_TRAINING_FEATURE_COLUMNS),
        "forbidden_source_columns": ["Dealer"],
        "quarantine_file": quarantine_path.name,
        "quarantine_sha256": quarantine_sha256,
        "quarantine_size_bytes": quarantine_size,
        "readiness_file": readiness_path.name,
        "metrics": metrics.to_dict(),
    }


_MANIFEST_KEYS: Final = {
    "schema_version",
    "source_id",
    "target_track",
    "review_id",
    "review_sha256",
    "reviewed_source_version",
    "raw_source_file",
    "raw_source_sha256",
    "raw_source_size_bytes",
    "raw_source_row_count",
    "market_country",
    "geography_basis",
    "currency",
    "mileage_unit",
    "price_kind",
    "sale_status",
    "approved_claim",
    "forbidden_claims",
    "label_mixing_policy",
    "observed_at_rule",
    "publication_status",
    "approved_for_acquisition",
    "acquisition_evidence",
    "approved_for_ml_training",
    "ml_training_evidence",
    "direct_marketplace_scraping",
    "parser_version",
    "normalization_version",
    "status_mapping",
    "candidate_file",
    "candidate_sha256",
    "candidate_size_bytes",
    "candidate_columns",
    "feature_allowlist",
    "forbidden_source_columns",
    "quarantine_file",
    "quarantine_sha256",
    "quarantine_size_bytes",
    "readiness_file",
    "metrics",
}


def _validate_manifest(
    manifest: dict[str, object],
    *,
    review: KaggleUSSalesCarsReview,
) -> None:
    _require_exact_keys(manifest, _MANIFEST_KEYS, label="candidate manifest")
    exact_values: dict[str, object] = {
        "schema_version": 1,
        "source_id": _SOURCE_ID,
        "target_track": "historical_us_retail_asking_price",
        "review_id": review.review_id,
        "review_sha256": review.review_sha256,
        "reviewed_source_version": review.source_version,
        "raw_source_file": review.expected_file_name,
        "raw_source_sha256": review.expected_sha256,
        "raw_source_size_bytes": review.expected_size_bytes,
        "raw_source_row_count": review.expected_row_count,
        "market_country": "US",
        "currency": "USD",
        "mileage_unit": "miles",
        "price_kind": PriceKind.ASKING.value,
        "sale_status": "active",
        "observed_at_rule": _OBSERVED_AT_RULE,
        "publication_status": "private_local_only",
        "approved_for_acquisition": True,
        "approved_for_ml_training": review.approved_for_ml_training,
        "direct_marketplace_scraping": "prohibited",
        "parser_version": _PARSER_VERSION,
        "normalization_version": _NORMALIZATION_VERSION,
    }
    if any(manifest[key] != value for key, value in exact_values.items()):
        raise KaggleUSSalesCarsError("candidate manifest semantics or lineage are invalid")
    if manifest["candidate_columns"] != list(_CANDIDATE_HEADER):
        raise KaggleUSSalesCarsError("candidate manifest columns are invalid")
    if manifest["feature_allowlist"] != list(_TRAINING_FEATURE_COLUMNS):
        raise KaggleUSSalesCarsError("candidate feature allowlist is invalid")
    if manifest["forbidden_source_columns"] != ["Dealer"]:
        raise KaggleUSSalesCarsError("candidate manifest does not exclude Dealer")
    if manifest["forbidden_claims"] != [
        "completed sale price",
        "wholesale auction value",
        "current live market value",
    ]:
        raise KaggleUSSalesCarsError("candidate manifest target limitations are invalid")
    if manifest["status_mapping"] != {
        "New": "vehicle_status=new",
        "Used": "vehicle_status=used",
        "Certified": "vehicle_status=certified",
    }:
        raise KaggleUSSalesCarsError("candidate manifest status mapping is invalid")
    for key in ("acquisition_evidence", "approved_claim"):
        _require_text(manifest[key], label=key)
    ml_evidence = manifest["ml_training_evidence"]
    if not isinstance(ml_evidence, str) or (
        review.approved_for_ml_training and not ml_evidence.strip()
    ):
        raise KaggleUSSalesCarsError("candidate ML permission evidence is invalid")
    for key in ("candidate_sha256", "quarantine_sha256"):
        _require_sha256(manifest[key], label=key)
    for key in ("candidate_size_bytes", "raw_source_size_bytes"):
        _require_positive_int(manifest[key], label=key)
    _require_nonnegative_int(manifest["quarantine_size_bytes"], label="quarantine_size_bytes")
    for key in ("candidate_file", "quarantine_file", "readiness_file"):
        _safe_artifact_name(manifest[key], label=key)
    _require_object(manifest["metrics"], label="manifest metrics")


def _validate_training_candidate_row(row: Mapping[str, str]) -> None:
    if (
        row["source_id"] != _SOURCE_ID
        or row["market_country"] != "US"
        or row["mileage_unit"] != "miles"
        or row["currency"] != "USD"
        or row["price_kind"] != PriceKind.ASKING.value
        or row["sale_status"] != "active"
        or row["parser_version"] != _PARSER_VERSION
        or row["normalization_version"] != _NORMALIZATION_VERSION
        or row["observed_at"] != _OBSERVED_AT.isoformat()
        or row["vehicle_status"] not in {"new", "used", "certified"}
        or not _SAFE_ROW_ID_PATTERN.fullmatch(row["source_listing_id"])
        or not _SHA256_PATTERN.fullmatch(row["raw_content_sha256"])
        or not row["make"].strip()
        or not row["model"].strip()
    ):
        raise KaggleUSSalesCarsError("candidate row violates asking-price semantics")
    year = _parse_candidate_integer(row["year"], label="year")
    if not 1886 <= year <= _OBSERVED_AT.year + 2:
        raise KaggleUSSalesCarsError("candidate row has an invalid year")
    if row["mileage"]:
        _parse_candidate_integer(row["mileage"], label="mileage")
    if _parse_candidate_integer(row["price_cents"], label="price_cents") <= 0:
        raise KaggleUSSalesCarsError("candidate row has an invalid price")
    if any(
        row[column] for column in ("trim", "condition", "engine", "drivetrain", "accident_status")
    ):
        raise KaggleUSSalesCarsError("candidate row contains unsupported source features")
    if any(row[column] for column in ("accident_count", "owner_count", "vehicle_type")):
        raise KaggleUSSalesCarsError("candidate row contains unsupported source features")
    parsed_url = urlsplit(row["canonical_url"])
    expected_query = {
        "select": ["cars.csv"],
        "record": [row["source_listing_id"]],
    }
    try:
        actual_query = parse_qs(
            parsed_url.query,
            keep_blank_values=True,
            strict_parsing=True,
        )
    except ValueError as error:
        raise KaggleUSSalesCarsError("candidate row has an invalid canonical URL") from error
    if (
        parsed_url.scheme != "https"
        or parsed_url.netloc != "www.kaggle.com"
        or parsed_url.path != "/datasets/juanmerinobermejo/us-sales-cars-dataset"
        or parsed_url.fragment
        or actual_query != expected_query
    ):
        raise KaggleUSSalesCarsError("candidate row has an invalid canonical URL")


def _parse_candidate_integer(value: str, *, label: str) -> int:
    if not value.isascii() or not value.isdigit():
        raise KaggleUSSalesCarsError(f"candidate row has invalid {label}")
    return int(value)


def _validate_permissions(value: object) -> dict[str, str]:
    permissions = _require_object(value, label="permissions")
    expected_keys = {
        "official_download",
        "local_storage",
        "private_transformation",
        "ml_training_and_evaluation",
        "public_aggregate_metrics_and_charts",
        "hosted_inference",
        "autovalue_direct_scraping_of_cars_com",
        "raw_row_redistribution",
        "processed_row_redistribution",
        "downloadable_trained_model",
        "sublicensing",
        "commercial_use",
    }
    _require_exact_keys(permissions, expected_keys, label="permissions")
    result: dict[str, str] = {}
    for key, raw_decision in permissions.items():
        decision = _require_text(raw_decision, label=f"permission {key}")
        if decision not in {"approved", "pending", "prohibited", "rejected"}:
            raise KaggleUSSalesCarsError(f"permission {key} has an invalid decision")
        result[key] = decision
    if result["autovalue_direct_scraping_of_cars_com"] != "prohibited":
        raise KaggleUSSalesCarsError("AutoValue direct marketplace scraping must stay prohibited")
    for key in (
        "raw_row_redistribution",
        "processed_row_redistribution",
        "downloadable_trained_model",
        "sublicensing",
        "commercial_use",
    ):
        if result[key] != "pending":
            raise KaggleUSSalesCarsError(f"permission {key} must remain pending")
    return result


def _validate_market_scope(value: object) -> None:
    market = _require_object(value, label="market_scope")
    _require_exact_keys(
        market,
        {
            "raw_scope",
            "approved_scope",
            "row_level_geography_available",
            "geography_status",
            "currency",
            "currency_status",
            "mileage_unit",
            "mileage_unit_status",
            "observation_period",
            "observation_period_status",
        },
        label="market_scope",
    )
    expected: dict[str, object] = {
        "raw_scope": "United States according to the Kaggle description and upstream repository",
        "approved_scope": "United States",
        "row_level_geography_available": False,
        "geography_status": "source_level_attestation_only",
        "currency": "USD",
        "currency_status": "declared_by_kaggle_description",
        "mileage_unit": "miles",
        "mileage_unit_status": "inferred_from_us_market_context",
        "observation_period": "2023 source snapshots; no row-level observation timestamp",
        "observation_period_status": "inferred_from_upstream extraction filenames",
    }
    if market != expected:
        raise KaggleUSSalesCarsError("market scope differs from the reviewed U.S. assumptions")


def _validate_target(value: object) -> None:
    target = _require_object(value, label="target")
    _require_exact_keys(
        target,
        {"source_column", "meaning", "approved_claim", "prohibited_claims"},
        label="target",
    )
    if (
        target["source_column"] != "Price"
        or target["meaning"] != "advertised retail asking price"
        or target["approved_claim"]
        != (
            "historical U.S. advertised asking-price estimate for New, Used, and Certified "
            "listings represented by the 2023 source snapshots"
        )
        or _require_text_list(target["prohibited_claims"], label="prohibited_claims")
        != ("completed sale price", "wholesale auction value", "current live market value")
    ):
        raise KaggleUSSalesCarsError("target semantics differ from the asking-price review")


def _validate_quality_profile(
    value: object,
    *,
    expected_rows: int,
) -> dict[str, int]:
    quality = _require_object(value, label="quality_profile")
    _require_exact_keys(
        quality,
        {
            "raw_rows",
            "status_counts",
            "target_valid_rows_before_deduplication",
            "target_valid_exact_duplicate_rows",
            "target_valid_rows_after_exact_deduplication",
            "rows_missing_or_invalid_price",
            "target_valid_rows_missing_mileage",
            "year_min_for_target_valid_rows",
            "year_max_for_target_valid_rows",
            "price_min_usd_for_target_valid_rows",
            "price_max_usd_for_target_valid_rows",
            "notes",
        },
        label="quality_profile",
    )
    raw_rows = _require_positive_int(quality["raw_rows"], label="quality raw_rows")
    if raw_rows != expected_rows:
        raise KaggleUSSalesCarsError("reviewed raw row counts are inconsistent")
    status_counts = _require_object(quality["status_counts"], label="status_counts")
    _require_exact_keys(status_counts, {"New", "Used", "Certified"}, label="status_counts")
    status_new = _require_nonnegative_int(status_counts["New"], label="New status count")
    status_used = _require_nonnegative_int(status_counts["Used"], label="Used status count")
    status_certified = _require_nonnegative_int(
        status_counts["Certified"], label="Certified status count"
    )
    if status_new + status_used + status_certified > raw_rows:
        raise KaggleUSSalesCarsError("reviewed status counts exceed the raw rows")
    target_valid = _require_positive_int(
        quality["target_valid_rows_before_deduplication"],
        label="target-valid rows",
    )
    duplicates = _require_nonnegative_int(
        quality["target_valid_exact_duplicate_rows"], label="duplicate rows"
    )
    rows_after = _require_positive_int(
        quality["target_valid_rows_after_exact_deduplication"],
        label="rows after deduplication",
    )
    invalid_price = _require_nonnegative_int(
        quality["rows_missing_or_invalid_price"], label="invalid price rows"
    )
    missing_mileage = _require_nonnegative_int(
        quality["target_valid_rows_missing_mileage"], label="missing mileage rows"
    )
    if (
        target_valid + invalid_price > raw_rows
        or target_valid - duplicates != rows_after
        or missing_mileage > target_valid
    ):
        raise KaggleUSSalesCarsError("reviewed target quality counts are inconsistent")
    year_min = _require_positive_int(
        quality["year_min_for_target_valid_rows"], label="minimum year"
    )
    year_max = _require_positive_int(
        quality["year_max_for_target_valid_rows"], label="maximum year"
    )
    price_min = _require_positive_int(
        quality["price_min_usd_for_target_valid_rows"], label="minimum price"
    )
    price_max = _require_positive_int(
        quality["price_max_usd_for_target_valid_rows"], label="maximum price"
    )
    if year_min > year_max or price_min > price_max:
        raise KaggleUSSalesCarsError("reviewed quality ranges are invalid")
    _require_text_list(quality["notes"], label="quality notes")
    return {
        "status_new": status_new,
        "status_used": status_used,
        "status_certified": status_certified,
        "target_valid_rows": target_valid,
        "duplicate_rows": duplicates,
        "rows_after_deduplication": rows_after,
        "invalid_price_rows": invalid_price,
        "missing_mileage_rows": missing_mileage,
        "year_min": year_min,
        "year_max": year_max,
        "price_min_cents": price_min * 100,
        "price_max_cents": price_max * 100,
    }


def _validate_publication_policy(value: object) -> None:
    policy = _require_object(value, label="publication_policy")
    _require_exact_keys(policy, {"allowed", "blocked_pending_review"}, label="publication_policy")
    _require_text_list(policy["allowed"], label="publication allowed")
    blocked = _require_text_list(policy["blocked_pending_review"], label="publication blocked")
    required_blocked = {
        "raw dataset files",
        "processed row-level datasets",
        "row-level examples copied from the dataset",
        "downloadable trained model artifacts",
        "commercial deployment",
    }
    if set(blocked) != required_blocked:
        raise KaggleUSSalesCarsError("row-level and model publication must remain blocked")


def _verify_raw_artifact(path: Path, review: KaggleUSSalesCarsReview) -> None:
    actual_sha256, actual_size = _hash_regular_file(path)
    if actual_size != review.expected_size_bytes:
        raise KaggleUSSalesCarsError("raw CSV byte size does not match the source review")
    if actual_sha256 != review.expected_sha256:
        raise KaggleUSSalesCarsError("raw CSV SHA-256 does not match the source review")


def _require_utf16_bom(path: Path) -> None:
    try:
        with path.open("rb") as stream:
            bom = stream.read(2)
    except OSError as error:
        raise KaggleUSSalesCarsError("raw CSV encoding could not be inspected") from error
    if bom not in {b"\xff\xfe", b"\xfe\xff"}:
        raise KaggleUSSalesCarsError("raw CSV must have a UTF-16 byte-order mark")


def _hash_regular_file(path: Path, *, allow_empty: bool = False) -> tuple[str, int]:
    resolved = _require_regular_file(path, label="artifact")
    try:
        path_before = resolved.stat()
        digest = hashlib.sha256()
        bytes_read = 0
        with resolved.open("rb") as stream:
            descriptor_before = os.fstat(stream.fileno())
            if not stat.S_ISREG(descriptor_before.st_mode):
                raise KaggleUSSalesCarsError("artifact is not a regular file")
            if _file_signature(descriptor_before) != _file_signature(path_before):
                raise KaggleUSSalesCarsError("artifact changed during verification")
            while chunk := stream.read(_HASH_CHUNK_BYTES):
                bytes_read += len(chunk)
                digest.update(chunk)
            descriptor_after = os.fstat(stream.fileno())
        path_after = resolved.stat()
    except KaggleUSSalesCarsError:
        raise
    except OSError as error:
        raise KaggleUSSalesCarsError("artifact could not be hashed") from error
    if (
        _file_signature(path_before) != _file_signature(path_after)
        or _file_signature(descriptor_before) != _file_signature(descriptor_after)
        or bytes_read != path_after.st_size
    ):
        raise KaggleUSSalesCarsError("artifact changed during verification")
    if bytes_read == 0 and not allow_empty:
        raise KaggleUSSalesCarsError("artifact must not be empty")
    return digest.hexdigest(), bytes_read


def _file_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_reviewed_source_path(path: Path, expected: PurePosixPath) -> None:
    expected_parts = tuple(part.casefold() for part in expected.parts)
    actual_parts = tuple(part.casefold() for part in path.parts)
    if (
        len(actual_parts) < len(expected_parts)
        or actual_parts[-len(expected_parts) :] != expected_parts
    ):
        raise KaggleUSSalesCarsError("raw CSV is not stored at the reviewed path")


def _require_regular_file(path: Path, *, label: str) -> Path:
    if not isinstance(path, Path) or path.is_symlink():
        raise KaggleUSSalesCarsError(f"{label} must be a non-symlink local file")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise KaggleUSSalesCarsError(f"{label} is missing or inaccessible") from error
    if not resolved.is_file():
        raise KaggleUSSalesCarsError(f"{label} is not a regular file")
    return resolved


def _read_bounded_bytes(path: Path, *, max_bytes: int, label: str) -> bytes:
    try:
        size = path.stat().st_size
        if size <= 0 or size > max_bytes:
            raise KaggleUSSalesCarsError(f"{label} is empty or exceeds its byte limit")
        payload = path.read_bytes()
    except KaggleUSSalesCarsError:
        raise
    except OSError as error:
        raise KaggleUSSalesCarsError(f"{label} could not be read") from error
    if len(payload) != size:
        raise KaggleUSSalesCarsError(f"{label} changed during read")
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
        raise KaggleUSSalesCarsError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise KaggleUSSalesCarsError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _require_object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise KaggleUSSalesCarsError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _require_exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise KaggleUSSalesCarsError(f"{label} fields do not match adapter v2")


def _require_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KaggleUSSalesCarsError(f"{label} must be non-empty text")
    return value.strip()


def _require_text_list(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise KaggleUSSalesCarsError(f"{label} must be a non-empty list")
    return tuple(_require_text(item, label=label) for item in value)


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise KaggleUSSalesCarsError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise KaggleUSSalesCarsError(f"{label} must be a nonnegative integer")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    text = _require_text(value, label=label)
    if not _SHA256_PATTERN.fullmatch(text):
        raise KaggleUSSalesCarsError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _parse_iso_date(value: object, *, label: str) -> date:
    text = _require_text(value, label=label)
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise KaggleUSSalesCarsError(f"{label} must be an ISO date") from error


def _require_public_url(value: object, *, label: str) -> str:
    text = _require_text(value, label=label)
    parsed = urlsplit(text)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise KaggleUSSalesCarsError(f"{label} must be a public HTTP(S) URL")
    return text


def _require_safe_relative_path(value: object, *, label: str) -> PurePosixPath:
    text = _require_text(value, label=label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise KaggleUSSalesCarsError(f"{label} must be a safe relative POSIX path")
    return path


def _safe_artifact_name(value: object, *, label: str) -> str:
    text = _require_text(value, label=label)
    if Path(text).name != text or text in {".", ".."}:
        raise KaggleUSSalesCarsError(f"{label} must be a safe filename")
    return text


def _require_hash_match(expected: object, actual: str, *, label: str) -> None:
    expected_hash = _require_sha256(expected, label=f"{label} hash")
    if not hmac.compare_digest(expected_hash, actual):
        raise KaggleUSSalesCarsError(f"{label} hash does not match")


def _write_fsynced(path: Path, payload: bytes) -> None:
    try:
        with path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise KaggleUSSalesCarsError("could not persist a staged artifact") from error


def _json_payload(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


__all__ = [
    "KAGGLE_US_SALES_CARS_HEADER",
    "KaggleUSSalesCarsArtifactSet",
    "KaggleUSSalesCarsError",
    "KaggleUSSalesCarsMetrics",
    "KaggleUSSalesCarsReview",
    "KaggleUSSalesCarsTrainingRows",
    "load_kaggle_us_sales_cars_review",
    "prepare_kaggle_us_sales_cars_training_rows",
    "process_kaggle_us_sales_cars_csv",
    "require_kaggle_us_sales_cars_ml_training_approval",
    "verify_kaggle_us_sales_cars_artifact_set",
]
