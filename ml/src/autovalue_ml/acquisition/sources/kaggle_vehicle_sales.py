"""Privacy-preserving adapter for the reviewed Kaggle vehicle-sales CSV.

The raw file is a mixed U.S./Canada/Puerto Rico artifact, so it deliberately
does not pass through the generic U.S.-only licensed-dataset loader. Instead,
this adapter consumes the project-owned source review, pins the exact raw file,
and emits only validated 50-state-plus-D.C. records in the common listing
schema. VIN is used transiently only for aggregate duplicate/group metrics.

Condition normalization is source-specific and explicit: values from 1 through
5 already represent the 1.0--5.0 scale; integral legacy values from 11 through
49 are divided by ten. Blank condition is retained as missing. Other values are
quarantined rather than guessed.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, NoReturn, cast
from urllib.parse import urlsplit

from autovalue_ml.acquisition.contracts import PriceKind, VehicleListingSnapshot

REVIEW_SCHEMA_VERSION: Final = 1
KAGGLE_VEHICLE_SALES_HEADER: Final[tuple[str, ...]] = (
    "year",
    "make",
    "model",
    "trim",
    "body",
    "transmission",
    "vin",
    "state",
    "condition",
    "odometer",
    "color",
    "interior",
    "seller",
    "mmr",
    "sellingprice",
    "saledate",
)
US_50_PLUS_DC: Final[frozenset[str]] = frozenset(
    {
        "AL",
        "AK",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "DC",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
    }
)

_SOURCE_ID: Final = "kaggle_vehicle_sales_v1"
_PARSER_VERSION: Final = "kaggle-vehicle-sales-v1.0.0"
_NORMALIZATION_VERSION: Final = "vehicle-listing-v1+kaggle-condition-v1"
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_STATE_PATTERN = re.compile(r"^[A-Z]{2}$")
_VIN_PATTERN = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")
_INTEGER_PATTERN = re.compile(r"^[0-9]+(?:\.0+)?$")
_PRICE_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]{1,2})?$")
_CONDITION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
_ROW_ID_PATTERN = re.compile(r"^row-([0-9]{9})$")
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
_TZ_OFFSETS_MINUTES: Final = {
    "PST": -8 * 60,
    "PDT": -7 * 60,
    "MST": -7 * 60,
    "MDT": -6 * 60,
    "CST": -6 * 60,
    "CDT": -5 * 60,
    "EST": -5 * 60,
    "EDT": -4 * 60,
}
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
_RIVER_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "year",
    "make",
    "model",
    "trim",
    "mileage",
    "condition",
    "vehicle_type",
)
_REQUIRED_PROCESSING_GATES: Final[frozenset[str]] = frozenset(
    {
        "Verify the exact archive and CSV byte sizes and SHA-256 hashes before parsing.",
        "Retain only records mapped to one of the 50 U.S. states or the District of "
        "Columbia; reject Canada and Puerto Rico.",
        "Require a parseable sale date and positive sellingprice target.",
        "Remove VIN and seller from every feature table and published artifact.",
        "Forbid MMR from model features, preprocessing, feature selection, tuning, "
        "explanations, and inference payloads because it is a competing valuation "
        "estimate and target-leakage risk.",
        "Use a chronological test boundary and keep every normalized VIN in only one split.",
        "Fit imputers, encoders, category grouping, and all learned transforms on "
        "training folds only.",
        "Label every result as historical wholesale-auction performance and disclose "
        "the 2014-2015 time window.",
        "Do not commit or redistribute raw rows, processed rows, or a downloadable "
        "model without a new permission decision.",
    }
)
_REVIEW_TOP_LEVEL_KEYS: Final = {
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
    "required_processing_gates",
    "publication_policy",
    "training_status",
    "notes",
}
_MAX_REVIEW_BYTES: Final = 256_000
_QUARANTINE_REASON_CODES: Final = frozenset(
    {
        "body_invalid",
        "condition_invalid",
        "csv_formula_injection",
        "make_missing",
        "market_not_us",
        "model_missing",
        "odometer_invalid",
        "row_width_invalid",
        "sale_date_invalid",
        "sale_date_outside_reviewed_range",
        "sellingprice_invalid",
        "state_invalid",
        "trim_invalid",
        "vin_missing_or_invalid",
        "year_invalid",
    }
)
_MANIFEST_KEYS: Final = {
    "schema_version",
    "source_id",
    "review_id",
    "review_sha256",
    "reviewed_source_version",
    "raw_source_file",
    "raw_source_sha256",
    "raw_source_size_bytes",
    "raw_source_row_count",
    "raw_market_scope",
    "market_country",
    "currency",
    "price_kind",
    "sale_status",
    "publication_status",
    "approved_for_acquisition",
    "acquisition_evidence",
    "approved_for_ml_training",
    "ml_training_evidence",
    "training_readiness",
    "parser_version",
    "normalization_version",
    "condition_rule",
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
_METRICS_KEYS: Final = {
    "rows_seen",
    "rows_accepted",
    "non_us_rows",
    "quarantined_rows",
    "exact_duplicate_rows",
    "repeated_vin_rows",
    "distinct_repeated_vins",
    "missing_or_invalid_vin_rows",
    "quarantine_reason_counts",
}
_TRAINING_BLOCKER: Final = "blocked_pending_reviewed_chronological_vin_isolated_split"


class KaggleVehicleSalesError(RuntimeError):
    """The source review, raw CSV, or derived artifact failed closed."""


class _RowRejected(ValueError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True)
class KaggleVehicleSalesReview:
    """Validated source-specific authorization and immutable artifact pin."""

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
    sale_date_min: date
    sale_date_max: date
    approved_for_acquisition: bool
    acquisition_evidence: str
    approved_for_ml_training: bool
    ml_training_evidence: str


@dataclass(frozen=True, slots=True)
class KaggleIngestionMetrics:
    """Aggregate-only metrics; no VIN or source row is retained."""

    rows_seen: int
    rows_accepted: int
    non_us_rows: int
    quarantined_rows: int
    exact_duplicate_rows: int
    repeated_vin_rows: int
    distinct_repeated_vins: int
    missing_or_invalid_vin_rows: int
    quarantine_reason_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "rows_seen": self.rows_seen,
            "rows_accepted": self.rows_accepted,
            "non_us_rows": self.non_us_rows,
            "quarantined_rows": self.quarantined_rows,
            "exact_duplicate_rows": self.exact_duplicate_rows,
            "repeated_vin_rows": self.repeated_vin_rows,
            "distinct_repeated_vins": self.distinct_repeated_vins,
            "missing_or_invalid_vin_rows": self.missing_or_invalid_vin_rows,
            "quarantine_reason_counts": dict(self.quarantine_reason_counts),
        }


@dataclass(frozen=True, slots=True)
class KaggleCandidateArtifactSet:
    candidate_path: Path
    quarantine_path: Path
    manifest_path: Path
    readiness_path: Path
    metrics: KaggleIngestionMetrics


def load_kaggle_vehicle_sales_review(
    review_path: Path,
    *,
    today: date | None = None,
) -> KaggleVehicleSalesReview:
    """Load and strictly validate the committed, project-owned review record."""

    resolved_path = _require_regular_file(review_path, label="source review")
    payload = _read_bounded_bytes(resolved_path, max_bytes=_MAX_REVIEW_BYTES, label="source review")
    review_sha256 = hashlib.sha256(payload).hexdigest()
    root = _strict_json_object(payload, label="source review")
    _require_exact_keys(root, _REVIEW_TOP_LEVEL_KEYS, label="source review")

    if root["review_schema_version"] != REVIEW_SCHEMA_VERSION:
        raise KaggleVehicleSalesError("unsupported Kaggle review schema")
    if root["decision"] != "approved_with_conditions":
        raise KaggleVehicleSalesError("source review decision does not approve processing")
    review_id = _require_text(root["review_id"], label="review_id")
    reviewed_on = _parse_iso_date(root["reviewed_on"], label="reviewed_on")
    if reviewed_on > (date.today() if today is None else today):
        raise KaggleVehicleSalesError("source review date cannot be in the future")
    _require_text(root["project_role"], label="project_role")
    if root["training_status"] != "not_started":
        raise KaggleVehicleSalesError(
            "adapter v1 only accepts a review whose training status is not_started"
        )
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
            "initial_release_date",
            "license_label",
            "license_label_source",
            "underlying_ownership_independently_verified",
            "permission_evidence",
        },
        label="source",
    )
    if source["platform"] != "Kaggle" or source["dataset_slug"] != "vehicle-sales-data":
        raise KaggleVehicleSalesError("review identifies an unsupported source")
    if type(source["underlying_ownership_independently_verified"]) is not bool:
        raise KaggleVehicleSalesError("source ownership review flag must be a boolean")
    dataset_url = _require_kaggle_dataset_url(source["dataset_url"])
    source_version = _require_text(source["version"], label="source version")
    _parse_iso_date(source["initial_release_date"], label="initial_release_date")
    for key in ("uploader", "license_label", "license_label_source"):
        _require_text(source[key], label=key)
    permission_evidence = _require_object(
        source["permission_evidence"], label="permission_evidence"
    )
    _require_exact_keys(
        permission_evidence,
        {"kind", "summary", "external_correspondence_committed"},
        label="permission_evidence",
    )
    evidence_kind = _require_text(permission_evidence["kind"], label="permission evidence kind")
    evidence_summary = _require_text(
        permission_evidence["summary"], label="permission evidence summary"
    )
    if type(permission_evidence["external_correspondence_committed"]) is not bool:
        raise KaggleVehicleSalesError("permission evidence committed flag must be a boolean")

    retrieval = _require_object(root["retrieval"], label="retrieval")
    _require_exact_keys(
        retrieval,
        {
            "retrieved_on",
            "reviewed_on",
            "required_method",
            "csv_path",
            "archive_path",
            "archive",
            "csv",
        },
        label="retrieval",
    )
    retrieved_on = _parse_iso_date(retrieval["retrieved_on"], label="retrieved_on")
    if retrieved_on > reviewed_on:
        raise KaggleVehicleSalesError("retrieval date cannot follow the review date")
    if _parse_iso_date(retrieval["reviewed_on"], label="retrieval reviewed_on") != reviewed_on:
        raise KaggleVehicleSalesError("retrieval and review dates differ")
    if retrieval["required_method"] != "official Kaggle download":
        raise KaggleVehicleSalesError("only the official Kaggle download is approved")
    csv_path = _require_safe_relative_path(retrieval["csv_path"], label="csv_path")
    _require_safe_relative_path(retrieval["archive_path"], label="archive_path")
    _validate_pinned_artifact(_require_object(retrieval["archive"], label="archive"), archive=True)
    csv_pin = _require_object(retrieval["csv"], label="csv")
    _require_exact_keys(csv_pin, {"file_name", "size_bytes", "sha256", "row_count"}, label="csv")
    expected_file_name = _require_text(csv_pin["file_name"], label="CSV file_name")
    if Path(expected_file_name).name != expected_file_name or csv_path.name != expected_file_name:
        raise KaggleVehicleSalesError("reviewed CSV filename is inconsistent")
    expected_size = _require_positive_int(csv_pin["size_bytes"], label="CSV size_bytes")
    expected_sha256 = _require_sha256(csv_pin["sha256"], label="CSV sha256")
    expected_row_count = _require_positive_int(csv_pin["row_count"], label="CSV row_count")

    permissions = _require_object(root["permissions"], label="permissions")
    permission_keys = {
        "official_download",
        "local_storage",
        "private_transformation",
        "ml_training_and_evaluation",
        "public_aggregate_metrics_and_charts",
        "hosted_inference",
        "raw_row_redistribution",
        "processed_row_redistribution",
        "downloadable_trained_model",
        "sublicensing",
        "commercial_use",
    }
    _require_exact_keys(permissions, permission_keys, label="permissions")
    for key, value in permissions.items():
        if value not in {"approved", "pending", "rejected"}:
            raise KaggleVehicleSalesError(f"permission {key} has an invalid decision")
    acquisition_approved = all(
        permissions[key] == "approved"
        for key in ("official_download", "local_storage", "private_transformation")
    )
    if not acquisition_approved:
        raise KaggleVehicleSalesError("source review does not approve private acquisition")

    market = _require_object(root["market_scope"], label="market_scope")
    _require_exact_keys(
        market,
        {
            "raw_scope",
            "approved_scope",
            "raw_row_count",
            "us_50_plus_dc_row_count",
            "us_50_plus_dc_valid_target_and_date_row_count",
            "us_50_plus_dc_core_complete_row_count",
            "sale_date_min",
            "sale_date_max",
            "currency",
            "currency_status",
            "currency_assumption",
        },
        label="market_scope",
    )
    if market["raw_scope"] != "mixed United States, Canada, and Puerto Rico":
        raise KaggleVehicleSalesError("raw market scope is not the reviewed mixed scope")
    if market["approved_scope"] != "50 United States plus the District of Columbia":
        raise KaggleVehicleSalesError("approved output scope must be the 50 states plus D.C.")
    if market["currency"] != "USD":
        raise KaggleVehicleSalesError("approved U.S. output currency must be USD")
    if market["raw_row_count"] != expected_row_count:
        raise KaggleVehicleSalesError("reviewed row counts are inconsistent")
    reviewed_counts = tuple(
        _require_nonnegative_int(market[count_key], label=count_key)
        for count_key in (
            "us_50_plus_dc_core_complete_row_count",
            "us_50_plus_dc_valid_target_and_date_row_count",
            "us_50_plus_dc_row_count",
        )
    )
    if not (reviewed_counts[0] <= reviewed_counts[1] <= reviewed_counts[2] <= expected_row_count):
        raise KaggleVehicleSalesError("reviewed market row counts are inconsistent")
    _require_text(market["currency_status"], label="currency_status")
    _require_text(market["currency_assumption"], label="currency_assumption")
    sale_date_min = _parse_iso_date(market["sale_date_min"], label="sale_date_min")
    sale_date_max = _parse_iso_date(market["sale_date_max"], label="sale_date_max")
    if sale_date_max < sale_date_min:
        raise KaggleVehicleSalesError("reviewed sale-date range is invalid")

    target = _require_object(root["target"], label="target")
    _require_exact_keys(
        target,
        {"source_column", "meaning", "approved_claim", "prohibited_claim"},
        label="target",
    )
    if target["source_column"] != "sellingprice":
        raise KaggleVehicleSalesError("reviewed target must be sellingprice")
    for key in ("meaning", "approved_claim", "prohibited_claim"):
        _require_text(target[key], label=f"target {key}")

    gates = frozenset(
        _require_text_list(root["required_processing_gates"], label="required_processing_gates")
    )
    if gates != _REQUIRED_PROCESSING_GATES:
        raise KaggleVehicleSalesError("required processing gates do not match adapter v1")
    publication = _require_object(root["publication_policy"], label="publication_policy")
    _require_exact_keys(
        publication,
        {"allowed", "blocked_pending_review"},
        label="publication_policy",
    )
    _require_text_list(publication["allowed"], label="publication allowed")
    blocked = _require_text_list(publication["blocked_pending_review"], label="publication blocked")
    if "processed row-level datasets" not in blocked:
        raise KaggleVehicleSalesError("processed-row publication must remain blocked")

    ml_approved = permissions["ml_training_and_evaluation"] == "approved"
    return KaggleVehicleSalesReview(
        review_id=review_id,
        reviewed_on=reviewed_on,
        review_sha256=review_sha256,
        review_path=resolved_path,
        dataset_url=dataset_url,
        source_version=source_version,
        expected_csv_path=csv_path,
        expected_file_name=expected_file_name,
        expected_size_bytes=expected_size,
        expected_sha256=expected_sha256,
        expected_row_count=expected_row_count,
        sale_date_min=sale_date_min,
        sale_date_max=sale_date_max,
        approved_for_acquisition=acquisition_approved,
        acquisition_evidence=f"{evidence_kind}: {evidence_summary}",
        approved_for_ml_training=ml_approved,
        ml_training_evidence=(
            "project source review permissions.ml_training_and_evaluation=approved"
            if ml_approved
            else ""
        ),
    )


def require_kaggle_ml_training_approval(
    review: KaggleVehicleSalesReview,
) -> KaggleVehicleSalesReview:
    """Fail closed at the independent ML-reuse decision."""

    if not review.approved_for_ml_training or not review.ml_training_evidence:
        raise KaggleVehicleSalesError("Kaggle source review does not approve ML training")
    return review


def process_kaggle_vehicle_sales_csv(
    source_path: Path,
    review_path: Path,
    output_path: Path,
    *,
    today: date | None = None,
) -> KaggleCandidateArtifactSet:
    """Stream the pinned raw CSV into a private, privacy-safe U.S. candidate set."""

    if output_path.suffix.lower() != ".csv":
        raise KaggleVehicleSalesError("candidate output must use the .csv suffix")
    review = load_kaggle_vehicle_sales_review(review_path, today=today)
    source = _require_regular_file(source_path, label="raw CSV")
    _require_reviewed_source_path(source, review.expected_csv_path)
    _verify_raw_artifact(source, review)
    resolved_review = _require_regular_file(review_path, label="source review")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_parent = output_path.parent.resolve(strict=True)
    candidate_path = output_parent / output_path.name
    quarantine_path = candidate_path.with_suffix(".quarantine.jsonl")
    manifest_path = candidate_path.with_suffix(".manifest.json")
    readiness_path = candidate_path.with_suffix(".ready.json")
    artifact_paths = (candidate_path, quarantine_path, manifest_path, readiness_path)
    _validate_output_targets(
        artifact_paths,
        protected_paths=(source, resolved_review),
    )
    readiness_path.unlink(missing_ok=True)

    staging_path = Path(tempfile.mkdtemp(prefix=f".{candidate_path.stem}.", dir=output_parent))
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
        if metrics.rows_seen != review.expected_row_count:
            raise KaggleVehicleSalesError("raw CSV row count does not match the source review")

        # A second complete source verification closes the parse-to-publish TOCTOU window.
        _verify_raw_artifact(source, review)
        candidate_sha256, candidate_size = _hash_regular_file(staged_candidate)
        quarantine_sha256, quarantine_size = _hash_regular_file(staged_quarantine, allow_empty=True)
        manifest = {
            "schema_version": 1,
            "source_id": _SOURCE_ID,
            "review_id": review.review_id,
            "review_sha256": review.review_sha256,
            "reviewed_source_version": review.source_version,
            "raw_source_file": review.expected_file_name,
            "raw_source_sha256": review.expected_sha256,
            "raw_source_size_bytes": review.expected_size_bytes,
            "raw_source_row_count": review.expected_row_count,
            "raw_market_scope": "mixed United States, Canada, and Puerto Rico",
            "market_country": "US",
            "currency": "USD",
            "price_kind": PriceKind.COMPLETED_SALE.value,
            "sale_status": "sold",
            "publication_status": "private_local_only",
            "approved_for_acquisition": review.approved_for_acquisition,
            "acquisition_evidence": review.acquisition_evidence,
            "approved_for_ml_training": review.approved_for_ml_training,
            "ml_training_evidence": review.ml_training_evidence,
            "training_readiness": _TRAINING_BLOCKER,
            "parser_version": _PARSER_VERSION,
            "normalization_version": _NORMALIZATION_VERSION,
            "condition_rule": "1-5 retained on a 1.0-5.0 scale; integral 11-49 divided by 10",
            "candidate_file": candidate_path.name,
            "candidate_sha256": candidate_sha256,
            "candidate_size_bytes": candidate_size,
            "candidate_columns": list(_CANDIDATE_HEADER),
            "feature_allowlist": list(_RIVER_FEATURE_COLUMNS),
            "forbidden_source_columns": ["vin", "seller", "mmr", "transmission"],
            "quarantine_file": quarantine_path.name,
            "quarantine_sha256": quarantine_sha256,
            "quarantine_size_bytes": quarantine_size,
            "readiness_file": readiness_path.name,
            "metrics": metrics.to_dict(),
        }
        staged_manifest = staging_path / manifest_path.name
        manifest_payload = _json_payload(manifest)
        _write_fsynced(staged_manifest, manifest_payload)
        manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        artifact_set_id = hashlib.sha256(
            "|".join((manifest_sha256, candidate_sha256, quarantine_sha256)).encode("utf-8")
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
        # The final marker is the only signal that all three artifacts are publishable.
        os.replace(staged_readiness, readiness_path)
        return KaggleCandidateArtifactSet(
            candidate_path=candidate_path,
            quarantine_path=quarantine_path,
            manifest_path=manifest_path,
            readiness_path=readiness_path,
            metrics=metrics,
        )
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)


def verify_kaggle_candidate_artifact_set(
    manifest_path: Path,
    review_path: Path,
    *,
    today: date | None = None,
) -> dict[str, object]:
    """Verify the final marker, all derived hashes, and source-review lineage."""

    review = load_kaggle_vehicle_sales_review(review_path, today=today)
    manifest_file = _require_regular_file(manifest_path, label="candidate manifest")
    manifest_payload = _read_bounded_bytes(
        manifest_file, max_bytes=5_000_000, label="candidate manifest"
    )
    manifest = _strict_json_object(manifest_payload, label="candidate manifest")
    _validate_candidate_manifest(manifest, manifest_file=manifest_file, review=review)

    readiness_name = _safe_artifact_name(manifest.get("readiness_file"), label="readiness_file")
    readiness_file = _require_regular_file(
        manifest_file.parent / readiness_name, label="readiness marker"
    )
    readiness = _strict_json_object(
        _read_bounded_bytes(readiness_file, max_bytes=1_000_000, label="readiness marker"),
        label="readiness marker",
    )
    expected_ready_keys = {
        "schema_version",
        "artifact_set_id",
        "manifest_file",
        "manifest_sha256",
        "candidate_file",
        "candidate_sha256",
        "quarantine_file",
        "quarantine_sha256",
    }
    if set(readiness) != expected_ready_keys or readiness.get("schema_version") != 1:
        raise KaggleVehicleSalesError("readiness marker schema is invalid")
    if readiness.get("manifest_file") != manifest_file.name:
        raise KaggleVehicleSalesError("readiness marker references another manifest")
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    _require_hash_match(readiness.get("manifest_sha256"), manifest_sha256, label="manifest")

    verified_hashes: list[str] = []
    verified_paths: dict[str, Path] = {}
    for name_key, hash_key, allow_empty in (
        ("candidate_file", "candidate_sha256", False),
        ("quarantine_file", "quarantine_sha256", True),
    ):
        file_name = _safe_artifact_name(readiness.get(name_key), label=name_key)
        if manifest.get(name_key) != file_name or manifest.get(hash_key) != readiness.get(hash_key):
            raise KaggleVehicleSalesError("manifest and readiness lineage differ")
        artifact = _require_regular_file(manifest_file.parent / file_name, label=name_key)
        actual_sha256, actual_size = _hash_regular_file(artifact, allow_empty=allow_empty)
        _require_hash_match(readiness.get(hash_key), actual_sha256, label=name_key)
        expected_size = _require_nonnegative_int(
            manifest.get(name_key.replace("_file", "_size_bytes")),
            label=f"{name_key} size",
        )
        if actual_size != expected_size:
            raise KaggleVehicleSalesError(f"{name_key} byte size does not match the manifest")
        verified_hashes.append(actual_sha256)
        verified_paths[name_key] = artifact
    expected_set_id = hashlib.sha256(
        "|".join((manifest_sha256, *verified_hashes)).encode("utf-8")
    ).hexdigest()
    _require_hash_match(readiness.get("artifact_set_id"), expected_set_id, label="artifact set")
    metrics = _require_object(manifest["metrics"], label="metrics")
    _verify_candidate_rows(verified_paths["candidate_file"], review=review, metrics=metrics)
    _verify_quarantine_rows(
        verified_paths["quarantine_file"],
        review=review,
        metrics=metrics,
    )
    for hash_key, path_key, allow_empty in (
        ("candidate_sha256", "candidate_file", False),
        ("quarantine_sha256", "quarantine_file", True),
    ):
        final_sha256, _ = _hash_regular_file(
            verified_paths[path_key],
            allow_empty=allow_empty,
        )
        _require_hash_match(readiness[hash_key], final_sha256, label=path_key)
    return readiness


def _validate_candidate_manifest(
    manifest: dict[str, object],
    *,
    manifest_file: Path,
    review: KaggleVehicleSalesReview,
) -> None:
    _require_exact_keys(manifest, _MANIFEST_KEYS, label="candidate manifest")
    exact_values: dict[str, object] = {
        "schema_version": 1,
        "source_id": _SOURCE_ID,
        "review_id": review.review_id,
        "review_sha256": review.review_sha256,
        "reviewed_source_version": review.source_version,
        "raw_source_file": review.expected_file_name,
        "raw_source_sha256": review.expected_sha256,
        "raw_source_size_bytes": review.expected_size_bytes,
        "raw_source_row_count": review.expected_row_count,
        "raw_market_scope": "mixed United States, Canada, and Puerto Rico",
        "market_country": "US",
        "currency": "USD",
        "price_kind": PriceKind.COMPLETED_SALE.value,
        "sale_status": "sold",
        "publication_status": "private_local_only",
        "approved_for_acquisition": review.approved_for_acquisition,
        "acquisition_evidence": review.acquisition_evidence,
        "approved_for_ml_training": review.approved_for_ml_training,
        "ml_training_evidence": review.ml_training_evidence,
        "training_readiness": _TRAINING_BLOCKER,
        "parser_version": _PARSER_VERSION,
        "normalization_version": _NORMALIZATION_VERSION,
        "condition_rule": ("1-5 retained on a 1.0-5.0 scale; integral 11-49 divided by 10"),
        "candidate_columns": list(_CANDIDATE_HEADER),
        "feature_allowlist": list(_RIVER_FEATURE_COLUMNS),
        "forbidden_source_columns": ["vin", "seller", "mmr", "transmission"],
    }
    for key, expected in exact_values.items():
        if manifest[key] != expected:
            raise KaggleVehicleSalesError(f"candidate manifest {key} is invalid")

    candidate_name = _safe_artifact_name(manifest["candidate_file"], label="candidate_file")
    quarantine_name = _safe_artifact_name(manifest["quarantine_file"], label="quarantine_file")
    readiness_name = _safe_artifact_name(manifest["readiness_file"], label="readiness_file")
    if not candidate_name.lower().endswith(".csv"):
        raise KaggleVehicleSalesError("candidate manifest must reference a CSV")
    candidate_stem = Path(candidate_name)
    if (
        candidate_stem.with_suffix(".manifest.json").name != manifest_file.name
        or candidate_stem.with_suffix(".quarantine.jsonl").name != quarantine_name
        or candidate_stem.with_suffix(".ready.json").name != readiness_name
    ):
        raise KaggleVehicleSalesError("derived artifact filenames are inconsistent")
    for hash_key in ("candidate_sha256", "quarantine_sha256"):
        _require_sha256(manifest[hash_key], label=hash_key)
    _require_positive_int(manifest["candidate_size_bytes"], label="candidate_size_bytes")
    _require_nonnegative_int(manifest["quarantine_size_bytes"], label="quarantine_size_bytes")
    _validate_metrics(_require_object(manifest["metrics"], label="metrics"), review=review)


def _validate_metrics(
    metrics: dict[str, object],
    *,
    review: KaggleVehicleSalesReview,
) -> None:
    _require_exact_keys(metrics, _METRICS_KEYS, label="metrics")
    counters = {
        key: _require_nonnegative_int(metrics[key], label=f"metrics {key}")
        for key in _METRICS_KEYS - {"quarantine_reason_counts"}
    }
    if counters["rows_seen"] != review.expected_row_count:
        raise KaggleVehicleSalesError("candidate metrics row count differs from the review")
    if counters["rows_seen"] != (
        counters["rows_accepted"] + counters["quarantined_rows"] + counters["exact_duplicate_rows"]
    ):
        raise KaggleVehicleSalesError("candidate metrics accounting is invalid")
    if counters["non_us_rows"] > counters["quarantined_rows"]:
        raise KaggleVehicleSalesError("candidate non-U.S. metrics are invalid")
    if counters["distinct_repeated_vins"] > counters["repeated_vin_rows"]:
        raise KaggleVehicleSalesError("candidate repeated-VIN metrics are invalid")
    if any(
        counters[key] > counters["rows_seen"]
        for key in (
            "repeated_vin_rows",
            "missing_or_invalid_vin_rows",
        )
    ):
        raise KaggleVehicleSalesError("candidate VIN metrics are invalid")

    reasons = _require_object(
        metrics["quarantine_reason_counts"],
        label="quarantine_reason_counts",
    )
    if any(reason not in _QUARANTINE_REASON_CODES for reason in reasons):
        raise KaggleVehicleSalesError("candidate quarantine reason is invalid")
    reason_total = 0
    for reason, count_value in reasons.items():
        count = _require_positive_int(count_value, label=f"quarantine reason {reason}")
        reason_total += count
    if reason_total != counters["quarantined_rows"]:
        raise KaggleVehicleSalesError("candidate quarantine metrics do not add up")
    if reasons.get("market_not_us", 0) != counters["non_us_rows"]:
        raise KaggleVehicleSalesError("candidate non-U.S. metrics do not add up")


def _verify_candidate_rows(
    candidate_path: Path,
    *,
    review: KaggleVehicleSalesReview,
    metrics: dict[str, object],
) -> None:
    expected_count = _require_nonnegative_int(
        metrics["rows_accepted"], label="metrics rows_accepted"
    )
    expected_run_id = f"kvs-{review.expected_sha256[:16]}-{review.review_sha256[:8]}"
    count = 0
    previous_row_number = 1
    try:
        with candidate_path.open("r", encoding="utf-8", newline="") as source_file:
            reader = csv.DictReader(source_file, strict=True)
            if tuple(reader.fieldnames or ()) != _CANDIDATE_HEADER:
                raise KaggleVehicleSalesError("candidate header is invalid")
            for row in reader:
                count += 1
                if set(row) != set(_CANDIDATE_HEADER) or any(
                    not isinstance(row[column], str) for column in _CANDIDATE_HEADER
                ):
                    raise KaggleVehicleSalesError("candidate row width is invalid")
                try:
                    _verify_candidate_row(
                        cast(dict[str, str], row),
                        review=review,
                        expected_run_id=expected_run_id,
                        previous_row_number=previous_row_number,
                    )
                except _RowRejected as error:
                    raise KaggleVehicleSalesError(
                        "candidate row violates the normalized schema"
                    ) from error
                match = _ROW_ID_PATTERN.fullmatch(row["source_listing_id"])
                if match is None:
                    raise KaggleVehicleSalesError("candidate source row ID is invalid")
                previous_row_number = int(match.group(1))
                if previous_row_number > review.expected_row_count + 1:
                    raise KaggleVehicleSalesError("candidate source row ID is out of range")
    except (csv.Error, UnicodeError) as error:
        raise KaggleVehicleSalesError("candidate CSV is malformed") from error
    if count != expected_count:
        raise KaggleVehicleSalesError("candidate row count differs from the manifest")


def _verify_candidate_row(
    row: dict[str, str],
    *,
    review: KaggleVehicleSalesReview,
    expected_run_id: str,
    previous_row_number: int,
) -> None:
    match = _ROW_ID_PATTERN.fullmatch(row["source_listing_id"])
    if match is None or int(match.group(1)) <= previous_row_number:
        raise KaggleVehicleSalesError("candidate source row IDs are not strictly increasing")
    exact_values = {
        "source_id": _SOURCE_ID,
        "canonical_url": review.dataset_url,
        "market_country": "US",
        "mileage_unit": "miles",
        "vehicle_status": "",
        "engine": "",
        "drivetrain": "",
        "accident_status": "",
        "accident_count": "",
        "owner_count": "",
        "currency": "USD",
        "price_kind": PriceKind.COMPLETED_SALE.value,
        "sale_status": "sold",
        "parser_version": _PARSER_VERSION,
        "normalization_version": _NORMALIZATION_VERSION,
        "ingestion_run_id": expected_run_id,
        "authorization_policy_id": review.review_id,
    }
    if any(row[key] != expected for key, expected in exact_values.items()):
        raise KaggleVehicleSalesError("candidate row contains invalid fixed fields")
    try:
        observed_at = datetime.fromisoformat(row["observed_at"])
    except ValueError as error:
        raise KaggleVehicleSalesError("candidate observed_at is invalid") from error
    if observed_at.tzinfo is None or observed_at.utcoffset() != timedelta(0):
        raise KaggleVehicleSalesError("candidate observed_at must use UTC")
    if not review.sale_date_min <= observed_at.date() <= review.sale_date_max + timedelta(days=1):
        raise KaggleVehicleSalesError("candidate observed_at is outside the reviewed period")
    year = _parse_integral(row["year"], reason_code="year_invalid")
    if str(year) != row["year"] or not 1886 <= year <= observed_at.year + 2:
        raise KaggleVehicleSalesError("candidate year is invalid")
    normalized_make = _required_source_text(row["make"], reason_code="make_missing", max_length=100)
    normalized_model = _required_source_text(
        row["model"], reason_code="model_missing", max_length=150
    )
    normalized_trim = _optional_source_text(row["trim"], reason_code="trim_invalid", max_length=150)
    normalized_body = _optional_source_text(
        row["vehicle_type"], reason_code="body_invalid", max_length=100
    )
    if (
        normalized_make != row["make"]
        or normalized_model != row["model"]
        or normalized_trim != (row["trim"] or None)
        or normalized_body != (row["vehicle_type"] or None)
    ):
        raise KaggleVehicleSalesError("candidate text fields are not normalized")
    condition = _parse_condition(row["condition"])
    if condition != (row["condition"] or None):
        raise KaggleVehicleSalesError("candidate condition is not normalized")
    mileage = _parse_odometer(row["mileage"])
    if row["mileage"] and (mileage is None or str(mileage) != row["mileage"]):
        raise KaggleVehicleSalesError("candidate mileage is not normalized")
    price_cents = _parse_integral(row["price_cents"], reason_code="sellingprice_invalid")
    if price_cents <= 0 or str(price_cents) != row["price_cents"]:
        raise KaggleVehicleSalesError("candidate price is not normalized integer cents")
    _require_sha256(row["raw_content_sha256"], label="candidate content hash")


def _verify_quarantine_rows(
    quarantine_path: Path,
    *,
    review: KaggleVehicleSalesReview,
    metrics: dict[str, object],
) -> None:
    expected_count = _require_nonnegative_int(
        metrics["quarantined_rows"], label="metrics quarantined_rows"
    )
    expected_reasons = _require_object(
        metrics["quarantine_reason_counts"], label="quarantine_reason_counts"
    )
    actual_reasons: dict[str, int] = {}
    count = 0
    previous_row_number = 1
    try:
        with quarantine_path.open("rb") as source_file:
            for payload in source_file:
                count += 1
                if len(payload) > 2_000:
                    raise KaggleVehicleSalesError("quarantine row exceeds its byte limit")
                row = _strict_json_object(payload, label="quarantine row")
                _require_exact_keys(
                    row,
                    {
                        "source_listing_id",
                        "row_number",
                        "reason_code",
                        "safe_record_sha256",
                    },
                    label="quarantine row",
                )
                row_number = _require_positive_int(row["row_number"], label="quarantine row_number")
                if row_number <= previous_row_number:
                    raise KaggleVehicleSalesError(
                        "quarantine row numbers are not strictly increasing"
                    )
                if row_number > review.expected_row_count + 1:
                    raise KaggleVehicleSalesError("quarantine row number is out of range")
                if row["source_listing_id"] != _safe_row_id(row_number):
                    raise KaggleVehicleSalesError("quarantine row ID is invalid")
                reason = _require_text(row["reason_code"], label="quarantine reason")
                if reason not in _QUARANTINE_REASON_CODES:
                    raise KaggleVehicleSalesError("quarantine reason is invalid")
                safe_hash = _require_sha256(
                    row["safe_record_sha256"], label="quarantine safe row hash"
                )
                if safe_hash != _safe_row_reference_sha256(
                    review.expected_sha256,
                    row_number,
                ):
                    raise KaggleVehicleSalesError("quarantine safe row hash is invalid")
                actual_reasons[reason] = actual_reasons.get(reason, 0) + 1
                previous_row_number = row_number
    except UnicodeError as error:
        raise KaggleVehicleSalesError("quarantine file is malformed") from error
    if count != expected_count or actual_reasons != expected_reasons:
        raise KaggleVehicleSalesError("quarantine rows differ from the manifest metrics")


def prepare_kaggle_training_rows(
    candidate_path: Path,
    manifest_path: Path,
    review_path: Path,
    *,
    today: date | None = None,
) -> NoReturn:
    """Fail closed until a reviewed chronological/VIN-isolated split exists."""

    require_kaggle_ml_training_approval(load_kaggle_vehicle_sales_review(review_path, today=today))
    verify_kaggle_candidate_artifact_set(manifest_path, review_path, today=today)
    candidate = _require_regular_file(candidate_path, label="candidate CSV")
    manifest = _strict_json_object(
        _read_bounded_bytes(
            _require_regular_file(manifest_path, label="candidate manifest"),
            max_bytes=5_000_000,
            label="candidate manifest",
        ),
        label="candidate manifest",
    )
    if candidate.name != manifest.get("candidate_file"):
        raise KaggleVehicleSalesError("training candidate differs from the verified manifest")
    raise KaggleVehicleSalesError(
        "unsplit candidate is not training-ready; create and review a chronological "
        "split that keeps every transiently normalized VIN in only one split"
    )


def _stream_transform(
    source_path: Path,
    review: KaggleVehicleSalesReview,
    candidate_path: Path,
    quarantine_path: Path,
    staging_path: Path,
) -> KaggleIngestionMetrics:
    rows_seen = rows_accepted = non_us_rows = quarantined_rows = exact_duplicates = 0
    missing_or_invalid_vins = 0
    reason_counts: dict[str, int] = {}
    secret = secrets.token_bytes(32)
    index_path = staging_path / "private-dedup.sqlite3"
    connection = sqlite3.connect(index_path)
    try:
        connection.execute("CREATE TABLE row_digests (digest BLOB PRIMARY KEY)")
        connection.execute(
            "CREATE TABLE vin_digests (digest BLOB PRIMARY KEY, occurrences INTEGER NOT NULL)"
        )
        with (
            source_path.open("r", encoding="utf-8-sig", newline="") as source_file,
            candidate_path.open("w", encoding="utf-8", newline="") as candidate_file,
            quarantine_path.open("w", encoding="utf-8", newline="") as quarantine_file,
        ):
            reader = csv.reader(source_file, strict=True)
            try:
                header = next(reader)
            except (StopIteration, csv.Error) as error:
                raise KaggleVehicleSalesError("raw CSV is missing its required header") from error
            if tuple(header) != KAGGLE_VEHICLE_SALES_HEADER:
                raise KaggleVehicleSalesError(
                    "raw CSV header does not match the exact 16-column schema"
                )
            writer = csv.DictWriter(
                candidate_file,
                fieldnames=list(_CANDIDATE_HEADER),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            try:
                for row_number, values in enumerate(reader, start=2):
                    rows_seen += 1
                    safe_record_sha256 = _safe_row_reference_sha256(
                        review.expected_sha256,
                        row_number,
                    )
                    row_digest = hmac.digest(secret, _canonical_row_bytes(values), "sha256")
                    inserted = connection.execute(
                        "INSERT OR IGNORE INTO row_digests (digest) VALUES (?)", (row_digest,)
                    ).rowcount
                    if inserted == 0:
                        exact_duplicates += 1
                        continue
                    if len(values) == len(KAGGLE_VEHICLE_SALES_HEADER):
                        normalized_vin = _normalize_vin(values[6])
                        if normalized_vin is None:
                            missing_or_invalid_vins += 1
                        else:
                            vin_digest = hmac.digest(
                                secret,
                                normalized_vin.encode("ascii"),
                                "sha256",
                            )
                            connection.execute(
                                "INSERT INTO vin_digests (digest, occurrences) VALUES (?, 1) "
                                "ON CONFLICT(digest) DO UPDATE SET occurrences = occurrences + 1",
                                (vin_digest,),
                            )
                    try:
                        listing = _normalize_row(values, row_number=row_number, review=review)
                    except _RowRejected as rejection:
                        if rejection.reason_code == "market_not_us":
                            non_us_rows += 1
                        quarantined_rows += 1
                        reason_counts[rejection.reason_code] = (
                            reason_counts.get(rejection.reason_code, 0) + 1
                        )
                        quarantine_file.write(
                            json.dumps(
                                {
                                    "source_listing_id": _safe_row_id(row_number),
                                    "row_number": row_number,
                                    "reason_code": rejection.reason_code,
                                    "safe_record_sha256": safe_record_sha256,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        continue
                    writer.writerow(_snapshot_to_csv_row(listing))
                    rows_accepted += 1
                    if rows_seen % 10_000 == 0:
                        connection.commit()
            except csv.Error as error:
                raise KaggleVehicleSalesError(
                    "raw CSV contains unrecoverable CSV syntax"
                ) from error
            candidate_file.flush()
            os.fsync(candidate_file.fileno())
            quarantine_file.flush()
            os.fsync(quarantine_file.fileno())
        connection.commit()
        repeated_vin_rows_value = connection.execute(
            "SELECT COALESCE(SUM(occurrences - 1), 0) FROM vin_digests WHERE occurrences > 1"
        ).fetchone()
        distinct_repeated_value = connection.execute(
            "SELECT COUNT(*) FROM vin_digests WHERE occurrences > 1"
        ).fetchone()
        repeated_vin_rows = _sqlite_integer_result(repeated_vin_rows_value)
        distinct_repeated_vins = _sqlite_integer_result(distinct_repeated_value)
    finally:
        connection.close()
        index_path.unlink(missing_ok=True)
    if rows_seen != rows_accepted + quarantined_rows + exact_duplicates:
        raise KaggleVehicleSalesError("ingestion accounting invariant failed")
    return KaggleIngestionMetrics(
        rows_seen=rows_seen,
        rows_accepted=rows_accepted,
        non_us_rows=non_us_rows,
        quarantined_rows=quarantined_rows,
        exact_duplicate_rows=exact_duplicates,
        repeated_vin_rows=repeated_vin_rows,
        distinct_repeated_vins=distinct_repeated_vins,
        missing_or_invalid_vin_rows=missing_or_invalid_vins,
        quarantine_reason_counts=MappingProxyType(dict(sorted(reason_counts.items()))),
    )


def _normalize_row(
    values: Sequence[str],
    *,
    row_number: int,
    review: KaggleVehicleSalesReview,
) -> VehicleListingSnapshot:
    if len(values) != len(KAGGLE_VEHICLE_SALES_HEADER):
        raise _RowRejected("row_width_invalid")
    state = values[7].strip().upper()
    if not _STATE_PATTERN.fullmatch(state):
        raise _RowRejected("state_invalid")
    if state not in US_50_PLUS_DC:
        raise _RowRejected("market_not_us")
    if _normalize_vin(values[6]) is None:
        raise _RowRejected("vin_missing_or_invalid")

    observed_at = _parse_sale_timestamp(values[15])
    local_sale_date = _sale_local_date(values[15])
    if not review.sale_date_min <= local_sale_date <= review.sale_date_max:
        raise _RowRejected("sale_date_outside_reviewed_range")
    year = _parse_integral(values[0], reason_code="year_invalid")
    if not 1886 <= year <= observed_at.year + 2:
        raise _RowRejected("year_invalid")
    make = _required_source_text(values[1], reason_code="make_missing", max_length=100)
    model = _required_source_text(values[2], reason_code="model_missing", max_length=150)
    trim = _optional_source_text(values[3], reason_code="trim_invalid", max_length=150)
    vehicle_type = _optional_source_text(values[4], reason_code="body_invalid", max_length=100)
    condition = _parse_condition(values[8])
    mileage = _parse_odometer(values[9])
    price_cents = _parse_selling_price_cents(values[14])

    safe_content = {
        "year": year,
        "make": make,
        "model": model,
        "trim": trim,
        "body": vehicle_type,
        "state": state,
        "condition": condition,
        "odometer": mileage,
        "sellingprice_cents": price_cents,
        "saledate_utc": observed_at.isoformat(),
    }
    safe_content_hash = hashlib.sha256(
        json.dumps(safe_content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    ingestion_run_id = f"kvs-{review.expected_sha256[:16]}-{review.review_sha256[:8]}"
    return VehicleListingSnapshot(
        source_id=_SOURCE_ID,
        source_listing_id=_safe_row_id(row_number),
        canonical_url=review.dataset_url,
        observed_at=observed_at,
        market_country="US",
        year=year,
        make=make,
        model=model,
        trim=trim,
        mileage=mileage,
        mileage_unit="miles",
        condition=condition,
        vehicle_status=None,
        engine=None,
        drivetrain=None,
        accident_status=None,
        accident_count=None,
        owner_count=None,
        vehicle_type=vehicle_type,
        price_cents=price_cents,
        currency="USD",
        price_kind=PriceKind.COMPLETED_SALE,
        sale_status="sold",
        raw_content_sha256=safe_content_hash,
        parser_version=_PARSER_VERSION,
        normalization_version=_NORMALIZATION_VERSION,
        ingestion_run_id=ingestion_run_id,
        authorization_policy_id=review.review_id,
    )


def _snapshot_to_csv_row(listing: VehicleListingSnapshot) -> dict[str, object]:
    row = listing.to_dict()
    return {column: "" if row[column] is None else row[column] for column in _CANDIDATE_HEADER}


def _parse_sale_timestamp(value: str) -> datetime:
    match = _SALE_DATE_PATTERN.fullmatch(value)
    if match is None:
        raise _RowRejected("sale_date_invalid")
    (
        weekday,
        month_name,
        day,
        year,
        hour,
        minute,
        second,
        sign,
        offset_hour,
        offset_minute,
        zone,
    ) = match.groups()
    offset_minutes = int(offset_hour) * 60 + int(offset_minute)
    if offset_minutes > 14 * 60 or int(offset_minute) >= 60:
        raise _RowRejected("sale_date_invalid")
    if sign == "-":
        offset_minutes = -offset_minutes
    if _TZ_OFFSETS_MINUTES.get(zone) != offset_minutes:
        raise _RowRejected("sale_date_invalid")
    try:
        local = datetime(
            int(year),
            _MONTHS[month_name],
            int(day),
            int(hour),
            int(minute),
            int(second),
            tzinfo=timezone(timedelta(minutes=offset_minutes)),
        )
    except ValueError as error:
        raise _RowRejected("sale_date_invalid") from error
    if local.weekday() != _WEEKDAYS[weekday]:
        raise _RowRejected("sale_date_invalid")
    return local.astimezone(UTC)


def _sale_local_date(value: str) -> date:
    match = _SALE_DATE_PATTERN.fullmatch(value)
    if match is None:
        raise _RowRejected("sale_date_invalid")
    try:
        return date(int(match.group(4)), _MONTHS[match.group(2)], int(match.group(3)))
    except ValueError as error:
        raise _RowRejected("sale_date_invalid") from error


def _parse_selling_price_cents(value: str) -> int:
    text = value.strip()
    if not _PRICE_PATTERN.fullmatch(text):
        raise _RowRejected("sellingprice_invalid")
    try:
        amount = Decimal(text)
        cents = amount * 100
    except InvalidOperation as error:
        raise _RowRejected("sellingprice_invalid") from error
    if amount <= 0 or cents != cents.to_integral_value():
        raise _RowRejected("sellingprice_invalid")
    return int(cents)


def _parse_odometer(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    odometer = _parse_integral(text, reason_code="odometer_invalid")
    if odometer == 999_999:
        return None
    return odometer


def _parse_integral(value: str, *, reason_code: str) -> int:
    text = value.strip()
    if not _INTEGER_PATTERN.fullmatch(text):
        raise _RowRejected(reason_code)
    try:
        return int(Decimal(text))
    except (InvalidOperation, ValueError) as error:
        raise _RowRejected(reason_code) from error


def _parse_condition(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    if not _CONDITION_PATTERN.fullmatch(text):
        raise _RowRejected("condition_invalid")
    try:
        numeric = Decimal(text)
    except InvalidOperation as error:
        raise _RowRejected("condition_invalid") from error
    if Decimal("1") <= numeric <= Decimal("5"):
        normalized = numeric
    elif numeric == numeric.to_integral_value() and Decimal("11") <= numeric <= Decimal("49"):
        normalized = numeric / Decimal("10")
    else:
        raise _RowRejected("condition_invalid")
    return f"{normalized.quantize(Decimal('0.1')):.1f}"


def _required_source_text(value: str, *, reason_code: str, max_length: int) -> str:
    normalized = _normalize_source_text(value, reason_code=reason_code, max_length=max_length)
    if normalized is None:
        raise _RowRejected(reason_code)
    return normalized


def _optional_source_text(value: str, *, reason_code: str, max_length: int) -> str | None:
    if not value.strip():
        return None
    return _normalize_source_text(value, reason_code=reason_code, max_length=max_length)


def _normalize_source_text(value: str, *, reason_code: str, max_length: int) -> str | None:
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise _RowRejected(reason_code)
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > max_length:
        raise _RowRejected(reason_code)
    if normalized.startswith(("=", "@")) or (
        len(normalized) > 1 and normalized.startswith(("+", "-"))
    ):
        raise _RowRejected("csv_formula_injection")
    return normalized


def _normalize_vin(value: str) -> str | None:
    normalized = value.strip().upper()
    if not normalized or normalized == "NAN" or not _VIN_PATTERN.fullmatch(normalized):
        return None
    return normalized


def _safe_row_id(row_number: int) -> str:
    return f"row-{row_number:09d}"


def _canonical_row_bytes(values: Sequence[str]) -> bytes:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _safe_row_reference_sha256(source_sha256: str, row_number: int) -> str:
    """Hash only the pinned source identity and row number, never private row values."""

    return hashlib.sha256(f"{source_sha256}:row:{row_number}".encode("ascii")).hexdigest()


def _sqlite_integer_result(value: object) -> int:
    if not isinstance(value, tuple) or len(value) != 1 or type(value[0]) is not int or value[0] < 0:
        raise KaggleVehicleSalesError("private deduplication index returned invalid metrics")
    return value[0]


def _verify_raw_artifact(path: Path, review: KaggleVehicleSalesReview) -> None:
    actual_sha256, actual_size = _hash_regular_file(path)
    if actual_size != review.expected_size_bytes:
        raise KaggleVehicleSalesError("raw CSV byte size does not match the source review")
    if actual_sha256 != review.expected_sha256:
        raise KaggleVehicleSalesError("raw CSV SHA-256 does not match the source review")


def _validate_output_targets(
    artifact_paths: Sequence[Path],
    *,
    protected_paths: Sequence[Path],
) -> None:
    normalized_artifacts = [
        os.path.normcase(str(path.resolve(strict=False))) for path in artifact_paths
    ]
    if len(normalized_artifacts) != len(set(normalized_artifacts)):
        raise KaggleVehicleSalesError("derived artifact paths must be distinct")
    normalized_protected = {
        os.path.normcase(str(path.resolve(strict=True))) for path in protected_paths
    }
    if any(path in normalized_protected for path in normalized_artifacts):
        raise KaggleVehicleSalesError("derived artifacts must not overwrite an input file")
    for path in artifact_paths:
        if path.is_symlink():
            raise KaggleVehicleSalesError("derived artifact targets must not be symbolic links")
        if path.exists() and not path.is_file():
            raise KaggleVehicleSalesError("derived artifact targets must be regular files")


def _hash_regular_file(path: Path, *, allow_empty: bool = False) -> tuple[str, int]:
    resolved = _require_regular_file(path, label="artifact")
    before = resolved.stat()
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    after = resolved.stat()
    signature_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    signature_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if signature_before != signature_after or size != after.st_size:
        raise KaggleVehicleSalesError("artifact changed during verification")
    if size == 0 and not allow_empty:
        raise KaggleVehicleSalesError("artifact must not be empty")
    return digest.hexdigest(), size


def _require_reviewed_source_path(path: Path, expected: PurePosixPath) -> None:
    expected_parts = expected.parts
    actual_parts = tuple(part.casefold() for part in path.parts)
    if len(actual_parts) < len(expected_parts) or actual_parts[-len(expected_parts) :] != tuple(
        part.casefold() for part in expected_parts
    ):
        raise KaggleVehicleSalesError("raw CSV is not stored at the reviewed path")


def _validate_pinned_artifact(value: dict[str, object], *, archive: bool) -> None:
    expected_keys = {"recommended_file_name", "size_bytes", "sha256"} if archive else set()
    _require_exact_keys(value, expected_keys, label="archive")
    file_name = _require_text(value["recommended_file_name"], label="archive filename")
    if Path(file_name).name != file_name:
        raise KaggleVehicleSalesError("archive filename must not contain a path")
    _require_positive_int(value["size_bytes"], label="archive size_bytes")
    _require_sha256(value["sha256"], label="archive sha256")


def _require_regular_file(path: Path, *, label: str) -> Path:
    if not isinstance(path, Path) or path.is_symlink():
        raise KaggleVehicleSalesError(f"{label} must be a non-symlink local file")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise KaggleVehicleSalesError(f"{label} is missing or inaccessible") from error
    if not resolved.is_file():
        raise KaggleVehicleSalesError(f"{label} is not a regular file")
    return resolved


def _read_bounded_bytes(path: Path, *, max_bytes: int, label: str) -> bytes:
    before = path.stat()
    if before.st_size <= 0 or before.st_size > max_bytes:
        raise KaggleVehicleSalesError(f"{label} is empty or exceeds its byte limit")
    payload = path.read_bytes()
    after = path.stat()
    signature_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    signature_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if signature_before != signature_after or len(payload) != after.st_size:
        raise KaggleVehicleSalesError(f"{label} changed during read")
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
        raise KaggleVehicleSalesError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise KaggleVehicleSalesError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _require_object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise KaggleVehicleSalesError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _require_exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise KaggleVehicleSalesError(f"{label} fields do not match adapter v1")


def _require_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KaggleVehicleSalesError(f"{label} must be non-empty text")
    return value.strip()


def _require_text_list(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise KaggleVehicleSalesError(f"{label} must be a non-empty list")
    return tuple(_require_text(item, label=label) for item in value)


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise KaggleVehicleSalesError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise KaggleVehicleSalesError(f"{label} must be a nonnegative integer")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    text = _require_text(value, label=label)
    if not _SHA256_PATTERN.fullmatch(text):
        raise KaggleVehicleSalesError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _parse_iso_date(value: object, *, label: str) -> date:
    text = _require_text(value, label=label)
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise KaggleVehicleSalesError(f"{label} must be an ISO date") from error


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
        raise KaggleVehicleSalesError(f"{label} must be a public HTTP(S) URL")
    return text


def _require_kaggle_dataset_url(value: object) -> str:
    text = _require_public_url(value, label="dataset_url")
    parsed = urlsplit(text)
    hostname = parsed.hostname
    if hostname is None or hostname.casefold() not in {"kaggle.com", "www.kaggle.com"}:
        raise KaggleVehicleSalesError("dataset_url must use the official Kaggle host")
    if parsed.path.rstrip("/") not in {
        "/datasets/syedanwarafridi/vehicle-sales-data",
        "/datasets/syedanwarafridi/vehicle-sales-data/data",
    }:
        raise KaggleVehicleSalesError("dataset_url must identify the reviewed Kaggle dataset")
    if parsed.query:
        raise KaggleVehicleSalesError("dataset_url must not contain a query string")
    return text


def _require_safe_relative_path(value: object, *, label: str) -> PurePosixPath:
    text = _require_text(value, label=label)
    if "\\" in text or ":" in text:
        raise KaggleVehicleSalesError(f"{label} must use safe POSIX path separators")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise KaggleVehicleSalesError(f"{label} must be a safe relative POSIX path")
    return path


def _safe_artifact_name(value: object, *, label: str) -> str:
    text = _require_text(value, label=label)
    if Path(text).name != text:
        raise KaggleVehicleSalesError(f"{label} must be a safe filename")
    return text


def _require_hash_match(expected: object, actual: str, *, label: str) -> None:
    if _require_sha256(expected, label=f"{label} hash") != actual:
        raise KaggleVehicleSalesError(f"{label} hash does not match")


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("wb") as file_handle:
        file_handle.write(payload)
        file_handle.flush()
        os.fsync(file_handle.fileno())


def _json_payload(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


__all__ = [
    "KAGGLE_VEHICLE_SALES_HEADER",
    "US_50_PLUS_DC",
    "KaggleCandidateArtifactSet",
    "KaggleIngestionMetrics",
    "KaggleVehicleSalesError",
    "KaggleVehicleSalesReview",
    "load_kaggle_vehicle_sales_review",
    "prepare_kaggle_training_rows",
    "process_kaggle_vehicle_sales_csv",
    "require_kaggle_ml_training_approval",
    "verify_kaggle_candidate_artifact_set",
]
