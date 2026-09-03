"""Reviewed mappings for the two quarantined Hugging Face candidates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Final

from autovalue_ml.acquisition.contracts import PriceKind
from autovalue_ml.acquisition.huggingface_dataset import (
    ApprovalStatus,
    DatasetUseApprovals,
    HuggingFaceArtifactSpec,
)
from autovalue_ml.acquisition.scalar_parsing import parse_price_text_cents
from autovalue_ml.acquisition.sources.kaggle_vehicle_sales import US_50_PLUS_DC

YOAD_SCHEMA_MAPPING_VERSION: Final = "hf-yoad-craigslist/1.0.0"
CARSON_SCHEMA_MAPPING_VERSION: Final = "hf-carson-used-car-price/1.0.0"
_MILEAGE_PATTERN = re.compile(
    r"^\s*(?P<amount>(?:\d{1,3}(?:,\d{3})+|\d+))\s*(?:mi\.?|miles?)\s*$",
    flags=re.IGNORECASE,
)


YOAD22_CRAIGSLIST_SPEC: Final = HuggingFaceArtifactSpec(
    source_id="hf_yoad22_craigslist_used_cars",
    repo_id="Yoad22/craigslist-used-cars-eda",
    revision="912f968086868effb8523537015fb6a107c8eb3a",
    file_path=PurePosixPath("vehicles_clean.csv"),
    expected_size_bytes=20_749_648,
    expected_sha256="f702408dee80181f1d003e9e7d2173340f26eb6eb013e76e9a47af6296833791",
    expected_row_count=250_361,
    declared_license="CC-BY-4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    upstream_source="Austin Reese Craigslist Cars and Trucks Data v10 (Craigslist US)",
    schema_mapping_version=YOAD_SCHEMA_MAPPING_VERSION,
    approvals=DatasetUseApprovals(
        acquisition=ApprovalStatus.APPROVED,
        batch_training=ApprovalStatus.APPROVED,
        online_learning=ApprovalStatus.BLOCKED,
        acquisition_evidence=(
            "Public ungated Hugging Face artifact reviewed at the pinned revision; "
            "CC-BY-4.0 attribution is retained."
        ),
        batch_training_evidence=(
            "Approved only for controlled, offline batch experimentation after independent "
            "quality, U.S./USD scope, license, lineage, source-overlap, common-feature, and "
            "grouped-split review. This is not production-model approval."
        ),
        online_learning_evidence=(
            "Blocked: the artifact has no stable listing ID or row timestamp and does not define "
            "append-only observations, delayed labels, or replay-safe online-learning semantics."
        ),
    ),
    usage_restrictions=(
        "Attribute Yoad22 and the documented Austin Reese/Craigslist upstream source.",
        "Do not combine with another Austin Reese/Craigslist derivative without a source-level "
        "partition or deduplication decision.",
        "Do not redistribute raw or processed rows from this repository without a new review.",
    ),
    attribution=(
        "Yoad22, craigslist-used-cars-eda, derived from the Austin Reese Craigslist vehicle data; "
        "licensed CC BY 4.0."
    ),
    config="default",
    split="train",
)


CARSON_SHIVELY_SPEC: Final = HuggingFaceArtifactSpec(
    source_id="hf_carson_shively_used_car_price",
    repo_id="Carson-Shively/used-car-price",
    revision="4f58418cafab4dff1bd273aae8c5da66cd2ed3f5",
    file_path=PurePosixPath("data/bronze/bronze.parquet"),
    expected_size_bytes=114_074,
    expected_sha256="b5530c96732db26d05e59d4d02c868a1facb1e1612a7eb1c8ee5d204d497962e",
    expected_row_count=4_009,
    declared_license="MIT stated in dataset card text; repository metadata is incomplete",
    license_url=(
        "https://huggingface.co/datasets/Carson-Shively/used-car-price/blob/"
        "4f58418cafab4dff1bd273aae8c5da66cd2ed3f5/README.md"
    ),
    upstream_source=(
        "Unknown; the repository card does not identify the original row source. The repository's "
        "bronze and silver files are processing layers, not 7,970 independent observations."
    ),
    schema_mapping_version=CARSON_SCHEMA_MAPPING_VERSION,
    approvals=DatasetUseApprovals(
        acquisition=ApprovalStatus.APPROVED,
        batch_training=ApprovalStatus.BLOCKED,
        online_learning=ApprovalStatus.BLOCKED,
        acquisition_evidence=(
            "Public ungated artifact may be retrieved privately for reproducible compatibility "
            "review."
        ),
        batch_training_evidence=(
            "Blocked pending confirmation of upstream provenance, U.S. market scope, USD target "
            "semantics, and license metadata consistency."
        ),
        online_learning_evidence=(
            "Blocked; upstream provenance and observation/label semantics are unresolved."
        ),
    ),
    usage_restrictions=(
        "Private candidate review only until upstream provenance is documented.",
        "Do not treat dollar-formatted price strings as confirmed U.S. market evidence.",
        "Do not concatenate bronze and silver processing layers as independent observations.",
        "Do not redistribute raw or processed rows from this repository without a new review.",
    ),
    attribution="Carson Shively, used-car-price Hugging Face dataset (archived project).",
    config="default",
    split="train",
)

CARSON_SHIVELY_SILVER_AUDIT_SPEC: Final = HuggingFaceArtifactSpec(
    source_id=CARSON_SHIVELY_SPEC.source_id,
    repo_id=CARSON_SHIVELY_SPEC.repo_id,
    revision=CARSON_SHIVELY_SPEC.revision,
    file_path=PurePosixPath("data/silver/silver.parquet"),
    expected_size_bytes=82_060,
    expected_sha256="26ed9d0d159ece7ab68b152e1355503ecd6bba46604523d21fe59fe506a7ffa7",
    expected_row_count=3_961,
    declared_license=CARSON_SHIVELY_SPEC.declared_license,
    license_url=CARSON_SHIVELY_SPEC.license_url,
    upstream_source=(
        "Transformed/filtered silver processing layer derived from the repository's bronze data; "
        "not an independent observation source"
    ),
    schema_mapping_version="hf-carson-used-car-price/silver-audit-1.0.0",
    approvals=DatasetUseApprovals(
        acquisition=ApprovalStatus.APPROVED,
        batch_training=ApprovalStatus.BLOCKED,
        online_learning=ApprovalStatus.BLOCKED,
        acquisition_evidence="Pinned auxiliary file inspected to audit the 7,970-row claim.",
        batch_training_evidence="Blocked because this is a transformed layer of the bronze source.",
        online_learning_evidence="Blocked because this is not an independent observation stream.",
    ),
    usage_restrictions=(
        "Audit-only processing layer; never concatenate with bronze as new observations.",
        "Do not redistribute raw or processed rows from this repository without a new review.",
    ),
    attribution=CARSON_SHIVELY_SPEC.attribution,
    config="default",
    split="train",
)


@dataclass(frozen=True, slots=True)
class CandidateVehicleRecord:
    """Pre-training canonical candidate with immutable local audit metadata."""

    source_id: str
    source_record_id: str
    observed_at: datetime
    market_country: str | None
    market_scope_status: str
    year: int
    make: str
    model: str | None
    mileage: int | None
    condition: str | None
    engine: str | None
    drivetrain: str | None
    accident_status: str | None
    title_status: str | None
    transmission: str | None
    fuel_type: str | None
    vehicle_type: str | None
    state: str | None
    price_cents: int
    currency: str
    currency_status: str
    price_kind: PriceKind
    schema_mapping_version: str
    raw_content_sha256: str
    raw_values: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.market_country not in {None, "US"}:
            raise ValueError("candidate market country must be US or unresolved")
        if self.currency != "USD" or self.price_kind is not PriceKind.ASKING:
            raise ValueError("candidate target must be a USD asking price")
        if not 1886 <= self.year <= self.observed_at.year + 2:
            raise ValueError("candidate year is outside the accepted range")
        if self.mileage is not None and self.mileage < 0:
            raise ValueError("candidate mileage cannot be negative")
        if self.price_cents <= 0:
            raise ValueError("candidate price must be positive")
        if not self.make.strip() or self.model is not None and not self.model.strip():
            raise ValueError("candidate make/model text is invalid")

    def feature_values(self) -> dict[str, str | int]:
        """Return candidate predictors without source identity or audit values."""
        features: dict[str, str | int] = {"year": self.year, "make": self.make}
        optional: tuple[tuple[str, str | int | None], ...] = (
            ("model", self.model),
            ("mileage", self.mileage),
            ("condition", self.condition),
            ("engine", self.engine),
            ("drivetrain", self.drivetrain),
            ("accident_status", self.accident_status),
            ("title_status", self.title_status),
            ("transmission", self.transmission),
            ("fuel_type", self.fuel_type),
            ("vehicle_type", self.vehicle_type),
            ("state", self.state),
        )
        features.update((name, value) for name, value in optional if value is not None)
        return features


@dataclass(frozen=True, slots=True)
class CandidateRecordRejection:
    row_number: int
    reason_code: str
    raw_content_sha256: str


@dataclass(frozen=True, slots=True)
class CandidateNormalizationBatch:
    records: tuple[CandidateVehicleRecord, ...]
    rejections: tuple[CandidateRecordRejection, ...]
    rows_seen: int
    duplicates_skipped: int


def normalize_carson_record(
    record: Mapping[str, object], *, observed_at: datetime
) -> CandidateVehicleRecord:
    """Normalize one Carson row while retaining source values outside features."""
    audit, raw_sha256 = _audit_record(record)
    return CandidateVehicleRecord(
        source_id=CARSON_SHIVELY_SPEC.source_id,
        source_record_id=f"derived-{raw_sha256[:24]}",
        observed_at=observed_at,
        market_country=None,
        market_scope_status="unverified_source_level_scope",
        year=_required_integer(record.get("model_year"), field="model_year"),
        make=_required_text(record.get("brand"), field="brand"),
        model=_required_text(record.get("model"), field="model"),
        mileage=parse_mileage_text(record.get("milage")),
        condition=None,
        engine=_optional_text(record.get("engine")),
        drivetrain=None,
        accident_status=normalize_accident_status(record.get("accident")),
        title_status=normalize_clean_title(record.get("clean_title")),
        transmission=_optional_text(record.get("transmission")),
        fuel_type=_optional_text(record.get("fuel_type")),
        vehicle_type=None,
        state=None,
        price_cents=parse_price_text_cents(
            _required_text(record.get("price"), field="price"),
            expected_currency="USD",
            price_kind=PriceKind.ASKING,
        ),
        currency="USD",
        currency_status="parsed_from_dollar_format_but_market_scope_unverified",
        price_kind=PriceKind.ASKING,
        schema_mapping_version=CARSON_SCHEMA_MAPPING_VERSION,
        raw_content_sha256=raw_sha256,
        raw_values=audit,
    )


def normalize_yoad_record(
    record: Mapping[str, object], *, observed_at: datetime
) -> CandidateVehicleRecord:
    """Normalize one Yoad/Austin-Reese derivative without inventing a model."""
    audit, raw_sha256 = _audit_record(record)
    state = _optional_text(record.get("state"))
    if state is not None:
        state = state.upper()
        if state not in US_50_PLUS_DC:
            raise ValueError("state is outside the 50 U.S. states and Washington, D.C.")
    return CandidateVehicleRecord(
        source_id=YOAD22_CRAIGSLIST_SPEC.source_id,
        source_record_id=f"derived-{raw_sha256[:24]}",
        observed_at=observed_at,
        market_country="US",
        market_scope_status="row_state_or_reviewed_us_craigslist_source",
        year=_required_integer(record.get("year"), field="year"),
        make=_required_text(record.get("manufacturer"), field="manufacturer"),
        model=None,
        mileage=_optional_integer(record.get("odometer"), field="odometer"),
        condition=_optional_text(record.get("condition")),
        engine=_optional_text(record.get("cylinders")),
        drivetrain=_optional_text(record.get("drive")),
        accident_status=None,
        title_status=_optional_text(record.get("title_status")),
        transmission=_optional_text(record.get("transmission")),
        fuel_type=_optional_text(record.get("fuel")),
        vehicle_type=_optional_text(record.get("type")),
        state=state,
        price_cents=_numeric_price_cents(record.get("price")),
        currency="USD",
        currency_status="reviewed_us_craigslist_source_semantics",
        price_kind=PriceKind.ASKING,
        schema_mapping_version=YOAD_SCHEMA_MAPPING_VERSION,
        raw_content_sha256=raw_sha256,
        raw_values=audit,
    )


def normalize_candidate_records(
    records: Iterable[Mapping[str, object]],
    *,
    source_id: str,
    observed_at: datetime,
) -> CandidateNormalizationBatch:
    """Normalize and exact-deduplicate one source without granting ML reuse."""
    if source_id == YOAD22_CRAIGSLIST_SPEC.source_id:
        normalizer = normalize_yoad_record
    elif source_id == CARSON_SHIVELY_SPEC.source_id:
        normalizer = normalize_carson_record
    else:
        raise ValueError("unreviewed Hugging Face source_id")

    accepted: dict[str, CandidateVehicleRecord] = {}
    rejected: list[CandidateRecordRejection] = []
    rows_seen = 0
    duplicates = 0
    for row_number, record in enumerate(records, start=1):
        rows_seen += 1
        _, fallback_sha256 = _audit_record(record)
        try:
            candidate = normalizer(record, observed_at=observed_at)
        except (InvalidOperation, TypeError, ValueError) as error:
            rejected.append(
                CandidateRecordRejection(
                    row_number=row_number,
                    reason_code=_reason_code(error),
                    raw_content_sha256=fallback_sha256,
                )
            )
            continue
        if candidate.raw_content_sha256 in accepted:
            duplicates += 1
            continue
        accepted[candidate.raw_content_sha256] = candidate
    return CandidateNormalizationBatch(
        records=tuple(accepted.values()),
        rejections=tuple(rejected),
        rows_seen=rows_seen,
        duplicates_skipped=duplicates,
    )


def parse_mileage_text(value: object) -> int | None:
    """Parse Carson strings such as ``51,000 mi.`` without partial matches."""
    if _is_missing(value):
        return None
    if not isinstance(value, str):
        raise ValueError("milage must be text")
    match = _MILEAGE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("milage does not match the reviewed miles format")
    return int(match.group("amount").replace(",", ""))


def normalize_accident_status(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = _required_text(value, field="accident").casefold()
    if "none reported" in text or text in {"none", "no"}:
        return "none_reported"
    if "accident" in text or "damage" in text:
        return "accident_or_damage_reported"
    return "unknown"


def normalize_clean_title(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = _required_text(value, field="clean_title").casefold()
    if text in {"yes", "y", "true", "clean"}:
        return "clean"
    if text in {"no", "n", "false", "not clean"}:
        return "not_clean"
    return "unknown"


def _audit_record(record: Mapping[str, object]) -> tuple[Mapping[str, object], str]:
    audit_values = {str(key): _audit_value(value) for key, value in record.items()}
    payload = json.dumps(
        audit_values, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return MappingProxyType(audit_values), hashlib.sha256(payload).hexdigest()


def _audit_value(value: object) -> object:
    if _is_missing(value):
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value
    return str(value)


def _is_missing(value: object) -> bool:
    return value is None or isinstance(value, float) and math.isnan(value)


def _required_text(value: object, *, field: str) -> str:
    if _is_missing(value):
        raise ValueError(f"{field} is missing")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} is missing")
    return text


def _optional_text(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _required_integer(value: object, *, field: str) -> int:
    if _is_missing(value) or isinstance(value, bool):
        raise ValueError(f"{field} is missing or invalid")
    try:
        parsed = Decimal(str(value).strip())
    except InvalidOperation as error:
        raise ValueError(f"{field} is invalid") from error
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise ValueError(f"{field} is invalid")
    return int(parsed)


def _optional_integer(value: object, *, field: str) -> int | None:
    if _is_missing(value) or isinstance(value, str) and not value.strip():
        return None
    parsed = _required_integer(value, field=field)
    if parsed < 0:
        raise ValueError(f"{field} cannot be negative")
    return parsed


def _numeric_price_cents(value: object) -> int:
    if _is_missing(value) or isinstance(value, bool):
        raise ValueError("price is missing")
    try:
        amount = Decimal(str(value).strip())
    except InvalidOperation as error:
        raise ValueError("price is invalid") from error
    if not amount.is_finite() or amount <= 0:
        raise ValueError("price must be positive")
    cents = amount * 100
    if cents != cents.to_integral_value():
        raise ValueError("price has more than two decimal places")
    return int(cents)


def _reason_code(error: Exception) -> str:
    message = str(error)
    for token in (
        "price",
        "year",
        "brand",
        "manufacturer",
        "model",
        "milage",
        "odometer",
        "state",
    ):
        if token in message:
            return f"invalid_{token}"
    return "normalization_failed"


__all__ = [
    "CARSON_SCHEMA_MAPPING_VERSION",
    "CARSON_SHIVELY_SILVER_AUDIT_SPEC",
    "CARSON_SHIVELY_SPEC",
    "CandidateNormalizationBatch",
    "CandidateRecordRejection",
    "CandidateVehicleRecord",
    "YOAD22_CRAIGSLIST_SPEC",
    "YOAD_SCHEMA_MAPPING_VERSION",
    "normalize_accident_status",
    "normalize_candidate_records",
    "normalize_carson_record",
    "normalize_clean_title",
    "normalize_yoad_record",
    "parse_mileage_text",
]
