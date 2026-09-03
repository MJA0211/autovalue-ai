"""Source-independent normalization into the AutoValue AI listing schema."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from urllib.parse import quote, urljoin, urlsplit

from autovalue_ml.acquisition.contracts import (
    PriceKind,
    RejectedListing,
    VehicleListingSnapshot,
)
from autovalue_ml.acquisition.scalar_parsing import parse_price_text_cents

NORMALIZER_VERSION = "mapped-records/1.0.0"


@dataclass(frozen=True, slots=True)
class VehicleFieldMapping:
    """Map one source's reviewed columns to the common listing schema."""

    year: str = "year"
    make: str = "make"
    model: str = "model"
    price: str = "price"
    source_listing_id: str | None = "source_listing_id"
    canonical_url: str | None = "canonical_url"
    trim: str | None = "trim"
    mileage: str | None = "mileage"
    condition: str | None = "condition"
    vehicle_status: str | None = "vehicle_status"
    engine: str | None = "engine"
    drivetrain: str | None = "drivetrain"
    accident_status: str | None = "accident_status"
    accident_count: str | None = "accident_count"
    owner_count: str | None = "owner_count"
    vehicle_type: str | None = "vehicle_type"


@dataclass(frozen=True, slots=True)
class NormalizationContext:
    """Run-level lineage supplied by a reviewed adapter."""

    source_id: str
    source_record_url_prefix: str
    observed_at: datetime
    ingestion_run_id: str
    authorization_policy_id: str
    parser_version: str
    market_country: str = "US"
    currency: str = "USD"
    price_kind: PriceKind = PriceKind.ASKING
    sale_status: str = "active"


@dataclass(frozen=True, slots=True)
class NormalizationBatch:
    """Accepted, quarantined, and deduplicated records from one source batch."""

    listings: tuple[VehicleListingSnapshot, ...]
    rejected_listings: tuple[RejectedListing, ...]
    records_seen: int
    duplicates_skipped: int


def normalize_vehicle_records(
    records: Iterable[Mapping[str, object]],
    *,
    mapping: VehicleFieldMapping,
    context: NormalizationContext,
) -> NormalizationBatch:
    """Normalize mappings independently; bad rows are quarantined, not coerced."""
    accepted: dict[tuple[str, str, str], VehicleListingSnapshot] = {}
    identities: dict[tuple[str, str], VehicleListingSnapshot] = {}
    rejected: list[RejectedListing] = []
    records_seen = 0
    duplicates_skipped = 0

    for row_number, record in enumerate(records, start=1):
        records_seen += 1
        try:
            raw_payload = _canonical_record_bytes(record)
        except ValueError as error:
            fallback_sha256 = hashlib.sha256(
                f"unserializable-row:{row_number}".encode()
            ).hexdigest()
            rejected.append(
                RejectedListing(
                    source_id=context.source_id,
                    page_url=context.source_record_url_prefix,
                    source_listing_id=f"row-{row_number}",
                    observed_at=context.observed_at,
                    reason_code="normalization_failed",
                    message=str(error),
                    raw_content_sha256=fallback_sha256,
                    parser_version=context.parser_version,
                    ingestion_run_id=context.ingestion_run_id,
                    authorization_policy_id=context.authorization_policy_id,
                )
            )
            continue
        raw_sha256 = hashlib.sha256(raw_payload).hexdigest()
        source_listing_id: str | None = None
        try:
            source_listing_id = _source_listing_id(record, mapping, raw_sha256)
            listing = _normalize_record(
                record,
                mapping=mapping,
                context=context,
                source_listing_id=source_listing_id,
                raw_sha256=raw_sha256,
            )
            identity = (listing.source_id, listing.source_listing_id)
            existing_identity = identities.get(identity)
            if existing_identity is not None and existing_identity != listing:
                raise ValueError("source listing ID conflicts with another record in this batch")
            content_key = (*identity, listing.raw_content_sha256)
            if content_key in accepted:
                duplicates_skipped += 1
                continue
            identities[identity] = listing
            accepted[content_key] = listing
        except (InvalidOperation, KeyError, TypeError, ValueError) as error:
            rejected.append(
                RejectedListing(
                    source_id=context.source_id,
                    page_url=context.source_record_url_prefix,
                    source_listing_id=source_listing_id or f"row-{row_number}",
                    observed_at=context.observed_at,
                    reason_code="normalization_failed",
                    message=str(error),
                    raw_content_sha256=raw_sha256,
                    parser_version=context.parser_version,
                    ingestion_run_id=context.ingestion_run_id,
                    authorization_policy_id=context.authorization_policy_id,
                )
            )

    return NormalizationBatch(
        listings=tuple(accepted.values()),
        rejected_listings=tuple(rejected),
        records_seen=records_seen,
        duplicates_skipped=duplicates_skipped,
    )


def _normalize_record(
    record: Mapping[str, object],
    *,
    mapping: VehicleFieldMapping,
    context: NormalizationContext,
    source_listing_id: str,
    raw_sha256: str,
) -> VehicleListingSnapshot:
    canonical_url = _canonical_url(record, mapping, context, source_listing_id)
    return VehicleListingSnapshot(
        source_id=context.source_id,
        source_listing_id=source_listing_id,
        canonical_url=canonical_url,
        observed_at=context.observed_at,
        market_country=context.market_country,
        year=_required_integer(record, mapping.year, field="year"),
        make=_required_text(record, mapping.make, field="make"),
        model=_required_text(record, mapping.model, field="model"),
        trim=_optional_text(record, mapping.trim),
        mileage=_optional_integer(record, mapping.mileage, field="mileage"),
        mileage_unit="miles",
        condition=_optional_text(record, mapping.condition),
        vehicle_status=_optional_vehicle_status(record, mapping.vehicle_status),
        engine=_optional_text(record, mapping.engine),
        drivetrain=_optional_text(record, mapping.drivetrain),
        accident_status=_optional_text(record, mapping.accident_status),
        accident_count=_optional_integer(
            record,
            mapping.accident_count,
            field="accident_count",
        ),
        owner_count=_optional_integer(record, mapping.owner_count, field="owner_count"),
        vehicle_type=_optional_text(record, mapping.vehicle_type),
        price_cents=_price_cents(
            _required_value(record, mapping.price, field="price"),
            expected_currency=context.currency,
            price_kind=context.price_kind,
        ),
        currency=context.currency,
        price_kind=context.price_kind,
        sale_status=context.sale_status,
        raw_content_sha256=raw_sha256,
        parser_version=context.parser_version,
        normalization_version=NORMALIZER_VERSION,
        ingestion_run_id=context.ingestion_run_id,
        authorization_policy_id=context.authorization_policy_id,
    )


def _source_listing_id(
    record: Mapping[str, object], mapping: VehicleFieldMapping, raw_sha256: str
) -> str:
    if mapping.source_listing_id is None:
        return f"derived-{raw_sha256[:24]}"
    value = record.get(mapping.source_listing_id)
    if value is None:
        return f"derived-{raw_sha256[:24]}"
    text = _identifier_text(value)
    return text if text else f"derived-{raw_sha256[:24]}"


def _canonical_url(
    record: Mapping[str, object],
    mapping: VehicleFieldMapping,
    context: NormalizationContext,
    source_listing_id: str,
) -> str:
    value = record.get(mapping.canonical_url) if mapping.canonical_url else None
    if value is None or (isinstance(value, str) and not _normalize_string(value)):
        value = urljoin(
            context.source_record_url_prefix.rstrip("/") + "/",
            quote(source_listing_id, safe="-._~"),
        )
    if not isinstance(value, str):
        raise ValueError("canonical_url must be a string")
    url = _normalize_string(value)
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("canonical_url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("canonical_url cannot contain credentials or a fragment")
    return url


def _required_value(record: Mapping[str, object], key: str, *, field: str) -> object:
    if key not in record or record[key] is None:
        raise KeyError(f"required field is missing: {field}")
    return record[key]


def _required_text(record: Mapping[str, object], key: str, *, field: str) -> str:
    raw_value = _required_value(record, key, field=field)
    if not isinstance(raw_value, str):
        raise ValueError(f"{field} must be a string")
    value = _normalize_string(raw_value)
    if not value:
        raise ValueError(f"required field is empty: {field}")
    return value


def _optional_text(record: Mapping[str, object], key: str | None) -> str | None:
    if key is None or key not in record or record[key] is None:
        return None
    raw_value = record[key]
    if not isinstance(raw_value, str):
        raise ValueError(f"{key} must be a string or null")
    value = _normalize_string(raw_value)
    return value or None


def _optional_vehicle_status(record: Mapping[str, object], key: str | None) -> str | None:
    value = _optional_text(record, key)
    if value is None:
        return None
    normalized = value.casefold()
    if normalized not in {"new", "used", "certified"}:
        raise ValueError("vehicle_status must be New, Used, or Certified")
    return normalized


def _required_integer(record: Mapping[str, object], key: str, *, field: str) -> int:
    return _integer(_required_value(record, key, field=field), field=field)


def _optional_integer(record: Mapping[str, object], key: str | None, *, field: str) -> int | None:
    if key is None or key not in record or record[key] is None:
        return None
    value = record[key]
    if isinstance(value, str) and value.strip().lower() in {"", "unknown", "n/a", "not reported"}:
        return None
    return _integer(value, field=field)


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} cannot be a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field} must be an integer")
        return int(value)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or numeric scalar")
    compact = re.sub(r"[,\s]", "", _normalize_string(value))
    match = re.fullmatch(
        r"(-?\d+)(?:mi|miles?|owners?|accidents?)?",
        compact,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"{field} is not an integer")
    return int(match.group(1))


def _price_cents(
    value: object,
    *,
    expected_currency: str,
    price_kind: PriceKind,
) -> int:
    if isinstance(value, bool):
        raise ValueError("price cannot be a boolean")
    if isinstance(value, (int, float, Decimal)):
        amount = Decimal(str(value))
    elif isinstance(value, str):
        return parse_price_text_cents(
            _normalize_string(value),
            expected_currency=expected_currency,
            price_kind=price_kind,
        )
    else:
        raise ValueError("price must be a string or numeric scalar")
    if not amount.is_finite() or amount <= 0:
        raise ValueError("price must be a positive finite amount")
    scaled = amount * 100
    if scaled != scaled.to_integral_value():
        raise ValueError("price has more than two decimal places")
    return int(scaled)


def _normalize_string(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split())


def _identifier_text(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("source listing ID must be a string or integer")
    return _normalize_string(str(value))


def _canonical_record_bytes(record: Mapping[str, object]) -> bytes:
    try:
        payload = json.dumps(
            dict(record),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("record is not JSON-serializable") from error
    return payload.encode("utf-8")
