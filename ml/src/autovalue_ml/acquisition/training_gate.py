"""Explicit bridge from approved acquisition records to future ML consumers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Any

from autovalue_ml.acquisition.contracts import PriceKind, ScrapeResult, VehicleListingSnapshot
from autovalue_ml.acquisition.policy import SourcePolicy
from autovalue_ml.acquisition.provenance import validate_scrape_result_provenance

TRAINING_EVENT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TrainingRecordEvent:
    """Append-only record shaped for River's ``learn_one(x, y)`` interface."""

    event_id: str
    schema_version: int
    emitted_at: datetime
    source_id: str
    source_listing_id: str
    market_country: str
    ingestion_run_id: str
    observed_at: datetime
    policy_sha256: str
    ml_reuse_permission_sha256: str
    raw_content_sha256: str
    parser_version: str
    normalization_version: str
    content_dedup_key: str
    features: Mapping[str, str | int | float]
    target_price: float
    target_currency: str
    target_kind: PriceKind

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "schema_version": self.schema_version,
            "emitted_at": self.emitted_at.isoformat(),
            "source_id": self.source_id,
            "source_listing_id": self.source_listing_id,
            "market_country": self.market_country,
            "ingestion_run_id": self.ingestion_run_id,
            "observed_at": self.observed_at.isoformat(),
            "policy_sha256": self.policy_sha256,
            "ml_reuse_permission_sha256": self.ml_reuse_permission_sha256,
            "raw_content_sha256": self.raw_content_sha256,
            "parser_version": self.parser_version,
            "normalization_version": self.normalization_version,
            "content_dedup_key": self.content_dedup_key,
            "features": dict(self.features),
            "target_price": self.target_price,
            "target_currency": self.target_currency,
            "target_kind": self.target_kind.value,
        }

    def to_river_example(self) -> tuple[dict[str, str | int | float], float]:
        """Return the exact feature/target pair accepted by ``River.learn_one``."""
        return dict(self.features), self.target_price


@dataclass(frozen=True, slots=True)
class TrainingExclusion:
    source_listing_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class TrainingEventBatch:
    events: tuple[TrainingRecordEvent, ...]
    exclusions: tuple[TrainingExclusion, ...]
    source_policy_sha256: str
    ml_reuse_permission_sha256: str


def build_training_event_batch(
    result: ScrapeResult,
    policy: SourcePolicy,
    *,
    today: date,
    allowed_price_kinds: frozenset[PriceKind] = frozenset({PriceKind.ASKING}),
    required_currency: str = "USD",
) -> TrainingEventBatch:
    """Build events only after a separate, current ML-reuse approval check."""
    validate_scrape_result_provenance(result, policy)
    policy.validate_for_ml_reuse(today=today)
    if not allowed_price_kinds:
        raise ValueError("at least one target price kind must be explicitly approved")
    if required_currency != "USD":
        raise ValueError("required_currency must be USD for the US market")

    events: list[TrainingRecordEvent] = []
    exclusions: list[TrainingExclusion] = []
    ml_permission_sha256 = policy.ml_reuse_permission_fingerprint()
    for listing in result.listings:
        reason = _training_exclusion_reason(
            listing,
            allowed_price_kinds=allowed_price_kinds,
            required_currency=required_currency,
        )
        if reason is not None:
            exclusions.append(
                TrainingExclusion(
                    source_listing_id=listing.source_listing_id,
                    reason=reason,
                )
            )
            continue
        features = _river_features(listing)
        content_dedup_key = _sha256_json(
            {
                "source_id": listing.source_id,
                "source_listing_id": listing.source_listing_id,
                "market_country": listing.market_country,
                "raw_content_sha256": listing.raw_content_sha256,
                "parser_version": listing.parser_version,
                "normalization_version": listing.normalization_version,
                "features": features,
                "target_price_cents": listing.price_cents,
                "target_currency": listing.currency,
                "target_kind": listing.price_kind.value,
            }
        )
        event_payload = {
            "schema_version": TRAINING_EVENT_SCHEMA_VERSION,
            "content_dedup_key": content_dedup_key,
            "emitted_at": result.completed_at.isoformat(),
            "observed_at": listing.observed_at.isoformat(),
            "ingestion_run_id": listing.ingestion_run_id,
            "policy_sha256": result.policy_sha256,
            "ml_reuse_permission_sha256": ml_permission_sha256,
        }
        events.append(
            TrainingRecordEvent(
                event_id=_sha256_json(event_payload),
                schema_version=TRAINING_EVENT_SCHEMA_VERSION,
                emitted_at=result.completed_at,
                source_id=listing.source_id,
                source_listing_id=listing.source_listing_id,
                market_country=listing.market_country,
                ingestion_run_id=listing.ingestion_run_id,
                observed_at=listing.observed_at,
                policy_sha256=result.policy_sha256,
                ml_reuse_permission_sha256=ml_permission_sha256,
                raw_content_sha256=listing.raw_content_sha256,
                parser_version=listing.parser_version,
                normalization_version=listing.normalization_version,
                content_dedup_key=content_dedup_key,
                features=MappingProxyType(features),
                target_price=listing.price_cents / 100,
                target_currency=listing.currency,
                target_kind=listing.price_kind,
            )
        )

    return TrainingEventBatch(
        events=tuple(events),
        exclusions=tuple(exclusions),
        source_policy_sha256=result.policy_sha256,
        ml_reuse_permission_sha256=ml_permission_sha256,
    )


def iter_river_examples(
    batch: TrainingEventBatch,
) -> Iterator[tuple[dict[str, str | int | float], float]]:
    """Yield records lazily without importing or updating River itself."""
    for event in batch.events:
        yield event.to_river_example()


def _training_exclusion_reason(
    listing: VehicleListingSnapshot,
    *,
    allowed_price_kinds: frozenset[PriceKind],
    required_currency: str,
) -> str | None:
    if listing.price_kind not in allowed_price_kinds:
        return f"price_kind_not_approved:{listing.price_kind.value}"
    if listing.currency != required_currency:
        return f"currency_not_approved:{listing.currency}"
    return None


def _river_features(listing: VehicleListingSnapshot) -> dict[str, str | int | float]:
    features: dict[str, str | int | float | None] = {
        "year": listing.year,
        "vehicle_age": max(0, listing.observed_at.year - listing.year),
        "make": listing.make,
        "model": listing.model,
        "trim": listing.trim,
        "mileage": listing.mileage,
        "condition": listing.condition,
        "vehicle_status": listing.vehicle_status,
        "engine": listing.engine,
        "drivetrain": listing.drivetrain,
        "accident_status": listing.accident_status,
        "accident_count": listing.accident_count,
        "owner_count": listing.owner_count,
        "vehicle_type": listing.vehicle_type,
    }
    return {name: value for name, value in features.items() if value is not None}


def _sha256_json(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
