"""Tests for the explicit batch/online ML reuse boundary."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime

import pytest
from autovalue_ml.acquisition.contracts import PriceKind, ScrapeResult, VehicleListingSnapshot
from autovalue_ml.acquisition.errors import PolicyViolationError
from autovalue_ml.acquisition.training_gate import (
    build_training_event_batch,
    iter_river_examples,
)

from tests.ml.test_acquisition import _ml_permission, _policy

_TODAY = date(2026, 8, 27)
_NOW = datetime(2026, 8, 27, tzinfo=UTC)


def _listing(
    *, currency: str = "USD", price_kind: PriceKind = PriceKind.ASKING
) -> VehicleListingSnapshot:
    return VehicleListingSnapshot(
        source_id="autovalue-synthetic-marketplace",
        source_listing_id=f"listing-{currency}-{price_kind.value}",
        canonical_url="http://127.0.0.1:8765/vehicles/listing-001",
        observed_at=_NOW,
        market_country="US",
        year=2021,
        make="Toyota",
        model="Camry",
        trim="SE",
        mileage=34_120,
        mileage_unit="miles",
        condition="Good",
        engine="2.5L I4",
        drivetrain="FWD",
        accident_status="None reported",
        accident_count=0,
        owner_count=1,
        vehicle_type="Sedan",
        price_cents=2_390_000,
        currency=currency,
        price_kind=price_kind,
        sale_status="active",
        raw_content_sha256="a" * 64,
        parser_version="synthetic/1",
        normalization_version="1",
        ingestion_run_id="run-001",
        authorization_policy_id="synthetic-marketplace-v1",
    )


def _result(policy_sha256: str, listings: tuple[VehicleListingSnapshot, ...]) -> ScrapeResult:
    return ScrapeResult(
        source_id="autovalue-synthetic-marketplace",
        policy_id="synthetic-marketplace-v1",
        policy_sha256=policy_sha256,
        ingestion_run_id="run-001",
        authorization_date=_TODAY,
        started_at=_NOW,
        completed_at=_NOW,
        pages_fetched=1,
        requests_made=2,
        retries=0,
        response_bytes=1_000,
        robots_url="http://127.0.0.1:8765/robots.txt",
        robots_sha256="b" * 64,
        duplicates_skipped=0,
        cache_hits=0,
        cache_misses=0,
        cache_backend="disabled",
        cache_persistent=False,
        cache_max_bytes=None,
        rejected_listings=(),
        http_status_counts=((200, 2),),
        listings=listings,
    )


def test_separate_ml_permission_blocks_events_after_successful_acquisition() -> None:
    policy = replace(_policy(), ml_training_permission=_ml_permission(approved=False))
    result = _result(policy.fingerprint(), (_listing(),))

    with pytest.raises(PolicyViolationError, match="not approved"):
        build_training_event_batch(result, policy, today=_TODAY)


def test_later_ml_approval_does_not_rewrite_acquisition_lineage() -> None:
    collection_policy = replace(_policy(), ml_training_permission=_ml_permission(approved=False))
    result = _result(collection_policy.fingerprint(), (_listing(),))
    approved_policy = replace(collection_policy, ml_training_permission=_ml_permission())

    assert collection_policy.fingerprint() == approved_policy.fingerprint()
    assert (
        collection_policy.ml_reuse_permission_fingerprint()
        != approved_policy.ml_reuse_permission_fingerprint()
    )
    assert len(build_training_event_batch(result, approved_policy, today=_TODAY).events) == 1


def test_ml_reuse_is_independent_of_the_current_collection_switch() -> None:
    collection_policy = _policy()
    result = _result(collection_policy.fingerprint(), (_listing(),))
    disabled_policy = replace(collection_policy, enabled=False)

    assert collection_policy.fingerprint() == disabled_policy.fingerprint()
    assert len(build_training_event_batch(result, disabled_policy, today=_TODAY).events) == 1


def test_lawfully_acquired_result_survives_later_collection_grant_expiry() -> None:
    policy = _policy()
    expiring_policy = replace(
        policy,
        scraping_permission=replace(policy.scraping_permission, expires_on=_TODAY),
    )
    result = _result(expiring_policy.fingerprint(), (_listing(),))

    batch = build_training_event_batch(
        result,
        expiring_policy,
        today=date(2026, 8, 28),
    )

    assert len(batch.events) == 1


def test_builds_deterministic_river_compatible_examples_only_when_approved() -> None:
    policy = _policy()
    result = _result(policy.fingerprint(), (_listing(),))

    first = build_training_event_batch(result, policy, today=_TODAY)
    second = build_training_event_batch(result, policy, today=_TODAY)
    examples = list(iter_river_examples(first))

    assert first == second
    assert len(first.events) == 1
    assert first.exclusions == ()
    assert len(first.events[0].event_id) == 64
    assert len(first.events[0].content_dedup_key) == 64
    assert first.events[0].ml_reuse_permission_sha256 == policy.ml_reuse_permission_fingerprint()
    assert first.ml_reuse_permission_sha256 == policy.ml_reuse_permission_fingerprint()
    assert examples == [
        (
            {
                "year": 2021,
                "vehicle_age": 5,
                "make": "Toyota",
                "model": "Camry",
                "trim": "SE",
                "mileage": 34_120,
                "condition": "Good",
                "engine": "2.5L I4",
                "drivetrain": "FWD",
                "accident_status": "None reported",
                "accident_count": 0,
                "owner_count": 1,
                "vehicle_type": "Sedan",
            },
            23_900.0,
        )
    ]


def test_unapproved_price_semantics_are_excluded_with_reasons() -> None:
    policy = _policy()
    result = _result(
        policy.fingerprint(),
        (_listing(price_kind=PriceKind.MONTHLY_PAYMENT), _listing()),
    )

    batch = build_training_event_batch(result, policy, today=_TODAY)

    assert len(batch.events) == 1
    assert [exclusion.reason for exclusion in batch.exclusions] == [
        "price_kind_not_approved:monthly_payment"
    ]


def test_foreign_nested_listing_is_rejected_before_event_creation() -> None:
    policy = _policy()
    foreign_listing = replace(_listing(), source_id="unapproved-source")
    result = _result(policy.fingerprint(), (foreign_listing,))

    with pytest.raises(PolicyViolationError, match="listing provenance"):
        build_training_event_batch(result, policy, today=_TODAY)


def test_event_identity_distinguishes_runs_but_content_key_deduplicates_content() -> None:
    policy = _policy()
    first_result = _result(policy.fingerprint(), (_listing(),))
    second_listing = replace(_listing(), ingestion_run_id="run-002")
    second_result = replace(
        first_result,
        ingestion_run_id="run-002",
        listings=(second_listing,),
    )

    first_event = build_training_event_batch(first_result, policy, today=_TODAY).events[0]
    second_event = build_training_event_batch(second_result, policy, today=_TODAY).events[0]

    assert first_event.event_id != second_event.event_id
    assert first_event.content_dedup_key == second_event.content_dedup_key


def test_training_gate_rejects_an_invalid_required_currency_contract() -> None:
    policy = _policy()
    result = _result(policy.fingerprint(), (_listing(),))

    for currency in ("usd", "CAD"):
        with pytest.raises(ValueError, match="required_currency must be USD"):
            build_training_event_batch(result, policy, today=_TODAY, required_currency=currency)


def test_training_gate_rejects_fabricated_metrics_over_the_acquisition_policy() -> None:
    policy = _policy()
    result = replace(
        _result(policy.fingerprint(), (_listing(),)),
        duplicates_skipped=policy.limits.max_records,
    )

    with pytest.raises(PolicyViolationError, match="metrics exceed"):
        build_training_event_batch(result, policy, today=_TODAY)
