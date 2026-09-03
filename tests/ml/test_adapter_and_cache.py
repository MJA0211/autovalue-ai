"""Tests for reusable adapter registration and bounded response caching."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date

import httpx
import pytest
from autovalue_ml.acquisition.adapter import AdapterRegistry, ReviewedScrapingAdapter
from autovalue_ml.acquisition.cache import MemoryResponseCache
from autovalue_ml.acquisition.contracts import ScrapeResult
from autovalue_ml.acquisition.errors import PolicyViolationError
from autovalue_ml.acquisition.parser import SyntheticMarketplaceParser
from autovalue_ml.acquisition.scraper import SafeVehicleScraper

from tests.ml.test_acquisition import (
    _OBSERVED_AT,
    _TODAY,
    _policy,
    _response,
    _robots_response,
    _synthetic_page,
    _unexpected_resolver,
)


def test_registry_reuses_a_reviewed_parser_policy_bundle_without_ambiguity() -> None:
    policy = _policy()
    adapter = ReviewedScrapingAdapter(
        adapter_id="synthetic-dealership-v1",
        adapter_version="1.0.0",
        start_path="/inventory",
        policy=policy,
        parser=SyntheticMarketplaceParser(),
    )
    registry = AdapterRegistry(today=lambda: _TODAY)

    registry.register(adapter)

    assert registry.get("synthetic-dealership-v1") is adapter
    assert registry.list_ids() == ("synthetic-dealership-v1",)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(adapter)
    with pytest.raises(KeyError, match="unknown acquisition adapter"):
        registry.get("missing-adapter")


def test_registry_rejects_adapter_with_expired_scraping_permission() -> None:
    policy = _policy()
    expired_permission = replace(
        policy.scraping_permission,
        expires_on=date(2026, 8, 26),
    )
    adapter = ReviewedScrapingAdapter(
        adapter_id="expired-synthetic-v1",
        adapter_version="1.0.0",
        start_path="/inventory",
        policy=replace(policy, scraping_permission=expired_permission),
        parser=SyntheticMarketplaceParser(),
    )
    registry = AdapterRegistry(today=lambda: _TODAY)

    with pytest.raises(PolicyViolationError, match="authorization has expired"):
        registry.register(adapter)

    assert registry.list_ids() == ()


def test_memory_cache_expires_entries_and_enforces_integrity() -> None:
    clock = [0.0]
    cache = MemoryResponseCache(
        max_entries=2,
        max_bytes=1_024,
        monotonic=lambda: clock[0],
    )

    assert cache.put("page", "vehicle data", ttl_seconds=10) is True
    assert cache.get("page") == "vehicle data"
    clock[0] = 10.0
    assert cache.get("page") is None
    assert cache.put("oversized", "x" * 1_025, ttl_seconds=10) is False
    with pytest.raises(ValueError, match="finite"):
        cache.put("never-expire", "vehicle data", ttl_seconds=math.nan)


def test_second_scrape_uses_cached_pages_but_refetches_robots() -> None:
    policy = replace(
        _policy(),
        raw_html_retention_days=1,
        cache_ttl_seconds=60,
        max_cache_bytes=50_000,
    )
    cache = MemoryResponseCache(max_bytes=policy.max_cache_bytes, monotonic=lambda: 0.0)
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(request, body=_synthetic_page())

    def run_once() -> ScrapeResult:
        with SafeVehicleScraper(
            policy,
            SyntheticMarketplaceParser(),
            transport=httpx.MockTransport(handler),
            cache=cache,
            sleep=lambda _seconds: None,
            monotonic=lambda: 0.0,
            now=lambda: _OBSERVED_AT,
            today=lambda: _TODAY,
            resolver=_unexpected_resolver,
        ) as scraper:
            return scraper.scrape(start_path="/inventory")

    first = run_once()
    second = run_once()

    assert first.cache_hits == 0
    assert first.cache_misses == 1
    assert second.cache_hits == 1
    assert second.cache_misses == 0
    assert second.requests_made == 1
    assert requested_paths == ["/robots.txt", "/inventory", "/robots.txt"]
    assert second.cache_backend == "memory"
    assert second.cache_persistent is False
    assert second.cache_max_bytes == policy.max_cache_bytes


def test_scraper_rejects_cache_capacity_above_the_reviewed_policy() -> None:
    policy = replace(_policy(), max_cache_bytes=10_000)
    cache = MemoryResponseCache(max_bytes=10_001)

    with (
        SafeVehicleScraper(
            policy,
            SyntheticMarketplaceParser(),
            transport=httpx.MockTransport(lambda request: _robots_response(request)),
            cache=cache,
            today=lambda: _TODAY,
        ) as scraper,
        pytest.raises(PolicyViolationError, match="cache exceeds"),
    ):
        scraper.scrape(start_path="/inventory")
