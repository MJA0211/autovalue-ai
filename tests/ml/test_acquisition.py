"""Synthetic-only tests for the permission-gated acquisition boundary."""

from __future__ import annotations

import hashlib
import json
import math
import socket
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest
from autovalue_ml.acquisition.contracts import ParsedPage, PriceKind, ScrapeResult
from autovalue_ml.acquisition.demo import DEMO_TERMS_SHA256, build_demo_policy
from autovalue_ml.acquisition.errors import (
    ContentValidationError,
    CrawlBudgetExceededError,
    DuplicateListingConflictError,
    FetchError,
    ListingParseError,
    PaginationLoopError,
    PolicyViolationError,
    RobotsDeniedError,
)
from autovalue_ml.acquisition.parser import SyntheticMarketplaceParser
from autovalue_ml.acquisition.policy import (
    CrawlLimits,
    PermissionGrant,
    Purpose,
    SourcePolicy,
)
from autovalue_ml.acquisition.scraper import SafeVehicleScraper
from autovalue_ml.acquisition.writer import verify_scrape_artifact_set, write_scrape_result

_TODAY = date(2026, 8, 27)
_OBSERVED_AT = datetime(2026, 8, 27, 14, 30, tzinfo=UTC)
_PORT = 8765
_BASE_URL = f"http://127.0.0.1:{_PORT}/inventory/"


def _synthetic_page(
    *,
    price: str = "$24,995.50",
    price_kind: str = "asking",
    next_url: str | None = None,
    listing_url: str = "/vehicles/syn-001",
) -> str:
    next_link = f'<a rel="next" href="{next_url}">Next</a>' if next_url else ""
    return f"""<!doctype html>
<html lang="en">
  <body>
    <article class="vehicle-card" data-listing-id="syn-001">
      <a class="vehicle-card__link" href="{listing_url}">Details</a>
      <span data-field="year">2021</span>
      <span data-field="make"> Subaru </span>
      <span data-field="model">Outback</span>
      <span data-field="trim">Limited XT</span>
      <span data-field="mileage">42,125 miles</span>
      <span data-field="condition">Excellent</span>
      <span data-field="engine">2.4L Turbo H4</span>
      <span data-field="drivetrain">All-Wheel Drive</span>
      <span data-field="accident-status">Reported</span>
      <span data-field="accident-count">1 accident</span>
      <span data-field="owner-count">2 owners</span>
      <span data-field="vehicle-type">SUV</span>
      <span data-field="price" data-currency="USD" data-price-kind="{price_kind}">
        {price}
      </span>
    </article>
    {next_link}
  </body>
</html>
"""


def _parse_synthetic_page(html: str) -> ParsedPage:
    return SyntheticMarketplaceParser().parse_page(
        html,
        page_url="http://127.0.0.1:8765/inventory",
        source_id="autovalue-synthetic-marketplace",
        observed_at=_OBSERVED_AT,
        ingestion_run_id="run-synthetic-001",
        authorization_policy_id="synthetic-marketplace-v1",
    )


def _policy(
    *,
    enabled: bool = True,
    permission: PermissionGrant | None = None,
    limits: CrawlLimits | None = None,
    allowed_fields: frozenset[str] | None = None,
) -> SourcePolicy:
    parser = SyntheticMarketplaceParser()
    return SourcePolicy(
        policy_id="synthetic-marketplace-v1",
        source_id="autovalue-synthetic-marketplace",
        source_owner="AutoValue AI repository maintainers",
        market_country="US",
        base_url=_BASE_URL,
        allowed_hosts=frozenset({"127.0.0.1"}),
        allowed_ports=frozenset({_PORT}),
        allowed_path_prefixes=("/inventory", "/vehicles", "/robots.txt", "/terms.html"),
        allowed_query_parameters=frozenset({"page"}),
        allowed_fields=allowed_fields if allowed_fields is not None else parser.output_fields,
        terms_url=f"http://127.0.0.1:{_PORT}/terms.html",
        terms_version="project-owned-fixture-v1",
        user_agent="AutoValueAIResearchBot/0.1 (synthetic unit tests)",
        scraping_permission=permission or _permission(),
        ml_training_permission=_ml_permission(),
        limits=limits or _limits(),
        raw_html_retention_days=0,
        demo_only=True,
        enabled=enabled,
    )


def _external_policy() -> SourcePolicy:
    parser = SyntheticMarketplaceParser()
    return SourcePolicy(
        policy_id="authorized-external-v1",
        source_id="authorized-external",
        source_owner="Authorized Example",
        market_country="US",
        base_url="https://authorized.example/inventory/",
        allowed_hosts=frozenset({"authorized.example"}),
        allowed_ports=frozenset({443}),
        allowed_path_prefixes=("/inventory", "/vehicles", "/robots.txt", "/terms.html"),
        allowed_query_parameters=frozenset({"page"}),
        allowed_fields=parser.output_fields,
        terms_url="https://authorized.example/terms.html",
        terms_version="written-authorization-v1",
        user_agent="AutoValueAIResearchBot/0.1 (security unit tests)",
        scraping_permission=_permission(),
        ml_training_permission=_ml_permission(),
        limits=replace(_limits(), request_delay_seconds=1),
        raw_html_retention_days=0,
        demo_only=False,
        enabled=True,
    )


def _permission() -> PermissionGrant:
    return PermissionGrant(
        approved=True,
        basis="project_owned_synthetic_fixture",
        evidence_reference="tests/ml/synthetic-inline-html",
        evidence_sha256="a" * 64,
        effective_on=date(2026, 1, 1),
        expires_on=None,
        approved_purposes=frozenset({Purpose.COLLECTION, Purpose.STORAGE}),
        contact="AutoValue AI repository maintainers",
    )


def _ml_permission(*, approved: bool = True) -> PermissionGrant:
    return PermissionGrant(
        approved=approved,
        basis="project_owned_synthetic_fixture",
        evidence_reference="tests/ml/synthetic-inline-html",
        evidence_sha256="b" * 64,
        effective_on=date(2026, 1, 1),
        expires_on=None,
        approved_purposes=frozenset({Purpose.ML_TRAINING, Purpose.PUBLIC_PORTFOLIO}),
        contact="AutoValue AI repository maintainers",
    )


def _limits() -> CrawlLimits:
    return CrawlLimits(
        request_delay_seconds=0,
        max_pages=3,
        max_records=10,
        max_requests=10,
        max_response_bytes=200_000,
        max_runtime_seconds=30,
        max_retries=2,
        max_retry_after_seconds=1,
    )


def _response(
    request: httpx.Request,
    *,
    status: int = 200,
    body: str = "",
    media_type: str = "text/html",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    response_headers = {"content-type": f"{media_type}; charset=utf-8"}
    if headers:
        response_headers.update(headers)
    return httpx.Response(
        status,
        content=body.encode("utf-8"),
        headers=response_headers,
        request=request,
    )


def _robots_response(
    request: httpx.Request, body: str = "User-agent: *\nAllow: /\n"
) -> httpx.Response:
    return _response(request, body=body, media_type="text/plain")


def _scrape(
    policy: SourcePolicy,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    sleep: Callable[[float], None] = lambda _seconds: None,
    monotonic: Callable[[], float] = lambda: 0.0,
    now: Callable[[], datetime] = lambda: _OBSERVED_AT,
    today: Callable[[], date] = lambda: _TODAY,
) -> ScrapeResult:
    transport = httpx.MockTransport(handler)
    with SafeVehicleScraper(
        policy,
        SyntheticMarketplaceParser(),
        transport=transport,
        sleep=sleep,
        monotonic=monotonic,
        now=now,
        today=today,
        resolver=_unexpected_resolver,
        random_fraction=lambda: 0.0,
    ) as scraper:
        return scraper.scrape(start_path="/inventory")


def _unexpected_request(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"request occurred before policy approval: {request.url}")


def _unexpected_resolver(host: str, port: int) -> tuple[str, ...]:
    raise AssertionError(f"DNS resolution must not occur in unit tests: {host}:{port}")


class _ControlledClock:
    """Advance monotonic time deterministically without wall-clock delays."""

    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        assert math.isfinite(seconds) and seconds >= 0
        self.sleeps.append(seconds)
        self.value += seconds

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _AdvancingByteStream(httpx.SyncByteStream):
    """Synthetic stream whose chunks consume controlled runtime."""

    def __init__(self, clock: _ControlledClock, chunks: tuple[bytes, ...]) -> None:
        self.clock = clock
        self.chunks = chunks
        self.yielded = 0

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self.chunks:
            self.clock.advance(0.6)
            self.yielded += 1
            yield chunk


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make an accidental socket path fail even if a test omits MockTransport."""

    def blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("real network access is forbidden in acquisition unit tests")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    yield


def test_complete_parser_normalizes_all_supported_fields_and_provenance() -> None:
    parsed = SyntheticMarketplaceParser().parse_page(
        _synthetic_page(next_url="/inventory?page=2"),
        page_url="http://127.0.0.1:8765/inventory",
        source_id="autovalue-synthetic-marketplace",
        observed_at=_OBSERVED_AT,
        ingestion_run_id="run-synthetic-001",
        authorization_policy_id="synthetic-marketplace-v1",
    )

    assert parsed.next_url == "http://127.0.0.1:8765/inventory?page=2"
    assert len(parsed.listings) == 1
    listing = parsed.listings[0]
    assert listing.source_listing_id == "syn-001"
    assert listing.canonical_url == "http://127.0.0.1:8765/vehicles/syn-001"
    assert (listing.year, listing.make, listing.model, listing.trim) == (
        2021,
        "Subaru",
        "Outback",
        "Limited XT",
    )
    assert listing.mileage == 42_125
    assert listing.mileage_unit == "miles"
    assert listing.condition == "Excellent"
    assert listing.engine == "2.4L Turbo H4"
    assert listing.drivetrain == "All-Wheel Drive"
    assert listing.accident_status == "Reported"
    assert listing.accident_count == 1
    assert listing.owner_count == 2
    assert listing.vehicle_type == "SUV"
    assert listing.price_cents == 2_499_550
    assert listing.currency == "USD"
    assert listing.price_kind is PriceKind.ASKING
    assert listing.observed_at == _OBSERVED_AT
    assert listing.ingestion_run_id == "run-synthetic-001"
    assert listing.authorization_policy_id == "synthetic-marketplace-v1"
    assert len(listing.raw_content_sha256) == 64
    assert set(listing.raw_content_sha256) <= set("0123456789abcdef")


def test_parser_fails_closed_when_page_layout_has_no_cards() -> None:
    with pytest.raises(ListingParseError, match="no vehicle cards"):
        _parse_synthetic_page("<html><body><p>No cards</p></body></html>")


def test_parser_quarantines_a_card_with_a_missing_required_field() -> None:
    html = _synthetic_page().replace('data-field="price"', 'data-field="other-price"', 1)

    parsed = _parse_synthetic_page(html)

    assert parsed.listings == ()
    assert len(parsed.rejected_listings) == 1
    assert "required field is missing: price" in parsed.rejected_listings[0].message


def test_parser_rejects_monthly_payment_as_vehicle_price() -> None:
    parsed = _parse_synthetic_page(_synthetic_page(price="$399/mo"))

    assert parsed.listings == ()
    assert len(parsed.rejected_listings) == 1
    assert "monthly payment" in parsed.rejected_listings[0].message


def test_parser_rejects_negative_price_instead_of_dropping_the_sign() -> None:
    parsed = _parse_synthetic_page(_synthetic_page(price="-$23,900"))

    assert parsed.listings == ()
    assert len(parsed.rejected_listings) == 1
    assert "approved numeric format" in parsed.rejected_listings[0].message


def test_parser_rejects_embedded_price_and_integer_garbage() -> None:
    bad_price = _parse_synthetic_page(_synthetic_page(price="$12abc34"))
    bad_mileage_html = _synthetic_page().replace("42,125 miles", "1 or 2 miles")
    bad_mileage = _parse_synthetic_page(bad_mileage_html)

    assert bad_price.listings == ()
    assert "approved numeric format" in bad_price.rejected_listings[0].message
    assert bad_mileage.listings == ()
    assert "mileage is not an integer" in bad_mileage.rejected_listings[0].message


def test_parser_quarantines_card_whose_listing_id_attribute_is_missing() -> None:
    html = _synthetic_page().replace(' data-listing-id="syn-001"', "")

    parsed = _parse_synthetic_page(html)

    assert parsed.listings == ()
    assert len(parsed.rejected_listings) == 1
    assert parsed.rejected_listings[0].source_listing_id is None
    assert "missing its source listing ID" in parsed.rejected_listings[0].message


def test_disabled_policy_refuses_before_any_request() -> None:
    with pytest.raises(PolicyViolationError, match="disabled"):
        _scrape(_policy(enabled=False), _unexpected_request)


def test_expired_permission_refuses_before_any_request() -> None:
    expired = replace(_permission(), expires_on=_TODAY - timedelta(days=1))

    with pytest.raises(PolicyViolationError, match="expired"):
        _scrape(_policy(permission=expired), _unexpected_request)


def test_unapproved_parser_field_refuses_before_any_request() -> None:
    allowed_fields = SyntheticMarketplaceParser.output_fields - {"owner_count"}

    with pytest.raises(PolicyViolationError, match="owner_count"):
        _scrape(_policy(allowed_fields=allowed_fields), _unexpected_request)


@pytest.mark.parametrize(
    "limits",
    [
        replace(_limits(), request_delay_seconds=math.nan),
        replace(_limits(), request_delay_seconds=math.inf),
        replace(_limits(), request_delay_seconds=-math.inf),
        replace(_limits(), max_runtime_seconds=math.nan),
        replace(_limits(), max_runtime_seconds=math.inf),
        replace(_limits(), max_runtime_seconds=-math.inf),
        replace(_limits(), max_retry_after_seconds=math.nan),
        replace(_limits(), max_retry_after_seconds=math.inf),
        replace(_limits(), max_retry_after_seconds=-math.inf),
    ],
    ids=(
        "delay-nan",
        "delay-positive-infinity",
        "delay-negative-infinity",
        "runtime-nan",
        "runtime-positive-infinity",
        "runtime-negative-infinity",
        "retry-cap-nan",
        "retry-cap-positive-infinity",
        "retry-cap-negative-infinity",
    ),
)
def test_nonfinite_float_limits_refuse_before_any_request(limits: CrawlLimits) -> None:
    with pytest.raises(PolicyViolationError):
        _scrape(_policy(limits=limits), _unexpected_request)


def test_external_source_rejects_shared_address_space_before_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _robots_response(request)

    transport = httpx.MockTransport(handler)
    with (
        SafeVehicleScraper(
            _external_policy(),
            SyntheticMarketplaceParser(),
            transport=transport,
            sleep=lambda _seconds: None,
            monotonic=lambda: 0.0,
            now=lambda: _OBSERVED_AT,
            today=lambda: _TODAY,
            resolver=lambda _host, _port: ("100.64.0.1",),
            random_fraction=lambda: 0.0,
        ) as scraper,
        pytest.raises(PolicyViolationError),
    ):
        scraper._ensure_network_target_safe(  # noqa: SLF001 - direct security boundary test
            "https://authorized.example/inventory"
        )

    assert requests == []


def test_policy_switches_and_approval_flags_require_exact_booleans() -> None:
    with pytest.raises(PolicyViolationError, match="approval must be a boolean"):
        replace(_policy(), scraping_permission=replace(_permission(), approved=1)).validate_for_run(  # type: ignore[arg-type]
            today=_TODAY
        )

    with pytest.raises(PolicyViolationError, match="enabled must be a boolean"):
        replace(_policy(), enabled=1).validate_for_run(today=_TODAY)  # type: ignore[arg-type]

    with pytest.raises(PolicyViolationError, match="switches must be booleans"):
        replace(_policy(), demo_only=1).validate_for_ml_reuse(today=_TODAY)  # type: ignore[arg-type]


def test_external_crawling_stays_disabled_without_address_pinning() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _robots_response(request)

    transport = httpx.MockTransport(handler)
    with (
        SafeVehicleScraper(
            _external_policy(),
            SyntheticMarketplaceParser(),
            transport=transport,
            today=lambda: _TODAY,
        ) as scraper,
        pytest.raises(PolicyViolationError, match="external crawling is disabled"),
    ):
        scraper.scrape(start_path="/inventory")

    assert requests == []


def test_policy_fingerprint_changes_with_a_reviewed_boundary() -> None:
    policy = _policy()
    changed_limits = replace(policy.limits, max_records=policy.limits.max_records + 1)

    assert len(policy.fingerprint()) == 64
    assert policy.fingerprint() == policy.fingerprint()
    assert policy.fingerprint() != replace(policy, limits=changed_limits).fingerprint()


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8765/inventory",
        "http://127.0.0.1:9999/inventory",
        "http://127.0.0.1:8765/inventory-archive",
        "http://127.0.0.1:8765/inventory/%2e%2e/admin",
        "http://127.0.0.1:8765/inventory?page=2&token=secret",
        "http://user@127.0.0.1:8765/inventory",
        "http://127.0.0.1:8765/inventory#fragment",
        "http://192.0.2.1:8765/inventory",
    ],
)
def test_url_boundary_rejects_unreviewed_targets(url: str) -> None:
    with pytest.raises(PolicyViolationError):
        _policy().ensure_url_allowed(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8765/inventory?page=2",
        "http://127.0.0.1:8765/inventory/archive?page=3",
        "http://127.0.0.1:8765/vehicles/syn-001",
    ],
)
def test_url_boundary_accepts_only_reviewed_targets(url: str) -> None:
    _policy().ensure_url_allowed(url)


def test_scraper_rejects_out_of_boundary_listing_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(request, body=_synthetic_page(listing_url="https://example.test/car"))

    with pytest.raises(PolicyViolationError, match="host is not allowlisted"):
        _scrape(_policy(), handler)


def test_robots_denial_blocks_page_request() -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path != "/robots.txt":
            raise AssertionError("disallowed inventory page was requested")
        return _robots_response(request, "User-agent: *\nDisallow: /inventory\n")

    with pytest.raises(RobotsDeniedError, match="disallows"):
        _scrape(_policy(), handler)

    assert requested_paths == ["/robots.txt"]


def test_redirect_is_not_followed() -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(
            request,
            status=302,
            headers={"location": "https://outside.example/inventory"},
        )

    with pytest.raises(PolicyViolationError, match="redirects are disabled"):
        _scrape(_policy(), handler)

    assert requested_paths == ["/robots.txt", "/inventory"]


def test_cookie_setting_response_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "cookie" not in request.headers
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(
            request,
            body=_synthetic_page(),
            headers={"set-cookie": "session=not-allowed; Secure"},
        )

    with pytest.raises(PolicyViolationError, match="cookie-requiring"):
        _scrape(_policy(), handler)


def test_set_cookie_on_missing_robots_file_aborts_before_inventory_request() -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        assert "cookie" not in request.headers
        if request.url.path != "/robots.txt":
            raise AssertionError("a cookie-bearing follow-up request must never occur")
        return _response(
            request,
            status=404,
            body="not found",
            media_type="text/plain",
            headers={"set-cookie": "session=forbidden; Secure"},
        )

    with pytest.raises(PolicyViolationError, match="cookie-requiring"):
        _scrape(_policy(), handler)

    assert requested_paths == ["/robots.txt"]


def test_set_cookie_on_429_aborts_without_retry_or_cookie_replay() -> None:
    inventory_attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal inventory_attempts
        assert "cookie" not in request.headers
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        inventory_attempts += 1
        if inventory_attempts > 1:
            raise AssertionError("a cookie-bearing retry must never occur")
        return _response(
            request,
            status=429,
            headers={
                "retry-after": "1",
                "set-cookie": "session=forbidden; Secure",
            },
        )

    with pytest.raises(PolicyViolationError, match="cookie-requiring"):
        _scrape(_policy(), handler, sleep=sleeps.append)

    assert inventory_attempts == 1
    assert sleeps == []


@pytest.mark.parametrize("status", [401, 403])
def test_authentication_challenge_is_rejected_without_credentials(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(request, status=status)

    with pytest.raises(PolicyViolationError, match=f"status {status}"):
        _scrape(_policy(), handler)


def test_429_retries_once_when_retry_after_is_within_cap() -> None:
    inventory_attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal inventory_attempts
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        inventory_attempts += 1
        if inventory_attempts == 1:
            return _response(request, status=429, headers={"retry-after": "1"})
        return _response(request, body=_synthetic_page())

    result = _scrape(_policy(), handler, sleep=sleeps.append)

    assert inventory_attempts == 2
    assert sleeps == [1.0]
    assert result.requests_made == 3
    assert len(result.listings) == 1


def test_retry_after_above_policy_cap_aborts_without_retrying() -> None:
    inventory_attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal inventory_attempts
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        inventory_attempts += 1
        return _response(request, status=429, headers={"retry-after": "1.001"})

    with pytest.raises(FetchError):
        _scrape(_policy(), handler, sleep=sleeps.append)

    assert inventory_attempts == 1
    assert sleeps == []


def test_http_date_retry_after_is_honored_when_within_cap() -> None:
    policy = _policy(limits=replace(_limits(), max_retry_after_seconds=3))
    retry_at = format_datetime(_OBSERVED_AT + timedelta(seconds=2), usegmt=True)
    inventory_attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal inventory_attempts
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        inventory_attempts += 1
        if inventory_attempts == 1:
            return _response(request, status=429, headers={"retry-after": retry_at})
        return _response(request, body=_synthetic_page())

    result = _scrape(policy, handler, sleep=sleeps.append)

    assert inventory_attempts == 2
    assert sleeps == [2.0]
    assert len(result.listings) == 1


def test_declared_oversized_response_is_rejected_before_parsing() -> None:
    limits = replace(_limits(), max_response_bytes=1_024)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(
            request,
            body="small body",
            headers={"content-length": "1025"},
        )

    with pytest.raises(CrawlBudgetExceededError, match="declared response size"):
        _scrape(_policy(limits=limits), handler)


def test_pagination_loop_stops_before_refetching_page() -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(request, body=_synthetic_page(next_url="/inventory"))

    with pytest.raises(PaginationLoopError, match="loop detected"):
        _scrape(_policy(), handler)

    assert requested_paths == ["/robots.txt", "/inventory"]


def test_rate_limit_delay_cannot_cross_hard_runtime_budget() -> None:
    limits = replace(_limits(), request_delay_seconds=2, max_runtime_seconds=1)
    clock = _ControlledClock()
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(request, body=_synthetic_page())

    with pytest.raises(CrawlBudgetExceededError, match="runtime"):
        _scrape(
            _policy(limits=limits),
            handler,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert clock.value <= limits.max_runtime_seconds
    assert requested_paths == ["/robots.txt"]


def test_retry_delay_cannot_cross_hard_runtime_budget() -> None:
    limits = replace(
        _limits(),
        max_runtime_seconds=1,
        max_retry_after_seconds=2,
    )
    clock = _ControlledClock()
    inventory_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal inventory_attempts
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        inventory_attempts += 1
        return _response(request, status=429, headers={"retry-after": "1.5"})

    with pytest.raises(CrawlBudgetExceededError, match="runtime"):
        _scrape(
            _policy(limits=limits),
            handler,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert clock.value <= limits.max_runtime_seconds
    assert inventory_attempts == 1


def test_streaming_stops_as_soon_as_hard_runtime_budget_is_exceeded() -> None:
    limits = replace(_limits(), max_runtime_seconds=1)
    clock = _ControlledClock()
    page_bytes = _synthetic_page().encode("utf-8")
    split = len(page_bytes) // 3
    stream = _AdvancingByteStream(
        clock,
        (page_bytes[:split], page_bytes[split : split * 2], page_bytes[split * 2 :]),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            stream=stream,
            request=request,
        )

    with pytest.raises(CrawlBudgetExceededError, match="runtime"):
        _scrape(
            _policy(limits=limits),
            handler,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert stream.yielded == 2


def test_writer_emits_checksum_verifiable_dataset_and_complete_provenance(tmp_path: Path) -> None:
    policy = _policy()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(request, body=_synthetic_page())

    result = _scrape(policy, handler)
    dataset_path, manifest_path = write_scrape_result(
        result,
        policy,
        tmp_path / "synthetic-listings.jsonl",
    )

    payload = dataset_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    readiness_path = dataset_path.with_suffix(".ready.json")
    readiness = verify_scrape_artifact_set(manifest_path)
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]

    assert manifest["normalized_file_sha256"] == hashlib.sha256(payload).hexdigest()
    assert readiness_path.exists()
    assert readiness["manifest_file"] == manifest_path.name
    assert len(readiness["artifact_set_id"]) == 64
    assert manifest["normalized_file"] == dataset_path.name
    assert manifest["source_id"] == policy.source_id == result.source_id
    assert manifest["source_owner"] == policy.source_owner
    assert manifest["policy_id"] == policy.policy_id == result.policy_id
    assert manifest["policy_sha256"] == policy.fingerprint() == result.policy_sha256
    assert manifest["ml_reuse_permission_sha256"] == policy.ml_reuse_permission_fingerprint()
    assert manifest["scraping_permission_approved"] is True
    assert manifest["scraping_permission_basis"] == policy.scraping_permission.basis
    assert (
        manifest["scraping_permission_evidence_reference"]
        == policy.scraping_permission.evidence_reference
    )
    assert (
        manifest["scraping_permission_evidence_sha256"]
        == policy.scraping_permission.evidence_sha256
    )
    assert manifest["scraping_permission_approved_purposes"] == ["collection", "storage"]
    assert manifest["ml_training_permission_approved"] is True
    assert manifest["terms_url"] == policy.terms_url
    assert manifest["terms_version"] == policy.terms_version
    assert manifest["ingestion_run_id"] == result.ingestion_run_id
    assert manifest["authorization_date"] == _TODAY.isoformat()
    assert manifest["started_at"] == _OBSERVED_AT.isoformat()
    assert manifest["completed_at"] == _OBSERVED_AT.isoformat()
    assert manifest["pages_fetched"] == 1
    assert manifest["requests_made"] == 2
    assert manifest["retries"] == 0
    assert manifest["response_bytes"] == result.response_bytes > 0
    assert manifest["robots_url"].endswith("/robots.txt")
    assert manifest["robots_sha256"] == result.robots_sha256
    assert manifest["record_count"] == 1
    assert manifest["crawl_limits"]["max_total_response_bytes"] == 10_000_000
    assert manifest["parser_versions"] == ["synthetic-marketplace/1.0.0"]
    assert manifest["normalization_versions"] == ["1.0.0"]
    assert manifest["raw_html_persisted"] is False
    assert manifest["response_cache"] == "disabled"
    assert manifest["response_cache_persistent"] is False
    assert manifest["response_cache_max_bytes"] is None

    assert len(rows) == 1
    assert rows[0]["source_id"] == result.source_id
    assert rows[0]["ingestion_run_id"] == result.ingestion_run_id
    assert rows[0]["authorization_policy_id"] == policy.policy_id
    assert rows[0]["raw_content_sha256"] == result.listings[0].raw_content_sha256
    assert rows[0]["price_kind"] == "asking"
    assert b"<article" not in payload


def test_artifact_set_is_not_ready_when_the_final_marker_is_missing(tmp_path: Path) -> None:
    policy = _policy()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(request, body=_synthetic_page())

    result = _scrape(policy, handler)
    dataset_path, manifest_path = write_scrape_result(
        result,
        policy,
        tmp_path / "incomplete-listings.jsonl",
    )
    dataset_path.with_suffix(".ready.json").unlink()

    with pytest.raises(ContentValidationError, match="missing or inaccessible"):
        verify_scrape_artifact_set(manifest_path)


def test_artifact_set_verifier_detects_dataset_tampering(tmp_path: Path) -> None:
    policy = _policy()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(request, body=_synthetic_page())

    result = _scrape(policy, handler)
    dataset_path, manifest_path = write_scrape_result(
        result,
        policy,
        tmp_path / "tampered-listings.jsonl",
    )
    dataset_path.write_bytes(dataset_path.read_bytes() + b"{}\n")

    with pytest.raises(ContentValidationError, match="does not match the artifact"):
        verify_scrape_artifact_set(manifest_path)


def test_writer_rejects_result_from_different_policy_before_writing(tmp_path: Path) -> None:
    policy = _policy()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(request, body=_synthetic_page())

    result = replace(_scrape(policy, handler), policy_id="another-policy-v1")
    output_path = tmp_path / "should-not-exist.jsonl"

    with pytest.raises(PolicyViolationError, match="provenance does not match"):
        write_scrape_result(result, policy, output_path)

    assert not output_path.exists()


def test_writer_rejects_foreign_nested_listing_before_writing(tmp_path: Path) -> None:
    policy = _policy()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(request, body=_synthetic_page())

    valid_result = _scrape(policy, handler)
    foreign_listing = replace(valid_result.listings[0], source_id="unapproved-source")
    result = replace(valid_result, listings=(foreign_listing,))
    output_path = tmp_path / "should-not-exist.jsonl"

    with pytest.raises(PolicyViolationError, match="listing provenance"):
        write_scrape_result(result, policy, output_path)

    assert not output_path.exists()


def test_result_preserves_the_exact_authorization_date_used_at_timezone_boundary(
    tmp_path: Path,
) -> None:
    expiring_permission = replace(_permission(), expires_on=_TODAY)
    policy = _policy(permission=expiring_permission)
    utc_next_day = datetime(2026, 8, 28, 0, 30, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(request, body=_synthetic_page())

    result = _scrape(
        policy,
        handler,
        now=lambda: utc_next_day,
        today=lambda: _TODAY,
    )
    _, manifest_path = write_scrape_result(
        result,
        policy,
        tmp_path / "boundary-listings.jsonl",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result.started_at.date() == date(2026, 8, 28)
    assert result.authorization_date == _TODAY
    assert manifest["authorization_date"] == "2026-08-27"


def test_missing_robots_file_does_not_replace_explicit_permission() -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/robots.txt":
            return _response(request, status=404, body="not found")
        return _response(request, body=_synthetic_page())

    result = _scrape(_policy(), handler)

    assert len(result.listings) == 1
    assert requested_paths == ["/robots.txt", "/inventory"]


@pytest.mark.parametrize("content_length", ["not-a-number", "-1"])
def test_invalid_content_length_is_rejected(content_length: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(
            request,
            body="small body",
            headers={"content-length": content_length},
        )

    with pytest.raises(ContentValidationError, match="Content-Length"):
        _scrape(_policy(), handler)


def test_actual_response_bytes_cannot_exceed_declared_safe_size() -> None:
    limits = replace(_limits(), max_response_bytes=1_024)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(
            request,
            body="x" * 1_025,
            headers={"content-length": "1024"},
        )

    with pytest.raises(CrawlBudgetExceededError, match="response exceeded"):
        _scrape(_policy(limits=limits), handler)


def test_unexpected_media_type_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(request, body="{}", media_type="application/json")

    with pytest.raises(ContentValidationError, match="unexpected content type"):
        _scrape(_policy(), handler)


def test_transport_failures_are_counted_and_bounded() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("synthetic connection failure", request=request)

    with pytest.raises(FetchError, match="network failure exceeded retry budget"):
        _scrape(_policy(), handler)

    assert attempts == 3


def test_transient_status_exhausts_retry_budget() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return _response(request, status=503, media_type="text/plain")

    with pytest.raises(FetchError, match="status 503"):
        _scrape(_policy(), handler)

    assert attempts == 3


def test_negative_retry_after_aborts_without_retrying() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        attempts += 1
        if attempts == 1:
            return _response(request, status=429, headers={"retry-after": "-10"})
        return _response(request, body=_synthetic_page())

    with pytest.raises(FetchError):
        _scrape(_policy(), handler, sleep=sleeps.append)

    assert attempts == 1
    assert sleeps == []


def test_page_limit_stops_before_fetching_an_extra_page() -> None:
    limits = replace(_limits(), max_pages=1)
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(request, body=_synthetic_page(next_url="/inventory?page=2"))

    with pytest.raises(CrawlBudgetExceededError, match="page limit"):
        _scrape(_policy(limits=limits), handler)

    assert requested_paths == ["/robots.txt", "/inventory"]


def test_record_limit_rejects_an_overfull_page() -> None:
    limits = replace(_limits(), max_records=1)
    two_listings = _synthetic_page() + _synthetic_page().replace("syn-001", "syn-002")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(request, body=two_listings)

    with pytest.raises(CrawlBudgetExceededError, match="record count"):
        _scrape(_policy(limits=limits), handler)


def test_duplicate_rows_count_toward_record_limit() -> None:
    limits = replace(_limits(), max_records=1)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        if request.url.query == b"page=2":
            return _response(request, body=_synthetic_page())
        return _response(request, body=_synthetic_page(next_url="/inventory?page=2"))

    with pytest.raises(CrawlBudgetExceededError, match="record count"):
        _scrape(_policy(limits=limits), handler)


def test_quarantined_rows_count_toward_record_limit() -> None:
    limits = replace(_limits(), max_records=1)
    valid_listing = _synthetic_page()
    malformed_listing = _synthetic_page(price="not a price").replace("syn-001", "syn-002")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(request, body=valid_listing + malformed_listing)

    with pytest.raises(CrawlBudgetExceededError, match="record count"):
        _scrape(_policy(limits=limits), handler)


def test_changed_duplicate_listing_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        if request.url.query == b"page=2":
            return _response(request, body=_synthetic_page(price="$23,000"))
        return _response(request, body=_synthetic_page(next_url="/inventory?page=2"))

    with pytest.raises(DuplicateListingConflictError, match="conflicting snapshots"):
        _scrape(_policy(), handler)


def test_parser_cannot_forge_run_provenance() -> None:
    parser = SyntheticMarketplaceParser()

    class ForgedParser:
        output_fields = parser.output_fields

        def parse_page(self, html: str, **kwargs: object) -> ParsedPage:
            parsed = parser.parse_page(html, **kwargs)  # type: ignore[arg-type]
            forged = replace(parsed.listings[0], ingestion_run_id="forged-run")
            return ParsedPage(listings=(forged,), next_url=None)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _robots_response(request)
        return _response(request, body=_synthetic_page())

    transport = httpx.MockTransport(handler)
    with (
        SafeVehicleScraper(
            _policy(),
            ForgedParser(),
            transport=transport,
            sleep=lambda _seconds: None,
            monotonic=lambda: 0.0,
            now=lambda: _OBSERVED_AT,
            today=lambda: _TODAY,
            resolver=_unexpected_resolver,
        ) as scraper,
        pytest.raises(PolicyViolationError, match="provenance"),
    ):
        scraper.scrape(start_path="/inventory")


def test_demo_permission_digest_is_pinned_to_owned_terms() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    terms = repository_root / "data" / "fixtures" / "scraper_site" / "terms.html"
    actual_digest = hashlib.sha256(terms.read_bytes()).hexdigest()

    assert actual_digest == DEMO_TERMS_SHA256
    policy = build_demo_policy(
        base_url="http://127.0.0.1:8765/",
        port=8765,
        evidence_sha256=actual_digest,
    )
    assert policy.scraping_permission.evidence_sha256 == actual_digest
    assert policy.ml_training_permission.evidence_sha256 == actual_digest

    with pytest.raises(PolicyViolationError, match="terms changed"):
        build_demo_policy(
            base_url="http://127.0.0.1:8765/",
            port=8765,
            evidence_sha256="0" * 64,
        )
