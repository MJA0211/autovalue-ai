"""A bounded HTTP scraper that fails closed outside a reviewed source policy."""

from __future__ import annotations

import hashlib
import ipaddress
import math
import random
import socket
import time
import urllib.robotparser
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Protocol, Self
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

import httpx

from autovalue_ml.acquisition.cache import ResponseCache
from autovalue_ml.acquisition.contracts import (
    ParsedPage,
    RejectedListing,
    ScrapeResult,
    VehicleListingSnapshot,
)
from autovalue_ml.acquisition.errors import (
    ContentValidationError,
    CrawlBudgetExceededError,
    DuplicateListingConflictError,
    FetchError,
    PaginationLoopError,
    PolicyViolationError,
    RobotsDeniedError,
)
from autovalue_ml.acquisition.policy import SourcePolicy

_TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class PageParser(Protocol):
    """Network-independent contract implemented by a reviewed source parser."""

    output_fields: frozenset[str]

    def parse_page(
        self,
        html: str,
        *,
        page_url: str,
        source_id: str,
        observed_at: datetime,
        ingestion_run_id: str,
        authorization_policy_id: str,
    ) -> ParsedPage: ...


Resolver = Callable[[str, int], Iterable[str]]


class SafeVehicleScraper:
    """Fetch one permitted source sequentially without credentials or browser automation."""

    def __init__(
        self,
        policy: SourcePolicy,
        parser: PageParser,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        today: Callable[[], date] = date.today,
        resolver: Resolver | None = None,
        random_fraction: Callable[[], float] = random.random,
        cache: ResponseCache | None = None,
    ) -> None:
        self.policy = policy
        self.parser = parser
        self._sleep = sleep
        self._monotonic = monotonic
        self._now = now
        self._today = today
        self._resolver = resolver or _resolve_addresses
        self._random_fraction = random_fraction
        self._cache = cache
        self._client = httpx.Client(
            transport=transport,
            follow_redirects=False,
            timeout=httpx.Timeout(policy.limits.max_runtime_seconds, connect=10.0),
            trust_env=False,
        )
        self._requests_made = 0
        self._run_deadline = 0.0
        self._last_request_at: float | None = None
        self._effective_delay = policy.limits.request_delay_seconds
        self._response_bytes = 0
        self._retries = 0
        self._duplicates_skipped = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._rejected_listings: list[RejectedListing] = []
        self._http_status_counts: Counter[int] = Counter()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._client.close()

    def scrape(self, *, start_path: str) -> ScrapeResult:
        """Fetch a bounded pagination chain and return normalized snapshots."""
        authorization_date = self._today()
        if type(authorization_date) is not date:
            raise PolicyViolationError("the authorization clock must return a date")
        self.policy.validate_for_run(today=authorization_date)
        self._validate_cache_policy()
        if not self.policy.demo_only:
            raise PolicyViolationError(
                "external crawling is disabled until an address-pinned transport is implemented"
            )
        unexpected_fields = self.parser.output_fields - self.policy.allowed_fields
        if unexpected_fields:
            names = ", ".join(sorted(unexpected_fields))
            raise PolicyViolationError(f"parser emits fields not authorized by policy: {names}")

        self._requests_made = 0
        self._run_deadline = self._monotonic() + self.policy.limits.max_runtime_seconds
        self._last_request_at = None
        self._effective_delay = self.policy.limits.request_delay_seconds
        self._response_bytes = 0
        self._retries = 0
        self._duplicates_skipped = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._rejected_listings = []
        self._http_status_counts.clear()
        self._client.cookies.clear()

        started_at = self._now()
        if started_at.tzinfo is None:
            raise PolicyViolationError("the run clock must return a timezone-aware datetime")
        ingestion_run_id = str(uuid.uuid4())
        initial_url = urljoin(self.policy.base_url, start_path)
        self.policy.ensure_url_allowed(initial_url)
        current_url: str | None = initial_url

        robots, robots_url, robots_sha256 = self._load_robots_policy()
        crawl_delay = robots.crawl_delay(self.policy.user_agent) or robots.crawl_delay("*") or 0
        request_rate = robots.request_rate(self.policy.user_agent) or robots.request_rate("*")
        request_rate_delay = 0.0
        if request_rate is not None and request_rate.requests > 0:
            request_rate_delay = request_rate.seconds / request_rate.requests
        self._effective_delay = max(
            self._effective_delay,
            float(crawl_delay),
            request_rate_delay,
        )

        visited_pages: set[str] = set()
        listings_by_id: dict[str, VehicleListingSnapshot] = {}
        pages_fetched = 0
        records_seen = 0

        while current_url is not None:
            self._check_runtime_budget()
            if pages_fetched >= self.policy.limits.max_pages:
                raise CrawlBudgetExceededError("pagination exceeded the approved page limit")
            current_page_key = _canonical_url_key(current_url)
            if current_page_key in visited_pages:
                raise PaginationLoopError(f"pagination loop detected at {current_url}")
            self.policy.ensure_url_allowed(current_url)
            self._ensure_network_target_safe(current_url)
            if not robots.can_fetch(self.policy.user_agent, current_url):
                raise RobotsDeniedError(f"robots.txt disallows {current_url}")

            html = self._fetch_text(
                current_url,
                expected_media_types={"text/html"},
                cacheable=True,
            )
            parsed = self.parser.parse_page(
                html,
                page_url=current_url,
                source_id=self.policy.source_id,
                observed_at=started_at,
                ingestion_run_id=ingestion_run_id,
                authorization_policy_id=self.policy.policy_id,
            )
            self._check_runtime_budget()
            visited_pages.add(current_page_key)
            pages_fetched += 1
            records_seen += len(parsed.listings) + len(parsed.rejected_listings)
            if records_seen > self.policy.limits.max_records:
                raise CrawlBudgetExceededError("record count exceeded the approved limit")

            for rejection in parsed.rejected_listings:
                if (
                    rejection.source_id != self.policy.source_id
                    or rejection.authorization_policy_id != self.policy.policy_id
                    or rejection.ingestion_run_id != ingestion_run_id
                    or rejection.observed_at != started_at
                ):
                    raise PolicyViolationError(
                        "parser rejection provenance does not match the active acquisition run"
                    )
                self._rejected_listings.append(rejection)

            for listing in parsed.listings:
                if (
                    listing.source_id != self.policy.source_id
                    or listing.authorization_policy_id != self.policy.policy_id
                    or listing.ingestion_run_id != ingestion_run_id
                    or listing.observed_at != started_at
                ):
                    raise PolicyViolationError(
                        "parser output provenance does not match the active acquisition run"
                    )
                self.policy.ensure_url_allowed(listing.canonical_url)
                existing = listings_by_id.get(listing.source_listing_id)
                if existing is not None:
                    if existing != listing:
                        raise DuplicateListingConflictError(
                            f"conflicting snapshots for listing {listing.source_listing_id}"
                        )
                    self._duplicates_skipped += 1
                    continue
                listings_by_id[listing.source_listing_id] = listing

            current_url = parsed.next_url
            if current_url is not None:
                self.policy.ensure_url_allowed(current_url)

        completed_at = self._now()
        if completed_at.tzinfo is None or completed_at < started_at:
            raise PolicyViolationError("the completion clock returned an invalid datetime")
        return ScrapeResult(
            source_id=self.policy.source_id,
            policy_id=self.policy.policy_id,
            policy_sha256=self.policy.fingerprint(),
            ingestion_run_id=ingestion_run_id,
            authorization_date=authorization_date,
            started_at=started_at,
            completed_at=completed_at,
            pages_fetched=pages_fetched,
            requests_made=self._requests_made,
            retries=self._retries,
            response_bytes=self._response_bytes,
            robots_url=robots_url,
            robots_sha256=robots_sha256,
            duplicates_skipped=self._duplicates_skipped,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            cache_backend=self._cache.backend_name if self._cache is not None else "disabled",
            cache_persistent=self._cache.persistent if self._cache is not None else False,
            cache_max_bytes=self._cache.max_bytes if self._cache is not None else None,
            rejected_listings=tuple(self._rejected_listings),
            http_status_counts=tuple(sorted(self._http_status_counts.items())),
            listings=tuple(listings_by_id.values()),
        )

    def _load_robots_policy(self) -> tuple[urllib.robotparser.RobotFileParser, str, str]:
        robots_url = urljoin(self.policy.base_url, "/robots.txt")
        self.policy.ensure_url_allowed(robots_url)
        self._ensure_network_target_safe(robots_url)
        content = self._fetch_text(
            robots_url,
            expected_media_types={"text/plain"},
            allow_not_found=True,
        )
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(content.splitlines() if content else [])
        content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return parser, robots_url, content_sha256

    def _fetch_text(
        self,
        url: str,
        *,
        expected_media_types: set[str],
        allow_not_found: bool = False,
        cacheable: bool = False,
    ) -> str:
        cache_key = self._cache_key(url, expected_media_types)
        if cacheable and self._cache is not None and self.policy.cache_ttl_seconds > 0:
            cached_body = self._cache.get(cache_key)
            if cached_body is not None:
                cached_size = len(cached_body.encode("utf-8"))
                if cached_size > self.policy.limits.max_response_bytes:
                    raise CrawlBudgetExceededError("cached response exceeds the byte limit")
                self._check_runtime_budget()
                self._cache_hits += 1
                return cached_body
            self._cache_misses += 1

        for attempt in range(self.policy.limits.max_retries + 1):
            self._check_request_budget()
            self._check_runtime_budget()
            self._wait_for_rate_limit()
            self._check_runtime_budget()
            self._client.cookies.clear()
            request = self._client.build_request(
                "GET",
                url,
                headers={
                    "Accept": ", ".join(sorted(expected_media_types)),
                    "User-Agent": self.policy.user_agent,
                },
            )
            if "cookie" in request.headers or "authorization" in request.headers:
                raise PolicyViolationError("authenticated or cookie-bearing requests are forbidden")
            remaining_runtime = self._remaining_runtime()
            request.extensions["timeout"] = {
                "connect": min(10.0, remaining_runtime),
                "read": remaining_runtime,
                "write": remaining_runtime,
                "pool": remaining_runtime,
            }
            self._requests_made += 1
            self._last_request_at = self._monotonic()
            retry_delay: float | None = None
            try:
                response = self._client.send(request, stream=True)
                try:
                    self._check_runtime_budget()
                    self._http_status_counts[response.status_code] += 1
                    if response.headers.get_list("set-cookie"):
                        raise PolicyViolationError(
                            "cookie-requiring or cookie-setting sources are not supported"
                        )
                    if response.is_redirect:
                        raise PolicyViolationError(
                            "redirects are disabled; review the new URL manually"
                        )
                    if response.status_code == 404 and allow_not_found:
                        return ""
                    if response.status_code in _TRANSIENT_STATUS_CODES:
                        if attempt >= self.policy.limits.max_retries:
                            raise FetchError(
                                f"transient status {response.status_code} exceeded retry budget"
                            )
                        retry_delay = self._retry_delay(response, attempt)
                    elif response.status_code in {401, 403}:
                        raise PolicyViolationError(
                            f"source denied anonymous access with status {response.status_code}"
                        )
                    elif response.status_code != 200:
                        raise FetchError(f"unexpected response status {response.status_code}")
                    else:
                        media_type = (
                            response.headers.get("content-type", "").split(";", 1)[0].lower()
                        )
                        if media_type not in expected_media_types:
                            raise ContentValidationError(
                                f"unexpected content type: {media_type or '<missing>'}"
                            )
                        declared_length = response.headers.get("content-length")
                        if declared_length:
                            try:
                                declared_bytes = int(declared_length)
                            except ValueError as error:
                                raise ContentValidationError(
                                    "response Content-Length is not an integer"
                                ) from error
                            if declared_bytes < 0:
                                raise ContentValidationError(
                                    "response Content-Length cannot be negative"
                                )
                            if declared_bytes > self.policy.limits.max_response_bytes:
                                raise CrawlBudgetExceededError(
                                    "declared response size exceeds the byte limit"
                                )

                        content = bytearray()
                        for chunk in response.iter_bytes():
                            self._check_runtime_budget()
                            remaining_bytes = self.policy.limits.max_response_bytes - len(content)
                            if len(chunk) > remaining_bytes:
                                raise CrawlBudgetExceededError("response exceeded the byte limit")
                            if (
                                self._response_bytes + len(chunk)
                                > self.policy.limits.max_total_response_bytes
                            ):
                                raise CrawlBudgetExceededError(
                                    "crawl exceeded the total response byte limit"
                                )
                            content.extend(chunk)
                            self._response_bytes += len(chunk)
                        self._check_runtime_budget()
                        encoding = response.encoding or "utf-8"
                        decoded_content = bytes(content).decode(encoding, errors="strict")
                        if cacheable and self._cache is not None:
                            self._cache.put(
                                cache_key,
                                decoded_content,
                                ttl_seconds=self.policy.cache_ttl_seconds,
                            )
                        return decoded_content
                finally:
                    response.close()
            except httpx.TransportError as error:
                if attempt >= self.policy.limits.max_retries:
                    raise FetchError("network failure exceeded retry budget") from error
                self._retries += 1
                self._sleep_with_budget(self._bounded_backoff(attempt))
            except UnicodeDecodeError as error:
                raise ContentValidationError(
                    "response is not valid text in its declared encoding"
                ) from error
            finally:
                self._client.cookies.clear()

            if retry_delay is not None:
                self._retries += 1
                self._sleep_with_budget(retry_delay)
                continue

        raise FetchError("request ended without a response")

    def _cache_key(self, url: str, expected_media_types: set[str]) -> str:
        material = "|".join(
            (
                self.policy.fingerprint(),
                _canonical_url_key(url),
                ",".join(sorted(expected_media_types)),
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _validate_cache_policy(self) -> None:
        if self._cache is None:
            return
        if type(self._cache.persistent) is not bool or self._cache.persistent:
            raise PolicyViolationError("persistent response caches are not approved")
        if not isinstance(self._cache.backend_name, str) or not self._cache.backend_name.strip():
            raise PolicyViolationError("response cache backend name is required")
        if (
            type(self._cache.max_bytes) is not int
            or self._cache.max_bytes < 1_024
            or self._cache.max_bytes > self.policy.max_cache_bytes
        ):
            raise PolicyViolationError("response cache exceeds the policy byte limit")

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("retry-after")
        if retry_after is not None:
            retry_after_seconds: float | None = None
            try:
                numeric_delay = float(retry_after)
                if not math.isfinite(numeric_delay) or numeric_delay < 0:
                    raise FetchError("Retry-After is not a valid nonnegative delay")
                retry_after_seconds = numeric_delay
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                except (TypeError, ValueError, OverflowError):
                    raise FetchError("Retry-After is not a valid delay or HTTP date") from None
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                wall_clock_now = self._now()
                if wall_clock_now.tzinfo is None:
                    raise PolicyViolationError(
                        "the retry clock must return a timezone-aware datetime"
                    ) from None
                retry_after_seconds = max(
                    0.0,
                    (retry_at - wall_clock_now).total_seconds(),
                )
            if retry_after_seconds is not None:
                if retry_after_seconds > self.policy.limits.max_retry_after_seconds:
                    raise FetchError("Retry-After exceeds the approved wait budget")
                return retry_after_seconds
        return self._bounded_backoff(attempt)

    def _bounded_backoff(self, attempt: int) -> float:
        base = min(2.0**attempt, self.policy.limits.max_retry_after_seconds)
        jitter = max(0.0, min(self._random_fraction(), 1.0)) * 0.25
        return min(base + jitter, self.policy.limits.max_retry_after_seconds)

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_at is None or self._effective_delay <= 0:
            return
        elapsed = self._monotonic() - self._last_request_at
        remaining = self._effective_delay - elapsed
        if remaining > 0:
            self._sleep_with_budget(remaining)

    def _sleep_with_budget(self, seconds: float) -> None:
        if seconds <= 0:
            return
        if seconds >= self._remaining_runtime():
            raise CrawlBudgetExceededError("wait would exceed the crawl runtime limit")
        self._sleep(seconds)
        self._check_runtime_budget()

    def _check_request_budget(self) -> None:
        if self._requests_made >= self.policy.limits.max_requests:
            raise CrawlBudgetExceededError("request count exceeded the approved limit")

    def _check_runtime_budget(self) -> None:
        self._remaining_runtime()

    def _remaining_runtime(self) -> float:
        remaining = self._run_deadline - self._monotonic()
        if remaining <= 0:
            raise CrawlBudgetExceededError("crawl runtime exceeded the approved limit")
        return remaining

    def _ensure_network_target_safe(self, url: str) -> None:
        parsed = urlsplit(url)
        host = parsed.hostname
        if host is None:
            raise PolicyViolationError("network target has no host")
        if self.policy.demo_only:
            try:
                if ipaddress.ip_address(host).is_loopback:
                    return
            except ValueError:
                pass
            raise PolicyViolationError("demo policies may connect only to loopback")

        port = parsed.port or 443
        try:
            addresses = tuple(self._resolver(host, port))
        except OSError as error:
            raise PolicyViolationError("source host could not be resolved safely") from error
        if not addresses:
            raise PolicyViolationError("source host resolved to no addresses")
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global:
                raise PolicyViolationError("source resolved to a non-public network address")


def _resolve_addresses(host: str, port: int) -> Iterable[str]:
    records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return {str(record[4][0]) for record in records}


def _canonical_url_key(url: str) -> str:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    default_port = 443 if parsed.scheme == "https" else 80
    port = parsed.port or default_port
    netloc_host = f"[{host}]" if ":" in host else host
    netloc = netloc_host if port == default_port else f"{netloc_host}:{port}"
    normalized_path = quote(unquote(parsed.path or "/"), safe="/-._~")
    normalized_query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    return urlunsplit((parsed.scheme.lower(), netloc, normalized_path, normalized_query, ""))
