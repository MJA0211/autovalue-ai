"""Normalized acquisition records independent of any source website."""

import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class PriceKind(StrEnum):
    """Semantic meaning of a monetary value."""

    ASKING = "asking"
    COMPLETED_SALE = "completed_sale"
    CURRENT_BID = "current_bid"
    HIGH_BID = "high_bid"
    RESERVE = "reserve"
    MONTHLY_PAYMENT = "monthly_payment"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class VehicleListingSnapshot:
    """One immutable, normalized observation of a vehicle listing."""

    source_id: str
    source_listing_id: str
    canonical_url: str
    observed_at: datetime
    market_country: str
    year: int
    make: str
    model: str
    trim: str | None
    mileage: int | None
    mileage_unit: str
    condition: str | None
    engine: str | None
    drivetrain: str | None
    accident_status: str | None
    accident_count: int | None
    owner_count: int | None
    vehicle_type: str | None
    price_cents: int
    currency: str
    price_kind: PriceKind
    sale_status: str
    raw_content_sha256: str
    parser_version: str
    normalization_version: str
    ingestion_run_id: str
    authorization_policy_id: str
    vehicle_status: str | None = None

    def __post_init__(self) -> None:
        required_text = {
            "source_id": self.source_id,
            "source_listing_id": self.source_listing_id,
            "canonical_url": self.canonical_url,
            "market_country": self.market_country,
            "make": self.make,
            "model": self.model,
            "mileage_unit": self.mileage_unit,
            "sale_status": self.sale_status,
            "parser_version": self.parser_version,
            "normalization_version": self.normalization_version,
            "ingestion_run_id": self.ingestion_run_id,
            "authorization_policy_id": self.authorization_policy_id,
        }
        if any(not isinstance(value, str) or not value.strip() for value in required_text.values()):
            raise ValueError("required listing text fields must be non-empty strings")
        optional_text = (
            self.trim,
            self.condition,
            self.engine,
            self.drivetrain,
            self.accident_status,
            self.vehicle_type,
            self.vehicle_status,
        )
        if any(value is not None and not isinstance(value, str) for value in optional_text):
            raise ValueError("optional listing text fields must be strings or null")
        if self.vehicle_status not in {None, "new", "used", "certified"}:
            raise ValueError("vehicle_status must be new, used, certified, or null")
        _validate_http_url(self.canonical_url, field="canonical_url")
        if self.market_country != "US":
            raise ValueError("market_country must be US")
        if not isinstance(self.observed_at, datetime):
            raise ValueError("observed_at must be a datetime")
        if type(self.year) is not int:
            raise ValueError("vehicle year must be an integer")
        if not 1886 <= self.year <= self.observed_at.year + 2:
            raise ValueError("vehicle year is outside the accepted range")
        _validate_optional_nonnegative_integer(self.mileage, field="mileage")
        _validate_optional_nonnegative_integer(self.owner_count, field="owner_count")
        _validate_optional_nonnegative_integer(self.accident_count, field="accident_count")
        if self.mileage_unit != "miles":
            raise ValueError("mileage_unit must be miles")
        if type(self.price_cents) is not int:
            raise ValueError("price must be represented as integer cents")
        if self.price_cents <= 0:
            raise ValueError("price must be positive")
        if not isinstance(self.price_kind, PriceKind):
            raise ValueError("price_kind must use the PriceKind enum")
        if self.currency != "USD":
            raise ValueError("currency must be USD for the US market")
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        if self.observed_at.utcoffset() != timedelta(0):
            raise ValueError("observed_at must use UTC")
        if not isinstance(self.raw_content_sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.raw_content_sha256
        ):
            raise ValueError("raw_content_sha256 must be a lowercase SHA-256 hex digest")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready representation."""
        value = asdict(self)
        value["observed_at"] = self.observed_at.isoformat()
        value["price_kind"] = self.price_kind.value
        return value


@dataclass(frozen=True, slots=True)
class RejectedListing:
    """A source record quarantined without retaining its raw HTML."""

    source_id: str
    page_url: str
    source_listing_id: str | None
    observed_at: datetime
    reason_code: str
    message: str
    raw_content_sha256: str
    parser_version: str
    ingestion_run_id: str
    authorization_policy_id: str

    def __post_init__(self) -> None:
        required_text = (
            self.source_id,
            self.page_url,
            self.reason_code,
            self.message,
            self.parser_version,
            self.ingestion_run_id,
            self.authorization_policy_id,
        )
        if any(not isinstance(value, str) or not value.strip() for value in required_text):
            raise ValueError("required rejection text fields must be non-empty strings")
        if self.source_listing_id is not None and (
            not isinstance(self.source_listing_id, str) or not self.source_listing_id.strip()
        ):
            raise ValueError("rejection listing ID must be non-empty when present")
        _validate_http_url(self.page_url, field="page_url")
        if not isinstance(self.observed_at, datetime):
            raise ValueError("rejection timestamp must be a datetime")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() != timedelta(0):
            raise ValueError("rejection timestamp must use UTC")
        if not isinstance(self.raw_content_sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.raw_content_sha256
        ):
            raise ValueError("rejection content hash must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["observed_at"] = self.observed_at.isoformat()
        return value


@dataclass(frozen=True, slots=True)
class ParsedPage:
    """Pure parser output for one HTML page."""

    listings: tuple[VehicleListingSnapshot, ...]
    next_url: str | None
    rejected_listings: tuple[RejectedListing, ...] = ()


@dataclass(frozen=True, slots=True)
class ScrapeResult:
    """Bounded acquisition result and its run-level provenance."""

    source_id: str
    policy_id: str
    policy_sha256: str
    ingestion_run_id: str
    authorization_date: date
    started_at: datetime
    completed_at: datetime
    pages_fetched: int
    requests_made: int
    retries: int
    response_bytes: int
    robots_url: str
    robots_sha256: str
    duplicates_skipped: int
    cache_hits: int
    cache_misses: int
    cache_backend: str
    cache_persistent: bool
    cache_max_bytes: int | None
    rejected_listings: tuple[RejectedListing, ...]
    http_status_counts: tuple[tuple[int, int], ...]
    listings: tuple[VehicleListingSnapshot, ...]

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self.source_id,
                self.policy_id,
                self.ingestion_run_id,
                self.robots_url,
                self.cache_backend,
            )
        ):
            raise ValueError("scrape result identifiers and robots URL are required")
        if type(self.authorization_date) is not date:
            raise ValueError("authorization_date must be a calendar date")
        _validate_http_url(self.robots_url, field="robots_url")
        if not isinstance(self.started_at, datetime) or not isinstance(self.completed_at, datetime):
            raise ValueError("scrape timestamps must be datetimes")
        if self.started_at.tzinfo is None or self.completed_at.tzinfo is None:
            raise ValueError("scrape timestamps must be timezone-aware")
        if self.started_at.utcoffset() != timedelta(
            0
        ) or self.completed_at.utcoffset() != timedelta(0):
            raise ValueError("scrape timestamps must use UTC")
        if self.completed_at < self.started_at:
            raise ValueError("scrape completion cannot precede its start")
        counters = (
            self.pages_fetched,
            self.requests_made,
            self.retries,
            self.response_bytes,
            self.duplicates_skipped,
            self.cache_hits,
            self.cache_misses,
        )
        if any(type(counter) is not int or counter < 0 for counter in counters):
            raise ValueError("scrape counters must be nonnegative integers")
        if type(self.cache_persistent) is not bool:
            raise ValueError("cache persistence must be a boolean")
        if self.cache_backend == "disabled":
            if self.cache_max_bytes is not None or self.cache_hits or self.cache_misses:
                raise ValueError("disabled cache metrics are inconsistent")
        elif type(self.cache_max_bytes) is not int or self.cache_max_bytes < 1_024:
            raise ValueError("enabled cache requires a bounded byte capacity")
        if not isinstance(self.robots_sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.robots_sha256
        ):
            raise ValueError("robots provenance is incomplete")
        if not isinstance(self.policy_sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.policy_sha256
        ):
            raise ValueError("policy provenance is incomplete")
        statuses = [status for status, _ in self.http_status_counts]
        if len(statuses) != len(set(statuses)) or any(
            type(status) is not int
            or type(count) is not int
            or not 100 <= status <= 599
            or count < 1
            for status, count in self.http_status_counts
        ):
            raise ValueError("HTTP status metrics are invalid")


def _validate_optional_nonnegative_integer(value: int | None, *, field: str) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{field} must be a nonnegative integer or null")


def _validate_http_url(value: str, *, field: str) -> None:
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an absolute HTTP(S) URL") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.fragment)
    ):
        raise ValueError(f"{field} must be an absolute HTTP(S) URL without credentials or fragment")
