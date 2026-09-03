"""Fail-closed authorization and crawl policy."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import posixpath
import re
from dataclasses import asdict, dataclass
from datetime import date
from enum import StrEnum
from urllib.parse import parse_qsl, unquote, urlsplit

from autovalue_ml.acquisition.errors import PolicyViolationError

_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_QUERY_PARAMETER_PATTERN = re.compile(r"^[A-Za-z0-9_.~-]{1,64}$")


class Purpose(StrEnum):
    """Explicitly approved uses required by the portfolio workflow."""

    COLLECTION = "collection"
    STORAGE = "storage"
    ML_TRAINING = "ml_training"
    PUBLIC_PORTFOLIO = "public_portfolio"


REQUIRED_SCRAPING_PURPOSES = frozenset({Purpose.COLLECTION, Purpose.STORAGE})
REQUIRED_ML_REUSE_PURPOSES = frozenset({Purpose.ML_TRAINING, Purpose.PUBLIC_PORTFOLIO})


@dataclass(frozen=True, slots=True)
class PermissionGrant:
    """Versioned evidence that a source may be used for this project."""

    approved: bool
    basis: str
    evidence_reference: str
    evidence_sha256: str
    effective_on: date
    expires_on: date | None
    approved_purposes: frozenset[Purpose]
    contact: str

    def validate(self, *, today: date, required_purposes: frozenset[Purpose]) -> None:
        if type(self.approved) is not bool:
            raise PolicyViolationError("source authorization approval must be a boolean")
        if not self.approved:
            raise PolicyViolationError("source authorization is not approved")
        if not self.basis.strip() or not self.evidence_reference.strip():
            raise PolicyViolationError("authorization basis and evidence are required")
        if not _SHA256_PATTERN.fullmatch(self.evidence_sha256):
            raise PolicyViolationError("authorization evidence requires a SHA-256 digest")
        if self.effective_on > today:
            raise PolicyViolationError("source authorization is not effective yet")
        if self.expires_on is not None and self.expires_on < today:
            raise PolicyViolationError("source authorization has expired")
        missing = required_purposes - self.approved_purposes
        if missing:
            missing_values = ", ".join(sorted(purpose.value for purpose in missing))
            raise PolicyViolationError(f"authorization does not cover: {missing_values}")
        if not self.contact.strip():
            raise PolicyViolationError("an authorization contact is required")


@dataclass(frozen=True, slots=True)
class CrawlLimits:
    """Hard limits that cannot be expanded by page content."""

    request_delay_seconds: float = 1.0
    max_pages: int = 10
    max_records: int = 500
    max_requests: int = 35
    max_response_bytes: int = 2_000_000
    max_total_response_bytes: int = 10_000_000
    max_runtime_seconds: float = 120.0
    max_retries: int = 2
    max_retry_after_seconds: float = 10.0

    def validate(self, *, demo_only: bool) -> None:
        integer_limits = {
            "max_pages": self.max_pages,
            "max_records": self.max_records,
            "max_requests": self.max_requests,
            "max_response_bytes": self.max_response_bytes,
            "max_total_response_bytes": self.max_total_response_bytes,
            "max_retries": self.max_retries,
        }
        for name, integer_value in integer_limits.items():
            if type(integer_value) is not int:
                raise PolicyViolationError(f"{name} must be an integer")

        float_limits = {
            "request_delay_seconds": self.request_delay_seconds,
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_retry_after_seconds": self.max_retry_after_seconds,
        }
        for name, float_value in float_limits.items():
            if isinstance(float_value, bool) or not isinstance(float_value, (int, float)):
                raise PolicyViolationError(f"{name} must be a finite number")
            if not math.isfinite(float(float_value)):
                raise PolicyViolationError(f"{name} must be a finite number")

        if self.request_delay_seconds < 0:
            raise PolicyViolationError("request delay cannot be negative")
        if not demo_only and self.request_delay_seconds < 1:
            raise PolicyViolationError("external sources require at least a one-second delay")
        if not 1 <= self.max_pages <= 25:
            raise PolicyViolationError("max_pages must be between 1 and 25")
        if not 1 <= self.max_records <= 5_000:
            raise PolicyViolationError("max_records must be between 1 and 5,000")
        if not 2 <= self.max_requests <= 100:
            raise PolicyViolationError("max_requests must be between 2 and 100")
        if not 1_024 <= self.max_response_bytes <= 5_000_000:
            raise PolicyViolationError("response byte limit is outside the safe range")
        if not self.max_response_bytes <= self.max_total_response_bytes <= 100_000_000:
            raise PolicyViolationError("total response byte limit is outside the safe range")
        if not 1 <= self.max_runtime_seconds <= 900:
            raise PolicyViolationError("runtime limit is outside the safe range")
        if not 0 <= self.max_retries <= 3:
            raise PolicyViolationError("max_retries must be between 0 and 3")
        if not 0 <= self.max_retry_after_seconds <= 30:
            raise PolicyViolationError("Retry-After cap is outside the safe range")


@dataclass(frozen=True, slots=True)
class SourcePolicy:
    """Exact network, permission, purpose, and data boundaries for one adapter."""

    policy_id: str
    source_id: str
    source_owner: str
    market_country: str
    base_url: str
    allowed_hosts: frozenset[str]
    allowed_ports: frozenset[int]
    allowed_path_prefixes: tuple[str, ...]
    allowed_query_parameters: frozenset[str]
    allowed_fields: frozenset[str]
    terms_url: str
    terms_version: str
    user_agent: str
    scraping_permission: PermissionGrant
    ml_training_permission: PermissionGrant
    limits: CrawlLimits
    raw_html_retention_days: int
    cache_ttl_seconds: float = 0
    max_cache_bytes: int = 5_000_000
    demo_only: bool = False
    enabled: bool = False

    def validate_for_run(self, *, today: date) -> None:
        """Validate structure plus the current permission to acquire records."""
        if type(self.enabled) is not bool:
            raise PolicyViolationError("scraping enabled must be a boolean")
        if not self.enabled:
            raise PolicyViolationError("scraping is disabled by this source policy")
        self._validate_structure()
        self.scraping_permission.validate(
            today=today,
            required_purposes=REQUIRED_SCRAPING_PURPOSES,
        )

    def validate_acquisition_lineage(self, *, acquired_on: date) -> None:
        """Validate that an immutable result was authorized when it was acquired."""
        self._validate_structure()
        self.scraping_permission.validate(
            today=acquired_on,
            required_purposes=REQUIRED_SCRAPING_PURPOSES,
        )

    def validate_for_ml_reuse(self, *, today: date) -> None:
        """Independently require current downstream model-training permission."""
        self._validate_structure()
        self.ml_training_permission.validate(
            today=today,
            required_purposes=REQUIRED_ML_REUSE_PURPOSES,
        )

    def _validate_structure(self) -> None:
        if type(self.enabled) is not bool or type(self.demo_only) is not bool:
            raise PolicyViolationError("policy switches must be booleans")
        if not _IDENTIFIER_PATTERN.fullmatch(self.policy_id):
            raise PolicyViolationError("policy_id is invalid")
        if not _IDENTIFIER_PATTERN.fullmatch(self.source_id):
            raise PolicyViolationError("source_id is invalid")
        if not self.source_owner.strip():
            raise PolicyViolationError("source owner is required")
        if self.market_country != "US":
            raise PolicyViolationError("source policy market_country must be US")
        if len(self.allowed_hosts) != 1 or len(self.allowed_ports) != 1:
            raise PolicyViolationError("each source policy must define exactly one crawl origin")
        if not self.allowed_path_prefixes:
            raise PolicyViolationError("path allowlist cannot be empty")
        if not self.allowed_fields:
            raise PolicyViolationError("field allowlist cannot be empty")
        if any(not _is_exact_host(host) for host in self.allowed_hosts):
            raise PolicyViolationError("allowed_hosts must contain exact lowercase hostnames")
        if any(not 1 <= port <= 65_535 for port in self.allowed_ports):
            raise PolicyViolationError("allowed_ports contains an invalid port")
        for prefix in self.allowed_path_prefixes:
            normalized_prefix = posixpath.normpath(prefix).rstrip("/") or "/"
            if (
                not prefix.startswith("/")
                or prefix.startswith("//")
                or unquote(prefix) != prefix
                or any(character in prefix for character in "?#")
                or (prefix.rstrip("/") or "/") != normalized_prefix
            ):
                raise PolicyViolationError(f"path allowlist entry is invalid: {prefix}")
        if any(
            not _QUERY_PARAMETER_PATTERN.fullmatch(parameter)
            for parameter in self.allowed_query_parameters
        ):
            raise PolicyViolationError("query parameter allowlist contains an invalid name")
        if not self.terms_url.strip() or not self.terms_version.strip():
            raise PolicyViolationError("versioned source terms are required")
        if not self.user_agent.startswith("AutoValueAIResearchBot/"):
            raise PolicyViolationError("use the identifying AutoValue AI user agent")
        if (
            type(self.raw_html_retention_days) is not int
            or not 0 <= self.raw_html_retention_days <= 30
        ):
            raise PolicyViolationError("raw HTML retention must be between zero and 30 days")
        if (
            isinstance(self.cache_ttl_seconds, bool)
            or not isinstance(self.cache_ttl_seconds, (int, float))
            or not math.isfinite(float(self.cache_ttl_seconds))
            or not 0 <= self.cache_ttl_seconds <= 86_400
        ):
            raise PolicyViolationError("cache TTL is outside the safe range")
        if (
            type(self.max_cache_bytes) is not int
            or not 1_024 <= self.max_cache_bytes <= 100_000_000
        ):
            raise PolicyViolationError("cache byte limit is outside the safe range")
        if self.cache_ttl_seconds > 0 and self.raw_html_retention_days == 0:
            raise PolicyViolationError("response caching requires an explicit raw retention period")
        if self.cache_ttl_seconds > self.raw_html_retention_days * 86_400:
            raise PolicyViolationError("cache TTL exceeds the approved raw retention period")
        self.limits.validate(demo_only=self.demo_only)
        base_parts = urlsplit(self.base_url)
        if base_parts.query or not base_parts.path.endswith("/"):
            raise PolicyViolationError("base_url must end in a path slash and contain no query")
        self.ensure_url_allowed(self.base_url)
        self.ensure_url_allowed(self.terms_url)

    def fingerprint(self) -> str:
        """Return the immutable acquisition-policy fingerprint.

        ML-reuse approval and the operational ``enabled`` switch are deliberately
        excluded: either may change after a lawful acquisition without rewriting
        the result's acquisition lineage.
        """
        return self.acquisition_fingerprint()

    def acquisition_fingerprint(self) -> str:
        """Hash the reviewed collection boundary independently of ML permission."""
        payload = {
            "fingerprint_schema_version": 1,
            "policy_id": self.policy_id,
            "source_id": self.source_id,
            "source_owner": self.source_owner,
            "market_country": self.market_country,
            "base_url": self.base_url,
            "allowed_hosts": sorted(self.allowed_hosts),
            "allowed_ports": sorted(self.allowed_ports),
            "allowed_path_prefixes": list(self.allowed_path_prefixes),
            "allowed_query_parameters": sorted(self.allowed_query_parameters),
            "allowed_fields": sorted(self.allowed_fields),
            "terms_url": self.terms_url,
            "terms_version": self.terms_version,
            "user_agent": self.user_agent,
            "scraping_permission": _grant_payload(self.scraping_permission),
            "limits": asdict(self.limits),
            "raw_html_retention_days": self.raw_html_retention_days,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "max_cache_bytes": self.max_cache_bytes,
            "demo_only": self.demo_only,
        }
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def ml_reuse_permission_fingerprint(self) -> str:
        """Hash the current ML grant and bind it to one acquisition boundary."""
        payload = {
            "fingerprint_schema_version": 1,
            "acquisition_policy_sha256": self.acquisition_fingerprint(),
            "ml_training_permission": _grant_payload(self.ml_training_permission),
        }
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def ensure_url_allowed(self, url: str) -> None:
        """Reject any URL outside the reviewed source boundary."""
        if len(url) > 2_048:
            raise PolicyViolationError("URL exceeds the approved length limit")
        try:
            parsed = urlsplit(url)
            host = (parsed.hostname or "").lower()
            default_port = 443 if parsed.scheme == "https" else 80
            port = parsed.port or default_port
        except ValueError as error:
            raise PolicyViolationError("URL authority is invalid") from error
        if parsed.username or parsed.password or parsed.fragment:
            raise PolicyViolationError("userinfo and URL fragments are not allowed")
        if host not in self.allowed_hosts:
            raise PolicyViolationError(f"host is not allowlisted: {host or '<missing>'}")

        expected_scheme = urlsplit(self.base_url).scheme
        if parsed.scheme != expected_scheme:
            raise PolicyViolationError("URL scheme does not match the reviewed crawl origin")

        is_loopback = _is_loopback_host(host)
        if parsed.scheme != "https" and not (
            self.demo_only and is_loopback and parsed.scheme == "http"
        ):
            raise PolicyViolationError("HTTPS is required outside the loopback demo")

        if port not in self.allowed_ports:
            raise PolicyViolationError(f"port is not allowlisted: {port}")

        decoded_path = unquote(parsed.path or "/")
        if ".." in decoded_path.split("/"):
            raise PolicyViolationError("path traversal is not allowed")
        normalized_path = posixpath.normpath(decoded_path)
        if not normalized_path.startswith("/"):
            normalized_path = f"/{normalized_path}"
        if not any(_path_matches(normalized_path, prefix) for prefix in self.allowed_path_prefixes):
            raise PolicyViolationError(f"path is not allowlisted: {normalized_path}")

        query_items = parse_qsl(parsed.query, keep_blank_values=True)
        query_keys = [key for key, _ in query_items]
        if len(query_keys) != len(set(query_keys)):
            raise PolicyViolationError("duplicate query parameters are not allowed")
        if len(query_items) > 10 or any(len(value) > 256 for _, value in query_items):
            raise PolicyViolationError("query parameters exceed the approved size limit")
        unexpected_query_keys = set(query_keys) - self.allowed_query_parameters
        if unexpected_query_keys:
            names = ", ".join(sorted(unexpected_query_keys))
            raise PolicyViolationError(f"query parameters are not allowlisted: {names}")


def _path_matches(path: str, prefix: str) -> bool:
    normalized_prefix = prefix.rstrip("/") or "/"
    return (
        normalized_prefix == "/"
        or path == normalized_prefix
        or path.startswith(f"{normalized_prefix}/")
    )


def _is_loopback_host(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_exact_host(host: str) -> bool:
    if not host or host != host.strip().lower() or host.endswith("."):
        return False
    if any(character in host for character in "/@"):
        return False
    if ":" not in host:
        return True
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _grant_payload(grant: PermissionGrant) -> dict[str, object]:
    return {
        "approved": grant.approved,
        "basis": grant.basis,
        "evidence_reference": grant.evidence_reference,
        "evidence_sha256": grant.evidence_sha256,
        "effective_on": grant.effective_on.isoformat(),
        "expires_on": grant.expires_on.isoformat() if grant.expires_on else None,
        "approved_purposes": sorted(purpose.value for purpose in grant.approved_purposes),
        "contact": grant.contact,
    }
