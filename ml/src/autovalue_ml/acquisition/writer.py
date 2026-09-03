"""Write normalized snapshots and a checksum-verifiable provenance manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from autovalue_ml.acquisition.contracts import ScrapeResult
from autovalue_ml.acquisition.errors import ContentValidationError
from autovalue_ml.acquisition.policy import SourcePolicy
from autovalue_ml.acquisition.provenance import validate_scrape_result_provenance

_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_MAX_ARTIFACT_BYTES = 100_000_000


def write_scrape_result(
    result: ScrapeResult,
    policy: SourcePolicy,
    output_path: Path,
) -> tuple[Path, Path]:
    """Atomically write JSON Lines plus a run manifest; raw HTML is never retained."""
    if output_path.suffix != ".jsonl":
        raise ValueError("normalized scraper output must use the .jsonl suffix")
    validate_scrape_result_provenance(result, policy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path = output_path.with_suffix(".quarantine.jsonl")
    manifest_path = output_path.with_suffix(".manifest.json")
    readiness_path = output_path.with_suffix(".ready.json")
    readiness_path.unlink(missing_ok=True)

    lines = [json.dumps(listing.to_dict(), sort_keys=True) for listing in result.listings]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    _atomic_write_bytes(output_path, payload)
    dataset_sha256 = hashlib.sha256(payload).hexdigest()

    quarantine_lines = [
        json.dumps(rejection.to_dict(), sort_keys=True) for rejection in result.rejected_listings
    ]
    quarantine_payload = (
        ("\n".join(quarantine_lines) + "\n").encode("utf-8") if quarantine_lines else b""
    )
    _atomic_write_bytes(quarantine_path, quarantine_payload)
    quarantine_sha256 = hashlib.sha256(quarantine_payload).hexdigest()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_id": result.source_id,
        "source_owner": policy.source_owner,
        "market_country": policy.market_country,
        "policy_id": result.policy_id,
        "policy_sha256": result.policy_sha256,
        "ml_reuse_permission_sha256": policy.ml_reuse_permission_fingerprint(),
        "scraping_permission_approved": policy.scraping_permission.approved,
        "scraping_permission_basis": policy.scraping_permission.basis,
        "scraping_permission_evidence_reference": (policy.scraping_permission.evidence_reference),
        "scraping_permission_evidence_sha256": policy.scraping_permission.evidence_sha256,
        "scraping_permission_effective_on": policy.scraping_permission.effective_on.isoformat(),
        "scraping_permission_expires_on": (
            policy.scraping_permission.expires_on.isoformat()
            if policy.scraping_permission.expires_on
            else None
        ),
        "scraping_permission_approved_purposes": sorted(
            purpose.value for purpose in policy.scraping_permission.approved_purposes
        ),
        "ml_training_permission_approved": policy.ml_training_permission.approved,
        "ml_training_permission_basis": policy.ml_training_permission.basis,
        "ml_training_permission_evidence_reference": (
            policy.ml_training_permission.evidence_reference
        ),
        "ml_training_permission_evidence_sha256": (policy.ml_training_permission.evidence_sha256),
        "ml_training_permission_effective_on": (
            policy.ml_training_permission.effective_on.isoformat()
        ),
        "ml_training_permission_expires_on": (
            policy.ml_training_permission.expires_on.isoformat()
            if policy.ml_training_permission.expires_on
            else None
        ),
        "ml_training_permission_approved_purposes": sorted(
            purpose.value for purpose in policy.ml_training_permission.approved_purposes
        ),
        "terms_url": policy.terms_url,
        "terms_version": policy.terms_version,
        "ingestion_run_id": result.ingestion_run_id,
        "authorization_date": result.authorization_date.isoformat(),
        "started_at": result.started_at.isoformat(),
        "completed_at": result.completed_at.isoformat(),
        "duration_seconds": (result.completed_at - result.started_at).total_seconds(),
        "pages_fetched": result.pages_fetched,
        "requests_made": result.requests_made,
        "retries": result.retries,
        "response_bytes": result.response_bytes,
        "duplicates_skipped": result.duplicates_skipped,
        "cache_hits": result.cache_hits,
        "cache_misses": result.cache_misses,
        "http_status_counts": {str(status): count for status, count in result.http_status_counts},
        "robots_url": result.robots_url,
        "robots_sha256": result.robots_sha256,
        "record_count": len(result.listings),
        "records_seen": (
            len(result.listings) + result.duplicates_skipped + len(result.rejected_listings)
        ),
        "quarantined_record_count": len(result.rejected_listings),
        "quarantine_file": quarantine_path.name,
        "quarantine_file_sha256": quarantine_sha256,
        "allowed_fields": sorted(policy.allowed_fields),
        "crawl_limits": asdict(policy.limits),
        "raw_html_retention_days": policy.raw_html_retention_days,
        "raw_html_persisted": False,
        "response_cache": result.cache_backend,
        "response_cache_persistent": result.cache_persistent,
        "response_cache_max_bytes": result.cache_max_bytes,
        "cache_ttl_seconds": policy.cache_ttl_seconds,
        "parser_versions": sorted({listing.parser_version for listing in result.listings}),
        "normalization_versions": sorted(
            {listing.normalization_version for listing in result.listings}
        ),
        "normalized_file": output_path.name,
        "normalized_file_sha256": dataset_sha256,
        "readiness_file": readiness_path.name,
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(manifest_path, manifest_payload)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()

    artifact_set_id = hashlib.sha256(
        "|".join((manifest_sha256, dataset_sha256, quarantine_sha256)).encode("utf-8")
    ).hexdigest()
    readiness = {
        "schema_version": 1,
        "artifact_set_id": artifact_set_id,
        "manifest_file": manifest_path.name,
        "manifest_sha256": manifest_sha256,
        "normalized_file": output_path.name,
        "normalized_file_sha256": dataset_sha256,
        "quarantine_file": quarantine_path.name,
        "quarantine_file_sha256": quarantine_sha256,
    }
    readiness_payload = (json.dumps(readiness, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(readiness_path, readiness_payload)
    return output_path, manifest_path


def verify_scrape_artifact_set(manifest_path: Path) -> dict[str, Any]:
    """Require the final marker and verify every file in a published artifact set."""
    manifest_path = _validate_artifact_path(manifest_path, label="manifest")
    manifest_payload = _read_bounded_bytes(manifest_path, label="manifest")
    manifest = _strict_json_object(manifest_payload, label="manifest")

    readiness_name = _required_file_name(manifest, "readiness_file")
    readiness_path = _validate_artifact_path(
        manifest_path.parent / readiness_name,
        label="readiness marker",
    )
    readiness = _strict_json_object(
        _read_bounded_bytes(readiness_path, label="readiness marker"),
        label="readiness marker",
    )
    expected_readiness_keys = {
        "schema_version",
        "artifact_set_id",
        "manifest_file",
        "manifest_sha256",
        "normalized_file",
        "normalized_file_sha256",
        "quarantine_file",
        "quarantine_file_sha256",
    }
    if set(readiness) != expected_readiness_keys or readiness.get("schema_version") != 1:
        raise ContentValidationError("readiness marker schema is invalid")
    if readiness.get("manifest_file") != manifest_path.name:
        raise ContentValidationError("readiness marker references a different manifest")

    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    _require_matching_hash(readiness, "manifest_sha256", manifest_sha256)

    verified_hashes: list[str] = []
    for file_key, hash_key, label in (
        ("normalized_file", "normalized_file_sha256", "normalized dataset"),
        ("quarantine_file", "quarantine_file_sha256", "quarantine dataset"),
    ):
        file_name = _required_file_name(readiness, file_key)
        if manifest.get(file_key) != file_name or manifest.get(hash_key) != readiness.get(hash_key):
            raise ContentValidationError(f"{label} lineage differs between marker and manifest")
        artifact_path = _validate_artifact_path(manifest_path.parent / file_name, label=label)
        actual_sha256 = hashlib.sha256(
            _read_bounded_bytes(
                artifact_path,
                label=label,
                allow_empty=label == "quarantine dataset",
            )
        ).hexdigest()
        _require_matching_hash(readiness, hash_key, actual_sha256)
        verified_hashes.append(actual_sha256)

    expected_set_id = hashlib.sha256(
        "|".join((manifest_sha256, *verified_hashes)).encode("utf-8")
    ).hexdigest()
    _require_matching_hash(readiness, "artifact_set_id", expected_set_id)
    return readiness


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _validate_artifact_path(path: Path, *, label: str) -> Path:
    if not isinstance(path, Path) or path.is_symlink():
        raise ContentValidationError(f"{label} must be a non-symlink local file")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ContentValidationError(f"{label} is missing or inaccessible") from error
    if not resolved.is_file():
        raise ContentValidationError(f"{label} is not a regular file")
    return resolved


def _read_bounded_bytes(path: Path, *, label: str, allow_empty: bool = False) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ContentValidationError(f"could not inspect {label}") from error
    if size > _MAX_ARTIFACT_BYTES or (size == 0 and not allow_empty):
        raise ContentValidationError(f"{label} is empty or exceeds the verification limit")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ContentValidationError(f"could not read {label}") from error
    if len(payload) != size:
        raise ContentValidationError(f"{label} changed during verification")
    return payload


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON value is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ContentValidationError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ContentValidationError(f"{label} must be a JSON object")
    return value


def _required_file_name(value: dict[str, Any], key: str) -> str:
    file_name = value.get(key)
    if not isinstance(file_name, str) or Path(file_name).name != file_name:
        raise ContentValidationError(f"{key} is not a safe artifact filename")
    return file_name


def _require_matching_hash(value: dict[str, Any], key: str, actual_sha256: str) -> None:
    expected = value.get(key)
    if not isinstance(expected, str) or not _SHA256_PATTERN.fullmatch(expected):
        raise ContentValidationError(f"{key} is not a valid SHA-256 digest")
    if expected != actual_sha256:
        raise ContentValidationError(f"{key} does not match the artifact")
