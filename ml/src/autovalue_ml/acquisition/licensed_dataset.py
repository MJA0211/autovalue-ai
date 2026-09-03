"""Fail-closed ingestion for locally supplied, licensed public datasets.

The loader deliberately performs no network access. A dataset is parsed only after
its versioned manifest is validated and its complete payload matches the declared
SHA-256 digest.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast
from urllib.parse import urlsplit

MANIFEST_SCHEMA_VERSION: Final = 1
DEFAULT_MAX_BYTES: Final = 50_000_000
DEFAULT_MAX_ROWS: Final = 1_000_000
_MAX_MANIFEST_BYTES: Final = 256_000
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_SPDX_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]{0,79}$")
_FORMATS = frozenset({"csv", "jsonl"})
_HASH_CHUNK_BYTES: Final = 1024 * 1024


class LicensedDatasetError(RuntimeError):
    """A local dataset or its permission manifest failed validation."""


@dataclass(frozen=True, slots=True)
class DatasetLineage:
    """Permission and artifact lineage retained with loaded records."""

    manifest_schema_version: int
    source_owner: str
    source_name: str
    source_version: str
    canonical_url: str
    market_country: str
    license_spdx_id: str
    license_url: str
    license_reviewed_on: date
    approved_for_acquisition: bool
    acquisition_evidence: str
    approved_for_ml_training: bool
    ml_training_evidence: str
    artifact_file_name: str
    artifact_format: str
    artifact_sha256: str
    artifact_size_bytes: int
    row_count: int
    manifest_sha256: str
    manifest_path: Path
    dataset_path: Path


@dataclass(frozen=True, slots=True)
class LoadedLicensedDataset:
    """Immutable container for raw records and their verified lineage."""

    rows: tuple[MappingProxyType[str, object], ...]
    lineage: DatasetLineage


@dataclass(frozen=True, slots=True)
class VerifiedLicensedArtifact:
    """Manifest and artifact metadata verified without loading dataset rows."""

    manifest_schema_version: int
    source_owner: str
    source_name: str
    source_version: str
    canonical_url: str
    market_country: str
    license_spdx_id: str
    license_url: str
    license_reviewed_on: date
    approved_for_acquisition: bool
    acquisition_evidence: str
    approved_for_ml_training: bool
    ml_training_evidence: str
    artifact_file_name: str
    artifact_format: str
    artifact_sha256: str
    artifact_size_bytes: int
    manifest_sha256: str
    manifest_path: Path
    dataset_path: Path


@dataclass(frozen=True, slots=True)
class _ValidatedManifest:
    source_owner: str
    source_name: str
    source_version: str
    canonical_url: str
    market_country: str
    license_spdx_id: str
    license_url: str
    license_reviewed_on: date
    acquisition_evidence: str
    approved_for_ml_training: bool
    ml_training_evidence: str
    artifact_file_name: str
    artifact_format: str
    artifact_sha256: str


def verify_licensed_dataset_artifact(
    dataset_path: Path,
    manifest_path: Path,
    *,
    trusted_manifest_sha256: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
    today: date | None = None,
) -> VerifiedLicensedArtifact:
    """Verify a licensed artifact without materializing its rows in memory.

    The externally trusted manifest digest is checked before the strict manifest
    is parsed. The dataset is then hashed in bounded chunks while its size and
    filesystem identity are monitored for changes.
    """

    _validate_positive_limit("max_bytes", max_bytes)
    effective_today = date.today() if today is None else today

    resolved_manifest_path = _validate_local_file(manifest_path, label="manifest")
    manifest_payload = _read_bounded_file(
        resolved_manifest_path,
        max_bytes=_MAX_MANIFEST_BYTES,
        label="manifest",
    )
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    _validate_sha256("trusted manifest SHA-256", trusted_manifest_sha256)
    if manifest_sha256 != trusted_manifest_sha256:
        raise LicensedDatasetError("manifest SHA-256 does not match the trusted review")
    manifest = _parse_manifest(manifest_payload, today=effective_today)

    resolved_dataset_path = _validate_local_file(dataset_path, label="dataset")
    if resolved_dataset_path.name != manifest.artifact_file_name:
        raise LicensedDatasetError("dataset filename does not match the manifest")
    expected_suffix = f".{manifest.artifact_format}"
    if resolved_dataset_path.suffix.lower() != expected_suffix:
        raise LicensedDatasetError("dataset suffix does not match the manifest format")

    actual_sha256, artifact_size_bytes = _hash_bounded_file(
        resolved_dataset_path,
        max_bytes=max_bytes,
        label="dataset",
    )
    if actual_sha256 != manifest.artifact_sha256:
        raise LicensedDatasetError("dataset SHA-256 does not match the manifest")

    return VerifiedLicensedArtifact(
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        source_owner=manifest.source_owner,
        source_name=manifest.source_name,
        source_version=manifest.source_version,
        canonical_url=manifest.canonical_url,
        market_country=manifest.market_country,
        license_spdx_id=manifest.license_spdx_id,
        license_url=manifest.license_url,
        license_reviewed_on=manifest.license_reviewed_on,
        approved_for_acquisition=True,
        acquisition_evidence=manifest.acquisition_evidence,
        approved_for_ml_training=manifest.approved_for_ml_training,
        ml_training_evidence=manifest.ml_training_evidence,
        artifact_file_name=manifest.artifact_file_name,
        artifact_format=manifest.artifact_format,
        artifact_sha256=actual_sha256,
        artifact_size_bytes=artifact_size_bytes,
        manifest_sha256=manifest_sha256,
        manifest_path=resolved_manifest_path,
        dataset_path=resolved_dataset_path,
    )


def load_licensed_dataset(
    dataset_path: Path,
    manifest_path: Path,
    *,
    trusted_manifest_sha256: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
    today: date | None = None,
) -> LoadedLicensedDataset:
    """Load a checksum-verified local CSV or JSONL dataset.

    ``trusted_manifest_sha256`` must come from an external review record rather
    than from the supplied manifest itself. The manifest is parsed only after its
    complete payload matches that trusted digest. Acquisition approval must be
    explicitly true. ML-training approval remains separate lineage and is
    enforced only by ``require_ml_training_approval``.
    """

    _validate_positive_limit("max_bytes", max_bytes)
    _validate_positive_limit("max_rows", max_rows)
    verified = verify_licensed_dataset_artifact(
        dataset_path,
        manifest_path,
        trusted_manifest_sha256=trusted_manifest_sha256,
        max_bytes=max_bytes,
        today=today,
    )

    dataset_payload = _read_bounded_file(
        verified.dataset_path, max_bytes=max_bytes, label="dataset"
    )
    actual_sha256 = hashlib.sha256(dataset_payload).hexdigest()
    if (
        len(dataset_payload) != verified.artifact_size_bytes
        or actual_sha256 != verified.artifact_sha256
    ):
        raise LicensedDatasetError("dataset file changed after artifact verification")

    rows = _parse_dataset(
        dataset_payload,
        artifact_format=verified.artifact_format,
        max_rows=max_rows,
    )
    lineage = DatasetLineage(
        manifest_schema_version=verified.manifest_schema_version,
        source_owner=verified.source_owner,
        source_name=verified.source_name,
        source_version=verified.source_version,
        canonical_url=verified.canonical_url,
        market_country=verified.market_country,
        license_spdx_id=verified.license_spdx_id,
        license_url=verified.license_url,
        license_reviewed_on=verified.license_reviewed_on,
        approved_for_acquisition=verified.approved_for_acquisition,
        acquisition_evidence=verified.acquisition_evidence,
        approved_for_ml_training=verified.approved_for_ml_training,
        ml_training_evidence=verified.ml_training_evidence,
        artifact_file_name=verified.artifact_file_name,
        artifact_format=verified.artifact_format,
        artifact_sha256=actual_sha256,
        artifact_size_bytes=len(dataset_payload),
        row_count=len(rows),
        manifest_sha256=verified.manifest_sha256,
        manifest_path=verified.manifest_path,
        dataset_path=verified.dataset_path,
    )
    return LoadedLicensedDataset(rows=rows, lineage=lineage)


def require_ml_training_approval(dataset: LoadedLicensedDataset) -> LoadedLicensedDataset:
    """Fail closed unless the verified manifest separately approves ML reuse."""
    if not dataset.lineage.approved_for_ml_training:
        raise LicensedDatasetError("dataset is not approved for ML training")
    if not dataset.lineage.ml_training_evidence.strip():
        raise LicensedDatasetError("ML training approval evidence is missing")
    return dataset


def require_verified_ml_training_approval(
    artifact: VerifiedLicensedArtifact,
) -> VerifiedLicensedArtifact:
    """Fail closed unless a row-free verified artifact separately permits ML reuse."""
    if not artifact.approved_for_ml_training:
        raise LicensedDatasetError("dataset is not approved for ML training")
    if not artifact.ml_training_evidence.strip():
        raise LicensedDatasetError("ML training approval evidence is missing")
    return artifact


def sample_manifest(*, file_name: str, artifact_format: str, sha256: str) -> dict[str, object]:
    """Return a non-approved manifest template that must be reviewed before use."""

    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "source": {
            "owner": "REPLACE_WITH_SOURCE_OWNER",
            "name": "REPLACE_WITH_DATASET_NAME",
            "version": "REPLACE_WITH_IMMUTABLE_VERSION",
            "canonical_url": "https://example.invalid/dataset/version",
            "market_country": "US",
        },
        "license": {
            "spdx_id": "REPLACE-WITH-SPDX-ID",
            "url": "https://example.invalid/license",
            "reviewed_on": "YYYY-MM-DD",
        },
        "artifact": {
            "file_name": file_name,
            "format": artifact_format,
            "sha256": sha256,
        },
        "approvals": {
            "approved_for_acquisition": False,
            "acquisition_evidence": "",
            "approved_for_ml_training": False,
            "ml_training_evidence": "",
        },
    }


def _validate_positive_limit(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise LicensedDatasetError(f"{name} must be a positive integer")


def _validate_sha256(label: str, value: object) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise LicensedDatasetError(f"{label} must be lowercase hexadecimal")


def _validate_local_file(path: Path, *, label: str) -> Path:
    if not isinstance(path, Path):
        raise LicensedDatasetError(f"{label} path must be a pathlib.Path")
    if path.is_symlink():
        raise LicensedDatasetError(f"{label} path must not be a symbolic link")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LicensedDatasetError(f"{label} file is not accessible") from exc
    if not resolved.is_file():
        raise LicensedDatasetError(f"{label} path must identify a regular file")
    return resolved


def _read_bounded_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    try:
        declared_size = path.stat().st_size
    except OSError as exc:
        raise LicensedDatasetError(f"could not inspect {label} file") from exc
    if declared_size < 1:
        raise LicensedDatasetError(f"{label} file must not be empty")
    if declared_size > max_bytes:
        raise LicensedDatasetError(f"{label} file exceeds the byte limit")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise LicensedDatasetError(f"could not read {label} file") from exc
    if len(payload) != declared_size:
        raise LicensedDatasetError(f"{label} file changed while it was being read")
    if len(payload) > max_bytes:
        raise LicensedDatasetError(f"{label} file exceeds the byte limit")
    return payload


def _hash_bounded_file(path: Path, *, max_bytes: int, label: str) -> tuple[str, int]:
    """Hash a stable regular file in bounded chunks and return its digest and size."""

    try:
        path_before = path.stat()
    except OSError as exc:
        raise LicensedDatasetError(f"could not inspect {label} file") from exc
    _validate_file_size(path_before.st_size, max_bytes=max_bytes, label=label)

    digest = hashlib.sha256()
    bytes_read = 0
    try:
        with path.open("rb") as stream:
            descriptor_before = os.fstat(stream.fileno())
            if not stat.S_ISREG(descriptor_before.st_mode):
                raise LicensedDatasetError(f"{label} path must identify a regular file")
            if _file_signature(descriptor_before) != _file_signature(path_before):
                raise LicensedDatasetError(f"{label} file changed while it was being read")

            while chunk := stream.read(_HASH_CHUNK_BYTES):
                bytes_read += len(chunk)
                if bytes_read > max_bytes:
                    raise LicensedDatasetError(f"{label} file exceeds the byte limit")
                digest.update(chunk)
            descriptor_after = os.fstat(stream.fileno())
    except LicensedDatasetError:
        raise
    except OSError as exc:
        raise LicensedDatasetError(f"could not read {label} file") from exc

    try:
        path_after = path.stat()
    except OSError as exc:
        raise LicensedDatasetError(f"{label} file changed while it was being read") from exc
    if (
        bytes_read != path_before.st_size
        or _file_signature(descriptor_after) != _file_signature(descriptor_before)
        or _file_signature(path_after) != _file_signature(path_before)
    ):
        raise LicensedDatasetError(f"{label} file changed while it was being read")
    return digest.hexdigest(), bytes_read


def _validate_file_size(size: int, *, max_bytes: int, label: str) -> None:
    if size < 1:
        raise LicensedDatasetError(f"{label} file must not be empty")
    if size > max_bytes:
        raise LicensedDatasetError(f"{label} file exceeds the byte limit")


def _file_signature(file_stat: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _parse_manifest(payload: bytes, *, today: date) -> _ValidatedManifest:
    manifest_object = _decode_json(payload, label="manifest")
    manifest = _require_object(manifest_object, label="manifest")
    _require_exact_keys(
        manifest,
        {"manifest_schema_version", "source", "license", "artifact", "approvals"},
        label="manifest",
    )
    schema_version = manifest["manifest_schema_version"]
    if type(schema_version) is not int or schema_version != MANIFEST_SCHEMA_VERSION:
        raise LicensedDatasetError("unsupported manifest_schema_version")

    source = _require_object(manifest["source"], label="source")
    _require_exact_keys(
        source,
        {"owner", "name", "version", "canonical_url", "market_country"},
        label="source",
    )
    license_data = _require_object(manifest["license"], label="license")
    _require_exact_keys(license_data, {"spdx_id", "url", "reviewed_on"}, label="license")
    artifact = _require_object(manifest["artifact"], label="artifact")
    _require_exact_keys(artifact, {"file_name", "format", "sha256"}, label="artifact")
    approvals = _require_object(manifest["approvals"], label="approvals")
    _require_exact_keys(
        approvals,
        {
            "approved_for_acquisition",
            "acquisition_evidence",
            "approved_for_ml_training",
            "ml_training_evidence",
        },
        label="approvals",
    )

    if approvals["approved_for_acquisition"] is not True:
        raise LicensedDatasetError("dataset is not approved for acquisition")
    ml_training_approved = approvals["approved_for_ml_training"]
    if type(ml_training_approved) is not bool:
        raise LicensedDatasetError("approved_for_ml_training must be a boolean")
    acquisition_evidence = _require_text(
        approvals["acquisition_evidence"], label="acquisition_evidence"
    )
    ml_evidence_value = approvals["ml_training_evidence"]
    if not isinstance(ml_evidence_value, str):
        raise LicensedDatasetError("ml_training_evidence must be a string")
    ml_training_evidence = ml_evidence_value.strip()
    if ml_training_approved and not ml_training_evidence:
        raise LicensedDatasetError("ml_training_evidence must be a non-empty string")

    reviewed_on_text = _require_text(license_data["reviewed_on"], label="reviewed_on")
    try:
        reviewed_on = date.fromisoformat(reviewed_on_text)
    except ValueError as exc:
        raise LicensedDatasetError("license reviewed_on must be an ISO date") from exc
    if reviewed_on > today:
        raise LicensedDatasetError("license review date cannot be in the future")

    spdx_id = _require_text(license_data["spdx_id"], label="license SPDX identifier")
    if not _SPDX_PATTERN.fullmatch(spdx_id):
        raise LicensedDatasetError("license SPDX identifier is invalid")
    artifact_format = _require_text(artifact["format"], label="artifact format").lower()
    if artifact_format not in _FORMATS:
        raise LicensedDatasetError("artifact format must be csv or jsonl")
    artifact_sha256 = _require_text(artifact["sha256"], label="artifact SHA-256")
    if not _SHA256_PATTERN.fullmatch(artifact_sha256):
        raise LicensedDatasetError("artifact SHA-256 must be lowercase hexadecimal")
    file_name = _require_text(artifact["file_name"], label="artifact file_name")
    if Path(file_name).name != file_name:
        raise LicensedDatasetError("artifact file_name must not contain a path")

    canonical_url = _require_url(source["canonical_url"], label="canonical_url")
    market_country = _require_text(source["market_country"], label="market_country")
    if market_country != "US":
        raise LicensedDatasetError("licensed dataset market_country must be US")
    license_url = _require_url(license_data["url"], label="license URL")
    return _ValidatedManifest(
        source_owner=_require_text(source["owner"], label="source owner"),
        source_name=_require_text(source["name"], label="source name"),
        source_version=_require_text(source["version"], label="source version"),
        canonical_url=canonical_url,
        market_country=market_country,
        license_spdx_id=spdx_id,
        license_url=license_url,
        license_reviewed_on=reviewed_on,
        acquisition_evidence=acquisition_evidence,
        approved_for_ml_training=ml_training_approved,
        ml_training_evidence=ml_training_evidence,
        artifact_file_name=file_name,
        artifact_format=artifact_format,
        artifact_sha256=artifact_sha256,
    )


def _decode_json(payload: bytes, *, label: str) -> object:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LicensedDatasetError(f"{label} must be UTF-8") from exc
    try:
        return cast(
            object,
            json.loads(
                text,
                parse_constant=lambda value: _reject_json_constant(value),
                object_pairs_hook=_reject_duplicate_keys,
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise LicensedDatasetError(f"{label} is not valid strict JSON") from exc


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def _require_object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise LicensedDatasetError(f"{label} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _require_exact_keys(value: dict[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise LicensedDatasetError(f"{label} fields do not match the required schema")


def _require_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LicensedDatasetError(f"{label} must be a non-empty string")
    return value.strip()


def _require_url(value: object, *, label: str) -> str:
    url = _require_text(value, label=label)
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        raise LicensedDatasetError(f"{label} must be an absolute public HTTP(S) URL")
    if parsed.password is not None or parsed.fragment:
        raise LicensedDatasetError(f"{label} must not contain credentials or a fragment")
    return url


def _parse_dataset(
    payload: bytes,
    *,
    artifact_format: str,
    max_rows: int,
) -> tuple[MappingProxyType[str, object], ...]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise LicensedDatasetError("dataset must be UTF-8") from exc
    if artifact_format == "csv":
        return _parse_csv(text, max_rows=max_rows)
    if artifact_format == "jsonl":
        return _parse_jsonl(text, max_rows=max_rows)
    raise LicensedDatasetError("unsupported artifact format")


def _parse_csv(text: str, *, max_rows: int) -> tuple[MappingProxyType[str, object], ...]:
    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader)
        if not header or any(not column.strip() for column in header):
            raise LicensedDatasetError("CSV requires non-empty column names")
        if len(set(header)) != len(header):
            raise LicensedDatasetError("CSV column names must be unique")
        rows: list[MappingProxyType[str, object]] = []
        for values in reader:
            if len(rows) >= max_rows:
                raise LicensedDatasetError("dataset exceeds the row limit")
            if len(values) != len(header):
                raise LicensedDatasetError("CSV row width does not match its header")
            rows.append(MappingProxyType(dict(zip(header, values, strict=True))))
    except (csv.Error, StopIteration) as exc:
        raise LicensedDatasetError("dataset is not valid CSV") from exc
    return _require_rows(rows)


def _parse_jsonl(text: str, *, max_rows: int) -> tuple[MappingProxyType[str, object], ...]:
    rows: list[MappingProxyType[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise LicensedDatasetError(f"JSONL line {line_number} is empty")
        if len(rows) >= max_rows:
            raise LicensedDatasetError("dataset exceeds the row limit")
        try:
            value = json.loads(
                line,
                parse_constant=lambda constant: _reject_json_constant(constant),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise LicensedDatasetError(f"JSONL line {line_number} is invalid") from exc
        row = _require_object(cast(object, value), label=f"JSONL line {line_number}")
        rows.append(MappingProxyType({key: _deep_freeze(item) for key, item in row.items()}))
    return _require_rows(rows)


def _require_rows(
    rows: list[MappingProxyType[str, object]],
) -> tuple[MappingProxyType[str, object], ...]:
    if not rows:
        raise LicensedDatasetError("dataset must contain at least one row")
    return tuple(rows)


def _deep_freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_ROWS",
    "MANIFEST_SCHEMA_VERSION",
    "DatasetLineage",
    "LicensedDatasetError",
    "LoadedLicensedDataset",
    "VerifiedLicensedArtifact",
    "load_licensed_dataset",
    "require_ml_training_approval",
    "require_verified_ml_training_approval",
    "sample_manifest",
    "verify_licensed_dataset_artifact",
]
