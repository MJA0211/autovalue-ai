"""Revision-pinned, fail-closed acquisition for Hugging Face dataset files.

This module deliberately downloads individual reviewed artifacts instead of
loading arbitrary repository code through ``datasets``.  A source can be
acquired without thereby becoming eligible for batch or online training.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final
from urllib.parse import quote

import httpx

_REPO_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_REVISION_PATTERN = re.compile(r"^[a-f0-9]{40}$")
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
_HASH_CHUNK_BYTES: Final = 1024 * 1024
_MAX_ARTIFACT_BYTES: Final = 100_000_000


class HuggingFaceDatasetError(RuntimeError):
    """A candidate specification, permission, or artifact failed validation."""


class ApprovalStatus(StrEnum):
    """Independent lifecycle decisions for one immutable source revision."""

    APPROVED = "approved"
    BLOCKED = "blocked"
    PENDING = "pending"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class DatasetUseApprovals:
    """Acquisition, batch, and online decisions that always fail closed."""

    acquisition: ApprovalStatus
    batch_training: ApprovalStatus
    online_learning: ApprovalStatus
    acquisition_evidence: str
    batch_training_evidence: str
    online_learning_evidence: str

    def validate(self) -> None:
        values = (self.acquisition, self.batch_training, self.online_learning)
        if any(not isinstance(value, ApprovalStatus) for value in values):
            raise HuggingFaceDatasetError("approval fields must use ApprovalStatus")
        evidence = (
            self.acquisition_evidence,
            self.batch_training_evidence,
            self.online_learning_evidence,
        )
        if any(not isinstance(value, str) or not value.strip() for value in evidence):
            raise HuggingFaceDatasetError("every approval decision requires evidence")
        if (
            self.online_learning is ApprovalStatus.APPROVED
            and self.batch_training is not ApprovalStatus.APPROVED
        ):
            raise HuggingFaceDatasetError("online approval requires batch-training approval")

    def require_acquisition(self) -> None:
        self.validate()
        if self.acquisition is not ApprovalStatus.APPROVED:
            raise HuggingFaceDatasetError("dataset is not approved for acquisition")

    def require_batch_training(self) -> None:
        self.validate()
        if self.batch_training is not ApprovalStatus.APPROVED:
            raise HuggingFaceDatasetError("dataset is not approved for batch training")

    def require_online_learning(self) -> None:
        self.validate()
        if self.online_learning is not ApprovalStatus.APPROVED:
            raise HuggingFaceDatasetError("dataset is not approved for online learning")


@dataclass(frozen=True, slots=True)
class HuggingFaceArtifactSpec:
    """Reviewed identity and expected bytes for one dataset-repository file."""

    source_id: str
    repo_id: str
    revision: str
    file_path: PurePosixPath
    expected_size_bytes: int
    expected_sha256: str
    expected_row_count: int
    declared_license: str
    license_url: str
    upstream_source: str
    schema_mapping_version: str
    approvals: DatasetUseApprovals
    usage_restrictions: tuple[str, ...]
    attribution: str
    config: str | None = None
    split: str | None = None
    market_country: str = "US"
    currency: str = "USD"

    def validate(self) -> None:
        if not _IDENTIFIER_PATTERN.fullmatch(self.source_id):
            raise HuggingFaceDatasetError("source_id is invalid")
        if not _REPO_PATTERN.fullmatch(self.repo_id):
            raise HuggingFaceDatasetError("Hugging Face repo_id must be owner/name")
        if not _REVISION_PATTERN.fullmatch(self.revision):
            raise HuggingFaceDatasetError("dataset revision must be a full commit SHA")
        if self.file_path.is_absolute() or ".." in self.file_path.parts:
            raise HuggingFaceDatasetError("dataset file path must be repository-relative")
        if self.file_path.name in {"", "."}:
            raise HuggingFaceDatasetError("dataset file path is required")
        if type(self.expected_size_bytes) is not int or self.expected_size_bytes < 1:
            raise HuggingFaceDatasetError("expected artifact size must be positive")
        if self.expected_size_bytes > _MAX_ARTIFACT_BYTES:
            raise HuggingFaceDatasetError("reviewed artifact exceeds the acquisition size cap")
        if not _SHA256_PATTERN.fullmatch(self.expected_sha256):
            raise HuggingFaceDatasetError("expected artifact SHA-256 is invalid")
        if type(self.expected_row_count) is not int or self.expected_row_count < 1:
            raise HuggingFaceDatasetError("expected row count must be positive")
        required_text = (
            self.declared_license,
            self.license_url,
            self.upstream_source,
            self.schema_mapping_version,
            self.attribution,
        )
        if any(not isinstance(value, str) or not value.strip() for value in required_text):
            raise HuggingFaceDatasetError("license, lineage, mapping, and attribution are required")
        if not self.usage_restrictions or any(
            not isinstance(value, str) or not value.strip() for value in self.usage_restrictions
        ):
            raise HuggingFaceDatasetError("usage restrictions must be explicit")
        if self.market_country != "US" or self.currency != "USD":
            raise HuggingFaceDatasetError("AutoValue candidate scope must be US and USD")
        self.approvals.validate()

    @property
    def resolve_url(self) -> str:
        """Return the immutable, revision-pinned Hugging Face resolve URL."""
        self.validate()
        path = "/".join(quote(part, safe="") for part in self.file_path.parts)
        return f"https://huggingface.co/datasets/{self.repo_id}/resolve/{self.revision}/{path}"


@dataclass(frozen=True, slots=True)
class HuggingFaceArtifactProvenance:
    """Local acquisition lineage plus aggregate transformation counts."""

    source_id: str
    repo_id: str
    revision: str
    file_path: str
    config: str | None
    split: str | None
    declared_license: str
    upstream_source: str
    acquired_at: datetime
    artifact_sha256: str
    artifact_size_bytes: int
    raw_row_count: int
    accepted_row_count: int
    rejected_row_count: int
    duplicate_row_count: int
    schema_mapping_version: str
    acquisition_approval: ApprovalStatus
    batch_training_approval: ApprovalStatus
    online_learning_approval: ApprovalStatus

    def __post_init__(self) -> None:
        if self.acquired_at.tzinfo is None or self.acquired_at.utcoffset() is None:
            raise HuggingFaceDatasetError("acquisition timestamp must be timezone-aware")
        if self.acquired_at.astimezone(UTC).utcoffset() != self.acquired_at.utcoffset():
            raise HuggingFaceDatasetError("acquisition timestamp must use UTC")
        counts = (
            self.raw_row_count,
            self.accepted_row_count,
            self.rejected_row_count,
            self.duplicate_row_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise HuggingFaceDatasetError("provenance row counts must be nonnegative integers")
        if self.accepted_row_count + self.rejected_row_count + self.duplicate_row_count != (
            self.raw_row_count
        ):
            raise HuggingFaceDatasetError("provenance row counts do not reconcile")


@dataclass(frozen=True, slots=True)
class VerifiedHuggingFaceArtifact:
    """A checksum-verified local candidate; this is not a training approval."""

    spec: HuggingFaceArtifactSpec
    path: Path
    acquired_at: datetime
    sha256: str
    size_bytes: int
    cache_hit: bool


class OverlapRisk(StrEnum):
    """Conservative source-level overlap classification."""

    NO_KNOWN_SHARED_ORIGIN = "no_known_shared_origin"
    CONFIRMED_SHARED_ORIGIN = "confirmed_shared_origin"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class SourceOrigin:
    source_id: str
    upstream_families: frozenset[str]
    provenance_known: bool


@dataclass(frozen=True, slots=True)
class SourceOverlapAssessment:
    left_source_id: str
    right_source_id: str
    risk: OverlapRisk
    shared_upstream_families: tuple[str, ...]
    merge_blocked: bool
    rationale: str


Downloader = Callable[[str, Path], None]


def assess_source_overlap(left: SourceOrigin, right: SourceOrigin) -> SourceOverlapAssessment:
    """Detect known common origin and fail closed when lineage is incomplete."""
    shared = tuple(sorted(left.upstream_families & right.upstream_families))
    if shared:
        risk = OverlapRisk.CONFIRMED_SHARED_ORIGIN
        rationale = "Both sources declare at least one identical upstream dataset family."
    elif not left.provenance_known or not right.provenance_known:
        risk = OverlapRisk.INDETERMINATE
        rationale = "At least one source lacks sufficient upstream provenance."
    else:
        risk = OverlapRisk.NO_KNOWN_SHARED_ORIGIN
        rationale = "Reviewed lineage contains no shared upstream dataset family."
    return SourceOverlapAssessment(
        left_source_id=left.source_id,
        right_source_id=right.source_id,
        risk=risk,
        shared_upstream_families=shared,
        merge_blocked=risk is not OverlapRisk.NO_KNOWN_SHARED_ORIGIN,
        rationale=rationale,
    )


def acquire_huggingface_artifact(
    spec: HuggingFaceArtifactSpec,
    raw_root: Path,
    *,
    downloader: Downloader | None = None,
    acquired_at: datetime | None = None,
) -> VerifiedHuggingFaceArtifact:
    """Download one exact reviewed file atomically and verify its bytes.

    Existing matching bytes are reused as a local cache. Existing mismatching
    bytes fail closed and are never overwritten silently.
    """
    spec.validate()
    spec.approvals.require_acquisition()
    root = raw_root.resolve()
    destination = (root / spec.source_id / spec.file_path.name).resolve()
    if root != destination and root not in destination.parents:
        raise HuggingFaceDatasetError("artifact destination escapes the raw-data root")
    destination.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC) if acquired_at is None else acquired_at
    if destination.exists():
        sha256, size_bytes = _hash_regular_file(destination)
        _require_expected_artifact(spec, sha256=sha256, size_bytes=size_bytes)
        return VerifiedHuggingFaceArtifact(
            spec=spec,
            path=destination,
            acquired_at=timestamp,
            sha256=sha256,
            size_bytes=size_bytes,
            cache_hit=True,
        )

    download = _download_with_httpx if downloader is None else downloader
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        download(spec.resolve_url, temporary_path)
        sha256, size_bytes = _hash_regular_file(temporary_path)
        _require_expected_artifact(spec, sha256=sha256, size_bytes=size_bytes)
        os.replace(temporary_path, destination)
        temporary_path = None
    except HuggingFaceDatasetError:
        raise
    except (OSError, httpx.HTTPError) as error:
        raise HuggingFaceDatasetError("Hugging Face artifact acquisition failed") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return VerifiedHuggingFaceArtifact(
        spec=spec,
        path=destination,
        acquired_at=timestamp,
        sha256=sha256,
        size_bytes=size_bytes,
        cache_hit=False,
    )


def build_huggingface_provenance(
    artifact: VerifiedHuggingFaceArtifact,
    *,
    raw_row_count: int,
    accepted_row_count: int,
    rejected_row_count: int,
    duplicate_row_count: int,
) -> HuggingFaceArtifactProvenance:
    """Bind aggregate processing metrics to the exact acquired bytes."""
    spec = artifact.spec
    if raw_row_count != spec.expected_row_count:
        raise HuggingFaceDatasetError("raw row count does not match the reviewed revision")
    return HuggingFaceArtifactProvenance(
        source_id=spec.source_id,
        repo_id=spec.repo_id,
        revision=spec.revision,
        file_path=spec.file_path.as_posix(),
        config=spec.config,
        split=spec.split,
        declared_license=spec.declared_license,
        upstream_source=spec.upstream_source,
        acquired_at=artifact.acquired_at,
        artifact_sha256=artifact.sha256,
        artifact_size_bytes=artifact.size_bytes,
        raw_row_count=raw_row_count,
        accepted_row_count=accepted_row_count,
        rejected_row_count=rejected_row_count,
        duplicate_row_count=duplicate_row_count,
        schema_mapping_version=spec.schema_mapping_version,
        acquisition_approval=spec.approvals.acquisition,
        batch_training_approval=spec.approvals.batch_training,
        online_learning_approval=spec.approvals.online_learning,
    )


def _require_expected_artifact(
    spec: HuggingFaceArtifactSpec, *, sha256: str, size_bytes: int
) -> None:
    if size_bytes != spec.expected_size_bytes:
        raise HuggingFaceDatasetError("artifact size does not match the reviewed revision")
    if sha256 != spec.expected_sha256:
        raise HuggingFaceDatasetError("artifact SHA-256 does not match the reviewed revision")


def _hash_regular_file(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise HuggingFaceDatasetError("artifact path must be a regular non-symlink file")
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(_HASH_CHUNK_BYTES):
                size_bytes += len(chunk)
                if size_bytes > _MAX_ARTIFACT_BYTES:
                    raise HuggingFaceDatasetError("artifact exceeds the acquisition size cap")
                digest.update(chunk)
    except OSError as error:
        raise HuggingFaceDatasetError("artifact cannot be read") from error
    return digest.hexdigest(), size_bytes


def _download_with_httpx(url: str, destination: Path) -> None:
    """Bounded direct retrieval with retry/backoff for 429 and transient 5xx."""
    delays = (0.0, 1.0, 2.0, 4.0)
    last_error: httpx.HTTPError | None = None
    for attempt, delay in enumerate(delays):
        if delay:
            import time

            time.sleep(delay)
        try:
            with httpx.stream(
                "GET",
                url,
                follow_redirects=True,
                timeout=httpx.Timeout(60.0, connect=15.0),
                headers={"User-Agent": "AutoValueAIResearch/1.0"},
            ) as response:
                if (
                    response.status_code == 429 or 500 <= response.status_code <= 599
                ) and attempt < len(delays) - 1:
                    continue
                response.raise_for_status()
                size_bytes = 0
                with destination.open("wb") as stream:
                    for chunk in response.iter_bytes(_HASH_CHUNK_BYTES):
                        size_bytes += len(chunk)
                        if size_bytes > _MAX_ARTIFACT_BYTES:
                            raise HuggingFaceDatasetError(
                                "artifact exceeds the acquisition size cap"
                            )
                        stream.write(chunk)
                return
        except httpx.HTTPError as error:
            last_error = error
            if attempt == len(delays) - 1:
                raise
    if last_error is not None:
        raise last_error
    raise HuggingFaceDatasetError("Hugging Face artifact acquisition exhausted retries")


__all__ = [
    "ApprovalStatus",
    "DatasetUseApprovals",
    "HuggingFaceArtifactProvenance",
    "HuggingFaceArtifactSpec",
    "HuggingFaceDatasetError",
    "OverlapRisk",
    "SourceOrigin",
    "SourceOverlapAssessment",
    "VerifiedHuggingFaceArtifact",
    "acquire_huggingface_artifact",
    "assess_source_overlap",
    "build_huggingface_provenance",
]
