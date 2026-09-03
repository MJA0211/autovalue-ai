"""Tests for the reusable revision-pinned Hugging Face acquisition gate."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest
from autovalue_ml.acquisition.huggingface_dataset import (
    ApprovalStatus,
    DatasetUseApprovals,
    HuggingFaceArtifactSpec,
    HuggingFaceDatasetError,
    OverlapRisk,
    SourceOrigin,
    acquire_huggingface_artifact,
    assess_source_overlap,
    build_huggingface_provenance,
)

_NOW = datetime(2026, 9, 1, tzinfo=UTC)


def _spec(payload: bytes = b"year,price\n2020,10000\n") -> HuggingFaceArtifactSpec:
    return HuggingFaceArtifactSpec(
        source_id="hf_example_vehicles",
        repo_id="Example/vehicles",
        revision="a" * 40,
        file_path=PurePosixPath("data/vehicles.csv"),
        expected_size_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_row_count=1,
        declared_license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        upstream_source="Example public vehicle registry",
        schema_mapping_version="example/1.0.0",
        approvals=DatasetUseApprovals(
            acquisition=ApprovalStatus.APPROVED,
            batch_training=ApprovalStatus.PENDING,
            online_learning=ApprovalStatus.BLOCKED,
            acquisition_evidence="Reviewed public download.",
            batch_training_evidence="Quality review pending.",
            online_learning_evidence="Online labels not reviewed.",
        ),
        usage_restrictions=("Attribution required.",),
        attribution="Example dataset authors.",
        config="default",
        split="train",
    )


def test_requires_full_revision_and_builds_pinned_url() -> None:
    spec = _spec()

    assert f"/resolve/{'a' * 40}/data/vehicles.csv" in spec.resolve_url
    with pytest.raises(HuggingFaceDatasetError, match="full commit SHA"):
        replace(spec, revision="main").validate()
    with pytest.raises(HuggingFaceDatasetError, match="repository-relative"):
        replace(spec, file_path=PurePosixPath("../secret.csv")).validate()


def test_acquisition_is_atomic_verified_and_cacheable(tmp_path: Path) -> None:
    payload = b"year,price\n2020,10000\n"
    requested_urls: list[str] = []

    def downloader(url: str, destination: Path) -> None:
        requested_urls.append(url)
        destination.write_bytes(payload)

    first = acquire_huggingface_artifact(
        _spec(payload), tmp_path, downloader=downloader, acquired_at=_NOW
    )
    second = acquire_huggingface_artifact(
        _spec(payload), tmp_path, downloader=downloader, acquired_at=_NOW
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.sha256 == hashlib.sha256(payload).hexdigest()
    assert first.path.read_bytes() == payload
    assert len(requested_urls) == 1


def test_mismatching_existing_artifact_is_not_overwritten(tmp_path: Path) -> None:
    spec = _spec()
    destination = tmp_path / spec.source_id / spec.file_path.name
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"wrong bytes")

    with pytest.raises(HuggingFaceDatasetError, match="size does not match"):
        acquire_huggingface_artifact(spec, tmp_path, downloader=lambda _url, _path: None)
    assert destination.read_bytes() == b"wrong bytes"


def test_batch_and_online_permissions_are_separate_fail_closed_gates() -> None:
    approvals = _spec().approvals

    approvals.require_acquisition()
    with pytest.raises(HuggingFaceDatasetError, match="batch training"):
        approvals.require_batch_training()
    with pytest.raises(HuggingFaceDatasetError, match="online learning"):
        approvals.require_online_learning()
    with pytest.raises(HuggingFaceDatasetError, match="online approval requires"):
        replace(
            approvals,
            batch_training=ApprovalStatus.BLOCKED,
            online_learning=ApprovalStatus.APPROVED,
        ).validate()


def test_provenance_reconciles_counts_and_retains_all_approval_states(tmp_path: Path) -> None:
    def downloader(_url: str, path: Path) -> None:
        path.write_bytes(b"year,price\n2020,10000\n")

    artifact = acquire_huggingface_artifact(
        _spec(),
        tmp_path,
        downloader=downloader,
        acquired_at=_NOW,
    )
    provenance = build_huggingface_provenance(
        artifact,
        raw_row_count=1,
        accepted_row_count=1,
        rejected_row_count=0,
        duplicate_row_count=0,
    )

    assert provenance.repo_id == "Example/vehicles"
    assert provenance.revision == "a" * 40
    assert provenance.batch_training_approval is ApprovalStatus.PENDING
    assert provenance.online_learning_approval is ApprovalStatus.BLOCKED
    with pytest.raises(HuggingFaceDatasetError, match="do not reconcile"):
        replace(provenance, rejected_row_count=1)


def test_overlap_detection_handles_shared_distinct_and_unknown_origins() -> None:
    current = SourceOrigin("current", frozenset({"cars_com"}), True)
    craigslist = SourceOrigin("craigslist", frozenset({"austin_reese"}), True)
    duplicate = SourceOrigin("duplicate", frozenset({"austin_reese"}), True)
    unknown = SourceOrigin("unknown", frozenset(), False)

    distinct = assess_source_overlap(current, craigslist)
    shared = assess_source_overlap(craigslist, duplicate)
    indeterminate = assess_source_overlap(current, unknown)

    assert distinct.risk is OverlapRisk.NO_KNOWN_SHARED_ORIGIN
    assert distinct.merge_blocked is False
    assert shared.risk is OverlapRisk.CONFIRMED_SHARED_ORIGIN
    assert shared.merge_blocked is True
    assert indeterminate.risk is OverlapRisk.INDETERMINATE
    assert indeterminate.merge_blocked is True
