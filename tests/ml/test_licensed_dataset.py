"""Tests for local, manifest-gated public dataset ingestion."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import cast

import autovalue_ml.acquisition.licensed_dataset as licensed_dataset_module
import pytest
from autovalue_ml.acquisition.licensed_dataset import (
    MANIFEST_SCHEMA_VERSION,
    LicensedDatasetError,
    load_licensed_dataset,
    require_ml_training_approval,
    require_verified_ml_training_approval,
    sample_manifest,
    verify_licensed_dataset_artifact,
)

_TODAY = date(2026, 8, 27)


def _write_dataset_and_manifest(
    tmp_path: Path,
    payload: bytes,
    *,
    artifact_format: str,
) -> tuple[Path, Path, dict[str, object]]:
    dataset_path = tmp_path / f"vehicles.{artifact_format}"
    dataset_path.write_bytes(payload)
    manifest: dict[str, object] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "source": {
            "owner": "Example Public Data Office",
            "name": "Versioned Vehicle Records",
            "version": "2026-08-01",
            "canonical_url": "https://data.example.org/vehicles/2026-08-01",
            "market_country": "US",
        },
        "license": {
            "spdx_id": "CC-BY-4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "reviewed_on": _TODAY.isoformat(),
        },
        "artifact": {
            "file_name": dataset_path.name,
            "format": artifact_format,
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "approvals": {
            "approved_for_acquisition": True,
            "acquisition_evidence": "review/data-source-approval.md#acquisition",
            "approved_for_ml_training": True,
            "ml_training_evidence": "review/data-source-approval.md#ml-training",
        },
    }
    manifest_path = tmp_path / "vehicles.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return dataset_path, manifest_path, manifest


def _manifest_section(manifest: dict[str, object], name: str) -> dict[str, object]:
    return cast(dict[str, object], manifest[name])


def _rewrite_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_loads_checksum_verified_csv_with_lineage(tmp_path: Path) -> None:
    payload = b"year,make,price\n2020,Toyota,21000\n2021,Honda,22500\n"
    dataset_path, manifest_path, _ = _write_dataset_and_manifest(
        tmp_path,
        payload,
        artifact_format="csv",
    )

    loaded = load_licensed_dataset(
        dataset_path,
        manifest_path,
        trusted_manifest_sha256=_file_sha256(manifest_path),
        today=_TODAY,
    )

    assert dict(loaded.rows[0]) == {"year": "2020", "make": "Toyota", "price": "21000"}
    assert loaded.lineage.row_count == 2
    assert loaded.lineage.artifact_size_bytes == len(payload)
    assert loaded.lineage.artifact_sha256 == hashlib.sha256(payload).hexdigest()
    assert loaded.lineage.source_version == "2026-08-01"
    assert loaded.lineage.market_country == "US"
    assert loaded.lineage.license_spdx_id == "CC-BY-4.0"
    assert loaded.lineage.approved_for_acquisition is True
    assert loaded.lineage.approved_for_ml_training is True
    assert loaded.lineage.dataset_path == dataset_path.resolve()
    assert require_ml_training_approval(loaded) is loaded


def test_verifies_artifact_metadata_without_loading_rows(tmp_path: Path) -> None:
    payload = b"year,make,price\n2020,Toyota,21000\n"
    dataset_path, manifest_path, _ = _write_dataset_and_manifest(
        tmp_path,
        payload,
        artifact_format="csv",
    )

    verified = verify_licensed_dataset_artifact(
        dataset_path,
        manifest_path,
        trusted_manifest_sha256=_file_sha256(manifest_path),
        today=_TODAY,
    )

    assert verified.source_name == "Versioned Vehicle Records"
    assert verified.market_country == "US"
    assert verified.artifact_file_name == "vehicles.csv"
    assert verified.artifact_format == "csv"
    assert verified.artifact_size_bytes == len(payload)
    assert verified.artifact_sha256 == hashlib.sha256(payload).hexdigest()
    assert verified.manifest_sha256 == _file_sha256(manifest_path)
    assert verified.dataset_path == dataset_path.resolve()
    assert require_verified_ml_training_approval(verified) is verified


def test_artifact_verifier_streams_dataset_instead_of_using_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"year,make\n2020,Toyota\n"
    dataset_path, manifest_path, _ = _write_dataset_and_manifest(
        tmp_path,
        payload,
        artifact_format="csv",
    )
    resolved_dataset_path = dataset_path.resolve()
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.resolve() == resolved_dataset_path:
            raise AssertionError("dataset must not be materialized by the artifact verifier")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    verified = verify_licensed_dataset_artifact(
        dataset_path,
        manifest_path,
        trusted_manifest_sha256=_file_sha256(manifest_path),
        today=_TODAY,
    )

    assert verified.artifact_size_bytes == len(payload)


def test_artifact_verifier_rejects_checksum_mismatch(tmp_path: Path) -> None:
    dataset_path, manifest_path, manifest = _write_dataset_and_manifest(
        tmp_path,
        b"year\n2020\n",
        artifact_format="csv",
    )
    _manifest_section(manifest, "artifact")["sha256"] = "0" * 64
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(LicensedDatasetError, match="SHA-256 does not match"):
        verify_licensed_dataset_artifact(
            dataset_path,
            manifest_path,
            trusted_manifest_sha256=_file_sha256(manifest_path),
            today=_TODAY,
        )


def test_artifact_verifier_rejects_tampering_after_manifest_creation(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path, _ = _write_dataset_and_manifest(
        tmp_path,
        b"year\n2020\n",
        artifact_format="csv",
    )
    dataset_path.write_bytes(b"year\n2021\n")

    with pytest.raises(LicensedDatasetError, match="SHA-256 does not match"):
        verify_licensed_dataset_artifact(
            dataset_path,
            manifest_path,
            trusted_manifest_sha256=_file_sha256(manifest_path),
            today=_TODAY,
        )


def test_artifact_verifier_enforces_byte_bound_before_hashing(tmp_path: Path) -> None:
    payload = b"year\n2020\n"
    dataset_path, manifest_path, _ = _write_dataset_and_manifest(
        tmp_path,
        payload,
        artifact_format="csv",
    )

    with pytest.raises(LicensedDatasetError, match="exceeds the byte limit"):
        verify_licensed_dataset_artifact(
            dataset_path,
            manifest_path,
            trusted_manifest_sha256=_file_sha256(manifest_path),
            max_bytes=len(payload) - 1,
            today=_TODAY,
        )


def test_artifact_verifier_detects_file_metadata_change_during_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path, manifest_path, _ = _write_dataset_and_manifest(
        tmp_path,
        b"year\n2020\n",
        artifact_format="csv",
    )
    original_signature = licensed_dataset_module._file_signature
    signature_calls = 0

    def changed_signature(file_stat: os.stat_result) -> tuple[int, int, int, int, int, int]:
        nonlocal signature_calls
        signature_calls += 1
        signature = original_signature(file_stat)
        if signature_calls == 3:
            return (*signature[:-1], signature[-1] + 1)
        return signature

    monkeypatch.setattr(licensed_dataset_module, "_file_signature", changed_signature)

    with pytest.raises(LicensedDatasetError, match="changed while it was being read"):
        verify_licensed_dataset_artifact(
            dataset_path,
            manifest_path,
            trusted_manifest_sha256=_file_sha256(manifest_path),
            today=_TODAY,
        )


@pytest.mark.parametrize(
    "trusted_digest",
    ["", "0" * 63, "A" * 64, "g" * 64],
)
def test_rejects_invalid_trusted_manifest_digest(
    tmp_path: Path,
    trusted_digest: str,
) -> None:
    dataset_path, manifest_path, _ = _write_dataset_and_manifest(
        tmp_path,
        b"year\n2020\n",
        artifact_format="csv",
    )

    with pytest.raises(LicensedDatasetError, match="must be lowercase hexadecimal"):
        load_licensed_dataset(
            dataset_path,
            manifest_path,
            trusted_manifest_sha256=trusted_digest,
            today=_TODAY,
        )


def test_rejects_unreviewed_manifest_bytes_before_parsing(tmp_path: Path) -> None:
    dataset_path, manifest_path, _ = _write_dataset_and_manifest(
        tmp_path,
        b"year\n2020\n",
        artifact_format="csv",
    )
    trusted_digest = _file_sha256(manifest_path)
    manifest_path.write_bytes(b"this is no longer the reviewed manifest")

    with pytest.raises(LicensedDatasetError, match="does not match the trusted review"):
        load_licensed_dataset(
            dataset_path,
            manifest_path,
            trusted_manifest_sha256=trusted_digest,
            today=_TODAY,
        )


def test_loads_jsonl_as_raw_immutable_mappings(tmp_path: Path) -> None:
    payload = b'{"year":2022,"make":"Ford","features":["awd"]}\n'
    dataset_path, manifest_path, _ = _write_dataset_and_manifest(
        tmp_path,
        payload,
        artifact_format="jsonl",
    )

    loaded = load_licensed_dataset(
        dataset_path,
        manifest_path,
        trusted_manifest_sha256=_file_sha256(manifest_path),
        today=_TODAY,
    )

    assert loaded.rows[0]["year"] == 2022
    assert loaded.rows[0]["features"] == ("awd",)
    with pytest.raises(TypeError):
        loaded.rows[0]["year"] = 2023  # type: ignore[index]


def test_rejects_nested_duplicate_keys_in_manifest(tmp_path: Path) -> None:
    dataset_path, manifest_path, manifest = _write_dataset_and_manifest(
        tmp_path,
        b"year\n2020\n",
        artifact_format="csv",
    )
    manifest_text = json.dumps(manifest)
    duplicate_manifest = manifest_text.replace(
        '"source": {',
        '"source": {"owner": "unreviewed replacement", ',
        1,
    )
    manifest_path.write_text(duplicate_manifest, encoding="utf-8")

    with pytest.raises(LicensedDatasetError, match="not valid strict JSON"):
        load_licensed_dataset(
            dataset_path,
            manifest_path,
            trusted_manifest_sha256=_file_sha256(manifest_path),
            today=_TODAY,
        )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"year":2020,"year":2021}\n',
        b'{"year":2020,"details":{"trim":"SE","trim":"SEL"}}\n',
    ],
)
def test_rejects_duplicate_keys_at_any_jsonl_depth(
    tmp_path: Path,
    payload: bytes,
) -> None:
    dataset_path, manifest_path, _ = _write_dataset_and_manifest(
        tmp_path,
        payload,
        artifact_format="jsonl",
    )

    with pytest.raises(LicensedDatasetError, match="JSONL line 1 is invalid"):
        load_licensed_dataset(
            dataset_path,
            manifest_path,
            trusted_manifest_sha256=_file_sha256(manifest_path),
            today=_TODAY,
        )


def test_rejects_nonfinite_number_in_manifest(tmp_path: Path) -> None:
    dataset_path, manifest_path, manifest = _write_dataset_and_manifest(
        tmp_path,
        b"year\n2020\n",
        artifact_format="csv",
    )
    _manifest_section(manifest, "source")["owner"] = float("nan")
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(LicensedDatasetError, match="not valid strict JSON"):
        load_licensed_dataset(
            dataset_path,
            manifest_path,
            trusted_manifest_sha256=_file_sha256(manifest_path),
            today=_TODAY,
        )


def test_rejects_nonfinite_number_in_jsonl(tmp_path: Path) -> None:
    dataset_path, manifest_path, _ = _write_dataset_and_manifest(
        tmp_path,
        b'{"year":2020,"price":Infinity}\n',
        artifact_format="jsonl",
    )

    with pytest.raises(LicensedDatasetError, match="JSONL line 1 is invalid"):
        load_licensed_dataset(
            dataset_path,
            manifest_path,
            trusted_manifest_sha256=_file_sha256(manifest_path),
            today=_TODAY,
        )


def test_fails_closed_when_acquisition_is_not_approved(tmp_path: Path) -> None:
    dataset_path, manifest_path, manifest = _write_dataset_and_manifest(
        tmp_path,
        b"year\n2020\n",
        artifact_format="csv",
    )
    approvals = _manifest_section(manifest, "approvals")
    approvals["approved_for_acquisition"] = False
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(LicensedDatasetError, match="not approved for acquisition"):
        load_licensed_dataset(
            dataset_path,
            manifest_path,
            trusted_manifest_sha256=_file_sha256(manifest_path),
            today=_TODAY,
        )


def test_ml_approval_is_a_separate_gate_after_loading(tmp_path: Path) -> None:
    dataset_path, manifest_path, manifest = _write_dataset_and_manifest(
        tmp_path,
        b"year\n2020\n",
        artifact_format="csv",
    )
    approvals = _manifest_section(manifest, "approvals")
    approvals["approved_for_ml_training"] = False
    approvals["ml_training_evidence"] = ""
    _rewrite_manifest(manifest_path, manifest)

    loaded = load_licensed_dataset(
        dataset_path,
        manifest_path,
        trusted_manifest_sha256=_file_sha256(manifest_path),
        today=_TODAY,
    )

    assert loaded.lineage.approved_for_acquisition is True
    assert loaded.lineage.approved_for_ml_training is False
    with pytest.raises(LicensedDatasetError, match="not approved for ML training"):
        require_ml_training_approval(loaded)

    verified = verify_licensed_dataset_artifact(
        dataset_path,
        manifest_path,
        trusted_manifest_sha256=_file_sha256(manifest_path),
        today=_TODAY,
    )
    assert verified.approved_for_ml_training is False
    with pytest.raises(LicensedDatasetError, match="not approved for ML training"):
        require_verified_ml_training_approval(verified)


@pytest.mark.parametrize("evidence_field", ["acquisition_evidence", "ml_training_evidence"])
def test_requires_separate_non_empty_approval_evidence(
    tmp_path: Path,
    evidence_field: str,
) -> None:
    dataset_path, manifest_path, manifest = _write_dataset_and_manifest(
        tmp_path,
        b"year\n2020\n",
        artifact_format="csv",
    )
    _manifest_section(manifest, "approvals")[evidence_field] = ""
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(LicensedDatasetError, match="must be a non-empty string"):
        load_licensed_dataset(
            dataset_path,
            manifest_path,
            trusted_manifest_sha256=_file_sha256(manifest_path),
            today=_TODAY,
        )


def test_verifies_checksum_before_attempting_to_parse_dataset(tmp_path: Path) -> None:
    invalid_csv = b'year,make\n2020,"unterminated\n'
    dataset_path, manifest_path, manifest = _write_dataset_and_manifest(
        tmp_path,
        invalid_csv,
        artifact_format="csv",
    )
    _manifest_section(manifest, "artifact")["sha256"] = "0" * 64
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(LicensedDatasetError, match="SHA-256 does not match"):
        load_licensed_dataset(
            dataset_path,
            manifest_path,
            trusted_manifest_sha256=_file_sha256(manifest_path),
            today=_TODAY,
        )


def test_enforces_dataset_byte_limit_before_loading(tmp_path: Path) -> None:
    payload = b"year\n2020\n"
    dataset_path, manifest_path, _ = _write_dataset_and_manifest(
        tmp_path,
        payload,
        artifact_format="csv",
    )

    with pytest.raises(LicensedDatasetError, match="exceeds the byte limit"):
        load_licensed_dataset(
            dataset_path,
            manifest_path,
            trusted_manifest_sha256=_file_sha256(manifest_path),
            max_bytes=len(payload) - 1,
            today=_TODAY,
        )


@pytest.mark.parametrize("artifact_format", ["csv", "jsonl"])
def test_enforces_row_limit(tmp_path: Path, artifact_format: str) -> None:
    payload = (
        b"year\n2020\n2021\n" if artifact_format == "csv" else b'{"year":2020}\n{"year":2021}\n'
    )
    dataset_path, manifest_path, _ = _write_dataset_and_manifest(
        tmp_path,
        payload,
        artifact_format=artifact_format,
    )

    with pytest.raises(LicensedDatasetError, match="exceeds the row limit"):
        load_licensed_dataset(
            dataset_path,
            manifest_path,
            trusted_manifest_sha256=_file_sha256(manifest_path),
            max_rows=1,
            today=_TODAY,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest.pop("license"), "manifest fields"),
        (
            lambda manifest: manifest.__setitem__("manifest_schema_version", 2),
            "unsupported manifest_schema_version",
        ),
        (
            lambda manifest: _manifest_section(manifest, "license").__setitem__(
                "reviewed_on", "2026-08-28"
            ),
            "cannot be in the future",
        ),
        (
            lambda manifest: _manifest_section(manifest, "license").__setitem__("spdx_id", ""),
            "non-empty string",
        ),
    ],
)
def test_rejects_incomplete_or_invalid_manifest(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    dataset_path, manifest_path, manifest = _write_dataset_and_manifest(
        tmp_path,
        b"year\n2020\n",
        artifact_format="csv",
    )
    mutation(manifest)
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(LicensedDatasetError, match=message):
        load_licensed_dataset(
            dataset_path,
            manifest_path,
            trusted_manifest_sha256=_file_sha256(manifest_path),
            today=_TODAY,
        )


def test_sample_manifest_is_non_approved_by_default() -> None:
    template = sample_manifest(file_name="vehicles.csv", artifact_format="csv", sha256="0" * 64)
    approvals = _manifest_section(template, "approvals")

    assert approvals["approved_for_acquisition"] is False
    assert approvals["approved_for_ml_training"] is False
    assert _manifest_section(template, "source")["market_country"] == "US"


def test_rejects_non_us_dataset_even_when_other_permissions_are_approved(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path, manifest = _write_dataset_and_manifest(
        tmp_path,
        b"year\n2020\n",
        artifact_format="csv",
    )
    _manifest_section(manifest, "source")["market_country"] = "MA"
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(LicensedDatasetError, match="must be US"):
        load_licensed_dataset(
            dataset_path,
            manifest_path,
            trusted_manifest_sha256=_file_sha256(manifest_path),
            today=_TODAY,
        )
