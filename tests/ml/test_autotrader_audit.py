from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import autovalue_ml.acquisition.autotrader_audit as audit_module
import pandas as pd
import pytest
from autovalue_ml.acquisition.autotrader_audit import (
    AUTOTRADER_PREVIEW_FILES,
    EXPECTED_COLUMNS,
    PREMIUM_FIELDS,
    audit_autotrader_preview,
    preview_artifact_specs,
    profile_autotrader_frame,
    write_autotrader_audit,
)
from autovalue_ml.acquisition.huggingface_dataset import (
    ApprovalStatus,
    VerifiedHuggingFaceArtifact,
)


def _review_frame() -> pd.DataFrame:
    base: dict[str, object] = {column: None for column in EXPECTED_COLUMNS}
    base.update(
        {
            "_primaryKey": "primary-1",
            "_firstSeenAt": "2026-07-20T01:00:00Z",
            "_lastSeenAt": "2026-07-21T01:00:00Z",
            "listingId": "listing-1",
            "stockNumber": "stock-1",
            "year": 2020,
            "makeName": "Example",
            "modelName": "Model",
            "trim": "Base",
            "listingType": "Used",
            "mileage": 20_000,
            "engine": "4-Cylinder",
            "transmission": "Automatic",
            "drivetrain": "Front-wheel drive",
            "sellerId": 10,
            "sellerCity": "Albany",
            "sellerState": "NY",
            "sellerZip": "12207",
            "vhrPreview": '["NO_ACCIDENTS_REPORTED"]',
            "kbbFairPriceLow": 10_000,
            "kbbFairPriceHigh": 12_000,
        }
    )
    base.update({field: "[PREMIUM]" for field in PREMIUM_FIELDS})
    first = dict(base, _snapshotFile="2026-07-20")
    exact_repeat = dict(base, _snapshotFile="2026-07-21")
    changed = dict(
        base,
        _snapshotFile="2026-07-22",
        kbbFairPriceHigh=13_000,
        vhrPreview="not-json",
    )
    invalid_range = dict(
        base,
        _primaryKey="primary-2",
        listingId="listing-2",
        stockNumber="stock-2",
        sellerState="ZZ",
        sellerZip="invalid",
        kbbFairPriceLow=30_000,
        kbbFairPriceHigh=20_000,
        vhrPreview='{"status": "clean"}',
        _snapshotFile="2026-07-20",
    )
    missing_range = dict(
        base,
        _primaryKey="primary-3",
        listingId="listing-3",
        stockNumber=None,
        sellerId=None,
        sellerState=None,
        sellerZip=None,
        kbbFairPriceLow=None,
        kbbFairPriceHigh=None,
        vhrPreview=None,
        _snapshotFile="2026-07-20",
    )
    return pd.DataFrame(
        [first, exact_repeat, changed, invalid_range, missing_range],
        columns=[*EXPECTED_COLUMNS, "_snapshotFile"],
    )


def test_preview_specs_are_pinned_and_fail_closed_for_training() -> None:
    specs = preview_artifact_specs()

    assert len(specs) == len(AUTOTRADER_PREVIEW_FILES) == 30
    assert sum(spec.expected_row_count for spec in specs) == 8_019
    assert all(spec.approvals.acquisition is ApprovalStatus.APPROVED for spec in specs)
    assert all(spec.approvals.batch_training is ApprovalStatus.BLOCKED for spec in specs)
    assert all(spec.approvals.online_learning is ApprovalStatus.BLOCKED for spec in specs)
    assert len({spec.revision for spec in specs}) == 1


def test_profile_reports_targets_repetition_scope_and_leakage(tmp_path: Path) -> None:
    report = profile_autotrader_frame(
        _review_frame(),
        manifest=[{"path": "example.parquet", "rows": 5, "size_bytes": 10, "sha256": "a" * 64}],
        generated_at=datetime(2026, 9, 2, tzinfo=UTC),
    )

    artifact = report["artifact"]
    targets = report["kbb_targets"]
    repetition = report["identifiers_and_repetition"]
    history = report["vehicle_history"]
    permissions = report["permissions"]
    assert isinstance(artifact, dict) and artifact["rows"] == 5
    assert isinstance(targets, dict) and targets["complete_valid_ranges"] == 3
    assert targets["low_greater_than_high"] == 1
    assert targets["both_missing"] == 1
    assert isinstance(repetition, dict) and repetition["listing_ids_repeated"] == 1
    assert repetition["listing_ids_across_multiple_snapshot_files"] == 1
    assert repetition["repeated_listing_ids_with_kbb_changes"] == 1
    assert repetition["exact_duplicate_rows_total"] == 2
    assert repetition["exact_duplicate_rows_beyond_first"] == 1
    assert isinstance(history, dict) and history["invalid_json_rows"] == 1
    assert history["non_list_json_rows"] == 1
    assert isinstance(permissions, dict) and permissions["batch_training"] == "blocked"
    assert report["decision"] == {
        "classification": "reference/analytics only",
        "model_training_run": False,
        "merge_with_cars_or_yoad": False,
        "requirements_before_reconsideration": [
            "document permission covering ML reuse of KBB-derived valuation targets",
            "document permission to publish aggregate results and any trained derivative",
            "freeze the exact approved artifact and attribution/redistribution terms",
            "preregister grouped temporal validation and the leakage denylist",
        ],
    }

    output = tmp_path / "audit.json"
    write_autotrader_audit(report, output)
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert "listing-1" not in output.read_text(encoding="utf-8")


def test_profile_rejects_naive_timestamp_and_wrong_schema() -> None:
    frame = _review_frame()
    with pytest.raises(ValueError, match="timezone-aware"):
        profile_autotrader_frame(frame, manifest=[], generated_at=datetime(2026, 9, 2))

    with pytest.raises(ValueError, match="schema"):
        profile_autotrader_frame(
            frame.drop(columns=["year"]),
            manifest=[],
            generated_at=datetime(2026, 9, 2, tzinfo=UTC),
        )


def test_full_audit_reads_only_manifest_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _review_frame().drop(columns=["_snapshotFile"])
    artifact_path = tmp_path / "one.parquet"
    source.to_parquet(artifact_path, index=False)
    payload = artifact_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    base_spec = preview_artifact_specs()[0]
    spec = replace(
        base_spec,
        file_path=PurePosixPath("car-listings/data/one.parquet"),
        expected_size_bytes=len(payload),
        expected_sha256=digest,
        expected_row_count=5,
    )
    verified = VerifiedHuggingFaceArtifact(
        spec=spec,
        path=artifact_path,
        acquired_at=datetime(2026, 9, 2, tzinfo=UTC),
        sha256=digest,
        size_bytes=len(payload),
        cache_hit=True,
    )
    monkeypatch.setattr(audit_module, "preview_artifact_specs", lambda: (spec,))

    report = audit_autotrader_preview((verified,), generated_at=datetime(2026, 9, 2, tzinfo=UTC))

    assert report["source"]["file_count"] == 1  # type: ignore[index]
    with pytest.raises(ValueError, match="manifest"):
        audit_autotrader_preview(())
