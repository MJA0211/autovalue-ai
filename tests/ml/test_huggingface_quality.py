"""Aggregate-only Hugging Face candidate quality report tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from autovalue_ml.acquisition.huggingface_dataset import acquire_huggingface_artifact
from autovalue_ml.acquisition.huggingface_quality import profile_huggingface_candidate
from autovalue_ml.acquisition.sources.huggingface_candidates import YOAD22_CRAIGSLIST_SPEC

_NOW = datetime(2026, 9, 1, 12, tzinfo=UTC)


def test_quality_report_reconciles_rows_and_retains_separate_use_decisions(
    tmp_path: Path,
) -> None:
    payload = (
        b"price,year,manufacturer,condition,cylinders,fuel,odometer,title_status,"
        b"transmission,drive,type,paint_color,state,car_age\n"
        b"15000,2020,toyota,good,4,gas,50000,clean,automatic,fwd,sedan,blue,ny,6\n"
        b"15000,2020,toyota,good,4,gas,50000,clean,automatic,fwd,sedan,blue,ny,6\n"
        b"9000,2018,honda,good,4,gas,70000,clean,automatic,fwd,sedan,red,on,8\n"
    )
    spec = replace(
        YOAD22_CRAIGSLIST_SPEC,
        file_path=PurePosixPath("fixture.csv"),
        expected_size_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_row_count=3,
    )

    def downloader(_url: str, path: Path) -> None:
        path.write_bytes(payload)

    artifact = acquire_huggingface_artifact(
        spec,
        tmp_path,
        downloader=downloader,
        acquired_at=_NOW,
    )

    report = profile_huggingface_candidate(artifact, generated_at=_NOW)
    quality = report["quality"]
    accounting = quality["row_accounting"]  # type: ignore[index]

    assert accounting == {
        "raw_rows": 3,
        "accepted_rows": 1,
        "rejected_rows": 1,
        "exact_duplicate_rows": 1,
        "acceptance_is_training_approval": False,
    }
    permissions = report["license_and_permissions"]
    assert permissions["batch_training"] == "approved"  # type: ignore[index]
    assert permissions["online_learning"] == "blocked"  # type: ignore[index]
    assert report["promotion_decision"]["training_experiment_started"] is True  # type: ignore[index]
    assert report["promotion_decision"]["merged_into_existing_training_data"] is False  # type: ignore[index]
    assert report["schema"]["source_id_is_predictive_feature"] is False  # type: ignore[index]
