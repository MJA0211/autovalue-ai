"""Tests for aggregate candidate-to-current-retail comparison."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath

import pandas as pd
import pytest
from autovalue_ml.acquisition.huggingface_comparison import (
    build_retail_candidate_comparison,
    write_comparison_report,
)
from autovalue_ml.acquisition.huggingface_dataset import VerifiedHuggingFaceArtifact
from autovalue_ml.acquisition.sources.huggingface_candidates import (
    CARSON_SHIVELY_SPEC,
    YOAD22_CRAIGSLIST_SPEC,
)

_NOW = datetime(2026, 9, 1, 12, tzinfo=UTC)


def _artifact(path: Path, *, carson: bool, rows: int) -> VerifiedHuggingFaceArtifact:
    payload = path.read_bytes()
    base = CARSON_SHIVELY_SPEC if carson else YOAD22_CRAIGSLIST_SPEC
    spec = replace(
        base,
        file_path=PurePosixPath(path.name),
        expected_size_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_row_count=rows,
    )
    return VerifiedHuggingFaceArtifact(
        spec=spec,
        path=path,
        acquired_at=_NOW,
        sha256=spec.expected_sha256,
        size_bytes=spec.expected_size_bytes,
        cache_hit=True,
    )


def test_comparison_reports_distribution_coverage_and_coarse_collisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_path = tmp_path / "current.csv"
    yoad_path = tmp_path / "yoad.csv"
    carson_path = tmp_path / "carson.csv"
    pd.DataFrame(
        [
            {
                "year": 2020,
                "make": "Ford",
                "model": "F-150",
                "mileage": 50_000,
                "vehicle_status": "used",
                "price_cents": 2_000_000,
            },
            {
                "year": 2023,
                "make": "Toyota",
                "model": "Camry",
                "mileage": None,
                "vehicle_status": "new",
                "price_cents": 3_000_000,
            },
        ]
    ).to_csv(current_path, index=False)
    pd.DataFrame(
        [
            {
                "price": 20_000,
                "year": 2020,
                "manufacturer": "ford",
                "condition": "good",
                "cylinders": 6,
                "fuel": "gas",
                "odometer": 50_000,
                "title_status": "clean",
                "transmission": "automatic",
                "drive": "4wd",
                "type": "truck",
                "paint_color": "blue",
                "state": "ny",
                "car_age": 6,
            }
        ]
    ).to_csv(yoad_path, index=False)
    pd.DataFrame(
        [
            {
                "brand": "Ford",
                "model": "F-150",
                "model_year": 2020,
                "milage": "50,000 mi.",
                "fuel_type": "Gasoline",
                "engine": "V6",
                "transmission": "Automatic",
                "ext_col": "Blue",
                "int_col": "Black",
                "accident": "None reported",
                "clean_title": "Yes",
                "price": "$20,000",
            },
            {
                "brand": "New Make",
                "model": "Rare Model",
                "model_year": 2018,
                "milage": "75,000 mi.",
                "fuel_type": "Gasoline",
                "engine": "I4",
                "transmission": "Automatic",
                "ext_col": "Red",
                "int_col": "Gray",
                "accident": "None reported",
                "clean_title": "Yes",
                "price": "$15,000",
            },
        ]
    ).to_csv(carson_path, index=False)
    monkeypatch.setattr(
        "autovalue_ml.acquisition.huggingface_comparison.verify_kaggle_us_sales_cars_artifact_set",
        lambda *_args, **_kwargs: {},
    )

    report = build_retail_candidate_comparison(
        current_candidate_path=current_path,
        current_manifest_path=tmp_path / "manifest.json",
        current_review_path=tmp_path / "review.json",
        yoad_artifact=_artifact(yoad_path, carson=False, rows=1),
        carson_artifact=_artifact(carson_path, carson=True, rows=2),
        generated_at=_NOW,
        today=date(2026, 9, 1),
    )

    summary = report["distribution_summary"]
    assert summary["current_retail"]["rows"] == 2  # type: ignore[index]
    assert summary["yoad22_craigslist"]["price_usd"]["median"] == 20_000  # type: ignore[index]
    collisions = report["coarse_cross_source_key_collisions"]
    assert collisions["yoad22_craigslist"]["shared_unique_keys"] == 1  # type: ignore[index]
    assert collisions["carson_shively"]["shared_unique_keys"] == 1  # type: ignore[index]
    assert report["merge_decision"]["merged"] is False  # type: ignore[index]

    output = tmp_path / "comparison.json"
    write_comparison_report(report, output)
    assert '"merged": false' in output.read_text(encoding="utf-8")
