"""Governed RF05 reconstruction orchestration and identity tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling.contracts import FeatureContractError
from autovalue_ml.modeling.phase4_confirmation import parse_phase4_confirmation_json
from autovalue_ml.modeling.phase4_evaluation import Phase4CandidateCVResult
from autovalue_ml.modeling.rf05_serving_bundle import (
    AuthorizedDevelopmentData,
    RF05ServingBundleError,
    _compare_reproduction,
    _validated_bundle_output,
    bundle_sha256,
    development_identity_sha256,
    reconstruct_rf05_serving_bundle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION = PROJECT_ROOT / "docs/experiments/retail-rf05-calibration-v1.artifact.json"
CONFIRMATION = PROJECT_ROOT / "docs/experiments/phase4-retail-full-development-v1.json"


def _tiny_development() -> AuthorizedDevelopmentData:
    features = pd.DataFrame(
        {
            "year": [2018, 2019, 2020, 2021, 2022, 2017, 2016, 2015],
            "make": ["Toyota", "Honda", "Ford", "Toyota", "Honda", "Ford", "Toyota", "Ford"],
            "model": ["Camry", "Civic", "F-150", "Camry", "Civic", "F-150", "Camry", "F-150"],
            "vehicle_status": ["used"] * 8,
            "mileage": [80_000, 70_000, 60_000, 40_000, 20_000, 90_000, 110_000, 120_000],
        }
    )
    target = np.asarray(
        [17_000, 18_000, 27_000, 23_000, 25_000, 22_000, 14_000, 19_000],
        dtype=np.float64,
    )
    return AuthorizedDevelopmentData(
        features=features,
        target=target,
        identity_sha256=development_identity_sha256(features, target),
        calibration_assignment_sha256=(
            "caa743681158c4eaccb2ec75ce17a1c5e20327a311f66c5e8e0d0c630c48e992"
        ),
        status_counts={"certified": 0, "new": 0, "used": 8},
    )


def test_development_identity_is_order_and_target_bound() -> None:
    development = _tiny_development()
    same = development_identity_sha256(development.features, development.target)
    changed_target = development.target.copy()
    changed_target[0] += 1

    assert same == development.identity_sha256
    assert development_identity_sha256(development.features, changed_target) != same
    assert (
        development_identity_sha256(
            development.features.iloc[::-1].reset_index(drop=True),
            development.target[::-1],
        )
        != same
    )


def test_development_identity_rejects_unapproved_features() -> None:
    development = _tiny_development()
    invalid = development.features.assign(dealer="forbidden")

    with pytest.raises(FeatureContractError, match="forbidden feature"):
        development_identity_sha256(invalid, development.target)


def test_bundle_fingerprint_is_domain_and_order_bound() -> None:
    first = bundle_sha256(b"manifest", b"model")

    assert first == bundle_sha256(b"manifest", b"model")
    assert first != bundle_sha256(b"model", b"manifest")
    with pytest.raises(RF05ServingBundleError, match="must not be empty"):
        bundle_sha256(b"", b"model")


def test_frozen_oof_evidence_compares_at_exact_tolerance() -> None:
    report = parse_phase4_confirmation_json(CONFIRMATION.read_bytes())
    rf05 = next(
        result
        for result in report.candidates
        if result.spec.candidate_id == "phase4-retail-random_forest-05"
    )

    comparison = _compare_reproduction(rf05, rf05)

    assert comparison["passed"] is True
    assert comparison["maximum_absolute_metric_delta"] == 0.0
    assert comparison["fold_and_status_metrics_compared"] is True


def test_reconstruction_orchestration_writes_private_bundle_and_aggregate_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for directory in (
        "ml",
        "models",
        "docs/experiments",
        "tests/fixtures",
    ):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (tmp_path / "docs/experiments" / CALIBRATION.name).write_bytes(CALIBRATION.read_bytes())
    development = _tiny_development()
    upstream = {
        "data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.csv": "1" * 64,
        "data/processed/kaggle_us_sales_cars_v2/split/split_assignments.csv": "2" * 64,
    }
    placeholder = cast(Phase4CandidateCVResult, object())

    monkeypatch.setattr(
        "autovalue_ml.modeling.rf05_serving_bundle._verify_policy_and_upstream",
        lambda _: (
            {"classification": "deterministic reconstruction test"},
            upstream,
        ),
    )
    monkeypatch.setattr(
        "autovalue_ml.modeling.rf05_serving_bundle._load_authorized_development",
        lambda _: development,
    )
    monkeypatch.setattr(
        "autovalue_ml.modeling.rf05_serving_bundle._load_rf05_reference",
        lambda _: placeholder,
    )
    monkeypatch.setattr(
        "autovalue_ml.modeling.rf05_serving_bundle._reproduce_oof",
        lambda _: placeholder,
    )
    monkeypatch.setattr(
        "autovalue_ml.modeling.rf05_serving_bundle._compare_reproduction",
        lambda _expected, _actual: {"passed": True, "sample_count": 98_552},
    )

    result = reconstruct_rf05_serving_bundle(
        project_root=tmp_path,
        bundle_dir=Path("models/retail-rf05-v1"),
        report_path=Path("docs/experiments/reconstruction.json"),
        golden_fixture_path=Path("tests/fixtures/golden.json"),
    )

    assert {path.name for path in result.bundle_dir.iterdir()} == {
        "manifest.json",
        "model.joblib",
    }
    report = json.loads((tmp_path / "docs/experiments/reconstruction.json").read_bytes())
    golden = json.loads((tmp_path / "tests/fixtures/golden.json").read_bytes())
    assert report["decision"] == "passed_for_private_local_serving"
    assert report["serving_bundle"]["raw_training_rows_included"] is False
    assert report["determinism"]["maximum_development_prediction_difference_usd"] == 0.0
    assert report["determinism"]["serialized_byte_identical"] is True
    assert golden["source_rows_included"] is False
    assert len(golden["fixtures"]) == 5


def test_bundle_output_must_be_exact_trusted_location(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()

    with pytest.raises(RF05ServingBundleError, match="must be models"):
        _validated_bundle_output(tmp_path, Path("other/model"), force=False)
