from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import autovalue_ml.modeling.retail_final_evaluation_cli as cli
import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling.calibration_artifact import (
    CALIBRATION_ASSIGNMENT_SHA256,
    RetailCalibrationArtifact,
)
from autovalue_ml.modeling.final_evaluation_policy import (
    FINAL_EVALUATION_POLICY_SHA256,
    FinalEvaluationPolicy,
    load_final_evaluation_policy_file,
)
from autovalue_ml.modeling.retail_final_evaluation import FinalEvaluationResult

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = PROJECT_ROOT / "docs" / "experiments" / "retail-rf05-final-evaluation-policy-v1.json"


class CountedRows:
    def __init__(self, count: int) -> None:
        self.expected_rows = count

    def __iter__(self) -> Iterator[tuple[Mapping[str, object], float]]:
        return iter(())


def _features(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2020] * rows,
            "make": ["Ford"] * rows,
            "model": ["F-150"] * rows,
            "vehicle_status": ["used"] * rows,
            "mileage": np.arange(rows, dtype=np.float64),
        }
    )


def test_actual_frozen_evidence_bindings_verify_without_holdout_access() -> None:
    evidence = cli._verify_frozen_evidence(PROJECT_ROOT)

    assert evidence.policy.policy_sha256 == FINAL_EVALUATION_POLICY_SHA256
    assert len(evidence.file_entries) == 26
    assert len(evidence.implementation_entries) == 3
    assert {entry["role"] for entry in evidence.file_entries} >= {
        "candidate",
        "split_assignments",
        "calibration_artifact",
        "sharpness_report",
    }


def test_output_preflight_blocks_reopening_before_input_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "docs" / "experiments").mkdir(parents=True)
    (tmp_path / "docs" / "model-cards").mkdir(parents=True)
    report = tmp_path / cli._REPORT
    report.write_text("already complete", encoding="utf-8")
    verified = False

    def forbidden_verify(project_root: Path) -> object:
        nonlocal verified
        verified = True
        return object()

    monkeypatch.setattr(cli, "_validate_project_root", lambda path: path)
    monkeypatch.setattr(cli, "_verify_frozen_evidence", forbidden_verify)

    with pytest.raises(SystemExit):
        cli.main(["--project-root", str(tmp_path)])
    assert verified is False


def test_main_orders_verification_train_and_single_holdout_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = load_final_evaluation_policy_file(POLICY_PATH)
    events: list[str] = []
    train = SimpleNamespace(features=_features(6), target=np.arange(6, dtype=np.float64) + 1.0)
    holdout = SimpleNamespace(features=_features(3), target=np.arange(3, dtype=np.float64) + 2.0)
    evidence = cli._VerifiedEvidence(
        policy=policy,
        artifact=cast(RetailCalibrationArtifact, object()),
        prior_report={},
        file_entries=(),
        implementation_entries=(),
    )
    outputs = {
        "report": tmp_path / "report.json",
        "markdown": tmp_path / "report.md",
        "model_card": tmp_path / "card.md",
        "manifest": tmp_path / "manifest.json",
    }

    monkeypatch.setattr(cli, "_validate_project_root", lambda path: path)

    def validate_outputs(project_root: Path) -> dict[str, Path]:
        events.append("outputs")
        return outputs

    def verify(project_root: Path) -> cli._VerifiedEvidence:
        events.append("verify")
        return evidence

    monkeypatch.setattr(cli, "_validated_output_paths", validate_outputs)
    monkeypatch.setattr(cli, "_verify_frozen_evidence", verify)

    def prepare(*paths: Path, partition: str) -> CountedRows:
        events.append(f"open-{partition}")
        return CountedRows(6 if partition == "train" else 3)

    monkeypatch.setattr(cli, "prepare_kaggle_us_sales_cars_split_training_rows", prepare)
    collections = iter((train, holdout))

    def collect(rows: object, *, expected_rows: int, label: str) -> object:
        events.append(f"collect-{label}")
        return next(collections)

    monkeypatch.setattr(cli, "_collect_retail_partition", collect)
    monkeypatch.setattr(
        cli,
        "retail_calibration_partition",
        lambda features, seed: SimpleNamespace(
            development_indices=np.asarray([0, 1, 2, 3]),
            calibration_indices=np.asarray([4, 5]),
        ),
    )
    monkeypatch.setattr(
        cli, "_partition_hash", lambda *args, **kwargs: CALIBRATION_ASSIGNMENT_SHA256
    )

    def fit(**kwargs: object) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        events.append("fit")
        assert "holdout_target" not in kwargs
        assert len(cast(pd.DataFrame, kwargs["development_features"])) == 4
        assert len(cast(pd.DataFrame, kwargs["holdout_features"])) == 3
        return np.full(3, 20_000.0)

    monkeypatch.setattr(cli, "fit_frozen_rf05_for_final", fit)

    def evaluate(**kwargs: object) -> FinalEvaluationResult:
        events.append("evaluate")
        np.testing.assert_array_equal(kwargs["holdout_target"], holdout.target)
        return FinalEvaluationResult(report={}, classification="classification")

    monkeypatch.setattr(cli, "evaluate_final_holdout", evaluate)
    monkeypatch.setattr(cli, "canonical_final_report_json", lambda report: "{}\n")
    monkeypatch.setattr(cli, "render_final_report", lambda result, report_sha256: "report\n")
    monkeypatch.setattr(cli, "render_model_card", lambda result, report_sha256: "card\n")
    monkeypatch.setattr(
        cli,
        "_revalidate_entries",
        lambda project_root, entries: events.append("revalidate"),
    )
    monkeypatch.setattr(
        cli,
        "_publish_outputs",
        lambda *args, **kwargs: events.append("publish"),
    )

    assert cli.main(["--project-root", str(tmp_path)]) == 0
    assert events == [
        "outputs",
        "verify",
        "open-train",
        "collect-retail train",
        "open-test",
        "collect-retail final holdout",
        "fit",
        "evaluate",
        "revalidate",
        "publish",
    ]


def test_manifest_is_published_last(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    outputs = {
        "report": tmp_path / "report.json",
        "markdown": tmp_path / "report.md",
        "model_card": tmp_path / "card.md",
        "manifest": tmp_path / "manifest.json",
    }
    order: list[Path] = []
    monkeypatch.setattr(cli, "_write_atomic", lambda path, payload: order.append(path))
    monkeypatch.setattr(
        cli,
        "_created_output",
        lambda path, expected_sha256: cli._CreatedOutput(path, expected_sha256, (1, 2, 3, 4)),
    )
    monkeypatch.setattr(cli, "_validate_created_output", lambda output: None)

    cli._publish_outputs(
        outputs,
        report_json="report",
        markdown="markdown",
        model_card="card",
        manifest_json="manifest",
    )

    assert order == [
        outputs["report"],
        outputs["markdown"],
        outputs["model_card"],
        outputs["manifest"],
    ]


def test_policy_binding_declarations_are_exact() -> None:
    policy: FinalEvaluationPolicy = load_final_evaluation_policy_file(POLICY_PATH)

    cli._validate_policy_file_bindings(policy)
    original = cli._BOUND_FILES["candidate"]
    try:
        cli._BOUND_FILES["candidate"] = (original[0], "0" * 64)
        with pytest.raises(cli.FinalEvaluationCLIError, match="runner file bindings"):
            cli._validate_policy_file_bindings(policy)
    finally:
        cli._BOUND_FILES["candidate"] = original
