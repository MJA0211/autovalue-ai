from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling import retail_uncertainty_diagnostics_cli
from autovalue_ml.modeling.calibration_artifact import (
    CALIBRATION_ASSIGNMENT_SHA256,
    CALIBRATION_POLICY_SHA256,
    PHASE4_RETAIL_CONFIRMATION_SHA256,
)
from autovalue_ml.modeling.retail_calibration_experiment import CALIBRATION_SEED


def test_cli_uses_only_frozen_development_rows_and_writes_one_immutable_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    experiments = project / "docs" / "experiments"
    experiments.mkdir(parents=True)
    confirmation_bytes = b"frozen phase 4 confirmation"
    confirmation = object()
    features = pd.DataFrame(
        {"row_marker": ["development-a", "calibration", "development-b", "development-c"]}
    )
    target = np.asarray([10_000.0, 20_000.0, 30_000.0, 40_000.0])
    requested_partitions: list[str] = []
    verified_inputs: list[tuple[Path, str, str]] = []
    validated_outputs: list[tuple[Path, Path, bool]] = []
    writes: list[tuple[Path, str, bool]] = []
    partition_inputs: list[pd.DataFrame] = []
    partition_hash_inputs: list[tuple[list[int], int, str, str]] = []
    diagnostic_inputs: list[tuple[pd.DataFrame, np.ndarray, object, str, object]] = []

    def validate_project_root(path: Path) -> Path:
        assert path == project
        return path

    def validate_output(
        path: Path,
        *,
        project_root: Path,
        force: bool,
    ) -> Path:
        validated_outputs.append((path, project_root, force))
        return path

    def verified_bytes(path: Path, *, expected: str, label: str) -> bytes:
        verified_inputs.append((path, expected, label))
        if label == "Phase 4 retail confirmation":
            return confirmation_bytes
        return b"verified immutable calibration v1 input"

    def parse_confirmation(payload: bytes) -> object:
        assert payload == confirmation_bytes
        return confirmation

    def prepare_rows(*paths: Path, partition: str) -> object:
        assert paths
        requested_partitions.append(partition)
        return SimpleNamespace(expected_rows=len(features))

    def collect_partition(stream: object, *, expected_rows: int, label: str) -> object:
        assert stream is not None
        assert expected_rows == len(features)
        assert label == "retail train"
        return SimpleNamespace(features=features, target=target)

    def calibration_partition(frame: pd.DataFrame, *, seed: int) -> object:
        assert seed == CALIBRATION_SEED
        partition_inputs.append(frame.copy(deep=True))
        return SimpleNamespace(
            development_indices=np.asarray([0, 2, 3], dtype=np.int64),
            calibration_indices=np.asarray([1], dtype=np.int64),
        )

    def partition_hash(
        indices: np.ndarray,
        *,
        population_count: int,
        selected_label: str,
        unselected_label: str,
    ) -> str:
        partition_hash_inputs.append(
            (indices.tolist(), population_count, selected_label, unselected_label)
        )
        return CALIBRATION_ASSIGNMENT_SHA256

    def build_diagnostics(
        *,
        development_features: object,
        development_target: object,
        confirmation: object,
        confirmation_sha256: str,
        progress: object,
    ) -> dict[str, object]:
        assert isinstance(development_features, pd.DataFrame)
        diagnostic_inputs.append(
            (
                development_features.copy(deep=True),
                np.asarray(development_target, dtype=np.float64),
                confirmation,
                confirmation_sha256,
                progress,
            )
        )
        return {"report_type": "synthetic-development-only-diagnostic"}

    def canonical_json(report: object) -> str:
        assert report == {"report_type": "synthetic-development-only-diagnostic"}
        return '{"report_type":"synthetic-development-only-diagnostic"}\n'

    def write_atomic(path: Path, serialized: str, *, force: bool) -> None:
        writes.append((path, serialized, force))

    monkeypatch.setattr(
        retail_uncertainty_diagnostics_cli,
        "_validate_project_root",
        validate_project_root,
    )
    monkeypatch.setattr(
        retail_uncertainty_diagnostics_cli,
        "_validate_output_path",
        validate_output,
    )
    monkeypatch.setattr(
        retail_uncertainty_diagnostics_cli,
        "_verified_bytes",
        verified_bytes,
    )
    monkeypatch.setattr(
        retail_uncertainty_diagnostics_cli,
        "parse_phase4_confirmation_json",
        parse_confirmation,
    )
    monkeypatch.setattr(
        retail_uncertainty_diagnostics_cli,
        "prepare_kaggle_us_sales_cars_split_training_rows",
        prepare_rows,
    )
    monkeypatch.setattr(
        retail_uncertainty_diagnostics_cli,
        "_collect_retail_partition",
        collect_partition,
    )
    monkeypatch.setattr(
        retail_uncertainty_diagnostics_cli,
        "retail_calibration_partition",
        calibration_partition,
    )
    monkeypatch.setattr(
        retail_uncertainty_diagnostics_cli,
        "_partition_hash",
        partition_hash,
    )
    monkeypatch.setattr(
        retail_uncertainty_diagnostics_cli,
        "build_development_residual_diagnostics",
        build_diagnostics,
    )
    monkeypatch.setattr(
        retail_uncertainty_diagnostics_cli,
        "canonical_diagnostics_json",
        canonical_json,
    )
    monkeypatch.setattr(retail_uncertainty_diagnostics_cli, "_write_atomic", write_atomic)

    assert retail_uncertainty_diagnostics_cli.main(["--project-root", str(project)]) == 0

    expected_output = experiments / "retail-rf05-development-residual-diagnostics-v1.json"
    assert requested_partitions == ["train"]
    assert all(
        "test" not in partition.lower() and "holdout" not in partition.lower()
        for partition in requested_partitions
    )
    assert validated_outputs == [(expected_output, project, False)]
    assert writes == [
        (
            expected_output,
            '{"report_type":"synthetic-development-only-diagnostic"}\n',
            False,
        )
    ]
    assert verified_inputs == [
        (
            experiments / "phase4-retail-full-development-v1.json",
            PHASE4_RETAIL_CONFIRMATION_SHA256,
            "Phase 4 retail confirmation",
        ),
        (
            experiments / "retail-rf05-calibration-policy-v1.json",
            CALIBRATION_POLICY_SHA256,
            "calibration v1 policy",
        ),
        (
            experiments / "retail-rf05-calibration-v1.artifact.json",
            retail_uncertainty_diagnostics_cli._CALIBRATION_ARTIFACT_SHA256,
            "calibration v1 serving artifact",
        ),
        (
            experiments / "retail-rf05-calibration-v1.report.json",
            retail_uncertainty_diagnostics_cli._CALIBRATION_REPORT_SHA256,
            "calibration v1 report",
        ),
    ]
    assert len(partition_inputs) == 1
    assert partition_inputs[0].equals(features)
    assert partition_hash_inputs == [([1], len(features), "calibration", "development")]
    assert len(diagnostic_inputs) == 1
    diagnostic_features, diagnostic_target, bound_confirmation, bound_hash, progress = (
        diagnostic_inputs[0]
    )
    assert diagnostic_features.index.tolist() == [0, 1, 2]
    assert diagnostic_features["row_marker"].tolist() == [
        "development-a",
        "development-b",
        "development-c",
    ]
    assert diagnostic_target.tolist() == [10_000.0, 30_000.0, 40_000.0]
    assert bound_confirmation is confirmation
    assert bound_hash == hashlib.sha256(confirmation_bytes).hexdigest()
    assert progress is retail_uncertainty_diagnostics_cli._print_progress
    expected_digest = hashlib.sha256(writes[0][1].encode("utf-8")).hexdigest()
    assert capsys.readouterr().out == (
        f"RF05 development residual diagnostics complete | sha256={expected_digest}\n"
    )
