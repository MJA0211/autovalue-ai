from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling import baseline_cli
from autovalue_ml.modeling import retail_uncertainty_sharpness_cli as cli
from autovalue_ml.modeling.calibration_artifact import (
    CALIBRATION_ASSIGNMENT_SHA256,
    CALIBRATION_POLICY_SHA256,
    PHASE4_PROTOCOL_SHA256,
    PHASE4_RETAIL_CONFIRMATION_SHA256,
)
from autovalue_ml.modeling.metrics import regression_metrics
from autovalue_ml.modeling.retail_calibration_experiment import CALIBRATION_SEED
from autovalue_ml.modeling.retail_uncertainty_sharpness import (
    BASELINE_METHOD,
    CALIBRATION_V1_ARTIFACT_SHA256,
    CALIBRATION_V1_REPORT_SHA256,
    DEVELOPMENT_DIAGNOSTICS_SHA256,
    GAMMA_METHOD,
    SHARPNESS_POLICY_SHA256,
    SMOOTH_METHOD,
)
from numpy.typing import NDArray


@dataclass
class _MainCapture:
    requested_partitions: list[str] = field(default_factory=list)
    source_paths: tuple[Path, ...] = ()
    policy_paths: list[Path] = field(default_factory=list)
    partition_features: pd.DataFrame | None = None
    partition_hash_call: tuple[list[int], int, str, str] | None = None
    oof_features: pd.DataFrame | None = None
    oof_target: NDArray[np.float64] | None = None
    oof_progress: object | None = None
    fit_development_features: pd.DataFrame | None = None
    fit_development_target: NDArray[np.float64] | None = None
    fit_calibration_features: pd.DataFrame | None = None
    compare_arguments: dict[str, object] | None = None
    gamma_payload_models: list[object] = field(default_factory=list)
    candidate_arguments: dict[str, object] | None = None
    candidate_loads: list[tuple[str, str, str, str]] = field(default_factory=list)
    rendered_reports: list[Mapping[str, object]] = field(default_factory=list)
    rendered_hashes: list[tuple[str, str | None, str | None]] = field(default_factory=list)
    text_writes: list[tuple[Path, str, bool]] = field(default_factory=list)
    binary_writes: list[tuple[Path, bytes]] = field(default_factory=list)
    loaded_policy: object | None = None


def _run_mocked_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    selected_method: str,
) -> tuple[Path, _MainCapture]:
    project = tmp_path / selected_method
    (project / "docs" / "experiments").mkdir(parents=True)
    capture = _MainCapture()
    features = pd.DataFrame(
        {
            "row_marker": [
                "development-alpha",
                "protected-calibration",
                "development-bravo",
                "development-charlie",
            ],
            "model_year": [2018, 2019, 2020, 2021],
        }
    )
    target = np.asarray([10_111.0, 20_222.0, 30_333.0, 40_444.0], dtype=np.float64)
    development_predictions = np.asarray([10_001.0, 30_003.0, 40_004.0], dtype=np.float64)
    calibration_predictions = np.asarray([20_002.0], dtype=np.float64)
    diagnostics = {
        "report_type": "retail_rf05_development_residual_diagnostics",
        "classification": "aggregate_development_diagnostic_only",
        "point_prediction_metrics": regression_metrics(
            np.asarray([10_111.0, 30_333.0, 40_444.0], dtype=np.float64),
            development_predictions,
        ).to_dict(),
        "overall_residual_distribution": {
            "support": 3,
            "mean_absolute_residual_usd": 200.0,
            "median_absolute_residual_usd": 220.0,
            "residual_variance_usd2": 1_234.0,
            "absolute_residual_quantiles_usd": {"p50": 220.0},
            "residual_to_actual_price_ratio": {"mean": 0.01},
        },
        "predicted_value_relationship": {
            "log_prediction_log_residual_pearson": 0.4,
            "prediction_residual_spearman": 0.3,
            "mean_absolute_residual_usd_by_predicted_value_quartile": {
                "predicted_value_1": 100.0,
                "predicted_value_2": 150.0,
                "predicted_value_3": 250.0,
                "predicted_value_4": 400.0,
            },
            "highest_to_lowest_quartile_mean_residual_ratio": 4.0,
        },
    }
    calibration_report: Mapping[str, object] = {"frozen_calibration_report": True}
    gamma_model = object()
    full_quantiles: Mapping[str, object] = {"0.9": {"quantile": 1_234.5}}
    base_report: Mapping[str, object] = {
        "report_type": "retail_rf05_uncertainty_sharpness_comparison",
        "generated_at": "2026-09-02T12:34:56+00:00",
        "classification": "controlled_experiment_only",
        "decision": {"selected_method": selected_method},
        "publication": {
            "aggregate_only": True,
            "raw_rows_predictions_residuals_or_category_vocabularies_in_report": False,
        },
    }

    def load_frozen_evidence(project_root: Path) -> dict[str, object]:
        assert project_root == project
        return {
            "confirmation": object(),
            "calibration_report": calibration_report,
            "diagnostics": diagnostics,
        }

    policy = SimpleNamespace(policy_sha256=SHARPNESS_POLICY_SHA256)

    def load_policy(path: Path) -> object:
        capture.policy_paths.append(path)
        capture.loaded_policy = policy
        return policy

    def prepare_rows(*paths: Path, partition: str) -> object:
        capture.source_paths = paths
        capture.requested_partitions.append(partition)
        return SimpleNamespace(expected_rows=len(features))

    def collect_partition(
        stream: object,
        *,
        expected_rows: int | None,
        label: str,
    ) -> object:
        assert stream is not None
        assert expected_rows == len(features)
        assert label == "retail train"
        return SimpleNamespace(features=features, target=target)

    def calibration_partition(frame: pd.DataFrame, *, seed: int) -> object:
        capture.partition_features = frame.copy(deep=True)
        assert seed == CALIBRATION_SEED
        return SimpleNamespace(
            development_indices=np.asarray([0, 2, 3], dtype=np.int64),
            calibration_indices=np.asarray([1], dtype=np.int64),
        )

    def partition_hash(
        indices: NDArray[np.int64],
        *,
        population_count: int,
        selected_label: str,
        unselected_label: str,
    ) -> str:
        capture.partition_hash_call = (
            indices.tolist(),
            population_count,
            selected_label,
            unselected_label,
        )
        return CALIBRATION_ASSIGNMENT_SHA256

    def reconstruct_oof(
        *,
        development_features: object,
        development_target: object,
        progress: object,
    ) -> tuple[NDArray[np.float64], object]:
        capture.oof_features = cast(pd.DataFrame, development_features).copy(deep=True)
        capture.oof_target = np.asarray(development_target, dtype=np.float64)
        capture.oof_progress = progress
        return development_predictions.copy(), object()

    def fit_calibration_predictions(
        *,
        development_features: object,
        development_target: object,
        calibration_features: object,
    ) -> NDArray[np.float64]:
        capture.fit_development_features = cast(pd.DataFrame, development_features).copy(deep=True)
        capture.fit_development_target = np.asarray(development_target, dtype=np.float64)
        capture.fit_calibration_features = cast(pd.DataFrame, calibration_features).copy(deep=True)
        return calibration_predictions.copy()

    def compare_methods(**arguments: object) -> object:
        capture.compare_arguments = arguments
        return SimpleNamespace(
            report=base_report,
            selected_method=selected_method,
            gamma_scale_model=gamma_model,
            full_quantiles=full_quantiles,
        )

    def gamma_payload(model: object) -> bytes:
        capture.gamma_payload_models.append(model)
        return b"immutable-gamma-model-payload"

    def build_artifact(**arguments: object) -> object:
        capture.candidate_arguments = arguments
        return arguments

    def canonical_candidate(artifact: object) -> str:
        return json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n"

    def load_candidate(
        serialized: str,
        *,
        active_model_identity_sha256: str,
        expected_artifact_sha256: str,
        expected_comparison_evidence_sha256: str,
    ) -> object:
        capture.candidate_loads.append(
            (
                serialized,
                active_model_identity_sha256,
                expected_artifact_sha256,
                expected_comparison_evidence_sha256,
            )
        )
        return object()

    def canonical_report(report: Mapping[str, object]) -> str:
        return json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"

    def render_report(
        report: Mapping[str, object],
        *,
        report_sha256: str,
        candidate_sha256: str | None,
        model_sha256: str | None,
    ) -> str:
        capture.rendered_reports.append(report)
        capture.rendered_hashes.append((report_sha256, candidate_sha256, model_sha256))
        return "# Aggregate-only sharpness report\n"

    def publish_outputs(
        outputs: Mapping[str, Path],
        *,
        report_json: str,
        markdown: str,
        candidate_json: str | None,
        model_payload: bytes | None,
    ) -> None:
        if model_payload is not None:
            capture.binary_writes.append((outputs["model"], model_payload))
        if candidate_json is not None:
            capture.text_writes.append((outputs["candidate"], candidate_json, False))
        capture.text_writes.append((outputs["markdown"], markdown, False))
        capture.text_writes.append((outputs["report"], report_json, False))

    monkeypatch.setattr(cli, "_load_frozen_evidence", load_frozen_evidence)
    monkeypatch.setattr(cli, "load_uncertainty_sharpness_policy_file", load_policy)
    monkeypatch.setattr(cli, "prepare_kaggle_us_sales_cars_split_training_rows", prepare_rows)
    monkeypatch.setattr(cli, "_collect_retail_partition", collect_partition)
    monkeypatch.setattr(cli, "retail_calibration_partition", calibration_partition)
    monkeypatch.setattr(cli, "_partition_hash", partition_hash)
    monkeypatch.setattr(cli, "reconstruct_rf05_development_oof", reconstruct_oof)
    monkeypatch.setattr(
        cli,
        "fit_frozen_rf05_calibration_predictions",
        fit_calibration_predictions,
    )
    monkeypatch.setattr(cli, "compare_uncertainty_methods", compare_methods)
    monkeypatch.setattr(cli, "_gamma_model_payload", gamma_payload)
    monkeypatch.setattr(cli, "build_candidate_artifact", build_artifact)
    monkeypatch.setattr(cli, "canonical_candidate_artifact_json", canonical_candidate)
    monkeypatch.setattr(cli, "load_candidate_artifact", load_candidate)
    monkeypatch.setattr(cli, "canonical_sharpness_report_json", canonical_report)
    monkeypatch.setattr(cli, "render_sharpness_markdown", render_report)
    monkeypatch.setattr(cli, "_publish_outputs", publish_outputs)

    assert cli.main(["--project-root", str(project)]) == 0
    return project, capture


@pytest.mark.parametrize("selected_method", [BASELINE_METHOD, SMOOTH_METHOD, GAMMA_METHOD])
def test_main_uses_only_train_and_preserves_exact_development_calibration_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_method: str,
) -> None:
    project, capture = _run_mocked_main(
        tmp_path,
        monkeypatch,
        selected_method=selected_method,
    )

    assert capture.requested_partitions == ["train"]
    assert not any(
        forbidden in requested.lower()
        for requested in capture.requested_partitions
        for forbidden in ("test", "holdout")
    )
    assert capture.source_paths
    assert capture.policy_paths == [
        project / "docs" / "experiments" / "retail-rf05-uncertainty-sharpness-policy-v1.json"
    ]
    assert capture.partition_features is not None
    assert capture.partition_features["row_marker"].tolist() == [
        "development-alpha",
        "protected-calibration",
        "development-bravo",
        "development-charlie",
    ]
    assert capture.partition_hash_call == ([1], 4, "calibration", "development")

    assert capture.oof_features is not None
    assert capture.oof_features.index.tolist() == [0, 1, 2]
    assert capture.oof_features["row_marker"].tolist() == [
        "development-alpha",
        "development-bravo",
        "development-charlie",
    ]
    assert capture.oof_target is not None
    np.testing.assert_array_equal(capture.oof_target, [10_111.0, 30_333.0, 40_444.0])
    assert capture.oof_progress is cli._print_oof_progress

    assert capture.fit_development_features is not None
    assert capture.fit_development_features.equals(capture.oof_features)
    assert capture.fit_development_target is not None
    np.testing.assert_array_equal(
        capture.fit_development_target,
        [10_111.0, 30_333.0, 40_444.0],
    )
    assert capture.fit_calibration_features is not None
    assert capture.fit_calibration_features.index.tolist() == [0]
    assert capture.fit_calibration_features["row_marker"].tolist() == ["protected-calibration"]

    assert capture.compare_arguments is not None
    comparison = capture.compare_arguments
    assert cast(pd.DataFrame, comparison["development_features"]).equals(capture.oof_features)
    np.testing.assert_array_equal(
        comparison["development_target"],
        [10_111.0, 30_333.0, 40_444.0],
    )
    np.testing.assert_array_equal(
        comparison["development_oof_predictions"],
        [10_001.0, 30_003.0, 40_004.0],
    )
    assert cast(pd.DataFrame, comparison["calibration_features"])["row_marker"].tolist() == [
        "protected-calibration"
    ]
    np.testing.assert_array_equal(comparison["calibration_target"], [20_222.0])
    np.testing.assert_array_equal(comparison["calibration_predictions"], [20_002.0])
    assert comparison["calibration_v1_report"] == {"frozen_calibration_report": True}
    assert comparison["policy"] is capture.loaded_policy


def test_baseline_writes_only_report_and_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, capture = _run_mocked_main(
        tmp_path,
        monkeypatch,
        selected_method=BASELINE_METHOD,
    )

    assert capture.gamma_payload_models == []
    assert capture.candidate_arguments is None
    assert capture.candidate_loads == []
    assert capture.binary_writes == []
    assert [(path, force) for path, _, force in capture.text_writes] == [
        (
            project / "docs" / "experiments" / "retail-rf05-uncertainty-sharpness-v1.md",
            False,
        ),
        (
            project / "docs" / "experiments" / "retail-rf05-uncertainty-sharpness-v1.report.json",
            False,
        ),
    ]
    artifacts = cast(Mapping[str, object], capture.rendered_reports[0]["artifacts"])
    assert artifacts["candidate_serving_artifact"] is None
    assert artifacts["residual_scale_model"] is None
    assert artifacts["new_serving_state_persisted"] is False


def test_smooth_selection_writes_checksum_bound_artifact_but_no_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, capture = _run_mocked_main(
        tmp_path,
        monkeypatch,
        selected_method=SMOOTH_METHOD,
    )

    assert capture.gamma_payload_models == []
    assert capture.binary_writes == []
    assert capture.candidate_arguments == {
        "selected_method": SMOOTH_METHOD,
        "full_quantiles": {"0.9": {"quantile": 1_234.5}},
        "generated_at": "2026-09-02T12:34:56+00:00",
        "comparison_evidence_sha256": hashlib.sha256(
            json.dumps(
                {
                    "classification": "controlled_experiment_only",
                    "decision": {"selected_method": SMOOTH_METHOD},
                    "generated_at": "2026-09-02T12:34:56+00:00",
                    "publication": {
                        "aggregate_only": True,
                        "raw_rows_predictions_residuals_or_category_vocabularies_in_report": False,
                    },
                    "report_type": "retail_rf05_uncertainty_sharpness_comparison",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        ).hexdigest(),
        "gamma_model_path": None,
        "gamma_model_sha256": None,
    }
    assert len(capture.candidate_loads) == 1
    serialized_candidate, active_identity, trusted_candidate_sha256, comparison_sha256 = (
        capture.candidate_loads[0]
    )
    assert active_identity == cli._RF05_IDENTITY_SHA256
    candidate_sha256 = hashlib.sha256(serialized_candidate.encode("utf-8")).hexdigest()
    assert trusted_candidate_sha256 == candidate_sha256
    assert comparison_sha256 == capture.candidate_arguments["comparison_evidence_sha256"]
    assert [path for path, _, _ in capture.text_writes] == [
        project / "docs" / "experiments" / "retail-rf05-uncertainty-sharpness-v2.artifact.json",
        project / "docs" / "experiments" / "retail-rf05-uncertainty-sharpness-v1.md",
        project / "docs" / "experiments" / "retail-rf05-uncertainty-sharpness-v1.report.json",
    ]
    artifacts = cast(Mapping[str, object], capture.rendered_reports[0]["artifacts"])
    assert artifacts["comparison_evidence_sha256"] == comparison_sha256
    assert artifacts["candidate_serving_artifact_sha256"] == candidate_sha256
    assert artifacts["residual_scale_model"] is None
    assert capture.rendered_hashes[0][1:] == (candidate_sha256, None)


def test_gamma_selection_writes_checksum_bound_artifact_and_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, capture = _run_mocked_main(
        tmp_path,
        monkeypatch,
        selected_method=GAMMA_METHOD,
    )

    model_payload = b"immutable-gamma-model-payload"
    model_sha256 = hashlib.sha256(model_payload).hexdigest()
    assert len(capture.gamma_payload_models) == 1
    assert capture.candidate_arguments == {
        "selected_method": GAMMA_METHOD,
        "full_quantiles": {"0.9": {"quantile": 1_234.5}},
        "generated_at": "2026-09-02T12:34:56+00:00",
        "comparison_evidence_sha256": cast(
            Mapping[str, object], capture.rendered_reports[0]["artifacts"]
        )["comparison_evidence_sha256"],
        "gamma_model_path": "models/uncertainty/retail-rf05-gamma-residual-scale-v1.joblib",
        "gamma_model_sha256": model_sha256,
    }
    assert capture.binary_writes == [
        (
            project / "models" / "uncertainty" / "retail-rf05-gamma-residual-scale-v1.joblib",
            model_payload,
        )
    ]
    assert len(capture.candidate_loads) == 1
    serialized_candidate, active_identity, trusted_candidate_sha256, comparison_sha256 = (
        capture.candidate_loads[0]
    )
    assert active_identity == cli._RF05_IDENTITY_SHA256
    assert model_sha256 in serialized_candidate
    candidate_sha256 = hashlib.sha256(serialized_candidate.encode("utf-8")).hexdigest()
    assert trusted_candidate_sha256 == candidate_sha256
    assert comparison_sha256 == capture.candidate_arguments["comparison_evidence_sha256"]
    artifacts = cast(Mapping[str, object], capture.rendered_reports[0]["artifacts"])
    assert artifacts["residual_scale_model_sha256"] == model_sha256
    assert artifacts["candidate_serving_artifact_sha256"] == candidate_sha256
    assert artifacts["comparison_evidence_sha256"] == comparison_sha256
    assert artifacts["new_serving_state_persisted"] is True
    assert capture.rendered_hashes[0][1:] == (candidate_sha256, model_sha256)


@pytest.mark.parametrize("selected_method", [BASELINE_METHOD, SMOOTH_METHOD, GAMMA_METHOD])
def test_written_report_is_aggregate_only_and_contains_no_input_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_method: str,
) -> None:
    _, capture = _run_mocked_main(
        tmp_path,
        monkeypatch,
        selected_method=selected_method,
    )

    report_payload = next(
        payload for path, payload, _ in capture.text_writes if path.name.endswith("report.json")
    )
    assert all(
        marker not in report_payload
        for marker in (
            "development-alpha",
            "development-bravo",
            "development-charlie",
            "protected-calibration",
        )
    )
    assert "development_target" not in report_payload
    assert "calibration_target" not in report_payload
    publication = cast(Mapping[str, object], capture.rendered_reports[0]["publication"])
    assert publication["aggregate_only"] is True
    assert publication["raw_rows_predictions_residuals_or_category_vocabularies_in_report"] is False
    diagnostic = cast(
        Mapping[str, object],
        capture.rendered_reports[0]["development_diagnostic_summary"],
    )
    assert diagnostic["classification"] == "aggregate_development_diagnostic_only"
    assert diagnostic["actual_price_used_for_evaluation_only"] is True


def test_output_prevalidation_uses_all_fixed_immutable_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    json_calls: list[tuple[Path, Path, bool]] = []
    markdown_calls: list[tuple[Path, Path]] = []
    binary_calls: list[tuple[Path, Path]] = []
    distinct_calls: list[tuple[tuple[Path, ...], tuple[Path, ...]]] = []

    def validate_json(path: Path, *, project_root: Path, force: bool) -> Path:
        json_calls.append((path, project_root, force))
        return path

    def validate_markdown(path: Path, *, project_root: Path) -> Path:
        markdown_calls.append((path, project_root))
        return path

    def validate_binary(path: Path, *, project_root: Path) -> Path:
        binary_calls.append((path, project_root))
        return path

    def ensure_distinct(outputs: tuple[Path, ...], inputs: tuple[Path, ...]) -> None:
        distinct_calls.append((outputs, inputs))

    monkeypatch.setattr(cli, "_validate_output_path", validate_json)
    monkeypatch.setattr(cli, "_validate_markdown_output", validate_markdown)
    monkeypatch.setattr(cli, "_validate_binary_output", validate_binary)
    monkeypatch.setattr(cli, "_ensure_paths_distinct", ensure_distinct)

    outputs = cli._validated_output_paths(project)

    experiments = project / "docs" / "experiments"
    assert outputs == {
        "report": experiments / "retail-rf05-uncertainty-sharpness-v1.report.json",
        "markdown": experiments / "retail-rf05-uncertainty-sharpness-v1.md",
        "candidate": experiments / "retail-rf05-uncertainty-sharpness-v2.artifact.json",
        "model": project / "models" / "uncertainty" / "retail-rf05-gamma-residual-scale-v1.joblib",
    }
    assert json_calls == [
        (outputs["report"], project, False),
        (outputs["candidate"], project, False),
    ]
    assert markdown_calls == [(outputs["markdown"], project)]
    assert binary_calls == [(outputs["model"], project)]
    assert len(distinct_calls) == 1
    validated_outputs, protected_inputs = distinct_calls[0]
    assert validated_outputs == (
        outputs["report"],
        outputs["markdown"],
        outputs["candidate"],
        outputs["model"],
    )
    assert protected_inputs == tuple(
        experiments / name
        for name in (
            "phase4-model-selection-v1.json",
            "phase4-retail-full-development-v1.json",
            "retail-rf05-calibration-policy-v1.json",
            "retail-rf05-calibration-v1.artifact.json",
            "retail-rf05-calibration-v1.report.json",
            "retail-rf05-development-residual-diagnostics-v1.json",
            "retail-rf05-uncertainty-sharpness-policy-v1.json",
        )
    )


@pytest.mark.parametrize(
    "relative_output",
    [
        "docs/experiments/retail-rf05-uncertainty-sharpness-v1.report.json",
        "docs/experiments/retail-rf05-uncertainty-sharpness-v1.md",
        "docs/experiments/retail-rf05-uncertainty-sharpness-v2.artifact.json",
        "models/uncertainty/retail-rf05-gamma-residual-scale-v1.joblib",
    ],
)
def test_existing_output_stops_main_before_frozen_reads_or_data_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_output: str,
) -> None:
    project = tmp_path / "project"
    (project / "docs" / "experiments").mkdir(parents=True)
    existing = project / Path(relative_output)
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"already immutable")
    frozen_read_attempted = False

    def forbidden_frozen_read(project_root: Path) -> dict[str, object]:
        nonlocal frozen_read_attempted
        frozen_read_attempted = True
        raise AssertionError(project_root)

    monkeypatch.setattr(cli, "_load_frozen_evidence", forbidden_frozen_read)

    with pytest.raises(SystemExit) as error:
        cli.main(["--project-root", str(project)])

    assert error.value.code == 2
    assert frozen_read_attempted is False


def _publication_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "model": tmp_path / "models" / "scale.joblib",
        "candidate": tmp_path / "experiments" / "candidate.json",
        "markdown": tmp_path / "experiments" / "report.md",
        "report": tmp_path / "experiments" / "report.json",
    }


@pytest.mark.parametrize(
    ("candidate_json", "model_payload", "expected_order"),
    [
        (None, None, ("markdown", "report")),
        ('{"candidate":true}\n', None, ("candidate", "markdown", "report")),
        (
            '{"candidate":true}\n',
            b"gamma-model",
            ("model", "candidate", "markdown", "report"),
        ),
    ],
)
def test_transactional_publication_writes_report_as_final_commit_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_json: str | None,
    model_payload: bytes | None,
    expected_order: tuple[str, ...],
) -> None:
    paths = _publication_paths(tmp_path)
    names = {path: name for name, path in paths.items()}
    writes: list[str] = []
    original_text_write = baseline_cli._write_atomic
    original_binary_write = cli._write_atomic_bytes

    def write_text(path: Path, payload: str, *, force: bool) -> None:
        writes.append(names[path])
        original_text_write(path, payload, force=force)

    def write_binary(path: Path, payload: bytes) -> None:
        writes.append(names[path])
        original_binary_write(path, payload)

    monkeypatch.setattr(cli, "_write_atomic", write_text)
    monkeypatch.setattr(cli, "_write_atomic_bytes", write_binary)

    cli._publish_outputs(
        paths,
        report_json='{"report":true}\n',
        markdown="# report\n",
        candidate_json=candidate_json,
        model_payload=model_payload,
    )

    assert tuple(writes) == expected_order
    assert paths["report"].read_text(encoding="utf-8") == '{"report":true}\n'
    assert paths["markdown"].read_text(encoding="utf-8") == "# report\n"
    assert paths["candidate"].exists() is (candidate_json is not None)
    assert paths["model"].exists() is (model_payload is not None)


@pytest.mark.parametrize("failure", ["model", "candidate", "markdown", "report"])
def test_publication_failure_rolls_back_only_outputs_created_by_this_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    paths = _publication_paths(tmp_path)
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("preserve me", encoding="utf-8")
    original_text_write = baseline_cli._write_atomic
    original_binary_write = cli._write_atomic_bytes

    def write_text(path: Path, payload: str, *, force: bool) -> None:
        if path == paths[failure]:
            raise OSError("injected publication failure")
        original_text_write(path, payload, force=force)

    def write_binary(path: Path, payload: bytes) -> None:
        if failure == "model":
            raise OSError("injected publication failure")
        original_binary_write(path, payload)

    monkeypatch.setattr(cli, "_write_atomic", write_text)
    monkeypatch.setattr(cli, "_write_atomic_bytes", write_binary)

    with pytest.raises(OSError, match="injected publication failure"):
        cli._publish_outputs(
            paths,
            report_json='{"report":true}\n',
            markdown="# report\n",
            candidate_json='{"candidate":true}\n',
            model_payload=b"gamma-model",
        )

    assert not any(path.exists() for path in paths.values())
    assert unrelated.read_text(encoding="utf-8") == "preserve me"


def test_rollback_refuses_digest_or_identity_drift_and_removes_other_owned_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _publication_paths(tmp_path)
    original_text_write = baseline_cli._write_atomic

    def write_text(path: Path, payload: str, *, force: bool) -> None:
        if path == paths["report"]:
            paths["candidate"].write_text("concurrent replacement\n", encoding="utf-8")
            raise OSError("injected final commit failure")
        original_text_write(path, payload, force=force)

    monkeypatch.setattr(cli, "_write_atomic", write_text)

    with pytest.raises(cli.SharpnessCLIError, match="safe rollback refused changed outputs"):
        cli._publish_outputs(
            paths,
            report_json='{"report":true}\n',
            markdown="# report\n",
            candidate_json='{"candidate":true}\n',
            model_payload=b"gamma-model",
        )

    assert paths["candidate"].read_text(encoding="utf-8") == "concurrent replacement\n"
    assert not paths["model"].exists()
    assert not paths["markdown"].exists()
    assert not paths["report"].exists()


def test_frozen_evidence_verifies_every_pinned_checksum_and_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    experiments = project / "docs" / "experiments"
    verified: list[tuple[Path, str, str]] = []
    confirmation = object()

    def verified_bytes(path: Path, *, expected: str, label: str) -> bytes:
        verified.append((path, expected, label))
        if label == "Phase 4 retail confirmation":
            return b"frozen-confirmation"
        if label == "calibration v1 report":
            return b'{"calibration":true}'
        if label == "development residual diagnostics":
            return b'{"report_type":"retail_rf05_development_residual_diagnostics"}'
        return b"unused-frozen-bytes"

    def parse_confirmation(payload: bytes) -> object:
        assert payload == b"frozen-confirmation"
        return confirmation

    monkeypatch.setattr(cli, "_verified_bytes", verified_bytes)
    monkeypatch.setattr(cli, "parse_phase4_confirmation_json", parse_confirmation)
    monkeypatch.setattr(
        cli,
        "active_rf05_identity",
        lambda: SimpleNamespace(identity_sha256=cli._RF05_IDENTITY_SHA256),
    )

    frozen = cli._load_frozen_evidence(project)

    assert frozen["confirmation"] is confirmation
    assert frozen["calibration_report"] == {"calibration": True}
    assert frozen["diagnostics"] == {"report_type": "retail_rf05_development_residual_diagnostics"}
    assert verified == [
        (
            experiments / "phase4-model-selection-v1.json",
            PHASE4_PROTOCOL_SHA256,
            "Phase 4 protocol",
        ),
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
            CALIBRATION_V1_ARTIFACT_SHA256,
            "calibration v1 serving artifact",
        ),
        (
            experiments / "retail-rf05-calibration-v1.report.json",
            CALIBRATION_V1_REPORT_SHA256,
            "calibration v1 report",
        ),
        (
            experiments / "retail-rf05-development-residual-diagnostics-v1.json",
            DEVELOPMENT_DIAGNOSTICS_SHA256,
            "development residual diagnostics",
        ),
    ]


def test_policy_identity_drift_stops_before_data_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    data_accessed = False

    monkeypatch.setattr(
        cli,
        "_validated_output_paths",
        lambda root: {
            "report": root / "report.json",
            "markdown": root / "report.md",
            "candidate": root / "candidate.json",
            "model": root / "model.joblib",
        },
    )
    monkeypatch.setattr(cli, "_load_frozen_evidence", lambda root: {})
    monkeypatch.setattr(
        cli,
        "load_uncertainty_sharpness_policy_file",
        lambda path: SimpleNamespace(policy_sha256="0" * 64),
    )

    def forbidden_prepare(*paths: Path, partition: str) -> object:
        nonlocal data_accessed
        del paths, partition
        data_accessed = True
        return object()

    monkeypatch.setattr(cli, "prepare_kaggle_us_sales_cars_split_training_rows", forbidden_prepare)

    with pytest.raises(SystemExit) as error:
        cli.main(["--project-root", str(project)])

    assert error.value.code == 2
    assert data_accessed is False


def test_calibration_assignment_drift_stops_before_any_model_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    features = pd.DataFrame({"row_marker": ["development", "calibration"]})
    model_called = False

    monkeypatch.setattr(
        cli,
        "_validated_output_paths",
        lambda root: {
            "report": root / "report.json",
            "markdown": root / "report.md",
            "candidate": root / "candidate.json",
            "model": root / "model.joblib",
        },
    )
    monkeypatch.setattr(
        cli,
        "_load_frozen_evidence",
        lambda root: {"calibration_report": {}, "diagnostics": {}},
    )
    monkeypatch.setattr(
        cli,
        "load_uncertainty_sharpness_policy_file",
        lambda path: SimpleNamespace(policy_sha256=SHARPNESS_POLICY_SHA256),
    )
    monkeypatch.setattr(
        cli,
        "prepare_kaggle_us_sales_cars_split_training_rows",
        lambda *paths, partition: SimpleNamespace(expected_rows=2),
    )
    monkeypatch.setattr(
        cli,
        "_collect_retail_partition",
        lambda stream, expected_rows, label: SimpleNamespace(
            features=features,
            target=np.asarray([1.0, 2.0]),
        ),
    )
    monkeypatch.setattr(
        cli,
        "retail_calibration_partition",
        lambda frame, seed: SimpleNamespace(
            development_indices=np.asarray([0], dtype=np.int64),
            calibration_indices=np.asarray([1], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(cli, "_partition_hash", lambda *args, **kwargs: "0" * 64)

    def forbidden_model(*args: object, **kwargs: object) -> object:
        nonlocal model_called
        del args, kwargs
        model_called = True
        return object()

    monkeypatch.setattr(cli, "reconstruct_rf05_development_oof", forbidden_model)

    with pytest.raises(SystemExit) as error:
        cli.main(["--project-root", str(project)])

    assert error.value.code == 2
    assert model_called is False


def test_frozen_rf05_identity_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_verified_bytes", lambda *args, **kwargs: b"{}")
    monkeypatch.setattr(cli, "parse_phase4_confirmation_json", lambda payload: object())
    monkeypatch.setattr(
        cli,
        "active_rf05_identity",
        lambda: SimpleNamespace(identity_sha256="0" * 64),
    )

    with pytest.raises(cli.SharpnessCLIError, match="active RF05"):
        cli._load_frozen_evidence(tmp_path)


def test_json_and_development_diagnostic_validation_reject_malformed_or_drifted_inputs() -> None:
    with pytest.raises(cli.SharpnessCLIError, match="valid UTF-8 JSON"):
        cli._json_mapping(b"{", "diagnostics")
    with pytest.raises(cli.SharpnessCLIError, match="string-keyed object"):
        cli._json_mapping(b"[]", "diagnostics")

    target = np.asarray([10.0, 20.0], dtype=np.float64)
    predictions = np.asarray([9.0, 18.0], dtype=np.float64)
    metrics = regression_metrics(target, predictions).to_dict()
    cli._validate_development_reconstruction(
        target,
        predictions,
        {
            "report_type": "retail_rf05_development_residual_diagnostics",
            "point_prediction_metrics": metrics,
        },
    )
    with pytest.raises(cli.SharpnessCLIError, match="report type"):
        cli._validate_development_reconstruction(
            target,
            predictions,
            {"report_type": "wrong", "point_prediction_metrics": metrics},
        )
    drifted = dict(metrics)
    drifted["mae"] = cast(float, drifted["mae"]) + 0.01
    with pytest.raises(cli.SharpnessCLIError, match="OOF mae differs"):
        cli._validate_development_reconstruction(
            target,
            predictions,
            {
                "report_type": "retail_rf05_development_residual_diagnostics",
                "point_prediction_metrics": drifted,
            },
        )


def test_missing_diagnostic_metric_mapping_raises_governed_error() -> None:
    target = np.asarray([10.0], dtype=np.float64)

    with pytest.raises(cli.SharpnessCLIError):
        cli._validate_development_reconstruction(
            target,
            target.copy(),
            {"report_type": "retail_rf05_development_residual_diagnostics"},
        )


def test_development_diagnostic_summary_is_allowlisted_and_fails_closed() -> None:
    diagnostics = {
        "classification": "aggregate_only",
        "overall_residual_distribution": {
            "support": 98_552,
            "mean_absolute_residual_usd": 13_000.0,
            "median_absolute_residual_usd": 7_500.0,
            "residual_variance_usd2": 123_456.0,
            "absolute_residual_quantiles_usd": {"p90": 30_000.0},
            "residual_to_actual_price_ratio": {"median": 0.25},
        },
        "predicted_value_relationship": {
            "log_prediction_log_residual_pearson": 0.5,
            "prediction_residual_spearman": 0.4,
            "mean_absolute_residual_usd_by_predicted_value_quartile": {
                "predicted_value_1": 4_000.0,
                "predicted_value_4": 20_000.0,
            },
            "highest_to_lowest_quartile_mean_residual_ratio": 5.0,
        },
        "raw_row_that_must_not_copy": {"vin": "forbidden"},
    }

    summary = cli._development_diagnostic_summary(diagnostics)

    assert set(summary) == {
        "classification",
        "overall_residual_distribution",
        "predicted_value_relationship",
        "actual_price_used_for_evaluation_only",
    }
    assert "raw_row_that_must_not_copy" not in summary
    assert "forbidden" not in json.dumps(summary)
    with pytest.raises(cli.SharpnessCLIError, match="summary fields differ"):
        cli._development_diagnostic_summary(
            {
                **diagnostics,
                "overall_residual_distribution": {
                    **cast(
                        Mapping[str, object],
                        diagnostics["overall_residual_distribution"],
                    ),
                    "unexpected": True,
                },
            }
        )


def _render_report() -> dict[str, object]:
    coverage_item = {
        "empirical_coverage": 0.9,
        "coverage_gap": 0.0,
        "displayed_width_usd": {
            "mean": 12_345.67,
            "median": 11_000.0,
            "p75": 13_000.0,
            "p90": 15_000.0,
            "p95": 17_000.0,
        },
        "fallback_rate": 0.0123,
    }
    focus_slices = {
        "actual_price_band": [
            {
                "label": "price_4",
                "coverages": {"0.9": {"empirical_coverage": 0.91}},
            }
        ],
        "manufacturer": [
            {
                "label": label,
                "coverages": {"0.9": {"empirical_coverage": 0.89}},
            }
            for label in ("gmc", "genesis", "bmw", "audi", "mercedes")
        ],
    }
    methods = {
        method: {
            "coverages": {level: dict(coverage_item) for level in ("0.8", "0.9", "0.95")},
            "slices": focus_slices,
        }
        for method in (BASELINE_METHOD, GAMMA_METHOD, SMOOTH_METHOD)
    }
    return {
        "classification": "controlled_experiment_only",
        "decision": {"selected_method": SMOOTH_METHOD},
        "methods": methods,
        "acceptance_gates": {
            GAMMA_METHOD: {
                "passed_all": False,
                "failed_gate_count": 2,
                "outcomes": [
                    {"gate": "width_reduction", "passed": False},
                    {"gate": "coverage", "passed": False},
                ],
            },
            SMOOTH_METHOD: {
                "passed_all": True,
                "failed_gate_count": 0,
                "outcomes": [{"gate": "width_reduction", "passed": True}],
            },
        },
        "data_boundaries": {
            "development_rows": 98_552,
            "calibration_rows": 10_958,
        },
        "development_diagnostic_summary": {
            "overall_residual_distribution": {
                "median_absolute_residual_usd": 7_500.0,
                "mean_absolute_residual_usd": 13_000.0,
            },
            "predicted_value_relationship": {
                "mean_absolute_residual_usd_by_predicted_value_quartile": {
                    "predicted_value_1": 4_000.0,
                    "predicted_value_4": 20_000.0,
                },
                "highest_to_lowest_quartile_mean_residual_ratio": 5.0,
            },
        },
    }


def test_markdown_render_covers_methods_levels_boundaries_and_absent_artifacts() -> None:
    markdown = cli.render_sharpness_markdown(
        _render_report(),
        report_sha256="a" * 64,
        candidate_sha256=None,
        model_sha256=None,
    )

    assert markdown.startswith("# Retail RF05 uncertainty sharpness experiment\n")
    assert "**Classification: `controlled_experiment_only`.**" in markdown
    assert f"Selected method: `{SMOOTH_METHOD}`" in markdown
    assert "Development OOF residual rows: 98,552" in markdown
    assert "Protected calibration rows: 10,958" in markdown
    assert "requested, loaded, or evaluated legacy-holdout rows: no" in markdown
    assert "Legacy holdout, Yoad, River, AutoTrader, or Carson-Shively accessed" not in markdown
    assert "Current status-conditional baseline" in markdown
    assert "Normalized Gamma residual scale" in markdown
    assert "Simple smooth predicted-value scale" in markdown
    assert "Why heteroscedastic methods were evaluated" in markdown
    assert "median of $7,500.00 and a mean of $13,000.00" in markdown
    assert "Failed gate names" in markdown
    assert "`normalized_gamma_scale_v1`: width_reduction, coverage" in markdown
    assert "Focus-slice 90% coverage" in markdown
    assert "| Highest price band | 91.00% | 91.00% | 91.00% |" in markdown
    assert "| GMC | 89.00% | 89.00% | 89.00% |" in markdown
    assert markdown.count("| 80% |") == 3
    assert markdown.count("| 90% |") == 3
    assert markdown.count("| 95% |") == 3
    assert "$12,345.67" in markdown
    assert "Candidate serving artifact: not created" in markdown
    assert "Gamma residual-scale model: not persisted" in markdown
    assert f"Comparison report SHA-256: `{'a' * 64}`" in markdown
    assert markdown.endswith("\n")


def test_markdown_render_includes_candidate_and_gamma_checksums() -> None:
    markdown = cli.render_sharpness_markdown(
        _render_report(),
        report_sha256="a" * 64,
        candidate_sha256="b" * 64,
        model_sha256="c" * 64,
    )

    assert f"Candidate serving artifact SHA-256: `{'b' * 64}`" in markdown
    assert f"Gamma residual-scale model SHA-256: `{'c' * 64}`" in markdown
    assert "not created" not in markdown
    assert "not persisted" not in markdown
