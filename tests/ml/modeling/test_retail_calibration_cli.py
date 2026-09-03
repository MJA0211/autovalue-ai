from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from autovalue_ml.modeling import retail_calibration_cli


def test_cli_requests_only_phase3_train_and_never_legacy_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / "docs" / "experiments").mkdir(parents=True)
    requested_partitions: list[str] = []
    writes: list[Path] = []

    monkeypatch.setattr(retail_calibration_cli, "_verify_regular_sha256", lambda *a, **k: None)
    monkeypatch.setattr(retail_calibration_cli, "load_phase4_protocol", lambda path: object())
    monkeypatch.setattr(
        retail_calibration_cli,
        "_verified_bytes",
        lambda *a, **k: b"confirmation",
    )
    monkeypatch.setattr(
        retail_calibration_cli,
        "parse_phase4_confirmation_json",
        lambda payload: object(),
    )

    def prepare(*paths: Path, partition: str) -> object:
        del paths
        requested_partitions.append(partition)
        return object()

    monkeypatch.setattr(
        retail_calibration_cli,
        "prepare_kaggle_us_sales_cars_split_training_rows",
        prepare,
    )
    monkeypatch.setattr(retail_calibration_cli, "_expected_count", lambda *a: 109_510)
    monkeypatch.setattr(
        retail_calibration_cli,
        "_collect_retail_partition",
        lambda *a, **k: SimpleNamespace(features=object(), target=object()),
    )
    monkeypatch.setattr(
        retail_calibration_cli,
        "run_retail_rf05_calibration",
        lambda **kwargs: SimpleNamespace(
            artifact=object(),
            report={
                "classification": "validated_for_calibrated_prediction_intervals",
                "decision": {"selected_method": "vehicle_status"},
            },
        ),
    )
    monkeypatch.setattr(
        retail_calibration_cli,
        "canonical_calibration_artifact_json",
        lambda artifact: '{"artifact":true}\n',
    )
    monkeypatch.setattr(
        retail_calibration_cli,
        "canonical_calibration_report_json",
        lambda report: '{"report":true}\n',
    )
    monkeypatch.setattr(
        retail_calibration_cli,
        "render_calibration_markdown",
        lambda *a, **k: "# report\n",
    )

    def write(path: Path, serialized: str, *, force: bool) -> None:
        del serialized
        assert force is False
        writes.append(path)

    monkeypatch.setattr(retail_calibration_cli, "_write_atomic", write)

    assert retail_calibration_cli.main(["--project-root", str(project)]) == 0
    assert requested_partitions == ["train"]
    assert len(writes) == 3
    assert not any("holdout" in path.name.lower() or "test" in path.name.lower() for path in writes)


def test_markdown_output_is_immutable_and_fixed_to_experiment_directory(tmp_path: Path) -> None:
    project = tmp_path / "project"
    experiments = project / "docs" / "experiments"
    experiments.mkdir(parents=True)
    output = experiments / "retail-rf05-calibration-v1.md"

    assert retail_calibration_cli._validate_markdown_output(output, project_root=project) == output
    output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(retail_calibration_cli.CalibrationCLIError, match="already exists"):
        retail_calibration_cli._validate_markdown_output(output, project_root=project)


def _accept_anything(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
