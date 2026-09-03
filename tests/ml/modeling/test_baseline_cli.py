from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling import baseline_cli


@dataclass(frozen=True)
class _FakeMetrics:
    mae: float = 1250.25
    rmse: float = 1840.5
    r2: float | None = 0.812345


@dataclass(frozen=True)
class _FakeHoldout:
    overall: _FakeMetrics = _FakeMetrics()


@dataclass(frozen=True)
class _FakeSelection:
    selected_model: str = "linear_regression"


@dataclass(frozen=True)
class _FakeExperiment:
    track: str
    outer_train_sample_count: int
    outer_test_sample_count: int
    selection: _FakeSelection = _FakeSelection()
    holdout: _FakeHoldout = _FakeHoldout()


@dataclass(frozen=True)
class _RetailStream:
    rows: tuple[tuple[Mapping[str, object], object], ...]
    expected_rows: int

    def __iter__(self) -> Iterator[tuple[Mapping[str, object], object]]:
        return iter(self.rows)


@dataclass(frozen=True)
class _WholesaleStream:
    rows: tuple[tuple[str, str | None, Mapping[str, object], object], ...]
    train_rows: int
    test_rows: int

    def __iter__(
        self,
    ) -> Iterator[tuple[str, str | None, Mapping[str, object], object]]:
        return iter(self.rows)


def _retail_rows(prefix: str) -> tuple[tuple[Mapping[str, object], object], ...]:
    return (
        (
            {
                "year": 2023,
                "make": f"{prefix} Secret Make",
                "model": "Alpha Private Marker",
                "vehicle_status": "new",
            },
            41_001.0,
        ),
        (
            {
                "year": 2020,
                "make": f"{prefix} Secret Make",
                "model": "Beta Private Marker",
                "vehicle_status": "used",
                "mileage": 44_500,
            },
            27_502.0,
        ),
        (
            {
                "year": 2022,
                "make": "Other Secret Make",
                "model": "Gamma Private Marker",
                "vehicle_status": "certified",
                "mileage": 12_000,
            },
            36_503.0,
        ),
    )


def _wholesale_rows() -> tuple[tuple[str, str | None, Mapping[str, object], object], ...]:
    result: list[tuple[str, str | None, Mapping[str, object], object]] = []
    for position, bucket in enumerate(("warmup", "2015_01", "2015_02", "2015_03_04", "2015_05")):
        result.append(
            (
                "train",
                bucket,
                {
                    "year": 2008 + position,
                    "make": "Wholesale Secret Make",
                    "model": f"Private Model {position}",
                    "trim": None if position == 0 else "Sport",
                    "mileage": None if position == 1 else 50_000 + position,
                    "condition": None if position == 2 else 3.2,
                    "vehicle_type": "Sedan",
                },
                9_000.0 + position,
            )
        )
    result.append(
        (
            "test",
            None,
            {
                "year": 2013,
                "make": "Holdout Secret Make",
                "model": "Holdout Private Model",
                "trim": "Base",
                "mileage": 60_000,
                "condition": 4.0,
                "vehicle_type": "SUV",
            },
            13_500.0,
        )
    )
    return tuple(result)


def _install_serializer(
    monkeypatch: pytest.MonkeyPatch,
    result: _FakeExperiment,
    payload: str = '{"report_type":"baseline_experiment","safe":true}\n',
) -> None:
    def serialize(received: object) -> str:
        assert received is result
        return payload

    monkeypatch.setattr(baseline_cli, "canonical_experiment_json", serialize)


def test_retail_routes_only_through_exact_split_aware_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "retail-report.json"
    gate_calls: list[tuple[tuple[Path, ...], str]] = []
    experiment = _FakeExperiment("retail", 3, 3)

    def split_gate(*paths: Path, partition: str) -> _RetailStream:
        gate_calls.append((paths, partition))
        return _RetailStream(_retail_rows(partition), 3)

    def run_experiment(**arguments: object) -> Any:
        train = arguments["outer_train_features"]
        test = arguments["outer_test_features"]
        train_target = arguments["outer_train_target"]
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert isinstance(train_target, np.ndarray)
        assert tuple(train.columns) == ("year", "make", "model", "vehicle_status", "mileage")
        assert tuple(test.columns) == tuple(train.columns)
        assert train["make"].dtype == object
        assert train["make"].iloc[0] is train["make"].iloc[1]
        assert train["mileage"].isna().sum() == 1
        assert train_target.tolist() == [41_001.0, 27_502.0, 36_503.0]
        return experiment

    monkeypatch.setattr(
        baseline_cli, "prepare_kaggle_us_sales_cars_split_training_rows", split_gate
    )
    monkeypatch.setattr(baseline_cli, "run_retail_baseline_experiment", run_experiment)
    _install_serializer(monkeypatch, experiment)

    assert (
        baseline_cli.main(
            ["retail", "--project-root", os.fspath(project), "--output", os.fspath(output)]
        )
        == 0
    )

    expected = (
        project / "data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.csv",
        project / "data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.manifest.json",
        project / "data/processed/kaggle_us_sales_cars_v2/split/split_assignments.manifest.json",
        project / "docs/data-reviews/kaggle-us-sales-cars-v2.review.json",
    )
    assert gate_calls == [(expected, "train"), (expected, "test")]
    report = output.read_text(encoding="utf-8")
    console = capsys.readouterr().out
    assert json.loads(report) == {"report_type": "baseline_experiment", "safe": True}
    for private_value in ("Secret Make", "Private Marker", "41001"):
        assert private_value not in report
        assert private_value not in console
    assert "selected=linear_regression" in console
    assert "holdout_mae_usd=1250.25" in console


def test_wholesale_routes_only_through_exact_split_aware_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "wholesale-report.json"
    gate_calls: list[tuple[Path, ...]] = []
    rows = _wholesale_rows()
    experiment = _FakeExperiment("wholesale", 5, 1)

    def split_gate(*paths: Path) -> _WholesaleStream:
        gate_calls.append(paths)
        return _WholesaleStream(rows, train_rows=5, test_rows=1)

    def run_experiment(**arguments: object) -> Any:
        train = arguments["outer_train_features"]
        test = arguments["outer_test_features"]
        buckets = arguments["train_cv_buckets"]
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert isinstance(buckets, pd.Series)
        assert tuple(train.columns) == (
            "year",
            "make",
            "model",
            "trim",
            "mileage",
            "condition",
            "vehicle_type",
        )
        assert buckets.tolist() == [
            "warmup",
            "2015_01",
            "2015_02",
            "2015_03_04",
            "2015_05",
        ]
        assert arguments["bucket_order"] == tuple(buckets.cat.categories)
        assert train["mileage"].isna().sum() == 1
        assert test.shape == (1, 7)
        return experiment

    monkeypatch.setattr(baseline_cli, "prepare_kaggle_vehicle_sales_training_rows", split_gate)
    monkeypatch.setattr(baseline_cli, "run_wholesale_baseline_experiment", run_experiment)
    _install_serializer(monkeypatch, experiment)

    assert (
        baseline_cli.main(["wholesale", "--project-root", str(project), "--output", str(output)])
        == 0
    )
    assert gate_calls == [
        (
            project / "data/processed/kaggle_vehicle_sales_v1/split_assignments.manifest.json",
            project / "data/raw/kaggle_vehicle_sales_v1/car_prices.csv",
            project / "data/interim/kaggle_vehicle_sales_v1.csv",
            project / "data/interim/kaggle_vehicle_sales_v1.manifest.json",
            project / "docs/data-reviews/kaggle-vehicle-sales-data-v1.review.json",
            project / "docs/data-reviews/kaggle-vehicle-sales-v1.split.json",
        )
    ]


def test_atomic_report_refuses_overwrite_then_force_replaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "reports" / "retail.json"
    experiment = _FakeExperiment("retail", 3, 3)
    payload = ['{"version":1}\n']

    def split_gate(*paths: Path, partition: str) -> _RetailStream:
        del paths
        return _RetailStream(_retail_rows(partition), 3)

    def serialize(result: object) -> str:
        assert result is experiment
        return payload[0]

    monkeypatch.setattr(
        baseline_cli, "prepare_kaggle_us_sales_cars_split_training_rows", split_gate
    )
    monkeypatch.setattr(
        baseline_cli,
        "run_retail_baseline_experiment",
        lambda **arguments: experiment,
    )
    monkeypatch.setattr(baseline_cli, "canonical_experiment_json", serialize)
    command = ["retail", "--project-root", str(project), "--output", str(output)]

    assert baseline_cli.main(command) == 0
    assert output.read_text(encoding="utf-8") == '{"version":1}\n'
    with pytest.raises(SystemExit, match="2"):
        baseline_cli.main(command)
    assert output.read_text(encoding="utf-8") == '{"version":1}\n'

    payload[0] = '{"version":2}\n'
    assert baseline_cli.main([*command, "--force"]) == 0
    assert output.read_text(encoding="utf-8") == '{"version":2}\n'
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []


@pytest.mark.parametrize("bad_output", ("report.txt", "report.JSON", "report"))
def test_non_json_output_is_rejected_before_source_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_output: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    called = False

    def split_gate(*paths: Path, partition: str) -> _RetailStream:
        nonlocal called
        del paths, partition
        called = True
        return _RetailStream((), 0)

    monkeypatch.setattr(
        baseline_cli, "prepare_kaggle_us_sales_cars_split_training_rows", split_gate
    )
    with pytest.raises(SystemExit, match="2"):
        baseline_cli.main(
            ["retail", "--project-root", str(project), "--output", str(tmp_path / bad_output)]
        )
    assert not called


@pytest.mark.parametrize(
    "protected_relative",
    (
        "docs/data-reviews/kaggle-us-sales-cars-v2.review.json",
        "data/processed/kaggle_us_sales_cars_v2/split/split_assignments.ready.json",
        "data/processed/kaggle_vehicle_sales_v1/split_assignments.ready.json",
    ),
)
def test_reviewed_input_cannot_be_used_as_forced_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected_relative: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    called = False

    def split_gate(*paths: Path, partition: str) -> _RetailStream:
        nonlocal called
        del paths, partition
        called = True
        return _RetailStream((), 0)

    monkeypatch.setattr(
        baseline_cli, "prepare_kaggle_us_sales_cars_split_training_rows", split_gate
    )
    protected = project / protected_relative
    with pytest.raises(SystemExit, match="2"):
        baseline_cli.main(
            [
                "retail",
                "--project-root",
                str(project),
                "--output",
                str(protected),
                "--force",
            ]
        )
    assert not called


@pytest.mark.parametrize(
    ("bad_features", "bad_target"),
    (
        (
            {
                "year": 2020,
                "make": "Safe",
                "model": "Model",
                "vehicle_status": "used",
                "dealer": "must-not-enter",
            },
            10_000.0,
        ),
        (
            {
                "year": 2020,
                "make": "Safe",
                "model": "Model",
                "vehicle_status": "used",
            },
            math.nan,
        ),
        (
            {
                "year": 2020,
                "make": "Safe",
                "model": "Model",
                "vehicle_status": "salvage",
            },
            10_000.0,
        ),
    ),
)
def test_invalid_retail_rows_fail_before_experiment_or_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    bad_features: Mapping[str, object],
    bad_target: object,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "report.json"

    def split_gate(*paths: Path, partition: str) -> _RetailStream:
        del paths
        if partition == "train":
            return _RetailStream(((bad_features, bad_target),), 1)
        return _RetailStream(_retail_rows("test"), 3)

    monkeypatch.setattr(
        baseline_cli, "prepare_kaggle_us_sales_cars_split_training_rows", split_gate
    )
    monkeypatch.setattr(
        baseline_cli,
        "run_retail_baseline_experiment",
        lambda **arguments: pytest.fail("experiment must not run"),
    )
    with pytest.raises(SystemExit, match="2"):
        baseline_cli.main(["retail", "--project-root", str(project), "--output", str(output)])
    captured = capsys.readouterr()
    assert "must-not-enter" not in captured.err
    assert not output.exists()


@pytest.mark.parametrize(
    "replacement",
    (
        ("train", "not_reviewed", {"year": 2010, "make": "A", "model": "B"}, 1.0),
        ("test", "2015_05", {"year": 2010, "make": "A", "model": "B"}, 1.0),
    ),
)
def test_invalid_wholesale_bucket_contract_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: tuple[str, str | None, Mapping[str, object], object],
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    rows = (replacement, *_wholesale_rows()[1:])
    expected_train = sum(row[0] == "train" for row in rows)
    expected_test = sum(row[0] == "test" for row in rows)

    monkeypatch.setattr(
        baseline_cli,
        "prepare_kaggle_vehicle_sales_training_rows",
        lambda *paths: _WholesaleStream(rows, expected_train, expected_test),
    )
    monkeypatch.setattr(
        baseline_cli,
        "run_wholesale_baseline_experiment",
        lambda **arguments: pytest.fail("experiment must not run"),
    )
    with pytest.raises(SystemExit, match="2"):
        baseline_cli.main(
            [
                "wholesale",
                "--project-root",
                str(project),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )


def test_verified_stream_count_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    def split_gate(*paths: Path, partition: str) -> _RetailStream:
        del paths
        rows = _retail_rows(partition)
        return _RetailStream(rows, 4 if partition == "train" else 3)

    monkeypatch.setattr(
        baseline_cli, "prepare_kaggle_us_sales_cars_split_training_rows", split_gate
    )
    with pytest.raises(SystemExit, match="2"):
        baseline_cli.main(
            [
                "retail",
                "--project-root",
                str(project),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )


def test_symlink_output_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = tmp_path / "target.json"
    target.write_text("original\n", encoding="utf-8")
    link = tmp_path / "linked.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    monkeypatch.setattr(
        baseline_cli,
        "prepare_kaggle_us_sales_cars_split_training_rows",
        lambda *paths, partition: _RetailStream(_retail_rows(partition), 3),
    )
    with pytest.raises(SystemExit, match="2"):
        baseline_cli.main(
            [
                "retail",
                "--project-root",
                str(project),
                "--output",
                str(link),
                "--force",
            ]
        )
    assert target.read_text(encoding="utf-8") == "original\n"
