from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from autovalue_ml.modeling import (
    AggregateModelReport,
    RegressionMetrics,
    ReportValidationError,
    canonical_report_json,
    parse_aggregate_report_json,
    regression_metrics,
    retail_status_metrics,
    validate_aggregate_report,
)


def test_regression_metrics_are_correct_and_do_not_clip_predictions() -> None:
    metrics = regression_metrics([100.0, 200.0, 300.0], [110.0, 190.0, 310.0])
    assert metrics.sample_count == 3
    assert metrics.mae == pytest.approx(10.0)
    assert metrics.rmse == pytest.approx(10.0)
    assert metrics.r2 == pytest.approx(0.985)

    negative_prediction = regression_metrics([100.0], [-100.0])
    assert negative_prediction.mae == 200.0
    assert negative_prediction.r2 is None


def test_retail_status_slices_have_null_r2_when_undefined() -> None:
    evaluation = retail_status_metrics(
        [10_000.0, 12_000.0, 30_000.0],
        [11_000.0, 11_000.0, 29_000.0],
        ["Used", "used", "New"],
    )

    assert evaluation.overall.sample_count == 3
    assert [item.status for item in evaluation.status_slices] == ["new", "used"]
    assert evaluation.status_slices[0].metrics.sample_count == 1
    assert evaluation.status_slices[0].metrics.r2 is None
    assert evaluation.status_slices[1].metrics.r2 == 0.0


@pytest.mark.parametrize(
    ("actual", "predicted", "message"),
    [
        ([], [], "at least one"),
        ([1, 2], [1], "same number"),
        ([1, float("nan")], [1, 2], "finite"),
        ([True], [1], "not boolean"),
        ([True, 1], [1, 2], "not boolean"),
        ([[1]], [1], "one-dimensional"),
    ],
)
def test_metrics_reject_invalid_vectors(actual: object, predicted: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        regression_metrics(actual, predicted)


def test_aggregate_report_is_deterministic_row_free_and_round_trips() -> None:
    evaluation = retail_status_metrics(
        [10_000.0, 12_000.0, 30_000.0],
        [11_000.0, 11_000.0, 29_000.0],
        ["used", "used", "new"],
    )
    report = AggregateModelReport(
        track="retail",
        model_name="linear_regression",
        evaluation_scope="cross_validation",
        overall=evaluation.overall,
        status_slices=evaluation.status_slices,
    )

    first = canonical_report_json(report)
    second = canonical_report_json(report)

    assert first == second
    assert first.endswith("\n")
    assert "timestamp" not in first
    assert "path" not in first
    assert "prediction" not in first
    assert "categories_" not in first
    assert parse_aggregate_report_json(first) == report
    assert canonical_report_json(parse_aggregate_report_json(first)) == first


def test_report_validator_rejects_rows_predictions_and_category_vocabulary() -> None:
    report = AggregateModelReport(
        track="wholesale",
        model_name="dummy_median",
        evaluation_scope="holdout",
        overall=RegressionMetrics(sample_count=4, mae=10.0, rmse=12.0, r2=0.5),
    )
    payload = report.to_dict()

    for forbidden_field in ("rows", "predictions", "category_vocabulary", "artifact_path"):
        contaminated = dict(payload)
        contaminated[forbidden_field] = []
        with pytest.raises(ReportValidationError, match="unexpected"):
            validate_aggregate_report(contaminated)


def test_report_validation_rejects_noncanonical_or_inconsistent_values() -> None:
    wholesale = AggregateModelReport(
        track="wholesale",
        model_name="linear_regression",
        evaluation_scope="holdout",
        overall=RegressionMetrics(sample_count=2, mae=1.0, rmse=1.0, r2=0.0),
    )
    payload = wholesale.to_dict()
    payload["feature_contract_version"] = "wrong"
    with pytest.raises(ReportValidationError, match="does not match"):
        validate_aggregate_report(payload)

    with pytest.raises(ReportValidationError, match="lowercase stable"):
        replace(wholesale, model_name="Linear Regression")
    with pytest.raises(ReportValidationError, match="must not contain"):
        replace(
            wholesale,
            status_slices=(retail_status_metrics([1], [1], ["used"]).status_slices[0],),
            overall=RegressionMetrics(sample_count=1, mae=0, rmse=0, r2=None),
        )

    duplicate_key_json = '{"schema_version":1,"schema_version":1}'
    with pytest.raises(ReportValidationError, match="duplicate JSON key"):
        parse_aggregate_report_json(duplicate_key_json)
    with pytest.raises(ReportValidationError, match="valid JSON"):
        parse_aggregate_report_json("{")
    with pytest.raises(ReportValidationError, match="maximum size"):
        parse_aggregate_report_json(" " * 100_001)
    with pytest.raises(ReportValidationError, match="sample_count"):
        replace(
            wholesale,
            overall=RegressionMetrics(
                sample_count=1.5,  # type: ignore[arg-type]
                mae=1.0,
                rmse=1.0,
                r2=0.0,
            ),
        )
    with pytest.raises(ReportValidationError, match="at most one"):
        replace(
            wholesale,
            overall=RegressionMetrics(sample_count=2, mae=1.0, rmse=1.0, r2=1.01),
        )
    with pytest.raises(ReportValidationError, match="finite and nonnegative"):
        replace(
            wholesale,
            overall=RegressionMetrics(
                sample_count=2,
                mae=np.float32(1.0),  # type: ignore[arg-type]
                rmse=1.0,
                r2=0.0,
            ),
        )


def test_report_json_has_exact_aggregate_schema() -> None:
    report = AggregateModelReport(
        track="wholesale",
        model_name="dummy_median",
        evaluation_scope="cross_validation",
        overall=RegressionMetrics(sample_count=10, mae=100.0, rmse=120.0, r2=None),
    )
    payload = json.loads(canonical_report_json(report))
    assert set(payload) == {
        "schema_version",
        "report_type",
        "track",
        "feature_contract_version",
        "target_semantics",
        "model_name",
        "evaluation_scope",
        "overall",
        "status_slices",
    }
    assert set(payload["overall"]) == {"sample_count", "mae", "rmse", "r2"}
    assert payload["status_slices"] == []
