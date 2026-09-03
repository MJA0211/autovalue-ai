"""Canonical, row-free model evaluation report contracts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, cast

from .contracts import TRACKS, TrackName
from .metrics import RegressionMetrics, StatusSliceMetrics

EvaluationScope = Literal["cross_validation", "holdout"]

_SCHEMA_VERSION: Final = 1
_REPORT_TYPE: Final = "aggregate_model_evaluation"
_MAX_REPORT_BYTES: Final = 100_000
_MODEL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ROOT_KEYS: Final = {
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
_METRIC_KEYS: Final = {"sample_count", "mae", "rmse", "r2"}
_SLICE_KEYS: Final = {"status", "metrics"}


class ReportValidationError(ValueError):
    """Raised when a report is noncanonical or is not aggregate-only."""


@dataclass(frozen=True, slots=True)
class AggregateModelReport:
    """The only row-free evaluation payload emitted by the modeling core."""

    track: TrackName
    model_name: str
    evaluation_scope: EvaluationScope
    overall: RegressionMetrics
    status_slices: tuple[StatusSliceMetrics, ...] = ()

    def __post_init__(self) -> None:
        _validate_report_fields(self)

    @property
    def feature_contract_version(self) -> str:
        return TRACKS[self.track].contract_version

    @property
    def target_semantics(self) -> str:
        return TRACKS[self.track].target_semantics

    def to_dict(self) -> dict[str, object]:
        """Return the exact public schema without paths, rows, or predictions."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "report_type": _REPORT_TYPE,
            "track": self.track,
            "feature_contract_version": self.feature_contract_version,
            "target_semantics": self.target_semantics,
            "model_name": self.model_name,
            "evaluation_scope": self.evaluation_scope,
            "overall": self.overall.to_dict(),
            "status_slices": [status_slice.to_dict() for status_slice in self.status_slices],
        }


def canonical_report_json(report: AggregateModelReport) -> str:
    """Serialize a validated report deterministically as canonical JSON text."""

    _validate_report_fields(report)
    return (
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def validate_aggregate_report(payload: object) -> AggregateModelReport:
    """Parse the exact aggregate schema and reject every extension field."""

    root = _object(payload, label="report")
    _exact_keys(root, _ROOT_KEYS, label="report")
    if _integer(root["schema_version"], label="schema_version") != _SCHEMA_VERSION:
        raise ReportValidationError("unsupported report schema_version")
    if root["report_type"] != _REPORT_TYPE:
        raise ReportValidationError("report_type is invalid")

    track_value = _text(root["track"], label="track")
    if track_value not in TRACKS:
        raise ReportValidationError("track is invalid")
    track: TrackName = track_value
    config = TRACKS[track]
    if root["feature_contract_version"] != config.contract_version:
        raise ReportValidationError("feature_contract_version does not match track")
    if root["target_semantics"] != config.target_semantics:
        raise ReportValidationError("target_semantics does not match track")

    model_name = _text(root["model_name"], label="model_name")
    scope_value = _text(root["evaluation_scope"], label="evaluation_scope")
    if scope_value not in {"cross_validation", "holdout"}:
        raise ReportValidationError("evaluation_scope is invalid")
    scope = cast(EvaluationScope, scope_value)
    overall = _parse_metrics(root["overall"], label="overall")

    raw_slices = root["status_slices"]
    if not isinstance(raw_slices, list):
        raise ReportValidationError("status_slices must be an array")
    slices: list[StatusSliceMetrics] = []
    for index, raw_slice in enumerate(raw_slices):
        slice_object = _object(raw_slice, label=f"status_slices[{index}]")
        _exact_keys(slice_object, _SLICE_KEYS, label=f"status_slices[{index}]")
        slices.append(
            StatusSliceMetrics(
                status=_text(slice_object["status"], label="status"),
                metrics=_parse_metrics(slice_object["metrics"], label="slice metrics"),
            )
        )

    return AggregateModelReport(
        track=track,
        model_name=model_name,
        evaluation_scope=scope,
        overall=overall,
        status_slices=tuple(slices),
    )


def parse_aggregate_report_json(serialized: str | bytes) -> AggregateModelReport:
    """Load canonical JSON while rejecting duplicate keys and oversized payloads."""

    if isinstance(serialized, bytes):
        if len(serialized) > _MAX_REPORT_BYTES:
            raise ReportValidationError("report exceeds the maximum size")
        try:
            text = serialized.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ReportValidationError("report must be UTF-8") from error
    elif isinstance(serialized, str):
        text = serialized
        if len(text.encode("utf-8")) > _MAX_REPORT_BYTES:
            raise ReportValidationError("report exceeds the maximum size")
    else:
        raise ReportValidationError("serialized report must be text or bytes")

    try:
        payload = json.loads(text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ReportValidationError("report is not valid JSON") from error
    return validate_aggregate_report(payload)


def _validate_report_fields(report: AggregateModelReport) -> None:
    if report.track not in TRACKS:
        raise ReportValidationError("track is invalid")
    if not _MODEL_NAME_PATTERN.fullmatch(report.model_name):
        raise ReportValidationError("model_name must be a lowercase stable identifier")
    if report.evaluation_scope not in {"cross_validation", "holdout"}:
        raise ReportValidationError("evaluation_scope is invalid")
    _validate_metrics(report.overall)

    statuses = tuple(status_slice.status for status_slice in report.status_slices)
    if statuses != tuple(sorted(statuses)) or len(statuses) != len(set(statuses)):
        raise ReportValidationError("status_slices must be unique and sorted by status")
    for status_slice in report.status_slices:
        if not status_slice.status.strip() or status_slice.status != status_slice.status.lower():
            raise ReportValidationError("status slice names must be non-empty lowercase text")
        _validate_metrics(status_slice.metrics)

    if report.track == "wholesale" and report.status_slices:
        raise ReportValidationError("wholesale reports must not contain retail status slices")
    if report.status_slices:
        slice_count = sum(item.metrics.sample_count for item in report.status_slices)
        if slice_count != report.overall.sample_count:
            raise ReportValidationError("status slice counts must sum to the overall count")


def _parse_metrics(value: object, *, label: str) -> RegressionMetrics:
    metrics = _object(value, label=label)
    _exact_keys(metrics, _METRIC_KEYS, label=label)
    r2_value = metrics["r2"]
    return RegressionMetrics(
        sample_count=_integer(metrics["sample_count"], label="sample_count"),
        mae=_number(metrics["mae"], label="mae"),
        rmse=_number(metrics["rmse"], label="rmse"),
        r2=None if r2_value is None else _number(r2_value, label="r2"),
    )


def _validate_metrics(metrics: RegressionMetrics) -> None:
    if type(metrics.sample_count) is not int or metrics.sample_count < 1:
        raise ReportValidationError("metric sample_count must be positive")
    if type(metrics.mae) not in {int, float} or not math.isfinite(metrics.mae) or metrics.mae < 0:
        raise ReportValidationError("metric MAE must be finite and nonnegative")
    if (
        type(metrics.rmse) not in {int, float}
        or not math.isfinite(metrics.rmse)
        or metrics.rmse < 0
    ):
        raise ReportValidationError("metric RMSE must be finite and nonnegative")
    if metrics.r2 is not None and (
        type(metrics.r2) not in {int, float} or not math.isfinite(metrics.r2) or metrics.r2 > 1.0
    ):
        raise ReportValidationError("metric R-squared must be finite, at most one, or null")


def _object(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReportValidationError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ReportValidationError(f"{label} keys must be strings")
    return cast(Mapping[str, object], value)


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise ReportValidationError(f"{label} has invalid fields: {'; '.join(details)}")


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportValidationError(f"{label} must be an integer")
    return value


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportValidationError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ReportValidationError(f"{label} must be finite")
    return number


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportValidationError(f"{label} must be non-empty text")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReportValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
