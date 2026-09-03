"""Strict serving artifact for a validated sharpness candidate, never calibration v1."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal, TypeAlias, cast

import joblib
import numpy as np
import pandas as pd

from .calibration import RETAIL_VEHICLE_STATUSES
from .calibration_artifact import (
    CALIBRATION_ASSIGNMENT_SHA256,
    CALIBRATION_SAMPLE_COUNT,
    COVERAGE_LEVELS,
    ConfidenceThresholds,
    active_rf05_identity,
)
from .retail_uncertainty_sharpness import (
    CALIBRATION_V1_ARTIFACT_SHA256,
    CALIBRATION_V1_REPORT_SHA256,
    CONFIDENCE_THRESHOLDS,
    GAMMA_METHOD,
    GAMMA_SCALE_CAP_USD,
    GAMMA_SCALE_FLOOR_USD,
    SCALE_VERSION,
    SHARPNESS_POLICY_SHA256,
    SMOOTH_METHOD,
    MethodId,
    ScaleEstimator,
    _predict_gamma_scale,
    smooth_value_scale,
)

CandidateMethod: TypeAlias = Literal[
    "normalized_gamma_scale_v1",
    "normalized_smooth_value_scale_v1",
]
ARTIFACT_TYPE: Final = "retail_rf05_uncertainty_candidate_artifact"
CALIBRATION_VERSION: Final = "retail-rf05-heteroscedastic-conformal-v2-candidate"
GAMMA_MODEL_RELATIVE_PATH: Final = "models/uncertainty/retail-rf05-gamma-residual-scale-v1.joblib"
MAXIMUM_ARTIFACT_BYTES: Final = 100_000
MAXIMUM_GAMMA_MODEL_BYTES: Final = 200_000_000


class UncertaintyCandidateArtifactError(ValueError):
    """A candidate artifact or serving input failed a frozen invariant."""


@dataclass(frozen=True, slots=True)
class StatusQuantile:
    status: str
    support: int
    quantile: float

    def __post_init__(self) -> None:
        if self.status not in RETAIL_VEHICLE_STATUSES:
            raise UncertaintyCandidateArtifactError("status quantile key is unsupported")
        _positive_int(self.support, label="status support")
        _positive_float(self.quantile, label="status quantile")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "support": self.support,
            "quantile": self.quantile,
        }


@dataclass(frozen=True, slots=True)
class NormalizedCoverage:
    coverage: float
    global_support: int
    global_quantile: float
    status_quantiles: tuple[StatusQuantile, ...]

    def __post_init__(self) -> None:
        if self.coverage not in COVERAGE_LEVELS:
            raise UncertaintyCandidateArtifactError("coverage level is not supported")
        if self.global_support != CALIBRATION_SAMPLE_COUNT:
            raise UncertaintyCandidateArtifactError("global support differs from calibration")
        _positive_float(self.global_quantile, label="global quantile")
        if tuple(item.status for item in self.status_quantiles) != RETAIL_VEHICLE_STATUSES:
            raise UncertaintyCandidateArtifactError("status quantiles are incomplete or unordered")
        if sum(item.support for item in self.status_quantiles) != CALIBRATION_SAMPLE_COUNT:
            raise UncertaintyCandidateArtifactError("status supports do not sum to calibration")

    def to_dict(self) -> dict[str, object]:
        return {
            "coverage": self.coverage,
            "global_support": self.global_support,
            "global_quantile": self.global_quantile,
            "status_quantiles": [item.to_dict() for item in self.status_quantiles],
        }


@dataclass(frozen=True, slots=True)
class ScaleBinding:
    kind: Literal["gamma_joblib", "smooth_formula"]
    version: str
    model_path: str | None = None
    model_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "smooth_formula":
            if self.version != "smooth-value-scale-v1" or self.model_path is not None:
                raise UncertaintyCandidateArtifactError("smooth scale binding is invalid")
            if self.model_sha256 is not None:
                raise UncertaintyCandidateArtifactError("smooth scale must not bind a model")
            return
        if self.kind != "gamma_joblib" or self.version != SCALE_VERSION:
            raise UncertaintyCandidateArtifactError("Gamma scale binding is invalid")
        if not isinstance(self.model_path, str) or not self.model_path:
            raise UncertaintyCandidateArtifactError("Gamma scale model path is required")
        _fixed_gamma_model_relative_path(self.model_path)
        _sha256(self.model_sha256, label="Gamma model")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": self.kind,
            "version": self.version,
        }
        if self.kind == "gamma_joblib":
            result.update(
                {
                    "model_path": self.model_path,
                    "model_sha256": self.model_sha256,
                    "scale_floor_usd": GAMMA_SCALE_FLOOR_USD,
                    "scale_cap_usd": GAMMA_SCALE_CAP_USD,
                }
            )
        else:
            result["formula"] = "1 + ln(1 + max(RF05 prediction USD, 0) / 10000)"
        return result


@dataclass(frozen=True, slots=True)
class UncertaintyCandidateArtifact:
    generated_at: str
    comparison_evidence_sha256: str
    selected_method: CandidateMethod
    coverage_calibrations: tuple[NormalizedCoverage, ...]
    scale_binding: ScaleBinding
    confidence_thresholds: ConfidenceThresholds = CONFIDENCE_THRESHOLDS

    def __post_init__(self) -> None:
        if not isinstance(self.generated_at, str) or not self.generated_at:
            raise UncertaintyCandidateArtifactError("generation timestamp is required")
        _sha256(self.comparison_evidence_sha256, label="comparison evidence")
        if self.selected_method not in (GAMMA_METHOD, SMOOTH_METHOD):
            raise UncertaintyCandidateArtifactError("artifact method is not a validated candidate")
        expected_kind = "gamma_joblib" if self.selected_method == GAMMA_METHOD else "smooth_formula"
        if self.scale_binding.kind != expected_kind:
            raise UncertaintyCandidateArtifactError("scale binding does not match selected method")
        if tuple(item.coverage for item in self.coverage_calibrations) != COVERAGE_LEVELS:
            raise UncertaintyCandidateArtifactError("coverage calibrations are incomplete")
        if self.confidence_thresholds != CONFIDENCE_THRESHOLDS:
            raise UncertaintyCandidateArtifactError("confidence thresholds differ from policy")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "artifact_type": ARTIFACT_TYPE,
            "calibration_version": CALIBRATION_VERSION,
            "generated_at": self.generated_at,
            "status": "validated_candidate_not_production_final",
            "bound_model": active_rf05_identity().to_dict(),
            "frozen_evidence": {
                "sharpness_policy_sha256": SHARPNESS_POLICY_SHA256,
                "calibration_assignment_sha256": CALIBRATION_ASSIGNMENT_SHA256,
                "calibration_v1_artifact_sha256": CALIBRATION_V1_ARTIFACT_SHA256,
                "calibration_v1_report_sha256": CALIBRATION_V1_REPORT_SHA256,
                "comparison_evidence_sha256": self.comparison_evidence_sha256,
            },
            "selected_method": self.selected_method,
            "interval": {
                "score": "absolute_error_divided_by_prediction_time_scale",
                "quantile_hierarchy": ["vehicle_status", "global"],
                "finite_sample_order": "ceil((n + 1) * coverage)",
                "lower_bound": "max(0, point_prediction - quantile * scale)",
                "upper_bound": "point_prediction + quantile * scale",
                "coverage_calibrations": [item.to_dict() for item in self.coverage_calibrations],
            },
            "scale": self.scale_binding.to_dict(),
            "confidence": {
                **self.confidence_thresholds.to_dict(),
                "semantics": "precision_and_support_label_not_probability",
                "data_quality_warnings_are_separate": True,
            },
            "publication": {
                "raw_rows_predictions_or_residuals_persisted": False,
                "legacy_holdout_opened": False,
            },
        }


@dataclass(frozen=True, slots=True)
class CandidateInterval:
    predicted_value: float
    interval_lower: float
    interval_upper: float
    interval_coverage: float
    interval_width: float
    confidence_label: str
    calibration_version: str
    calibration_support: int
    calibration_method: str


@dataclass(frozen=True, slots=True)
class BoundGammaScaleModel:
    """A Gamma scale model loaded only after its artifact checksum binding passes."""

    estimator: ScaleEstimator
    model_path: str
    model_sha256: str
    scale_version: str

    def __post_init__(self) -> None:
        if not callable(getattr(self.estimator, "predict", None)):
            raise UncertaintyCandidateArtifactError("bound Gamma model cannot predict")
        ScaleBinding(
            kind="gamma_joblib",
            version=self.scale_version,
            model_path=self.model_path,
            model_sha256=self.model_sha256,
        )


def build_candidate_artifact(
    *,
    selected_method: MethodId,
    full_quantiles: Mapping[str, object],
    generated_at: str,
    comparison_evidence_sha256: str,
    gamma_model_path: str | None = None,
    gamma_model_sha256: str | None = None,
) -> UncertaintyCandidateArtifact:
    if selected_method not in (GAMMA_METHOD, SMOOTH_METHOD):
        raise UncertaintyCandidateArtifactError("baseline may not create a candidate artifact")
    coverages = tuple(
        _coverage_from_mapping(cast(Mapping[str, object], full_quantiles[str(level)]))
        for level in COVERAGE_LEVELS
    )
    binding = (
        ScaleBinding(
            kind="gamma_joblib",
            version=SCALE_VERSION,
            model_path=gamma_model_path,
            model_sha256=gamma_model_sha256,
        )
        if selected_method == GAMMA_METHOD
        else ScaleBinding(kind="smooth_formula", version="smooth-value-scale-v1")
    )
    return UncertaintyCandidateArtifact(
        generated_at=generated_at,
        comparison_evidence_sha256=comparison_evidence_sha256,
        selected_method=cast(CandidateMethod, selected_method),
        coverage_calibrations=coverages,
        scale_binding=binding,
    )


def canonical_candidate_artifact_json(artifact: UncertaintyCandidateArtifact) -> str:
    if not isinstance(artifact, UncertaintyCandidateArtifact):
        raise UncertaintyCandidateArtifactError("candidate artifact has an invalid type")
    artifact.__post_init__()
    try:
        return (
            json.dumps(
                artifact.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise UncertaintyCandidateArtifactError("candidate artifact is not JSON-safe") from error


def load_candidate_artifact(
    serialized: str | bytes,
    *,
    active_model_identity_sha256: str,
    expected_artifact_sha256: str,
    expected_comparison_evidence_sha256: str,
) -> UncertaintyCandidateArtifact:
    expected_artifact = _sha256(expected_artifact_sha256, label="candidate artifact")
    expected_comparison = _sha256(
        expected_comparison_evidence_sha256,
        label="comparison evidence",
    )
    payload, text = _bounded_payload_and_text(serialized)
    if hashlib.sha256(payload).hexdigest() != expected_artifact:
        raise UncertaintyCandidateArtifactError(
            "candidate artifact checksum differs from trusted evidence"
        )
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise UncertaintyCandidateArtifactError("candidate artifact is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise UncertaintyCandidateArtifactError("candidate artifact root must be an object")
    try:
        artifact = _artifact_from_mapping(
            value,
            active_model_identity_sha256,
            expected_comparison,
        )
    except UncertaintyCandidateArtifactError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise UncertaintyCandidateArtifactError(
            "candidate artifact structure is invalid"
        ) from error
    if canonical_candidate_artifact_json(artifact) != text:
        raise UncertaintyCandidateArtifactError("candidate artifact is not canonical")
    return artifact


def load_bound_gamma_scale_model(
    *,
    artifact: UncertaintyCandidateArtifact,
    project_root: str | Path,
) -> BoundGammaScaleModel:
    """Verify a trusted local Gamma model's immutable binding before deserialization."""

    if artifact.selected_method != GAMMA_METHOD:
        raise UncertaintyCandidateArtifactError("artifact does not bind a Gamma scale model")
    binding = artifact.scale_binding
    if binding.kind != "gamma_joblib" or binding.model_path is None or binding.model_sha256 is None:
        raise UncertaintyCandidateArtifactError("Gamma scale binding is incomplete")
    root = Path(os.path.abspath(os.fspath(project_root)))
    if root.is_symlink() or not root.is_dir():
        raise UncertaintyCandidateArtifactError(
            "Gamma model project root must be a non-symlink directory"
        )
    relative = _fixed_gamma_model_relative_path(binding.model_path)
    model_path = _normalized_model_path(root, relative)
    _reject_model_symlink_components(model_path, root)
    try:
        before = model_path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise UncertaintyCandidateArtifactError(
                "Gamma model must be a regular non-symlink file"
            )
        if before.st_size <= 0 or before.st_size > MAXIMUM_GAMMA_MODEL_BYTES:
            raise UncertaintyCandidateArtifactError("Gamma model file size is invalid")
        payload = model_path.read_bytes()
        after = model_path.lstat()
    except OSError as error:
        raise UncertaintyCandidateArtifactError("Gamma model could not be read") from error
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise UncertaintyCandidateArtifactError("Gamma model changed while it was read")
    observed = hashlib.sha256(payload).hexdigest()
    if observed != binding.model_sha256:
        raise UncertaintyCandidateArtifactError("Gamma model checksum differs from artifact")
    try:
        loaded = joblib.load(io.BytesIO(payload))
    except Exception as error:
        raise UncertaintyCandidateArtifactError(
            "verified Gamma model could not be loaded"
        ) from error
    if not callable(getattr(loaded, "predict", None)):
        raise UncertaintyCandidateArtifactError("verified Gamma model cannot predict")
    return BoundGammaScaleModel(
        estimator=cast(ScaleEstimator, loaded),
        model_path=binding.model_path,
        model_sha256=binding.model_sha256,
        scale_version=binding.version,
    )


def candidate_interval(
    *,
    point_prediction: float,
    vehicle_status: str,
    coverage: float,
    artifact: UncertaintyCandidateArtifact,
    vehicle_features: pd.DataFrame | None = None,
    gamma_model: BoundGammaScaleModel | None = None,
) -> CandidateInterval:
    point = _nonnegative_float(point_prediction, label="point prediction")
    status = vehicle_status.strip().lower() if isinstance(vehicle_status, str) else ""
    selected = next(
        (item for item in artifact.coverage_calibrations if item.coverage == coverage),
        None,
    )
    if selected is None:
        raise UncertaintyCandidateArtifactError("coverage is unsupported")
    status_quantile = next(
        (item for item in selected.status_quantiles if item.status == status),
        None,
    )
    if status_quantile is None:
        quantile = selected.global_quantile
        support = selected.global_support
        calibration_method = "global_fallback"
    else:
        quantile = status_quantile.quantile
        support = status_quantile.support
        calibration_method = artifact.selected_method
    if artifact.selected_method == SMOOTH_METHOD:
        scale = float(smooth_value_scale(np.asarray([point], dtype=np.float64))[0])
    else:
        if not isinstance(gamma_model, BoundGammaScaleModel):
            raise UncertaintyCandidateArtifactError(
                "Gamma candidate requires one vehicle row and a checksum-bound scale model"
            )
        if vehicle_features is None or len(vehicle_features) != 1:
            raise UncertaintyCandidateArtifactError(
                "Gamma candidate requires exactly one vehicle feature row"
            )
        binding = artifact.scale_binding
        if (
            binding.model_path != gamma_model.model_path
            or binding.model_sha256 != gamma_model.model_sha256
            or binding.version != gamma_model.scale_version
        ):
            raise UncertaintyCandidateArtifactError("Gamma model binding differs from artifact")
        scale = float(
            _predict_gamma_scale(
                gamma_model.estimator,
                vehicle_features,
                np.asarray([point], dtype=np.float64),
            ).clipped[0]
        )
    radius = quantile * scale
    lower = max(0.0, point - radius)
    upper = point + radius
    relative_width = (upper - lower) / max(point, 1.0)
    confidence = _confidence_label(relative_width, support, artifact.confidence_thresholds)
    return CandidateInterval(
        predicted_value=point,
        interval_lower=lower,
        interval_upper=upper,
        interval_coverage=coverage,
        interval_width=upper - lower,
        confidence_label=confidence,
        calibration_version=CALIBRATION_VERSION,
        calibration_support=support,
        calibration_method=calibration_method,
    )


def _coverage_from_mapping(value: Mapping[str, object]) -> NormalizedCoverage:
    status = cast(Mapping[str, object], value["status"])
    return NormalizedCoverage(
        coverage=float(cast(float, value["coverage"])),
        global_support=cast(int, value["global_support"]),
        global_quantile=float(cast(float, value["global_quantile"])),
        status_quantiles=tuple(
            StatusQuantile(
                status=name,
                support=cast(int, cast(Mapping[str, object], status[name])["support"]),
                quantile=float(cast(float, cast(Mapping[str, object], status[name])["quantile"])),
            )
            for name in RETAIL_VEHICLE_STATUSES
        ),
    )


def _artifact_from_mapping(
    value: Mapping[str, object],
    active_model_identity_sha256: str,
    expected_comparison_evidence_sha256: str,
) -> UncertaintyCandidateArtifact:
    expected_root = {
        "schema_version",
        "artifact_type",
        "calibration_version",
        "generated_at",
        "status",
        "bound_model",
        "frozen_evidence",
        "selected_method",
        "interval",
        "scale",
        "confidence",
        "publication",
    }
    if set(value) != expected_root:
        raise UncertaintyCandidateArtifactError("candidate artifact root fields are invalid")
    if (
        value["schema_version"] != 2
        or value["artifact_type"] != ARTIFACT_TYPE
        or value["calibration_version"] != CALIBRATION_VERSION
        or value["status"] != "validated_candidate_not_production_final"
    ):
        raise UncertaintyCandidateArtifactError("candidate artifact metadata is invalid")
    bound_model = cast(Mapping[str, object], value["bound_model"])
    if (
        bound_model != active_rf05_identity().to_dict()
        or bound_model.get("identity_sha256") != active_model_identity_sha256
    ):
        raise UncertaintyCandidateArtifactError("candidate artifact does not bind active RF05")
    evidence = cast(Mapping[str, object], value["frozen_evidence"])
    if evidence != {
        "sharpness_policy_sha256": SHARPNESS_POLICY_SHA256,
        "calibration_assignment_sha256": CALIBRATION_ASSIGNMENT_SHA256,
        "calibration_v1_artifact_sha256": CALIBRATION_V1_ARTIFACT_SHA256,
        "calibration_v1_report_sha256": CALIBRATION_V1_REPORT_SHA256,
        "comparison_evidence_sha256": expected_comparison_evidence_sha256,
    }:
        raise UncertaintyCandidateArtifactError("candidate frozen evidence differs")
    interval = cast(Mapping[str, object], value["interval"])
    expected_interval = {
        "score",
        "quantile_hierarchy",
        "finite_sample_order",
        "lower_bound",
        "upper_bound",
        "coverage_calibrations",
    }
    if set(interval) != expected_interval:
        raise UncertaintyCandidateArtifactError("candidate interval fields are invalid")
    coverage_values = cast(list[Mapping[str, object]], interval["coverage_calibrations"])
    coverages = tuple(_parsed_coverage(item) for item in coverage_values)
    method = value["selected_method"]
    if method not in (GAMMA_METHOD, SMOOTH_METHOD):
        raise UncertaintyCandidateArtifactError("candidate method is invalid")
    scale = cast(Mapping[str, object], value["scale"])
    binding = _parsed_scale(scale, cast(CandidateMethod, method))
    confidence = cast(Mapping[str, object], value["confidence"])
    if confidence != {
        **CONFIDENCE_THRESHOLDS.to_dict(),
        "semantics": "precision_and_support_label_not_probability",
        "data_quality_warnings_are_separate": True,
    }:
        raise UncertaintyCandidateArtifactError("candidate confidence policy differs")
    publication = cast(Mapping[str, object], value["publication"])
    if publication != {
        "raw_rows_predictions_or_residuals_persisted": False,
        "legacy_holdout_opened": False,
    }:
        raise UncertaintyCandidateArtifactError("candidate publication boundary differs")
    generated_at = value["generated_at"]
    if not isinstance(generated_at, str):
        raise UncertaintyCandidateArtifactError("candidate timestamp is invalid")
    return UncertaintyCandidateArtifact(
        generated_at=generated_at,
        comparison_evidence_sha256=expected_comparison_evidence_sha256,
        selected_method=cast(CandidateMethod, method),
        coverage_calibrations=coverages,
        scale_binding=binding,
    )


def _parsed_coverage(value: Mapping[str, object]) -> NormalizedCoverage:
    if set(value) != {"coverage", "global_support", "global_quantile", "status_quantiles"}:
        raise UncertaintyCandidateArtifactError("coverage calibration fields are invalid")
    items = cast(list[Mapping[str, object]], value["status_quantiles"])
    return NormalizedCoverage(
        coverage=float(cast(float, value["coverage"])),
        global_support=cast(int, value["global_support"]),
        global_quantile=float(cast(float, value["global_quantile"])),
        status_quantiles=tuple(
            StatusQuantile(
                status=cast(str, item["status"]),
                support=cast(int, item["support"]),
                quantile=float(cast(float, item["quantile"])),
            )
            for item in items
        ),
    )


def _parsed_scale(value: Mapping[str, object], method: CandidateMethod) -> ScaleBinding:
    if method == SMOOTH_METHOD:
        if value != {
            "kind": "smooth_formula",
            "version": "smooth-value-scale-v1",
            "formula": "1 + ln(1 + max(RF05 prediction USD, 0) / 10000)",
        }:
            raise UncertaintyCandidateArtifactError("smooth scale fields differ")
        return ScaleBinding(kind="smooth_formula", version="smooth-value-scale-v1")
    expected = {
        "kind",
        "version",
        "model_path",
        "model_sha256",
        "scale_floor_usd",
        "scale_cap_usd",
    }
    if (
        set(value) != expected
        or value["scale_floor_usd"] != GAMMA_SCALE_FLOOR_USD
        or value["scale_cap_usd"] != GAMMA_SCALE_CAP_USD
    ):
        raise UncertaintyCandidateArtifactError("Gamma scale fields differ")
    return ScaleBinding(
        kind="gamma_joblib",
        version=cast(str, value["version"]),
        model_path=cast(str, value["model_path"]),
        model_sha256=cast(str, value["model_sha256"]),
    )


def _confidence_label(
    relative_width: float,
    support: int,
    thresholds: ConfidenceThresholds,
) -> str:
    if (
        support >= thresholds.high_minimum_support
        and relative_width <= thresholds.high_max_relative_width
    ):
        return "High confidence"
    if (
        support >= thresholds.moderate_minimum_support
        and relative_width <= thresholds.moderate_max_relative_width
    ):
        return "Moderate confidence"
    return "Low confidence"


def _bounded_payload_and_text(serialized: str | bytes) -> tuple[bytes, str]:
    if isinstance(serialized, bytes):
        if len(serialized) > MAXIMUM_ARTIFACT_BYTES:
            raise UncertaintyCandidateArtifactError("candidate artifact exceeds maximum size")
        try:
            return serialized, serialized.decode("utf-8")
        except UnicodeDecodeError as error:
            raise UncertaintyCandidateArtifactError("candidate artifact must be UTF-8") from error
    if isinstance(serialized, str):
        try:
            payload = serialized.encode("utf-8")
        except UnicodeEncodeError as error:
            raise UncertaintyCandidateArtifactError("candidate artifact must be UTF-8") from error
        if len(payload) > MAXIMUM_ARTIFACT_BYTES:
            raise UncertaintyCandidateArtifactError("candidate artifact exceeds maximum size")
        return payload, serialized
    raise UncertaintyCandidateArtifactError("candidate artifact must be text or bytes")


def _fixed_gamma_model_relative_path(value: str) -> PurePosixPath:
    if (
        value != GAMMA_MODEL_RELATIVE_PATH
        or "\\" in value
        or ":" in value
        or value.startswith(("/", "//"))
    ):
        raise UncertaintyCandidateArtifactError(
            "Gamma model path must use the fixed safe relative joblib path"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise UncertaintyCandidateArtifactError(
            "Gamma model path must use the fixed safe relative joblib path"
        )
    return path


def _normalized_model_path(root: Path, relative: PurePosixPath) -> Path:
    model_path = Path(os.path.abspath(os.fspath(root.joinpath(*relative.parts))))
    try:
        common = Path(os.path.commonpath((os.fspath(root), os.fspath(model_path))))
    except ValueError as error:
        raise UncertaintyCandidateArtifactError("Gamma model path escaped project root") from error
    if os.path.normcase(os.fspath(common)) != os.path.normcase(os.fspath(root)):
        raise UncertaintyCandidateArtifactError("Gamma model path escaped project root")
    return model_path


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise UncertaintyCandidateArtifactError(
                f"candidate artifact has duplicate field: {key}"
            )
        output[key] = value
    return output


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UncertaintyCandidateArtifactError(f"{label} must be a positive integer")
    return value


def _positive_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UncertaintyCandidateArtifactError(f"{label} must be positive finite")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise UncertaintyCandidateArtifactError(f"{label} must be positive finite")
    return result


def _nonnegative_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UncertaintyCandidateArtifactError(f"{label} must be nonnegative finite")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise UncertaintyCandidateArtifactError(f"{label} must be nonnegative finite")
    return result


def _reject_model_symlink_components(path: Path, root: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise UncertaintyCandidateArtifactError("Gamma model path must not contain symlinks")
        if current == root:
            return
        if current == current.parent:
            raise UncertaintyCandidateArtifactError("Gamma model path escaped project root")
        current = current.parent


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise UncertaintyCandidateArtifactError(f"{label} SHA-256 is invalid")
    return value


__all__ = [
    "ARTIFACT_TYPE",
    "BoundGammaScaleModel",
    "CALIBRATION_VERSION",
    "CandidateInterval",
    "GAMMA_MODEL_RELATIVE_PATH",
    "NormalizedCoverage",
    "ScaleBinding",
    "StatusQuantile",
    "UncertaintyCandidateArtifact",
    "UncertaintyCandidateArtifactError",
    "build_candidate_artifact",
    "candidate_interval",
    "canonical_candidate_artifact_json",
    "load_candidate_artifact",
    "load_bound_gamma_scale_model",
]
