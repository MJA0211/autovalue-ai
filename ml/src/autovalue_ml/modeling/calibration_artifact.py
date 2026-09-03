"""Strict aggregate-only calibration artifact and serving-time intervals."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Final, Literal, TypeAlias, cast

CALIBRATION_VERSION: Final = "retail-rf05-split-conformal-v1"
CALIBRATION_POLICY_SHA256: Final = (
    "1398519c699bd129ef4fbb552813c064839c6c1e1c4ecd35c7f5d42bcf8e1ca2"
)
PHASE4_PROTOCOL_SHA256: Final = "6e517acb29634d676155c80fb73f4f126db492eba12a4281e9216dc568b1d384"
PHASE4_RETAIL_CONFIRMATION_SHA256: Final = (
    "07cf667e2e325f0bbb9b0fca1d62f4f3cdb54db4d607a03ff603142ee5fbc54f"
)
CALIBRATION_ASSIGNMENT_SHA256: Final = (
    "caa743681158c4eaccb2ec75ce17a1c5e20327a311f66c5e8e0d0c630c48e992"
)
RF05_CANDIDATE_ID: Final = "phase4-retail-random_forest-05"
RF05_PARAMETERS: Final = (96, 1024, 5, 1.0, 0.6)
RF05_RANDOM_STATE: Final = 1_254_777_149
FEATURE_CONTRACT_VERSION: Final = "retail-historical-asking-price-v2"
CALIBRATION_SAMPLE_COUNT: Final = 10_958
DEVELOPMENT_SAMPLE_COUNT: Final = 98_552
MINIMUM_BUCKET_SUPPORT: Final = 400
COVERAGE_LEVELS: Final = (0.8, 0.9, 0.95)
ARTIFACT_TYPE: Final = "retail_rf05_split_conformal_calibration"
_MAX_ARTIFACT_BYTES: Final = 500_000
_DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")

CalibrationMethod: TypeAlias = Literal[
    "global",
    "vehicle_status",
    "vehicle_status_and_predicted_value_band_hierarchy",
]
ConfidenceLabel: TypeAlias = Literal["High confidence", "Moderate confidence", "Low confidence"]


class CalibrationArtifactError(ValueError):
    """A calibration artifact or model binding is invalid."""


@dataclass(frozen=True, slots=True)
class BoundRF05Identity:
    """Logical identity of the frozen estimator, independent of serialization."""

    identity_sha256: str
    candidate_id: str
    parameters: tuple[int, int, int, float, float]
    random_state: int
    feature_contract_version: str
    phase4_protocol_sha256: str
    phase4_confirmation_sha256: str

    def __post_init__(self) -> None:
        if self.to_dict() != _expected_rf05_identity_fields():
            raise CalibrationArtifactError("bound RF05 identity differs from frozen evidence")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["parameters"] = list(self.parameters)
        return payload


@dataclass(frozen=True, slots=True)
class ConditionalRadius:
    """One aggregate calibration radius and its support."""

    key: str
    support: int
    radius_usd: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key or len(self.key) > 300:
            raise CalibrationArtifactError("conditional radius key is invalid")
        if type(self.support) is not int or self.support < 1:
            raise CalibrationArtifactError("conditional radius support must be positive")
        if self.support < MINIMUM_BUCKET_SUPPORT and self.radius_usd is not None:
            raise CalibrationArtifactError("undersupported conditional radius must be omitted")
        if self.support >= MINIMUM_BUCKET_SUPPORT and self.radius_usd is None:
            raise CalibrationArtifactError("supported conditional radius must be populated")
        if self.radius_usd is not None:
            _finite_nonnegative(self.radius_usd, label="conditional radius")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CoverageCalibration:
    """All full-calibration radii for one preregistered coverage level."""

    coverage: float
    global_radius_usd: float
    status_radii: tuple[ConditionalRadius, ...]
    predicted_value_band_radii: tuple[ConditionalRadius, ...]
    status_value_band_radii: tuple[ConditionalRadius, ...]

    def __post_init__(self) -> None:
        if self.coverage not in COVERAGE_LEVELS:
            raise CalibrationArtifactError("coverage is outside the preregistered levels")
        _finite_nonnegative(self.global_radius_usd, label="global radius")
        for label, entries in (
            ("status", self.status_radii),
            ("predicted value band", self.predicted_value_band_radii),
            ("status-value band", self.status_value_band_radii),
        ):
            if not isinstance(entries, tuple):
                raise CalibrationArtifactError(f"{label} radii must be a tuple")
            if any(not isinstance(entry, ConditionalRadius) for entry in entries):
                raise CalibrationArtifactError(f"{label} radii contain an invalid entry")
            if len({entry.key for entry in entries}) != len(entries):
                raise CalibrationArtifactError(f"{label} radii must have unique keys")

    def to_dict(self) -> dict[str, object]:
        return {
            "coverage": self.coverage,
            "global_radius_usd": self.global_radius_usd,
            "status_radii": [entry.to_dict() for entry in self.status_radii],
            "predicted_value_band_radii": [
                entry.to_dict() for entry in self.predicted_value_band_radii
            ],
            "status_value_band_radii": [entry.to_dict() for entry in self.status_value_band_radii],
        }


@dataclass(frozen=True, slots=True)
class ConfidenceThresholds:
    """Empirical relative-width thresholds fixed by the preregistered percentiles."""

    coverage: float
    high_max_relative_width: float
    moderate_max_relative_width: float
    high_minimum_support: int = 1_000
    moderate_minimum_support: int = MINIMUM_BUCKET_SUPPORT

    def __post_init__(self) -> None:
        if self.coverage != 0.9:
            raise CalibrationArtifactError("confidence labels must use the 90% interval")
        high = _finite_nonnegative(self.high_max_relative_width, label="high threshold")
        moderate = _finite_nonnegative(self.moderate_max_relative_width, label="moderate threshold")
        if high > moderate:
            raise CalibrationArtifactError("confidence relative-width thresholds are unordered")
        if self.high_minimum_support != 1_000:
            raise CalibrationArtifactError("high-confidence support threshold differs from policy")
        if self.moderate_minimum_support != MINIMUM_BUCKET_SUPPORT:
            raise CalibrationArtifactError(
                "moderate-confidence support threshold differs from policy"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetailCalibrationArtifact:
    """Versioned row-free state needed for RF05 interval generation."""

    generated_at: str
    bound_model: BoundRF05Identity
    selected_method: CalibrationMethod
    predicted_value_cutpoints_usd: tuple[float, float, float]
    coverage_calibrations: tuple[CoverageCalibration, ...]
    confidence_thresholds: ConfidenceThresholds
    calibration_version: str = CALIBRATION_VERSION
    policy_sha256: str = CALIBRATION_POLICY_SHA256
    calibration_assignment_sha256: str = CALIBRATION_ASSIGNMENT_SHA256
    calibration_sample_count: int = CALIBRATION_SAMPLE_COUNT
    development_sample_count: int = DEVELOPMENT_SAMPLE_COUNT
    legacy_holdout_accessed: bool = False
    raw_rows_persisted: bool = False

    def __post_init__(self) -> None:
        _validate_generated_at(self.generated_at)
        if not isinstance(self.bound_model, BoundRF05Identity):
            raise CalibrationArtifactError("bound_model is invalid")
        if self.selected_method not in {
            "global",
            "vehicle_status",
            "vehicle_status_and_predicted_value_band_hierarchy",
        }:
            raise CalibrationArtifactError("selected calibration method is invalid")
        if self.calibration_version != CALIBRATION_VERSION:
            raise CalibrationArtifactError("calibration version is invalid")
        if self.policy_sha256 != CALIBRATION_POLICY_SHA256:
            raise CalibrationArtifactError("calibration policy checksum differs")
        if self.calibration_assignment_sha256 != CALIBRATION_ASSIGNMENT_SHA256:
            raise CalibrationArtifactError("calibration assignment checksum differs")
        if self.calibration_sample_count != CALIBRATION_SAMPLE_COUNT:
            raise CalibrationArtifactError("calibration population count differs")
        if self.development_sample_count != DEVELOPMENT_SAMPLE_COUNT:
            raise CalibrationArtifactError("development population count differs")
        if self.legacy_holdout_accessed is not False or self.raw_rows_persisted is not False:
            raise CalibrationArtifactError("artifact violates protected-data policy")
        if not isinstance(self.predicted_value_cutpoints_usd, tuple):
            raise CalibrationArtifactError("predicted value cutpoints must be a tuple")
        cutpoints = tuple(
            _finite_nonnegative(value, label="predicted value cutpoint")
            for value in self.predicted_value_cutpoints_usd
        )
        if len(cutpoints) != 3 or not cutpoints[0] < cutpoints[1] < cutpoints[2]:
            raise CalibrationArtifactError("predicted value cutpoints must be strictly increasing")
        if not isinstance(self.coverage_calibrations, tuple) or any(
            not isinstance(item, CoverageCalibration) for item in self.coverage_calibrations
        ):
            raise CalibrationArtifactError("coverage calibrations are invalid")
        if tuple(item.coverage for item in self.coverage_calibrations) != COVERAGE_LEVELS:
            raise CalibrationArtifactError("artifact must contain 80%, 90%, and 95% calibration")
        if not isinstance(self.confidence_thresholds, ConfidenceThresholds):
            raise CalibrationArtifactError("confidence thresholds are invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_type": ARTIFACT_TYPE,
            "calibration_version": self.calibration_version,
            "generated_at": self.generated_at,
            "policy_sha256": self.policy_sha256,
            "bound_model": self.bound_model.to_dict(),
            "calibration_population": {
                "calibration_assignment_sha256": self.calibration_assignment_sha256,
                "calibration_sample_count": self.calibration_sample_count,
                "development_sample_count": self.development_sample_count,
                "first_authorized_use": True,
                "legacy_holdout_accessed": self.legacy_holdout_accessed,
                "raw_rows_persisted": self.raw_rows_persisted,
            },
            "method": {
                "score": "absolute_error_usd",
                "finite_sample_order": "ceil((n + 1) * coverage)",
                "selected_method": self.selected_method,
                "minimum_bucket_support": MINIMUM_BUCKET_SUPPORT,
                "fallback_order": [
                    "vehicle_status_and_predicted_value_band",
                    "vehicle_status",
                    "predicted_value_band",
                    "global",
                ],
                "predicted_value_cutpoints_usd": list(self.predicted_value_cutpoints_usd),
            },
            "coverage_calibrations": [item.to_dict() for item in self.coverage_calibrations],
            "confidence_thresholds": self.confidence_thresholds.to_dict(),
        }

    def calibration_for(self, coverage: float) -> CoverageCalibration:
        for calibration in self.coverage_calibrations:
            if calibration.coverage == coverage:
                return calibration
        raise CalibrationArtifactError("requested coverage is not available")


@dataclass(frozen=True, slots=True)
class PredictionDataQuality:
    """Feature-quality signals kept distinct from conformal uncertainty."""

    mileage_missing: bool = False
    rare_or_unseen_category: bool = False
    unsupported_feature_combination: bool = False

    def warnings(self) -> tuple[str, ...]:
        values: list[str] = []
        if self.mileage_missing:
            values.append("missing_mileage")
        if self.rare_or_unseen_category:
            values.append("rare_or_unseen_category")
        if self.unsupported_feature_combination:
            values.append("unsupported_feature_combination")
        return tuple(values)


@dataclass(frozen=True, slots=True)
class CalibratedValuation:
    """User-facing-ready values with neutral uncertainty language."""

    predicted_value: float
    interval_lower: float
    interval_upper: float
    interval_coverage: float
    interval_width: float
    confidence_label: ConfidenceLabel
    calibration_version: str
    warnings: tuple[str, ...]
    calibration_method: str
    calibration_support: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        return payload


def active_rf05_identity() -> BoundRF05Identity:
    """Return the checksum-bound logical identity of the frozen RF05 predictor."""
    payload = _expected_rf05_identity_fields()
    return BoundRF05Identity(
        identity_sha256=cast(str, payload["identity_sha256"]),
        candidate_id=RF05_CANDIDATE_ID,
        parameters=RF05_PARAMETERS,
        random_state=RF05_RANDOM_STATE,
        feature_contract_version=FEATURE_CONTRACT_VERSION,
        phase4_protocol_sha256=PHASE4_PROTOCOL_SHA256,
        phase4_confirmation_sha256=PHASE4_RETAIL_CONFIRMATION_SHA256,
    )


def _expected_rf05_identity_fields() -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_id": RF05_CANDIDATE_ID,
        "parameters": list(RF05_PARAMETERS),
        "random_state": RF05_RANDOM_STATE,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "phase4_protocol_sha256": PHASE4_PROTOCOL_SHA256,
        "phase4_confirmation_sha256": PHASE4_RETAIL_CONFIRMATION_SHA256,
    }
    identity_sha256 = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return {"identity_sha256": identity_sha256, **payload}


def calibrated_valuation(
    *,
    point_prediction: float,
    vehicle_status: str,
    coverage: float,
    artifact: RetailCalibrationArtifact,
    data_quality: PredictionDataQuality | None = None,
) -> CalibratedValuation:
    """Apply the selected empirical radius and separate data-quality warnings."""
    point = _finite_nonnegative(point_prediction, label="point prediction")
    status = vehicle_status.strip().lower() if isinstance(vehicle_status, str) else ""
    if status not in {"certified", "new", "used"}:
        status = "__unknown__"
    calibration = artifact.calibration_for(coverage)
    band = predicted_value_band(point, artifact.predicted_value_cutpoints_usd)
    radius, support, method = _select_radius(
        calibration,
        selected_method=artifact.selected_method,
        status=status,
        band=band,
    )
    lower = max(0.0, point - radius)
    upper = point + radius
    width = upper - lower
    relative_width = width / max(point, 1.0)
    thresholds = artifact.confidence_thresholds
    if support >= thresholds.high_minimum_support and relative_width <= (
        thresholds.high_max_relative_width
    ):
        confidence: ConfidenceLabel = "High confidence"
    elif support >= thresholds.moderate_minimum_support and relative_width <= (
        thresholds.moderate_max_relative_width
    ):
        confidence = "Moderate confidence"
    else:
        confidence = "Low confidence"
    return CalibratedValuation(
        predicted_value=point,
        interval_lower=lower,
        interval_upper=upper,
        interval_coverage=coverage,
        interval_width=width,
        confidence_label=confidence,
        calibration_version=artifact.calibration_version,
        warnings=(data_quality or PredictionDataQuality()).warnings(),
        calibration_method=method,
        calibration_support=support,
    )


def predicted_value_band(point_prediction: float, cutpoints: Sequence[float]) -> str:
    point = _finite_nonnegative(point_prediction, label="point prediction")
    if len(cutpoints) != 3:
        raise CalibrationArtifactError("predicted value bands require three cutpoints")
    first, second, third = (float(value) for value in cutpoints)
    if point <= first:
        return "band_1"
    if point <= second:
        return "band_2"
    if point <= third:
        return "band_3"
    return "band_4"


def canonical_calibration_artifact_json(artifact: RetailCalibrationArtifact) -> str:
    if not isinstance(artifact, RetailCalibrationArtifact):
        raise CalibrationArtifactError("artifact has an invalid type")
    artifact.__post_init__()
    return _canonical_bytes(artifact.to_dict()).decode("utf-8") + "\n"


def load_calibration_artifact(
    serialized: str | bytes,
    *,
    active_model_identity_sha256: str,
) -> RetailCalibrationArtifact:
    """Strictly load row-free state and fail on an RF05 identity mismatch."""
    text = _bounded_text(serialized)
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise CalibrationArtifactError("calibration artifact is not valid JSON") from error
    root = _exact_mapping(
        value,
        {
            "schema_version",
            "artifact_type",
            "calibration_version",
            "generated_at",
            "policy_sha256",
            "bound_model",
            "calibration_population",
            "method",
            "coverage_calibrations",
            "confidence_thresholds",
        },
        label="artifact",
    )
    if root["schema_version"] != 1 or root["artifact_type"] != ARTIFACT_TYPE:
        raise CalibrationArtifactError("calibration artifact metadata is invalid")
    bound_model = _parse_bound_model(root["bound_model"])
    if bound_model.identity_sha256 != active_model_identity_sha256:
        raise CalibrationArtifactError("calibration artifact does not match the active RF05 model")
    population = _exact_mapping(
        root["calibration_population"],
        {
            "calibration_assignment_sha256",
            "calibration_sample_count",
            "development_sample_count",
            "first_authorized_use",
            "legacy_holdout_accessed",
            "raw_rows_persisted",
        },
        label="calibration_population",
    )
    if population["first_authorized_use"] is not True:
        raise CalibrationArtifactError("first calibration use is not recorded")
    method = _parse_method(root["method"])
    coverage_value = root["coverage_calibrations"]
    if not isinstance(coverage_value, list):
        raise CalibrationArtifactError("coverage_calibrations must be an array")
    confidence = _parse_confidence(root["confidence_thresholds"])
    artifact = RetailCalibrationArtifact(
        generated_at=_text(root["generated_at"], label="generated_at"),
        bound_model=bound_model,
        selected_method=cast(CalibrationMethod, method["selected_method"]),
        predicted_value_cutpoints_usd=_cutpoint_tuple(method["predicted_value_cutpoints_usd"]),
        coverage_calibrations=tuple(_parse_coverage(item) for item in coverage_value),
        confidence_thresholds=confidence,
        calibration_version=_text(root["calibration_version"], label="calibration_version"),
        policy_sha256=_digest(root["policy_sha256"], label="policy_sha256"),
        calibration_assignment_sha256=_digest(
            population["calibration_assignment_sha256"],
            label="calibration_assignment_sha256",
        ),
        calibration_sample_count=_integer(
            population["calibration_sample_count"], label="calibration_sample_count"
        ),
        development_sample_count=_integer(
            population["development_sample_count"], label="development_sample_count"
        ),
        legacy_holdout_accessed=_boolean(
            population["legacy_holdout_accessed"], label="legacy_holdout_accessed"
        ),
        raw_rows_persisted=_boolean(population["raw_rows_persisted"], label="raw_rows_persisted"),
    )
    if canonical_calibration_artifact_json(artifact) != text:
        raise CalibrationArtifactError("calibration artifact differs from canonical encoding")
    return artifact


def _select_radius(
    calibration: CoverageCalibration,
    *,
    selected_method: CalibrationMethod,
    status: str,
    band: str,
) -> tuple[float, int, str]:
    if selected_method == "global":
        return calibration.global_radius_usd, CALIBRATION_SAMPLE_COUNT, "global"
    status_entries = {entry.key: entry for entry in calibration.status_radii}
    band_entries = {entry.key: entry for entry in calibration.predicted_value_band_radii}
    if selected_method == "vehicle_status":
        status_entry = status_entries.get(status)
        if status_entry is not None and status_entry.radius_usd is not None:
            return status_entry.radius_usd, status_entry.support, "vehicle_status"
        return calibration.global_radius_usd, CALIBRATION_SAMPLE_COUNT, "global_fallback"
    exact_entries = {entry.key: entry for entry in calibration.status_value_band_radii}
    exact_entry = exact_entries.get(f"{status}|{band}")
    if exact_entry is not None and exact_entry.radius_usd is not None:
        return (
            exact_entry.radius_usd,
            exact_entry.support,
            "vehicle_status_and_predicted_value_band",
        )
    status_entry = status_entries.get(status)
    if status_entry is not None and status_entry.radius_usd is not None:
        return status_entry.radius_usd, status_entry.support, "vehicle_status_fallback"
    band_entry = band_entries.get(band)
    if band_entry is not None and band_entry.radius_usd is not None:
        return band_entry.radius_usd, band_entry.support, "predicted_value_band_fallback"
    return calibration.global_radius_usd, CALIBRATION_SAMPLE_COUNT, "global_fallback"


def _parse_bound_model(value: object) -> BoundRF05Identity:
    payload = _exact_mapping(
        value,
        {
            "identity_sha256",
            "candidate_id",
            "parameters",
            "random_state",
            "feature_contract_version",
            "phase4_protocol_sha256",
            "phase4_confirmation_sha256",
        },
        label="bound_model",
    )
    parameters = payload["parameters"]
    if not isinstance(parameters, list) or len(parameters) != 5:
        raise CalibrationArtifactError("RF05 parameters are invalid")
    resolved_parameters = (
        _integer(parameters[0], label="n_estimators"),
        _integer(parameters[1], label="max_leaf_nodes"),
        _integer(parameters[2], label="min_samples_leaf"),
        _number(parameters[3], label="max_features"),
        _number(parameters[4], label="max_samples"),
    )
    return BoundRF05Identity(
        identity_sha256=_digest(payload["identity_sha256"], label="identity_sha256"),
        candidate_id=_text(payload["candidate_id"], label="candidate_id"),
        parameters=resolved_parameters,
        random_state=_integer(payload["random_state"], label="random_state"),
        feature_contract_version=_text(
            payload["feature_contract_version"], label="feature_contract_version"
        ),
        phase4_protocol_sha256=_digest(
            payload["phase4_protocol_sha256"], label="phase4_protocol_sha256"
        ),
        phase4_confirmation_sha256=_digest(
            payload["phase4_confirmation_sha256"], label="phase4_confirmation_sha256"
        ),
    )


def _parse_method(value: object) -> Mapping[str, object]:
    payload = _exact_mapping(
        value,
        {
            "score",
            "finite_sample_order",
            "selected_method",
            "minimum_bucket_support",
            "fallback_order",
            "predicted_value_cutpoints_usd",
        },
        label="method",
    )
    expected_literals = {
        "score": "absolute_error_usd",
        "finite_sample_order": "ceil((n + 1) * coverage)",
        "minimum_bucket_support": MINIMUM_BUCKET_SUPPORT,
        "fallback_order": [
            "vehicle_status_and_predicted_value_band",
            "vehicle_status",
            "predicted_value_band",
            "global",
        ],
    }
    for field, expected in expected_literals.items():
        if payload[field] != expected:
            raise CalibrationArtifactError(f"method {field} differs from policy")
    if payload["selected_method"] not in {
        "global",
        "vehicle_status",
        "vehicle_status_and_predicted_value_band_hierarchy",
    }:
        raise CalibrationArtifactError("selected_method is invalid")
    cutpoints = payload["predicted_value_cutpoints_usd"]
    if not isinstance(cutpoints, list) or len(cutpoints) != 3:
        raise CalibrationArtifactError("predicted value cutpoints are invalid")
    return {
        **payload,
        "predicted_value_cutpoints_usd": tuple(
            _number(item, label="predicted value cutpoint") for item in cutpoints
        ),
    }


def _parse_coverage(value: object) -> CoverageCalibration:
    payload = _exact_mapping(
        value,
        {
            "coverage",
            "global_radius_usd",
            "status_radii",
            "predicted_value_band_radii",
            "status_value_band_radii",
        },
        label="coverage calibration",
    )
    return CoverageCalibration(
        coverage=_number(payload["coverage"], label="coverage"),
        global_radius_usd=_number(payload["global_radius_usd"], label="global radius"),
        status_radii=_parse_radius_entries(payload["status_radii"]),
        predicted_value_band_radii=_parse_radius_entries(payload["predicted_value_band_radii"]),
        status_value_band_radii=_parse_radius_entries(payload["status_value_band_radii"]),
    )


def _parse_radius_entries(value: object) -> tuple[ConditionalRadius, ...]:
    if not isinstance(value, list):
        raise CalibrationArtifactError("conditional radii must be an array")
    entries: list[ConditionalRadius] = []
    for item in value:
        payload = _exact_mapping(item, {"key", "support", "radius_usd"}, label="radius entry")
        raw_radius = payload["radius_usd"]
        radius = None if raw_radius is None else _number(raw_radius, label="radius_usd")
        entries.append(
            ConditionalRadius(
                key=_text(payload["key"], label="radius key"),
                support=_integer(payload["support"], label="radius support"),
                radius_usd=radius,
            )
        )
    return tuple(entries)


def _parse_confidence(value: object) -> ConfidenceThresholds:
    payload = _exact_mapping(
        value,
        {
            "coverage",
            "high_max_relative_width",
            "moderate_max_relative_width",
            "high_minimum_support",
            "moderate_minimum_support",
        },
        label="confidence_thresholds",
    )
    return ConfidenceThresholds(
        coverage=_number(payload["coverage"], label="confidence coverage"),
        high_max_relative_width=_number(
            payload["high_max_relative_width"], label="high confidence threshold"
        ),
        moderate_max_relative_width=_number(
            payload["moderate_max_relative_width"], label="moderate confidence threshold"
        ),
        high_minimum_support=_integer(
            payload["high_minimum_support"], label="high confidence support"
        ),
        moderate_minimum_support=_integer(
            payload["moderate_minimum_support"], label="moderate confidence support"
        ),
    )


def _cutpoint_tuple(value: object) -> tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise CalibrationArtifactError("predicted value cutpoints are invalid")
    return cast(tuple[float, float, float], value)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _bounded_text(serialized: str | bytes) -> str:
    if isinstance(serialized, bytes):
        if len(serialized) > _MAX_ARTIFACT_BYTES:
            raise CalibrationArtifactError("calibration artifact exceeds maximum size")
        try:
            return serialized.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CalibrationArtifactError("calibration artifact must be UTF-8") from error
    if isinstance(serialized, str):
        if len(serialized.encode("utf-8")) > _MAX_ARTIFACT_BYTES:
            raise CalibrationArtifactError("calibration artifact exceeds maximum size")
        return serialized
    raise CalibrationArtifactError("calibration artifact must be text or bytes")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationArtifactError(f"calibration artifact has duplicate field: {key}")
        result[key] = value
    return result


def _exact_mapping(value: object, keys: set[str], *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise CalibrationArtifactError(f"{label} must be an object")
    if set(value) != keys:
        raise CalibrationArtifactError(f"{label} fields are invalid")
    return cast(Mapping[str, object], value)


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationArtifactError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise CalibrationArtifactError(f"{label} must be finite")
    return number


def _finite_nonnegative(value: object, *, label: str) -> float:
    number = _number(value, label=label)
    if number < 0:
        raise CalibrationArtifactError(f"{label} must be nonnegative")
    return number


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise CalibrationArtifactError(f"{label} must be an integer")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationArtifactError(f"{label} must be non-empty text")
    return value


def _digest(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if not _DIGEST_PATTERN.fullmatch(text):
        raise CalibrationArtifactError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise CalibrationArtifactError(f"{label} must be a boolean")
    return value


def _validate_generated_at(value: str) -> None:
    text = _text(value, label="generated_at")
    try:
        timestamp = datetime.fromisoformat(text)
    except ValueError as error:
        raise CalibrationArtifactError("generated_at must be ISO-8601") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise CalibrationArtifactError("generated_at must be timezone-aware UTC")
