from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling.calibration_artifact import (
    CALIBRATION_SAMPLE_COUNT,
    COVERAGE_LEVELS,
    active_rf05_identity,
)
from autovalue_ml.modeling.retail_uncertainty_sharpness import (
    GAMMA_METHOD,
    GAMMA_SCALE_CAP_USD,
    GAMMA_SCALE_FLOOR_USD,
    SCALE_VERSION,
    SMOOTH_METHOD,
    ScaleEstimator,
)
from autovalue_ml.modeling.uncertainty_candidate_artifact import (
    CALIBRATION_VERSION,
    GAMMA_MODEL_RELATIVE_PATH,
    BoundGammaScaleModel,
    CandidateMethod,
    ScaleBinding,
    UncertaintyCandidateArtifact,
    UncertaintyCandidateArtifactError,
    build_candidate_artifact,
    candidate_interval,
    canonical_candidate_artifact_json,
    load_bound_gamma_scale_model,
    load_candidate_artifact,
)
from numpy.typing import NDArray

GENERATED_AT = "2026-09-02T12:34:56+00:00"
COMPARISON_EVIDENCE_SHA256 = "b" * 64
GAMMA_MODEL_PATH = GAMMA_MODEL_RELATIVE_PATH
GAMMA_MODEL_SHA256 = "a" * 64


def _full_quantiles() -> dict[str, object]:
    status_supports = {
        "certified": 300,
        "new": 4_000,
        "used": 6_658,
    }
    quantiles = {
        0.8: (110.0, 80.0, 90.0, 100.0),
        0.9: (220.0, 160.0, 180.0, 200.0),
        0.95: (440.0, 320.0, 360.0, 400.0),
    }
    return {
        str(coverage): {
            "coverage": coverage,
            "global_support": CALIBRATION_SAMPLE_COUNT,
            "global_quantile": values[0],
            "status": {
                status: {
                    "support": status_supports[status],
                    "quantile": quantile,
                }
                for status, quantile in zip(
                    ("certified", "new", "used"),
                    values[1:],
                    strict=True,
                )
            },
        }
        for coverage, values in quantiles.items()
    }


def _smooth_artifact() -> UncertaintyCandidateArtifact:
    return build_candidate_artifact(
        selected_method=SMOOTH_METHOD,
        full_quantiles=_full_quantiles(),
        generated_at=GENERATED_AT,
        comparison_evidence_sha256=COMPARISON_EVIDENCE_SHA256,
    )


def _gamma_artifact(
    *,
    model_path: str = GAMMA_MODEL_PATH,
    model_sha256: str = GAMMA_MODEL_SHA256,
) -> UncertaintyCandidateArtifact:
    return build_candidate_artifact(
        selected_method=GAMMA_METHOD,
        full_quantiles=_full_quantiles(),
        generated_at=GENERATED_AT,
        comparison_evidence_sha256=COMPARISON_EVIDENCE_SHA256,
        gamma_model_path=model_path,
        gamma_model_sha256=model_sha256,
    )


def _payload(artifact: UncertaintyCandidateArtifact) -> dict[str, object]:
    parsed: object = json.loads(canonical_candidate_artifact_json(artifact))
    assert isinstance(parsed, dict)
    return cast(dict[str, object], parsed)


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _canonical_payload(payload: Mapping[str, object]) -> str:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _load(serialized: str | bytes) -> UncertaintyCandidateArtifact:
    payload = serialized.encode("utf-8") if isinstance(serialized, str) else serialized
    return load_candidate_artifact(
        serialized,
        active_model_identity_sha256=active_rf05_identity().identity_sha256,
        expected_artifact_sha256=hashlib.sha256(payload).hexdigest(),
        expected_comparison_evidence_sha256=COMPARISON_EVIDENCE_SHA256,
    )


@pytest.mark.parametrize("method", [SMOOTH_METHOD, GAMMA_METHOD])
def test_candidate_artifact_round_trip_is_canonical_and_deterministic(method: str) -> None:
    artifact = _smooth_artifact() if method == SMOOTH_METHOD else _gamma_artifact()

    first = canonical_candidate_artifact_json(artifact)
    second = canonical_candidate_artifact_json(artifact)
    loaded_from_text = _load(first)
    loaded_from_bytes = _load(first.encode("utf-8"))

    assert first == second
    assert first.endswith("\n")
    assert loaded_from_text == artifact
    assert loaded_from_bytes == artifact
    assert canonical_candidate_artifact_json(loaded_from_text) == first
    assert (
        tuple(item.coverage for item in loaded_from_text.coverage_calibrations) == COVERAGE_LEVELS
    )


def test_serialized_artifact_binds_exact_rf05_identity() -> None:
    serialized = canonical_candidate_artifact_json(_smooth_artifact())

    with pytest.raises(UncertaintyCandidateArtifactError, match="active RF05"):
        load_candidate_artifact(
            serialized,
            active_model_identity_sha256="0" * 64,
            expected_artifact_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            expected_comparison_evidence_sha256=COMPARISON_EVIDENCE_SHA256,
        )

    payload = _payload(_smooth_artifact())
    _mapping(payload["bound_model"])["candidate_id"] = "retail-random_forest-99-99"
    with pytest.raises(UncertaintyCandidateArtifactError, match="active RF05"):
        _load(_canonical_payload(payload))


def test_serialized_artifact_binds_policy_evidence_and_calibration_version() -> None:
    policy_payload = _payload(_smooth_artifact())
    _mapping(policy_payload["frozen_evidence"])["sharpness_policy_sha256"] = "0" * 64
    with pytest.raises(UncertaintyCandidateArtifactError, match="frozen evidence"):
        _load(_canonical_payload(policy_payload))

    version_payload = _payload(_smooth_artifact())
    version_payload["calibration_version"] = CALIBRATION_VERSION + "-drift"
    with pytest.raises(UncertaintyCandidateArtifactError, match="metadata"):
        _load(_canonical_payload(version_payload))

    schema_payload = _payload(_smooth_artifact())
    schema_payload["schema_version"] = 3
    with pytest.raises(UncertaintyCandidateArtifactError, match="metadata"):
        _load(_canonical_payload(schema_payload))


def test_candidate_checksum_is_verified_before_json_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serialized = canonical_candidate_artifact_json(_smooth_artifact())

    def forbidden_json_loads(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("untrusted candidate bytes reached JSON parsing")

    monkeypatch.setattr(
        "autovalue_ml.modeling.uncertainty_candidate_artifact.json.loads",
        forbidden_json_loads,
    )
    with pytest.raises(UncertaintyCandidateArtifactError, match="checksum differs"):
        load_candidate_artifact(
            serialized,
            active_model_identity_sha256=active_rf05_identity().identity_sha256,
            expected_artifact_sha256="0" * 64,
            expected_comparison_evidence_sha256=COMPARISON_EVIDENCE_SHA256,
        )


def test_candidate_artifact_binds_trusted_comparison_evidence() -> None:
    serialized = canonical_candidate_artifact_json(_smooth_artifact())
    artifact_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    with pytest.raises(UncertaintyCandidateArtifactError, match="frozen evidence"):
        load_candidate_artifact(
            serialized,
            active_model_identity_sha256=active_rf05_identity().identity_sha256,
            expected_artifact_sha256=artifact_sha256,
            expected_comparison_evidence_sha256="c" * 64,
        )

    payload = _payload(_smooth_artifact())
    _mapping(payload["frozen_evidence"])["comparison_evidence_sha256"] = "c" * 64
    with pytest.raises(UncertaintyCandidateArtifactError, match="frozen evidence"):
        _load(_canonical_payload(payload))


@pytest.mark.parametrize(
    "field",
    ["expected_artifact_sha256", "expected_comparison_evidence_sha256"],
)
def test_candidate_loader_rejects_invalid_trusted_digest(field: str) -> None:
    serialized = canonical_candidate_artifact_json(_smooth_artifact())
    arguments = {
        "active_model_identity_sha256": active_rf05_identity().identity_sha256,
        "expected_artifact_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "expected_comparison_evidence_sha256": COMPARISON_EVIDENCE_SHA256,
    }
    arguments[field] = "A" * 64

    with pytest.raises(UncertaintyCandidateArtifactError, match="SHA-256"):
        load_candidate_artifact(serialized, **arguments)


def test_rejects_corrupted_duplicate_extra_and_noncanonical_artifacts() -> None:
    serialized = canonical_candidate_artifact_json(_smooth_artifact())

    with pytest.raises(UncertaintyCandidateArtifactError, match="valid JSON"):
        _load("{")
    with pytest.raises(UncertaintyCandidateArtifactError, match="root must be an object"):
        _load("[]")

    duplicate = serialized.replace(
        '{"artifact_type"',
        '{"artifact_type":"duplicate","artifact_type"',
        1,
    )
    with pytest.raises(UncertaintyCandidateArtifactError, match="duplicate field"):
        _load(duplicate)

    extra_payload = _payload(_smooth_artifact())
    extra_payload["unexpected"] = True
    with pytest.raises(UncertaintyCandidateArtifactError, match="root fields"):
        _load(_canonical_payload(extra_payload))

    with pytest.raises(UncertaintyCandidateArtifactError, match="canonical"):
        _load(json.dumps(_payload(_smooth_artifact()), indent=2))


@pytest.mark.parametrize(
    "field",
    ["bound_model", "frozen_evidence", "interval", "scale", "confidence", "publication"],
)
def test_rejects_malformed_nested_containers_with_governed_error(field: str) -> None:
    payload = _payload(_smooth_artifact())
    payload[field] = None

    with pytest.raises(UncertaintyCandidateArtifactError):
        _load(_canonical_payload(payload))


def test_rejects_invalid_encoding_size_and_input_type() -> None:
    with pytest.raises(UncertaintyCandidateArtifactError, match="UTF-8"):
        _load(b"\xff")
    with pytest.raises(UncertaintyCandidateArtifactError, match="maximum size"):
        _load(b" " * 100_001)
    with pytest.raises(UncertaintyCandidateArtifactError, match="text or bytes"):
        load_candidate_artifact(
            cast(str, 123),
            active_model_identity_sha256=active_rf05_identity().identity_sha256,
            expected_artifact_sha256="0" * 64,
            expected_comparison_evidence_sha256=COMPARISON_EVIDENCE_SHA256,
        )


def test_rejects_drifted_interval_confidence_and_publication_contracts() -> None:
    interval_payload = _payload(_smooth_artifact())
    _mapping(interval_payload["interval"])["score"] = "absolute_error"
    with pytest.raises(UncertaintyCandidateArtifactError):
        _load(_canonical_payload(interval_payload))

    confidence_payload = _payload(_smooth_artifact())
    _mapping(confidence_payload["confidence"])["coverage"] = 0.8
    with pytest.raises(UncertaintyCandidateArtifactError, match="confidence policy"):
        _load(_canonical_payload(confidence_payload))

    publication_payload = _payload(_smooth_artifact())
    _mapping(publication_payload["publication"])["legacy_holdout_opened"] = True
    with pytest.raises(UncertaintyCandidateArtifactError, match="publication boundary"):
        _load(_canonical_payload(publication_payload))


def test_smooth_scale_binding_is_exact_and_cannot_bind_model_bytes() -> None:
    binding = ScaleBinding(kind="smooth_formula", version="smooth-value-scale-v1")

    assert binding.to_dict() == {
        "kind": "smooth_formula",
        "version": "smooth-value-scale-v1",
        "formula": "1 + ln(1 + max(RF05 prediction USD, 0) / 10000)",
    }
    with pytest.raises(UncertaintyCandidateArtifactError, match="smooth scale binding"):
        ScaleBinding(kind="smooth_formula", version="smooth-value-scale-v2")
    with pytest.raises(UncertaintyCandidateArtifactError, match="must not bind a model"):
        ScaleBinding(
            kind="smooth_formula",
            version="smooth-value-scale-v1",
            model_sha256=GAMMA_MODEL_SHA256,
        )

    payload = _payload(_smooth_artifact())
    _mapping(payload["scale"])["formula"] = "drifted formula"
    with pytest.raises(UncertaintyCandidateArtifactError, match="smooth scale fields"):
        _load(_canonical_payload(payload))


@pytest.mark.parametrize(
    ("path", "sha256", "message"),
    [
        (None, GAMMA_MODEL_SHA256, "path is required"),
        ("../gamma.joblib", GAMMA_MODEL_SHA256, "fixed safe relative"),
        ("/models/gamma.joblib", GAMMA_MODEL_SHA256, "fixed safe relative"),
        ("models/gamma.pkl", GAMMA_MODEL_SHA256, "fixed safe relative"),
        (r"..\outside.joblib", GAMMA_MODEL_SHA256, "fixed safe relative"),
        (
            r"models\uncertainty\retail-rf05-gamma-residual-scale-v1.joblib",
            GAMMA_MODEL_SHA256,
            "fixed safe relative",
        ),
        (r"C:\outside.joblib", GAMMA_MODEL_SHA256, "fixed safe relative"),
        (r"\\server\share\outside.joblib", GAMMA_MODEL_SHA256, "fixed safe relative"),
        (
            "models/uncertainty/retail-rf05-gamma-residual-scale-v1.joblib:payload.joblib",
            GAMMA_MODEL_SHA256,
            "fixed safe relative",
        ),
        (
            "models/uncertainty/other-gamma-scale.joblib",
            GAMMA_MODEL_SHA256,
            "fixed safe relative",
        ),
        (GAMMA_MODEL_PATH, "A" * 64, "SHA-256"),
        (GAMMA_MODEL_PATH, "a" * 63, "SHA-256"),
    ],
)
def test_gamma_scale_binding_rejects_unsafe_path_or_invalid_checksum(
    path: str | None,
    sha256: str,
    message: str,
) -> None:
    with pytest.raises(UncertaintyCandidateArtifactError, match=message):
        ScaleBinding(
            kind="gamma_joblib",
            version=SCALE_VERSION,
            model_path=path,
            model_sha256=sha256,
        )


def test_gamma_binding_rejects_version_floor_cap_and_method_drift() -> None:
    with pytest.raises(UncertaintyCandidateArtifactError, match="binding is invalid"):
        ScaleBinding(
            kind="gamma_joblib",
            version=SCALE_VERSION + "-drift",
            model_path=GAMMA_MODEL_PATH,
            model_sha256=GAMMA_MODEL_SHA256,
        )

    floor_payload = _payload(_gamma_artifact())
    _mapping(floor_payload["scale"])["scale_floor_usd"] = GAMMA_SCALE_FLOOR_USD + 1.0
    with pytest.raises(UncertaintyCandidateArtifactError, match="scale fields"):
        _load(_canonical_payload(floor_payload))

    cap_payload = _payload(_gamma_artifact())
    _mapping(cap_payload["scale"])["scale_cap_usd"] = GAMMA_SCALE_CAP_USD - 1.0
    with pytest.raises(UncertaintyCandidateArtifactError, match="scale fields"):
        _load(_canonical_payload(cap_payload))

    method_payload = _payload(_gamma_artifact())
    method_payload["selected_method"] = SMOOTH_METHOD
    with pytest.raises(UncertaintyCandidateArtifactError, match="smooth scale fields"):
        _load(_canonical_payload(method_payload))


def test_artifact_constructor_rejects_method_binding_mismatch() -> None:
    smooth = _smooth_artifact()
    with pytest.raises(UncertaintyCandidateArtifactError, match="does not match"):
        UncertaintyCandidateArtifact(
            generated_at=GENERATED_AT,
            comparison_evidence_sha256=COMPARISON_EVIDENCE_SHA256,
            selected_method=cast(CandidateMethod, GAMMA_METHOD),
            coverage_calibrations=smooth.coverage_calibrations,
            scale_binding=smooth.scale_binding,
        )


def test_build_gamma_artifact_requires_complete_model_binding() -> None:
    with pytest.raises(UncertaintyCandidateArtifactError, match="path is required"):
        build_candidate_artifact(
            selected_method=GAMMA_METHOD,
            full_quantiles=_full_quantiles(),
            generated_at=GENERATED_AT,
            comparison_evidence_sha256=COMPARISON_EVIDENCE_SHA256,
        )


def test_smooth_intervals_are_ordered_nested_zero_clipped_and_deterministic() -> None:
    artifact = _smooth_artifact()
    results = [
        candidate_interval(
            point_prediction=100.0,
            vehicle_status=" USED ",
            coverage=coverage,
            artifact=artifact,
        )
        for coverage in COVERAGE_LEVELS
    ]

    assert [item.interval_width for item in results] == sorted(
        item.interval_width for item in results
    )
    assert len({item.interval_width for item in results}) == len(COVERAGE_LEVELS)
    assert all(item.interval_lower == 0.0 for item in results)
    assert all(item.predicted_value <= item.interval_upper for item in results)
    assert all(item.interval_width == item.interval_upper for item in results)
    assert results[1].calibration_support == 6_658
    assert results[1].calibration_method == SMOOTH_METHOD
    assert results[1] == candidate_interval(
        point_prediction=100.0,
        vehicle_status="used",
        coverage=0.9,
        artifact=artifact,
    )


@pytest.mark.parametrize("status", ["", "unknown", "salvage"])
def test_unknown_status_uses_global_quantile_and_support_fallback(status: str) -> None:
    artifact = _smooth_artifact()

    result = candidate_interval(
        point_prediction=1_000.0,
        vehicle_status=status,
        coverage=0.9,
        artifact=artifact,
    )
    scale = 1.0 + np.log1p(1_000.0 / 10_000.0)
    expected_radius = 220.0 * scale

    assert result.interval_lower == pytest.approx(1_000.0 - expected_radius)
    assert result.interval_upper == pytest.approx(1_000.0 + expected_radius)
    assert result.calibration_support == CALIBRATION_SAMPLE_COUNT
    assert result.calibration_method == "global_fallback"


@pytest.mark.parametrize("prediction", [-1.0, float("nan"), float("inf"), True])
def test_interval_rejects_invalid_point_prediction(prediction: object) -> None:
    with pytest.raises(UncertaintyCandidateArtifactError, match="point prediction"):
        candidate_interval(
            point_prediction=cast(float, prediction),
            vehicle_status="used",
            coverage=0.9,
            artifact=_smooth_artifact(),
        )


def test_interval_rejects_unsupported_coverage() -> None:
    with pytest.raises(UncertaintyCandidateArtifactError, match="coverage is unsupported"):
        candidate_interval(
            point_prediction=20_000.0,
            vehicle_status="used",
            coverage=0.85,
            artifact=_smooth_artifact(),
        )


class _ConstantGammaEstimator:
    def __init__(self, prediction: float) -> None:
        self.prediction = prediction

    def fit(
        self,
        features: pd.DataFrame,
        target: NDArray[np.float64],
    ) -> _ConstantGammaEstimator:
        return self

    def predict(self, features: pd.DataFrame) -> object:
        return np.full(len(features), self.prediction, dtype=np.float64)


def _vehicle_frame(rows: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2020] * rows,
            "make": ["Ford"] * rows,
            "model": ["F-150"] * rows,
            "vehicle_status": ["used"] * rows,
            "mileage": [50_000.0] * rows,
        }
    )


def _write_gamma_model(
    project_root: Path,
    estimator: ScaleEstimator,
    *,
    model_path: str = GAMMA_MODEL_PATH,
) -> tuple[Path, str]:
    path = project_root / model_path
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, sha256


def test_gamma_interval_requires_checksum_bound_loader_and_is_deterministic(tmp_path: Path) -> None:
    _, sha256 = _write_gamma_model(tmp_path, _ConstantGammaEstimator(100.0))
    artifact = _gamma_artifact(model_sha256=sha256)
    bound_model = load_bound_gamma_scale_model(artifact=artifact, project_root=tmp_path)

    first = candidate_interval(
        point_prediction=300_000.0,
        vehicle_status="used",
        coverage=0.9,
        artifact=artifact,
        vehicle_features=_vehicle_frame(),
        gamma_model=bound_model,
    )
    second = candidate_interval(
        point_prediction=300_000.0,
        vehicle_status="used",
        coverage=0.9,
        artifact=artifact,
        vehicle_features=_vehicle_frame(),
        gamma_model=bound_model,
    )

    assert first == second
    assert first.interval_lower == 200_000.0
    assert first.interval_upper == 400_000.0
    assert first.interval_width == 200_000.0
    assert first.calibration_method == GAMMA_METHOD

    raw_model = cast(BoundGammaScaleModel, _ConstantGammaEstimator(100.0))
    with pytest.raises(UncertaintyCandidateArtifactError, match="checksum-bound"):
        candidate_interval(
            point_prediction=300_000.0,
            vehicle_status="used",
            coverage=0.9,
            artifact=artifact,
            vehicle_features=_vehicle_frame(),
            gamma_model=raw_model,
        )


def test_bound_gamma_loader_rejects_checksum_drift(tmp_path: Path) -> None:
    model_path, sha256 = _write_gamma_model(tmp_path, _ConstantGammaEstimator(1_000.0))
    artifact = _gamma_artifact(model_sha256=sha256)
    model_path.write_bytes(model_path.read_bytes() + b"drift")

    with pytest.raises(UncertaintyCandidateArtifactError, match="checksum"):
        load_bound_gamma_scale_model(artifact=artifact, project_root=tmp_path)


def test_model_checksum_failure_stops_before_joblib_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path, sha256 = _write_gamma_model(tmp_path, _ConstantGammaEstimator(1_000.0))
    artifact = _gamma_artifact(model_sha256=sha256)
    model_path.write_bytes(model_path.read_bytes() + b"drift")
    loaded = False

    def forbidden_joblib_load(*args: object, **kwargs: object) -> object:
        nonlocal loaded
        del args, kwargs
        loaded = True
        raise AssertionError("untrusted model bytes reached joblib.load")

    monkeypatch.setattr(
        "autovalue_ml.modeling.uncertainty_candidate_artifact.joblib.load",
        forbidden_joblib_load,
    )
    with pytest.raises(UncertaintyCandidateArtifactError, match="checksum"):
        load_bound_gamma_scale_model(artifact=artifact, project_root=tmp_path)
    assert loaded is False


def test_bound_gamma_loader_rejects_wrong_method_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(UncertaintyCandidateArtifactError, match="Gamma"):
        load_bound_gamma_scale_model(artifact=_smooth_artifact(), project_root=tmp_path)

    with pytest.raises(UncertaintyCandidateArtifactError, match="could not be read"):
        load_bound_gamma_scale_model(artifact=_gamma_artifact(), project_root=tmp_path)


def test_gamma_interval_rejects_wrapper_bound_to_another_artifact(tmp_path: Path) -> None:
    _, first_sha256 = _write_gamma_model(tmp_path, _ConstantGammaEstimator(1_000.0))
    first_artifact = _gamma_artifact(model_sha256=first_sha256)
    _, other_sha256 = _write_gamma_model(
        tmp_path,
        _ConstantGammaEstimator(2_000.0),
    )
    other_artifact = _gamma_artifact(model_sha256=other_sha256)
    other_bound_model = load_bound_gamma_scale_model(
        artifact=other_artifact,
        project_root=tmp_path,
    )

    with pytest.raises(UncertaintyCandidateArtifactError, match="binding differs"):
        candidate_interval(
            point_prediction=300_000.0,
            vehicle_status="used",
            coverage=0.9,
            artifact=first_artifact,
            vehicle_features=_vehicle_frame(),
            gamma_model=other_bound_model,
        )


def test_gamma_interval_rejects_missing_or_wrong_shaped_serving_inputs(tmp_path: Path) -> None:
    _, sha256 = _write_gamma_model(tmp_path, _ConstantGammaEstimator(1_000.0))
    artifact = _gamma_artifact(model_sha256=sha256)
    bound_model = load_bound_gamma_scale_model(artifact=artifact, project_root=tmp_path)

    with pytest.raises(UncertaintyCandidateArtifactError, match="checksum-bound"):
        candidate_interval(
            point_prediction=20_000.0,
            vehicle_status="used",
            coverage=0.9,
            artifact=artifact,
        )
    with pytest.raises(UncertaintyCandidateArtifactError, match="exactly one vehicle feature row"):
        candidate_interval(
            point_prediction=20_000.0,
            vehicle_status="used",
            coverage=0.9,
            artifact=artifact,
            gamma_model=bound_model,
        )
    with pytest.raises(UncertaintyCandidateArtifactError, match="exactly one vehicle feature row"):
        candidate_interval(
            point_prediction=20_000.0,
            vehicle_status="used",
            coverage=0.9,
            artifact=artifact,
            vehicle_features=_vehicle_frame(rows=2),
            gamma_model=bound_model,
        )
