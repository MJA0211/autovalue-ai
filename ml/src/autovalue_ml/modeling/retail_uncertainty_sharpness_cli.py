"""Run the preregistered RF05 uncertainty-sharpness comparison exactly once."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

import joblib
import numpy as np

from autovalue_ml.acquisition.sources.kaggle_us_sales_cars import KaggleUSSalesCarsError
from autovalue_ml.acquisition.sources.kaggle_us_sales_cars_split import (
    KaggleUSSalesCarsSplitError,
    prepare_kaggle_us_sales_cars_split_training_rows,
)

from .baseline_cli import (
    BaselineCLIError,
    RetailTrainingRow,
    _collect_retail_partition,
    _expected_count,
    _project_path,
    _validate_output_path,
    _validate_project_root,
    _write_atomic,
)
from .calibration import retail_calibration_partition
from .calibration_artifact import (
    CALIBRATION_ASSIGNMENT_SHA256,
    CALIBRATION_POLICY_SHA256,
    PHASE4_PROTOCOL_SHA256,
    PHASE4_RETAIL_CONFIRMATION_SHA256,
    active_rf05_identity,
)
from .metrics import regression_metrics
from .phase4_confirmation import Phase4ConfirmationError, parse_phase4_confirmation_json
from .phase4_screening_cli import _RETAIL_PATHS
from .phase4_screening_experiment import _partition_hash
from .retail_calibration_cli import (
    CalibrationCLIError,
    _validate_markdown_output,
    _verified_bytes,
)
from .retail_calibration_experiment import (
    CALIBRATION_SEED,
    CalibrationExperimentError,
    fit_frozen_rf05_calibration_predictions,
)
from .retail_uncertainty_diagnostics import (
    ResidualDiagnosticsError,
    reconstruct_rf05_development_oof,
)
from .retail_uncertainty_sharpness import (
    BASELINE_METHOD,
    CALIBRATION_V1_ARTIFACT_SHA256,
    CALIBRATION_V1_REPORT_SHA256,
    DEVELOPMENT_DIAGNOSTICS_SHA256,
    GAMMA_METHOD,
    SHARPNESS_POLICY_SHA256,
    SMOOTH_METHOD,
    UncertaintySharpnessError,
    canonical_sharpness_report_json,
    compare_uncertainty_methods,
)
from .uncertainty_candidate_artifact import (
    GAMMA_MODEL_RELATIVE_PATH,
    UncertaintyCandidateArtifactError,
    build_candidate_artifact,
    canonical_candidate_artifact_json,
    load_candidate_artifact,
)
from .uncertainty_sharpness_policy import (
    UncertaintySharpnessPolicyError,
    load_uncertainty_sharpness_policy_file,
)

_PROTOCOL: Final = PurePosixPath("docs/experiments/phase4-model-selection-v1.json")
_CONFIRMATION: Final = PurePosixPath("docs/experiments/phase4-retail-full-development-v1.json")
_CALIBRATION_POLICY: Final = PurePosixPath(
    "docs/experiments/retail-rf05-calibration-policy-v1.json"
)
_CALIBRATION_ARTIFACT: Final = PurePosixPath(
    "docs/experiments/retail-rf05-calibration-v1.artifact.json"
)
_CALIBRATION_REPORT: Final = PurePosixPath(
    "docs/experiments/retail-rf05-calibration-v1.report.json"
)
_DIAGNOSTICS: Final = PurePosixPath(
    "docs/experiments/retail-rf05-development-residual-diagnostics-v1.json"
)
_POLICY: Final = PurePosixPath("docs/experiments/retail-rf05-uncertainty-sharpness-policy-v1.json")
_REPORT: Final = PurePosixPath("docs/experiments/retail-rf05-uncertainty-sharpness-v1.report.json")
_MARKDOWN: Final = PurePosixPath("docs/experiments/retail-rf05-uncertainty-sharpness-v1.md")
_CANDIDATE_ARTIFACT: Final = PurePosixPath(
    "docs/experiments/retail-rf05-uncertainty-sharpness-v2.artifact.json"
)
_GAMMA_MODEL: Final = PurePosixPath(GAMMA_MODEL_RELATIVE_PATH)
_RF05_IDENTITY_SHA256: Final = "3bbd73d6442387496b05253dd20bc749db24aa482d56fa6ba73ec2702de8b513"


class SharpnessCLIError(RuntimeError):
    """A protected input, output, or report boundary is invalid."""


@dataclass(frozen=True, slots=True)
class _CreatedOutput:
    path: Path
    sha256: str
    identity: tuple[int, int, int, int]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m autovalue_ml.modeling.retail_uncertainty_sharpness_cli",
        description="Compare the three frozen RF05 conformal sharpness methods.",
    )
    parser.add_argument("--project-root", required=True, metavar="PATH")
    arguments = parser.parse_args(argv)
    try:
        project_root = _validate_project_root(Path(cast(str, arguments.project_root)))
        outputs = _validated_output_paths(project_root)
        frozen = _load_frozen_evidence(project_root)
        policy = load_uncertainty_sharpness_policy_file(_project_path(project_root, _POLICY))
        if policy.policy_sha256 != SHARPNESS_POLICY_SHA256:
            raise SharpnessCLIError("loaded sharpness policy identity differs")

        print("Loading the authorized retail training partition only", flush=True)
        source_paths = tuple(_project_path(project_root, relative) for relative in _RETAIL_PATHS)
        stream = prepare_kaggle_us_sales_cars_split_training_rows(
            *source_paths,
            partition="train",
        )
        phase3_train = _collect_retail_partition(
            cast(Iterable[RetailTrainingRow], stream),
            expected_rows=_expected_count(stream, "expected_rows"),
            label="retail train",
        )
        partition = retail_calibration_partition(
            phase3_train.features,
            seed=CALIBRATION_SEED,
        )
        assignment_hash = _partition_hash(
            partition.calibration_indices,
            population_count=len(phase3_train.features),
            selected_label="calibration",
            unselected_label="development",
        )
        if assignment_hash != CALIBRATION_ASSIGNMENT_SHA256:
            raise SharpnessCLIError("calibration boundary differs from frozen evidence")

        development_indices = partition.development_indices
        calibration_indices = partition.calibration_indices
        development_features = phase3_train.features.iloc[development_indices].reset_index(
            drop=True
        )
        development_target = phase3_train.target[development_indices]
        calibration_features = phase3_train.features.iloc[calibration_indices].reset_index(
            drop=True
        )
        calibration_target = phase3_train.target[calibration_indices]

        print("Reconstructing leakage-safe development RF05 OOF predictions", flush=True)
        development_predictions, _ = reconstruct_rf05_development_oof(
            development_features=development_features,
            development_target=development_target,
            progress=_print_oof_progress,
        )
        _validate_development_reconstruction(
            development_target,
            development_predictions,
            cast(Mapping[str, object], frozen["diagnostics"]),
        )

        print(
            "Fitting the frozen RF05 definition on development for calibration scoring", flush=True
        )
        calibration_predictions = fit_frozen_rf05_calibration_predictions(
            development_features=development_features,
            development_target=development_target,
            calibration_features=calibration_features,
        )
        print("Evaluating the three preregistered interval methods", flush=True)
        result = compare_uncertainty_methods(
            policy=policy,
            development_features=development_features,
            development_target=development_target,
            development_oof_predictions=development_predictions,
            calibration_features=calibration_features,
            calibration_target=calibration_target,
            calibration_predictions=calibration_predictions,
            calibration_v1_report=cast(Mapping[str, object], frozen["calibration_report"]),
        )
        comparison_evidence_json = canonical_sharpness_report_json(result.report)
        comparison_evidence_sha256 = hashlib.sha256(
            comparison_evidence_json.encode("utf-8")
        ).hexdigest()

        model_payload = (
            _gamma_model_payload(result.gamma_scale_model)
            if (result.selected_method == GAMMA_METHOD)
            else None
        )
        model_sha256 = hashlib.sha256(model_payload).hexdigest() if model_payload else None
        candidate_json: str | None = None
        candidate_sha256: str | None = None
        if result.selected_method != BASELINE_METHOD:
            candidate = build_candidate_artifact(
                selected_method=result.selected_method,
                full_quantiles=result.full_quantiles,
                generated_at=cast(str, result.report["generated_at"]),
                comparison_evidence_sha256=comparison_evidence_sha256,
                gamma_model_path=_GAMMA_MODEL.as_posix() if model_payload else None,
                gamma_model_sha256=model_sha256,
            )
            candidate_json = canonical_candidate_artifact_json(candidate)
            candidate_sha256 = hashlib.sha256(candidate_json.encode("utf-8")).hexdigest()
            load_candidate_artifact(
                candidate_json,
                active_model_identity_sha256=_RF05_IDENTITY_SHA256,
                expected_artifact_sha256=candidate_sha256,
                expected_comparison_evidence_sha256=comparison_evidence_sha256,
            )

        report = {
            **result.report,
            "development_diagnostic_summary": _development_diagnostic_summary(
                cast(Mapping[str, object], frozen["diagnostics"])
            ),
            "artifacts": {
                "comparison_evidence_sha256": comparison_evidence_sha256,
                "comparison_report": _REPORT.as_posix(),
                "human_readable_report": _MARKDOWN.as_posix(),
                "candidate_serving_artifact": (
                    _CANDIDATE_ARTIFACT.as_posix() if candidate_json else None
                ),
                "candidate_serving_artifact_sha256": candidate_sha256,
                "residual_scale_model": _GAMMA_MODEL.as_posix() if model_payload else None,
                "residual_scale_model_sha256": model_sha256,
                "new_serving_state_persisted": candidate_json is not None,
                "original_calibration_v1_modified": False,
            },
        }
        report_json = canonical_sharpness_report_json(report)
        report_sha256 = hashlib.sha256(report_json.encode("utf-8")).hexdigest()
        markdown = render_sharpness_markdown(
            report,
            report_sha256=report_sha256,
            candidate_sha256=candidate_sha256,
            model_sha256=model_sha256,
        )

        _publish_outputs(
            outputs,
            report_json=report_json,
            markdown=markdown,
            candidate_json=candidate_json,
            model_payload=model_payload,
        )
    except (
        BaselineCLIError,
        CalibrationCLIError,
        CalibrationExperimentError,
        KaggleUSSalesCarsError,
        KaggleUSSalesCarsSplitError,
        OSError,
        Phase4ConfirmationError,
        ResidualDiagnosticsError,
        SharpnessCLIError,
        UncertaintyCandidateArtifactError,
        UncertaintySharpnessError,
        UncertaintySharpnessPolicyError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(
        "retail RF05 uncertainty-sharpness comparison complete | "
        f"classification={report['classification']} | "
        f"selected={cast(Mapping[str, object], report['decision'])['selected_method']} | "
        f"report_sha256={report_sha256}",
        flush=True,
    )
    return 0


def _load_frozen_evidence(project_root: Path) -> dict[str, object]:
    _verified_bytes(
        _project_path(project_root, _PROTOCOL),
        expected=PHASE4_PROTOCOL_SHA256,
        label="Phase 4 protocol",
    )
    confirmation_bytes = _verified_bytes(
        _project_path(project_root, _CONFIRMATION),
        expected=PHASE4_RETAIL_CONFIRMATION_SHA256,
        label="Phase 4 retail confirmation",
    )
    confirmation = parse_phase4_confirmation_json(confirmation_bytes)
    _verified_bytes(
        _project_path(project_root, _CALIBRATION_POLICY),
        expected=CALIBRATION_POLICY_SHA256,
        label="calibration v1 policy",
    )
    _verified_bytes(
        _project_path(project_root, _CALIBRATION_ARTIFACT),
        expected=CALIBRATION_V1_ARTIFACT_SHA256,
        label="calibration v1 serving artifact",
    )
    calibration_report_bytes = _verified_bytes(
        _project_path(project_root, _CALIBRATION_REPORT),
        expected=CALIBRATION_V1_REPORT_SHA256,
        label="calibration v1 report",
    )
    diagnostics_bytes = _verified_bytes(
        _project_path(project_root, _DIAGNOSTICS),
        expected=DEVELOPMENT_DIAGNOSTICS_SHA256,
        label="development residual diagnostics",
    )
    if active_rf05_identity().identity_sha256 != _RF05_IDENTITY_SHA256:
        raise SharpnessCLIError("active RF05 logical identity differs from frozen evidence")
    return {
        "confirmation": confirmation,
        "calibration_report": _json_mapping(calibration_report_bytes, "calibration v1 report"),
        "diagnostics": _json_mapping(diagnostics_bytes, "development diagnostics"),
    }


def _json_mapping(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SharpnessCLIError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise SharpnessCLIError(f"{label} root must be a string-keyed object")
    return cast(Mapping[str, object], value)


def _validate_development_reconstruction(
    target: np.ndarray[tuple[int], np.dtype[np.float64]],
    predictions: np.ndarray[tuple[int], np.dtype[np.float64]],
    diagnostics: Mapping[str, object],
) -> None:
    if diagnostics.get("report_type") != "retail_rf05_development_residual_diagnostics":
        raise SharpnessCLIError("development diagnostic report type differs")
    raw_expected = diagnostics.get("point_prediction_metrics")
    if not isinstance(raw_expected, Mapping):
        raise SharpnessCLIError("development diagnostic point metrics are invalid")
    expected = cast(Mapping[str, object], raw_expected)
    if expected.get("sample_count") != len(target):
        raise SharpnessCLIError("development diagnostic row count differs")
    observed = regression_metrics(target, predictions).to_dict()
    for field in ("mae", "rmse", "r2"):
        expected_value = expected.get(field)
        if (
            isinstance(expected_value, bool)
            or not isinstance(expected_value, (int, float))
            or not np.isfinite(float(expected_value))
        ):
            raise SharpnessCLIError(f"development diagnostic {field} is invalid")
        if not np.isclose(
            cast(float, observed[field]),
            float(expected_value),
            rtol=0.0,
            atol=1e-8,
        ):
            raise SharpnessCLIError(f"development OOF {field} differs from frozen diagnostics")


def _development_diagnostic_summary(
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    overall = diagnostics.get("overall_residual_distribution")
    relationship = diagnostics.get("predicted_value_relationship")
    if not isinstance(overall, Mapping) or not isinstance(relationship, Mapping):
        raise SharpnessCLIError("development diagnostic summary is invalid")
    required_overall = {
        "support",
        "mean_absolute_residual_usd",
        "median_absolute_residual_usd",
        "residual_variance_usd2",
        "absolute_residual_quantiles_usd",
        "residual_to_actual_price_ratio",
    }
    required_relationship = {
        "log_prediction_log_residual_pearson",
        "prediction_residual_spearman",
        "mean_absolute_residual_usd_by_predicted_value_quartile",
        "highest_to_lowest_quartile_mean_residual_ratio",
    }
    if set(overall) != required_overall or set(relationship) != required_relationship:
        raise SharpnessCLIError("development diagnostic summary fields differ")
    return {
        "classification": diagnostics.get("classification"),
        "overall_residual_distribution": dict(overall),
        "predicted_value_relationship": dict(relationship),
        "actual_price_used_for_evaluation_only": True,
    }


def _validated_output_paths(project_root: Path) -> dict[str, Path]:
    paths = {
        "report": _validate_output_path(
            _project_path(project_root, _REPORT), project_root=project_root, force=False
        ),
        "markdown": _validate_markdown_output(
            _project_path(project_root, _MARKDOWN), project_root=project_root
        ),
        "candidate": _validate_output_path(
            _project_path(project_root, _CANDIDATE_ARTIFACT),
            project_root=project_root,
            force=False,
        ),
        "model": _validate_binary_output(
            _project_path(project_root, _GAMMA_MODEL), project_root=project_root
        ),
    }
    inputs = tuple(
        _project_path(project_root, relative)
        for relative in (
            _PROTOCOL,
            _CONFIRMATION,
            _CALIBRATION_POLICY,
            _CALIBRATION_ARTIFACT,
            _CALIBRATION_REPORT,
            _DIAGNOSTICS,
            _POLICY,
        )
    )
    _ensure_paths_distinct(tuple(paths.values()), inputs)
    return paths


def _validate_binary_output(path: Path, *, project_root: Path) -> Path:
    resolved = Path(os.path.abspath(os.fspath(path)))
    expected_parent = project_root / "models" / "uncertainty"
    if resolved.parent != expected_parent or resolved.suffix != ".joblib":
        raise SharpnessCLIError("scale-model output must use the fixed models path")
    _reject_symlink_components(resolved)
    try:
        mode = resolved.lstat().st_mode
    except FileNotFoundError:
        return resolved
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise SharpnessCLIError("scale-model output must be a regular non-symlink file")
    raise SharpnessCLIError("scale-model output already exists")


def _ensure_paths_distinct(outputs: tuple[Path, ...], inputs: tuple[Path, ...]) -> None:
    all_paths = outputs + inputs
    keys = [os.path.normcase(os.path.normpath(os.fspath(path))) for path in all_paths]
    if len(keys) != len(set(keys)):
        raise SharpnessCLIError("experiment paths must be distinct")
    existing = [path for path in all_paths if path.exists()]
    for index, first in enumerate(existing):
        for second in existing[index + 1 :]:
            try:
                if os.path.samefile(first, second):
                    raise SharpnessCLIError("experiment paths must not alias")
            except OSError as error:
                raise SharpnessCLIError("experiment path identity could not be verified") from error


def _gamma_model_payload(model: object) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    payload = buffer.getvalue()
    if not payload:
        raise SharpnessCLIError("Gamma scale model serialization is empty")
    return payload


def _publish_outputs(
    outputs: Mapping[str, Path],
    *,
    report_json: str,
    markdown: str,
    candidate_json: str | None,
    model_payload: bytes | None,
) -> None:
    publications: list[tuple[Path, str | bytes]] = []
    if model_payload is not None:
        publications.append((outputs["model"], model_payload))
    if candidate_json is not None:
        publications.append((outputs["candidate"], candidate_json))
    publications.extend(
        (
            (outputs["markdown"], markdown),
            (outputs["report"], report_json),
        )
    )
    created: list[_CreatedOutput] = []
    try:
        for path, payload in publications:
            expected_sha256 = _publication_sha256(payload)
            if isinstance(payload, bytes):
                _write_atomic_bytes(path, payload)
            else:
                _write_atomic(path, payload, force=False)
            output = _created_output(path, expected_sha256=expected_sha256)
            created.append(output)
            _validate_created_output(output)
    except Exception as error:
        refused = _rollback_created_outputs(created)
        if refused:
            joined = ", ".join(os.fspath(path) for path in refused)
            raise SharpnessCLIError(
                f"publication failed and safe rollback refused changed outputs: {joined}"
            ) from error
        raise


def _publication_sha256(payload: str | bytes) -> str:
    encoded = payload.encode("utf-8", errors="strict") if isinstance(payload, str) else payload
    return hashlib.sha256(encoded).hexdigest()


def _created_output(path: Path, *, expected_sha256: str) -> _CreatedOutput:
    current = path.lstat()
    if not stat.S_ISREG(current.st_mode) or stat.S_ISLNK(current.st_mode):
        raise SharpnessCLIError("published output must be a regular non-symlink file")
    return _CreatedOutput(
        path=path,
        sha256=expected_sha256,
        identity=_file_identity(current),
    )


def _validate_created_output(output: _CreatedOutput) -> None:
    payload = output.path.read_bytes()
    after = output.path.lstat()
    if _file_identity(after) != output.identity:
        raise SharpnessCLIError("published output changed while it was verified")
    if hashlib.sha256(payload).hexdigest() != output.sha256:
        raise SharpnessCLIError("published output checksum differs from generated payload")


def _rollback_created_outputs(created: Sequence[_CreatedOutput]) -> tuple[Path, ...]:
    refused: list[Path] = []
    for output in reversed(created):
        try:
            before = output.path.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or _file_identity(before) != output.identity
            ):
                refused.append(output.path)
                continue
            payload = output.path.read_bytes()
            after = output.path.lstat()
            if (
                _file_identity(after) != output.identity
                or hashlib.sha256(payload).hexdigest() != output.sha256
            ):
                refused.append(output.path)
                continue
            output.path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            refused.append(output.path)
    return tuple(reversed(refused))


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _write_atomic_bytes(path: Path, payload: bytes) -> None:
    if not payload:
        raise SharpnessCLIError("binary model payload is empty")
    _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    committed = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise SharpnessCLIError("scale-model output was created concurrently") from error
        temporary.unlink()
        committed = True
    finally:
        if not committed and temporary.exists():
            temporary.unlink()


def _reject_symlink_components(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise SharpnessCLIError("output path must not contain symlinks")
        if current == current.parent:
            return
        current = current.parent


def _print_oof_progress(fold_number: int, fold_count: int) -> None:
    print(f"RF05 development OOF reconstruction | fold={fold_number}/{fold_count}", flush=True)


def render_sharpness_markdown(
    report: Mapping[str, object],
    *,
    report_sha256: str,
    candidate_sha256: str | None,
    model_sha256: str | None,
) -> str:
    decision = cast(Mapping[str, object], report["decision"])
    methods = cast(Mapping[str, object], report["methods"])
    gates = cast(Mapping[str, object], report["acceptance_gates"])
    boundaries = cast(Mapping[str, object], report["data_boundaries"])
    selected = cast(str, decision["selected_method"])
    labels = {
        BASELINE_METHOD: "Current status-conditional baseline",
        GAMMA_METHOD: "Normalized Gamma residual scale",
        SMOOTH_METHOD: "Simple smooth predicted-value scale",
    }
    lines = [
        "# Retail RF05 uncertainty sharpness experiment",
        "",
        "## Decision",
        "",
        f"**Classification: `{report['classification']}`.** Selected method: `{selected}`.",
        "This is uncertainty calibration around the unchanged Phase 4 RF05 point estimator; ",
        "it does not promote a production-final system or authorize opening the legacy holdout.",
        "",
        "## Protected design",
        "",
        f"- Development OOF residual rows: {boundaries['development_rows']:,}",
        f"- Protected calibration rows: {boundaries['calibration_rows']:,}",
        "- Gamma residual-scale fit used calibration targets: no",
        "- RF05 retuned or replaced: no",
        "- This experiment requested, loaded, or evaluated legacy-holdout rows: no",
        "- Yoad, River, AutoTrader, and Carson-Shively were outside this experiment.",
        "- Raw rows, row-level predictions, or residuals persisted: no",
        "- Displayed lower bounds are explicitly clipped at $0; coverage equivalence is audited.",
        "",
        "## Why heteroscedastic methods were evaluated",
        "",
    ]
    diagnostic = cast(Mapping[str, object], report["development_diagnostic_summary"])
    residuals = cast(Mapping[str, object], diagnostic["overall_residual_distribution"])
    relationship = cast(Mapping[str, object], diagnostic["predicted_value_relationship"])
    quartiles = cast(
        Mapping[str, object],
        relationship["mean_absolute_residual_usd_by_predicted_value_quartile"],
    )
    lines.extend(
        [
            (
                f"Development OOF absolute residuals have a median of "
                f"${cast(float, residuals['median_absolute_residual_usd']):,.2f} and a mean of "
                f"${cast(float, residuals['mean_absolute_residual_usd']):,.2f}."
            ),
            (
                "Mean error rises from "
                f"${cast(float, quartiles['predicted_value_1']):,.2f} in the lowest predicted-"
                f"value quartile to ${cast(float, quartiles['predicted_value_4']):,.2f} in the "
                "highest, a "
                f"{cast(float, relationship['highest_to_lowest_quartile_mean_residual_ratio']):.2f}"
                "x "
                "ratio. This is the preregistered basis for testing scaled conformal scores."
            ),
            "",
            "Actual price was used only to evaluate development residual behavior, never as a "
            "residual-scale predictor.",
            "",
            "## Direct comparison",
            "",
            "All widths are USD. Sharpness gates use unclipped symmetric mean width; displayed ",
            "widths reflect the physical $0 lower bound.",
            "",
            "| Method | Nominal | Coverage | Gap | Mean width | Median | p75 | p90 | p95 | "
            "Fallback |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method_id in (BASELINE_METHOD, GAMMA_METHOD, SMOOTH_METHOD):
        method = cast(Mapping[str, object], methods[method_id])
        coverages = cast(Mapping[str, object], method["coverages"])
        for coverage in ("0.8", "0.9", "0.95"):
            item = cast(Mapping[str, object], coverages[coverage])
            width = cast(Mapping[str, object], item["displayed_width_usd"])
            lines.append(
                f"| {labels[method_id]} | {float(coverage):.0%} | "
                f"{cast(float, item['empirical_coverage']):.2%} | "
                f"{cast(float, item['coverage_gap']):+.2%} | "
                f"${cast(float, width['mean']):,.2f} | "
                f"${cast(float, width['median']):,.2f} | "
                f"${cast(float, width['p75']):,.2f} | "
                f"${cast(float, width['p90']):,.2f} | "
                f"${cast(float, width['p95']):,.2f} | "
                f"{cast(float, item['fallback_rate']):.2%} |"
            )
    lines.extend(
        [
            "",
            "## Preregistered gate result",
            "",
            "| Candidate | Passed every gate | Failed gates |",
            "|---|---:|---:|",
        ]
    )
    failed_gate_details: list[str] = []
    for method_id in (GAMMA_METHOD, SMOOTH_METHOD):
        gate = cast(Mapping[str, object], gates[method_id])
        lines.append(
            f"| {labels[method_id]} | {'yes' if gate['passed_all'] else 'no'} | "
            f"{gate['failed_gate_count']} |"
        )
        failed = [
            cast(str, item["gate"])
            for item in cast(list[Mapping[str, object]], gate["outcomes"])
            if not cast(bool, item["passed"])
        ]
        if failed:
            failed_gate_details.append(f"- `{method_id}`: {', '.join(failed)}")
    lines.extend(
        [
            "",
            "Failed gate names (full observed values and thresholds are in the JSON report):",
            "",
            *(failed_gate_details or ["- None"]),
            "",
            "## Focus-slice 90% coverage",
            "",
            "These prespecified diagnostics were not individually tuned.",
            "",
            "| Slice | Baseline | Gamma scale | Smooth scale |",
            "|---|---:|---:|---:|",
        ]
    )
    focus = (
        ("Highest price band", "actual_price_band", "price_4"),
        ("GMC", "manufacturer", "gmc"),
        ("Genesis", "manufacturer", "genesis"),
        ("BMW", "manufacturer", "bmw"),
        ("Audi", "manufacturer", "audi"),
        ("Mercedes", "manufacturer", "mercedes"),
    )
    for label, dimension, slice_label in focus:
        values = [
            _slice_coverage_value(
                cast(Mapping[str, object], methods[method_id]),
                dimension=dimension,
                label=slice_label,
                coverage="0.9",
            )
            for method_id in (BASELINE_METHOD, GAMMA_METHOD, SMOOTH_METHOD)
        ]
        lines.append(f"| {label} | {values[0]:.2%} | {values[1]:.2%} | {values[2]:.2%} |")
    lines.extend(
        [
            "",
            "Coverage, fold stability, status coverage, all supported broad slices and ",
            "manufacturers, focus slices, clipping, interval validity, relative widths, and ",
            "paired predictor-group bootstrap uncertainty are retained in the JSON report.",
            "Confidence labels remain precision/support labels, not probabilities; data-quality ",
            "warnings remain separate.",
            "",
            "## Reproducibility",
            "",
            f"- Sharpness policy SHA-256: `{SHARPNESS_POLICY_SHA256}`",
            f"- Development diagnostics SHA-256: `{DEVELOPMENT_DIAGNOSTICS_SHA256}`",
            f"- Frozen RF05 identity SHA-256: `{_RF05_IDENTITY_SHA256}`",
            f"- Comparison report SHA-256: `{report_sha256}`",
            (
                f"- Candidate serving artifact SHA-256: `{candidate_sha256}`"
                if candidate_sha256
                else "- Candidate serving artifact: not created"
            ),
            (
                f"- Gamma residual-scale model SHA-256: `{model_sha256}`"
                if model_sha256
                else "- Gamma residual-scale model: not persisted"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _slice_coverage_value(
    method: Mapping[str, object],
    *,
    dimension: str,
    label: str,
    coverage: str,
) -> float:
    slices = cast(Mapping[str, object], method["slices"])
    items = cast(list[Mapping[str, object]], slices[dimension])
    selected = next((item for item in items if item.get("label") == label), None)
    if selected is None:
        raise SharpnessCLIError(f"required focus slice is missing: {label}")
    coverages = cast(Mapping[str, object], selected["coverages"])
    metrics = cast(Mapping[str, object], coverages[coverage])
    return cast(float, metrics["empirical_coverage"])


if __name__ == "__main__":
    raise SystemExit(main())
