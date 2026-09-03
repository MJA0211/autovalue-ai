"""Run the first governed calibration use for the frozen retail RF05 model."""

from __future__ import annotations

import argparse
import hashlib
import stat
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Final, cast

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
from .calibration_artifact import (
    CALIBRATION_POLICY_SHA256,
    PHASE4_RETAIL_CONFIRMATION_SHA256,
    canonical_calibration_artifact_json,
)
from .phase4_confirmation import Phase4ConfirmationError, parse_phase4_confirmation_json
from .phase4_protocol import Phase4ProtocolError, load_phase4_protocol
from .phase4_screening_cli import _RETAIL_PATHS
from .retail_calibration_experiment import (
    CalibrationExperimentError,
    canonical_calibration_report_json,
    run_retail_rf05_calibration,
)

_PROTOCOL: Final = PurePosixPath("docs/experiments/phase4-model-selection-v1.json")
_CONFIRMATION: Final = PurePosixPath("docs/experiments/phase4-retail-full-development-v1.json")
_POLICY: Final = PurePosixPath("docs/experiments/retail-rf05-calibration-policy-v1.json")
_ARTIFACT: Final = PurePosixPath("docs/experiments/retail-rf05-calibration-v1.artifact.json")
_REPORT: Final = PurePosixPath("docs/experiments/retail-rf05-calibration-v1.report.json")
_MARKDOWN: Final = PurePosixPath("docs/experiments/retail-rf05-calibration-v1.md")


class CalibrationCLIError(RuntimeError):
    """A protected input or immutable output boundary is invalid."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m autovalue_ml.modeling.retail_calibration_cli",
        description="Calibrate frozen retail RF05 with the reserved Phase 4 population.",
    )
    parser.add_argument("--project-root", required=True, metavar="PATH")
    arguments = parser.parse_args(argv)
    try:
        project_root = _validate_project_root(Path(cast(str, arguments.project_root)))
        paths = {
            "artifact": _validate_output_path(
                _project_path(project_root, _ARTIFACT), project_root=project_root, force=False
            ),
            "report": _validate_output_path(
                _project_path(project_root, _REPORT), project_root=project_root, force=False
            ),
            "markdown": _validate_markdown_output(
                _project_path(project_root, _MARKDOWN), project_root=project_root
            ),
        }
        policy_path = _project_path(project_root, _POLICY)
        _verify_regular_sha256(
            policy_path,
            expected=CALIBRATION_POLICY_SHA256,
            label="calibration policy",
        )
        protocol = load_phase4_protocol(_project_path(project_root, _PROTOCOL))
        confirmation_path = _project_path(project_root, _CONFIRMATION)
        confirmation_bytes = _verified_bytes(
            confirmation_path,
            expected=PHASE4_RETAIL_CONFIRMATION_SHA256,
            label="Phase 4 retail confirmation",
        )
        confirmation = parse_phase4_confirmation_json(confirmation_bytes)
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
        result = run_retail_rf05_calibration(
            phase3_train_features=phase3_train.features,
            phase3_train_target=phase3_train.target,
            protocol=protocol,
            confirmation=confirmation,
            confirmation_sha256=hashlib.sha256(confirmation_bytes).hexdigest(),
        )
        artifact_json = canonical_calibration_artifact_json(result.artifact)
        artifact_sha256 = hashlib.sha256(artifact_json.encode("utf-8")).hexdigest()
        report = {
            **result.report,
            "artifacts": {
                "calibration_artifact": _ARTIFACT.as_posix(),
                "calibration_artifact_sha256": artifact_sha256,
                "report_contains_raw_rows_predictions_or_residuals": False,
            },
        }
        report_json = canonical_calibration_report_json(report)
        report_sha256 = hashlib.sha256(report_json.encode("utf-8")).hexdigest()
        markdown = render_calibration_markdown(
            report,
            artifact_sha256=artifact_sha256,
            report_sha256=report_sha256,
        )
        _write_atomic(paths["artifact"], artifact_json, force=False)
        _write_atomic(paths["report"], report_json, force=False)
        _write_atomic(paths["markdown"], markdown, force=False)
    except (
        BaselineCLIError,
        CalibrationCLIError,
        CalibrationExperimentError,
        KaggleUSSalesCarsError,
        KaggleUSSalesCarsSplitError,
        OSError,
        Phase4ConfirmationError,
        Phase4ProtocolError,
        ValueError,
    ) as error:
        parser.error(str(error))
    selected = cast(Mapping[str, object], report["decision"])["selected_method"]
    print(
        "retail RF05 calibration complete | "
        f"classification={report['classification']} | selected={selected} | "
        f"artifact_sha256={artifact_sha256} | report_sha256={report_sha256}",
        flush=True,
    )
    return 0


def render_calibration_markdown(
    report: Mapping[str, object],
    *,
    artifact_sha256: str,
    report_sha256: str,
) -> str:
    """Render a concise human-readable view of aggregate calibration evidence."""

    decision = cast(Mapping[str, object], report["decision"])
    boundaries = cast(Mapping[str, object], report["data_boundaries"])
    point = cast(Mapping[str, object], report["point_prediction_metrics_on_calibration"])
    point_overall = cast(Mapping[str, object], point["overall"])
    cross = cast(Mapping[str, object], report["cross_calibration"])
    methods = cast(Mapping[str, object], cross["methods"])
    selected_method = cast(str, decision["selected_method"])
    selected = cast(Mapping[str, object], methods[selected_method])
    coverages = cast(Mapping[str, object], selected["coverages"])
    artifact = cast(Mapping[str, object], report["artifacts"])
    confidence = cast(Mapping[str, object], report["confidence"])
    confidence_counts = cast(Mapping[str, object], confidence["counts"])
    evidence = cast(Mapping[str, object], report["frozen_evidence"])
    lines = [
        "# Retail RF05 calibrated prediction intervals",
        "",
        "## Decision",
        "",
        f"**Classification: `{report['classification']}`.** The selected calibration method is ",
        f"`{selected_method}`. This calibrates the frozen Phase 4 RF05 point predictor; it does ",
        "not promote, replace, retrain, or retune it. The legacy final holdout remains unopened.",
        "",
        "These are empirical intervals around a historical U.S. advertised asking-price model. ",
        "They are not a probability for one vehicle, a guaranteed sale price, or a Kelley Blue ",
        "Book/third-party valuation.",
        "",
        "## Protected boundary",
        "",
        f"- Development-only RF05 fit: {boundaries['development_fit_rows']:,} rows",
        f"- Reserved calibration population: {boundaries['calibration_rows']:,} rows",
        "- Calibration rows used for RF fitting, model choice, or tuning: no",
        "- Legacy holdout, Yoad, and River data accessed: no",
        "- Raw rows, row-level predictions, or residuals persisted: no",
        "",
        "## Calibration point performance",
        "",
        "All price errors are USD.",
        "",
        "| Rows | MAE | RMSE | R-squared |",
        "|---:|---:|---:|---:|",
        (
            f"| {point_overall['sample_count']:,} | ${point_overall['mae']:,.2f} | "
            f"${point_overall['rmse']:,.2f} | {point_overall['r2']:.4f} |"
        ),
        "",
        "## Cross-fitted interval results",
        "",
        "Each diagnostic row is scored with radii derived from the other four predictor-group ",
        "folds. All three preregistered levels are reported.",
        "",
        "| Nominal | Empirical | Gap | Average width | Median width | Fold coverage SD |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for coverage in ("0.8", "0.9", "0.95"):
        item = cast(Mapping[str, object], coverages[coverage])
        lines.append(
            f"| {float(coverage):.0%} | {cast(float, item['empirical_coverage']):.2%} | "
            f"{cast(float, item['coverage_gap']):+.2%} | "
            f"${cast(float, item['average_width_usd']):,.2f} | "
            f"${cast(float, item['median_width_usd']):,.2f} | "
            f"{cast(float, item['fold_coverage_standard_deviation']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## Confidence labels",
            "",
            "Confidence labels use the preregistered empirical 90% relative-width thresholds ",
            "and applicable calibration support. They are not probabilities. Data-quality "
            "warnings ",
            "remain separate and do not silently widen the interval.",
            "",
            (
                f"High: {confidence_counts['High confidence']:,}; moderate: "
                f"{confidence_counts['Moderate confidence']:,}; low: "
                f"{confidence_counts['Low confidence']:,}."
            ),
            "",
            "## Reproducibility",
            "",
            f"- Calibration policy SHA-256: `{evidence['policy_sha256']}`",
            (f"- Calibration assignment SHA-256: `{boundaries['calibration_assignment_sha256']}`"),
            f"- Serving artifact: `{artifact['calibration_artifact']}`",
            f"- Serving artifact SHA-256: `{artifact_sha256}`",
            f"- Aggregate report SHA-256: `{report_sha256}`",
            "",
            "The JSON report contains fold-level coverage, interval widths, under/overcoverage, ",
            "vehicle-status diagnostics, and aggregate price/mileage/age/manufacturer slices.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify_regular_sha256(path: Path, *, expected: str, label: str) -> None:
    _verified_bytes(path, expected=expected, label=label)


def _validate_markdown_output(path: Path, *, project_root: Path) -> Path:
    if path.suffix != ".md" or path.parent != project_root / "docs" / "experiments":
        raise CalibrationCLIError("Markdown output must use the fixed experiment path")
    current = path
    while True:
        if current.is_symlink():
            raise CalibrationCLIError("Markdown output path must not contain symlinks")
        if current == current.parent:
            break
        current = current.parent
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return path
    if not stat.S_ISREG(mode):
        raise CalibrationCLIError("Markdown output must be a regular file")
    raise CalibrationCLIError("Markdown output already exists")


def _verified_bytes(path: Path, *, expected: str, label: str) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise CalibrationCLIError(f"{label} must be a regular non-symlink file")
    payload = path.read_bytes()
    after = path.lstat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise CalibrationCLIError(f"{label} changed while it was read")
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise CalibrationCLIError(f"{label} checksum differs from frozen evidence")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
