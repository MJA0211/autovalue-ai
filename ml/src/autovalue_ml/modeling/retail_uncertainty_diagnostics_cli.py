"""Run the development-only RF05 residual diagnostic reconstruction once."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterable, Sequence
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
from .calibration import retail_calibration_partition
from .calibration_artifact import (
    CALIBRATION_ASSIGNMENT_SHA256,
    CALIBRATION_POLICY_SHA256,
    PHASE4_RETAIL_CONFIRMATION_SHA256,
)
from .phase4_confirmation import Phase4ConfirmationError, parse_phase4_confirmation_json
from .phase4_screening_cli import _RETAIL_PATHS
from .phase4_screening_experiment import _partition_hash
from .retail_calibration_cli import CalibrationCLIError, _verified_bytes
from .retail_calibration_experiment import CALIBRATION_SEED
from .retail_uncertainty_diagnostics import (
    ResidualDiagnosticsError,
    build_development_residual_diagnostics,
    canonical_diagnostics_json,
)

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
_OUTPUT: Final = PurePosixPath(
    "docs/experiments/retail-rf05-development-residual-diagnostics-v1.json"
)
_CALIBRATION_ARTIFACT_SHA256: Final = (
    "b7eb5970b164ec68fb76cf8314f36080d085cda02968d3570d11f724490a6da0"
)
_CALIBRATION_REPORT_SHA256: Final = (
    "e7fafff505603669e73cfbff2fe1cf5e04f9c5d896666470fe212411aa1b3084"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m autovalue_ml.modeling.retail_uncertainty_diagnostics_cli",
        description="Reconstruct development-only RF05 OOF residual diagnostics.",
    )
    parser.add_argument("--project-root", required=True, metavar="PATH")
    arguments = parser.parse_args(argv)
    try:
        project_root = _validate_project_root(Path(cast(str, arguments.project_root)))
        output = _validate_output_path(
            _project_path(project_root, _OUTPUT),
            project_root=project_root,
            force=False,
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
            expected=_CALIBRATION_ARTIFACT_SHA256,
            label="calibration v1 serving artifact",
        )
        _verified_bytes(
            _project_path(project_root, _CALIBRATION_REPORT),
            expected=_CALIBRATION_REPORT_SHA256,
            label="calibration v1 report",
        )
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
            raise ResidualDiagnosticsError("calibration boundary differs from frozen evidence")
        development = partition.development_indices
        report = build_development_residual_diagnostics(
            development_features=phase3_train.features.iloc[development].reset_index(drop=True),
            development_target=phase3_train.target[development],
            confirmation=confirmation,
            confirmation_sha256=hashlib.sha256(confirmation_bytes).hexdigest(),
            progress=_print_progress,
        )
        serialized = canonical_diagnostics_json(report)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        _write_atomic(output, serialized, force=False)
    except (
        BaselineCLIError,
        CalibrationCLIError,
        KaggleUSSalesCarsError,
        KaggleUSSalesCarsSplitError,
        OSError,
        Phase4ConfirmationError,
        ResidualDiagnosticsError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(f"RF05 development residual diagnostics complete | sha256={digest}", flush=True)
    return 0


def _print_progress(fold_number: int, fold_count: int) -> None:
    print(f"RF05 development OOF reconstruction | fold={fold_number}/{fold_count}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
