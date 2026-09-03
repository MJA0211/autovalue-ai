"""Run or resume full-development confirmation from pinned local evidence."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Final, cast

from autovalue_ml.acquisition.sources.kaggle_us_sales_cars import KaggleUSSalesCarsError
from autovalue_ml.acquisition.sources.kaggle_us_sales_cars_split import (
    prepare_kaggle_us_sales_cars_split_training_rows,
)
from autovalue_ml.acquisition.sources.kaggle_vehicle_sales import KaggleVehicleSalesError
from autovalue_ml.acquisition.sources.kaggle_vehicle_sales_split import (
    prepare_kaggle_vehicle_sales_training_rows,
)

from .baseline_cli import (
    BaselineCLIError,
    RetailTrainingRow,
    WholesaleTrainingRow,
    _collect_retail_partition,
    _collect_wholesale_partitions,
    _expected_count,
    _path_exists,
    _project_path,
    _validate_output_path,
    _validate_project_root,
    _write_atomic,
)
from .contracts import FeatureContractError, TrackName
from .phase4_confirmation import (
    ConfirmationProgressCallback,
    Phase4ConfirmationError,
    Phase4ConfirmationReport,
    canonical_phase4_confirmation_checkpoint_json,
    canonical_phase4_confirmation_json,
    make_phase4_confirmation_checkpoint,
    parse_phase4_confirmation_checkpoint_json,
    run_retail_phase4_confirmation,
    run_wholesale_phase4_confirmation,
)
from .phase4_evaluation import Phase4CandidateCVResult, Phase4EvaluationError
from .phase4_protocol import Phase4Protocol, Phase4ProtocolError, load_phase4_protocol
from .phase4_screening_cli import _RETAIL_PATHS, _WHOLESALE_PATHS
from .phase4_screening_experiment import (
    Phase4ScreeningError,
    Phase4ScreeningReport,
    parse_phase4_screening_json,
)

_PROTOCOL: Final = PurePosixPath("docs/experiments/phase4-model-selection-v1.json")
_SCREENING_REPORTS: Final[dict[TrackName, PurePosixPath]] = {
    "retail": PurePosixPath("docs/experiments/phase4-retail-screening-v1.json"),
    "wholesale": PurePosixPath("docs/experiments/phase4-wholesale-screening-v1.json"),
}


class Phase4ConfirmationCLIError(RuntimeError):
    """A confirmation CLI path or evidence boundary is invalid."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    track = cast(TrackName, arguments.track)
    try:
        project_root = _validate_project_root(Path(cast(str, arguments.project_root)))
        force = cast(bool, arguments.force)
        output = _validate_output_path(
            Path(cast(str, arguments.output)),
            project_root=project_root,
            force=force,
        )
        checkpoint_argument = cast(str | None, arguments.checkpoint)
        checkpoint = (
            Path(checkpoint_argument)
            if checkpoint_argument is not None
            else output.with_suffix(".checkpoint.json")
        )
        checkpoint = _validate_output_path(checkpoint, project_root=project_root, force=True)
        protocol_path = _project_path(project_root, _PROTOCOL)
        screening_path = _project_path(project_root, _SCREENING_REPORTS[track])
        for first, second in (
            (output, checkpoint),
            (output, protocol_path),
            (checkpoint, protocol_path),
            (output, screening_path),
            (checkpoint, screening_path),
        ):
            _validate_distinct_paths(first, second)
        protocol = load_phase4_protocol(protocol_path)
        screening_bytes = screening_path.read_bytes()
        screening_hash = hashlib.sha256(screening_bytes).hexdigest()
        screening_report = parse_phase4_screening_json(screening_bytes)
        completed = _load_checkpoint(checkpoint, track=track)
        if completed:
            print(
                f"{track} confirmation resume | completed={len(completed)}/5 | "
                f"next={len(completed) + 1}",
                flush=True,
            )

        def persist_progress(results: tuple[Phase4CandidateCVResult, ...]) -> None:
            progress = make_phase4_confirmation_checkpoint(track, results)
            _write_atomic(
                checkpoint,
                canonical_phase4_confirmation_checkpoint_json(progress),
                force=True,
            )
            latest = results[-1]
            print(
                f"{track} full-development candidate complete | {len(results)}/5 | "
                f"id={latest.spec.candidate_id} | cv_mae_usd={latest.overall.mae:.2f}",
                flush=True,
            )

        report = (
            _run_retail(
                project_root,
                protocol,
                screening_report,
                screening_hash,
                completed,
                persist_progress,
            )
            if track == "retail"
            else _run_wholesale(
                project_root,
                protocol,
                screening_report,
                screening_hash,
                completed,
                persist_progress,
            )
        )
        _write_atomic(
            output,
            canonical_phase4_confirmation_json(report),
            force=force,
        )
    except (
        BaselineCLIError,
        FeatureContractError,
        KaggleUSSalesCarsError,
        KaggleVehicleSalesError,
        OSError,
        Phase4ConfirmationCLIError,
        Phase4ConfirmationError,
        Phase4EvaluationError,
        Phase4ProtocolError,
        Phase4ScreeningError,
        ValueError,
    ) as error:
        parser.error(str(error))
    _print_summary(report, output=output, checkpoint=checkpoint)
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autovalue_ml.modeling.phase4_confirmation_cli",
        description="Confirm the frozen five-model shortlist on complete development CV.",
    )
    parser.add_argument("track", choices=("retail", "wholesale"))
    parser.add_argument("--project-root", required=True, metavar="PATH")
    parser.add_argument("--output", required=True, metavar="PATH")
    parser.add_argument(
        "--checkpoint",
        metavar="PATH",
        help="aggregate progress JSON; defaults beside output with .checkpoint.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="atomically replace an existing final confirmation report",
    )
    return parser


def _run_retail(
    project_root: Path,
    protocol: Phase4Protocol,
    screening_report: Phase4ScreeningReport,
    screening_hash: str,
    completed: tuple[Phase4CandidateCVResult, ...],
    on_progress: ConfirmationProgressCallback,
) -> Phase4ConfirmationReport:
    paths = tuple(_project_path(project_root, relative) for relative in _RETAIL_PATHS)
    stream = prepare_kaggle_us_sales_cars_split_training_rows(*paths, partition="train")
    train = _collect_retail_partition(
        cast(Iterable[RetailTrainingRow], stream),
        expected_rows=_expected_count(stream, "expected_rows"),
        label="retail train",
    )
    return run_retail_phase4_confirmation(
        phase3_train_features=train.features,
        phase3_train_target=train.target,
        protocol=protocol,
        screening_report=screening_report,
        screening_report_sha256=screening_hash,
        completed_candidates=completed,
        on_progress=on_progress,
    )


def _run_wholesale(
    project_root: Path,
    protocol: Phase4Protocol,
    screening_report: Phase4ScreeningReport,
    screening_hash: str,
    completed: tuple[Phase4CandidateCVResult, ...],
    on_progress: ConfirmationProgressCallback,
) -> Phase4ConfirmationReport:
    paths = tuple(_project_path(project_root, relative) for relative in _WHOLESALE_PATHS)
    stream = prepare_kaggle_vehicle_sales_training_rows(*paths)
    train, _, buckets = _collect_wholesale_partitions(
        cast(Iterable[WholesaleTrainingRow], stream),
        expected_train_rows=_expected_count(stream, "train_rows"),
        expected_test_rows=_expected_count(stream, "test_rows"),
    )
    return run_wholesale_phase4_confirmation(
        phase3_train_features=train.features,
        phase3_train_target=train.target,
        phase3_train_cv_buckets=buckets,
        protocol=protocol,
        screening_report=screening_report,
        screening_report_sha256=screening_hash,
        completed_candidates=completed,
        on_progress=on_progress,
    )


def _load_checkpoint(
    path: Path,
    *,
    track: TrackName,
) -> tuple[Phase4CandidateCVResult, ...]:
    if not _path_exists(path):
        return ()
    if not stat.S_ISREG(path.lstat().st_mode):
        raise Phase4ConfirmationCLIError("confirmation checkpoint must be a regular file")
    checkpoint = parse_phase4_confirmation_checkpoint_json(path.read_bytes())
    if checkpoint.track != track:
        raise Phase4ConfirmationCLIError("confirmation checkpoint track differs")
    return checkpoint.completed_candidates


def _validate_distinct_paths(first: Path, second: Path) -> None:
    first_key = os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(first))))
    second_key = os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(second))))
    if first_key == second_key:
        raise Phase4ConfirmationCLIError("output paths must not replace input evidence")
    if _path_exists(first) and _path_exists(second):
        try:
            if os.path.samefile(first, second):
                raise Phase4ConfirmationCLIError("output paths must not alias input evidence")
        except OSError as error:
            raise Phase4ConfirmationCLIError("path identity could not be verified") from error


def _print_summary(
    report: Phase4ConfirmationReport,
    *,
    output: Path,
    checkpoint: Path,
) -> None:
    leader = report.metric_ranking[0]
    leader_result = next(item for item in report.candidates if item.spec.candidate_id == leader)
    print(
        f"{report.track} full-development confirmation complete | candidates=5 | "
        f"metric_leader={leader} | cv_mae_usd={leader_result.overall.mae:.2f} | "
        f"promotion=pending_deployment_measurements_and_gates | "
        f"report={output} | checkpoint={checkpoint}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
