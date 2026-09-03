"""Run resumable Phase 4 screening from reviewed, source-pinned artifacts only."""

from __future__ import annotations

import argparse
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
from .phase4_evaluation import Phase4CandidateCVResult, Phase4EvaluationError
from .phase4_protocol import Phase4ProtocolError, load_phase4_protocol
from .phase4_screening_experiment import (
    Phase4ScreeningError,
    Phase4ScreeningReport,
    canonical_phase4_checkpoint_json,
    canonical_phase4_screening_json,
    make_phase4_screening_checkpoint,
    parse_phase4_checkpoint_json,
    run_retail_phase4_screening,
    run_wholesale_phase4_screening,
)

_PROTOCOL: Final = PurePosixPath("docs/experiments/phase4-model-selection-v1.json")
_RETAIL_PATHS: Final = (
    PurePosixPath("data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.csv"),
    PurePosixPath("data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.manifest.json"),
    PurePosixPath("data/processed/kaggle_us_sales_cars_v2/split/split_assignments.manifest.json"),
    PurePosixPath("docs/data-reviews/kaggle-us-sales-cars-v2.review.json"),
)
_WHOLESALE_PATHS: Final = (
    PurePosixPath("data/processed/kaggle_vehicle_sales_v1/split_assignments.manifest.json"),
    PurePosixPath("data/raw/kaggle_vehicle_sales_v1/car_prices.csv"),
    PurePosixPath("data/interim/kaggle_vehicle_sales_v1.csv"),
    PurePosixPath("data/interim/kaggle_vehicle_sales_v1.manifest.json"),
    PurePosixPath("docs/data-reviews/kaggle-vehicle-sales-data-v1.review.json"),
    PurePosixPath("docs/data-reviews/kaggle-vehicle-sales-v1.split.json"),
)


class Phase4ScreeningCLIError(RuntimeError):
    """A CLI path, checkpoint, or source boundary is unsafe or inconsistent."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run or resume all 13 candidates for one price track."""

    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    track = cast(TrackName, arguments.track)
    try:
        project_root = _validate_project_root(Path(cast(str, arguments.project_root)))
        output = _validate_output_path(
            Path(cast(str, arguments.output)),
            project_root=project_root,
            force=cast(bool, arguments.force),
        )
        checkpoint_argument = cast(str | None, arguments.checkpoint)
        checkpoint = (
            Path(checkpoint_argument)
            if checkpoint_argument is not None
            else output.with_suffix(".checkpoint.json")
        )
        checkpoint = _validate_output_path(
            checkpoint,
            project_root=project_root,
            force=True,
        )
        _validate_distinct_paths(output, checkpoint)
        protocol_path = _project_path(project_root, _PROTOCOL)
        _validate_distinct_paths(output, protocol_path)
        _validate_distinct_paths(checkpoint, protocol_path)
        protocol = load_phase4_protocol(protocol_path)
        completed = _load_completed_checkpoint(checkpoint, track=track)
        if completed:
            print(
                f"{track} screening resume | completed={len(completed)}/13 | "
                f"next={len(completed) + 1}",
                flush=True,
            )

        def persist_progress(results: tuple[Phase4CandidateCVResult, ...]) -> None:
            progress = make_phase4_screening_checkpoint(track, results)
            _write_atomic(
                checkpoint,
                canonical_phase4_checkpoint_json(progress),
                force=True,
            )
            latest = results[-1]
            print(
                f"{track} candidate complete | {len(results)}/13 | "
                f"id={latest.spec.candidate_id} | cv_mae_usd={latest.overall.mae:.2f}",
                flush=True,
            )

        report = (
            _run_retail(project_root, protocol, completed, persist_progress)
            if track == "retail"
            else _run_wholesale(project_root, protocol, completed, persist_progress)
        )
        _write_atomic(
            output,
            canonical_phase4_screening_json(report),
            force=cast(bool, arguments.force),
        )
    except (
        BaselineCLIError,
        FeatureContractError,
        KaggleUSSalesCarsError,
        KaggleVehicleSalesError,
        OSError,
        Phase4EvaluationError,
        Phase4ProtocolError,
        Phase4ScreeningCLIError,
        Phase4ScreeningError,
        ValueError,
    ) as error:
        parser.error(str(error))

    _print_summary(report, checkpoint=checkpoint, output=output)
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autovalue_ml.modeling.phase4_screening_cli",
        description=(
            "Screen all frozen Phase 4 candidates with aggregate-only resumable checkpoints."
        ),
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
        help="atomically replace an existing final screening report",
    )
    return parser


def _run_retail(
    project_root: Path,
    protocol: object,
    completed: tuple[Phase4CandidateCVResult, ...],
    on_progress: object,
) -> Phase4ScreeningReport:
    from .phase4_protocol import Phase4Protocol
    from .phase4_screening_experiment import CandidateProgressCallback

    if not isinstance(protocol, Phase4Protocol) or not callable(on_progress):
        raise Phase4ScreeningCLIError("retail screening dependencies are invalid")
    paths = tuple(_project_path(project_root, relative) for relative in _RETAIL_PATHS)
    stream = prepare_kaggle_us_sales_cars_split_training_rows(
        *paths,
        partition="train",
    )
    train = _collect_retail_partition(
        cast(Iterable[RetailTrainingRow], stream),
        expected_rows=_expected_count(stream, "expected_rows"),
        label="retail train",
    )
    return run_retail_phase4_screening(
        phase3_train_features=train.features,
        phase3_train_target=train.target,
        protocol=protocol,
        completed_candidates=completed,
        on_progress=cast(CandidateProgressCallback, on_progress),
    )


def _run_wholesale(
    project_root: Path,
    protocol: object,
    completed: tuple[Phase4CandidateCVResult, ...],
    on_progress: object,
) -> Phase4ScreeningReport:
    from .phase4_protocol import Phase4Protocol
    from .phase4_screening_experiment import CandidateProgressCallback

    if not isinstance(protocol, Phase4Protocol) or not callable(on_progress):
        raise Phase4ScreeningCLIError("wholesale screening dependencies are invalid")
    paths = tuple(_project_path(project_root, relative) for relative in _WHOLESALE_PATHS)
    stream = prepare_kaggle_vehicle_sales_training_rows(*paths)
    train, _, buckets = _collect_wholesale_partitions(
        cast(Iterable[WholesaleTrainingRow], stream),
        expected_train_rows=_expected_count(stream, "train_rows"),
        expected_test_rows=_expected_count(stream, "test_rows"),
    )
    return run_wholesale_phase4_screening(
        phase3_train_features=train.features,
        phase3_train_target=train.target,
        phase3_train_cv_buckets=buckets,
        protocol=protocol,
        completed_candidates=completed,
        on_progress=cast(CandidateProgressCallback, on_progress),
    )


def _load_completed_checkpoint(
    path: Path,
    *,
    track: TrackName,
) -> tuple[Phase4CandidateCVResult, ...]:
    if not _path_exists(path):
        return ()
    if not stat.S_ISREG(path.lstat().st_mode):
        raise Phase4ScreeningCLIError("checkpoint must be a regular file")
    checkpoint = parse_phase4_checkpoint_json(path.read_bytes())
    if checkpoint.track != track:
        raise Phase4ScreeningCLIError("checkpoint track differs from requested track")
    return checkpoint.completed_candidates


def _validate_distinct_paths(first: Path, second: Path) -> None:
    first_key = os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(first))))
    second_key = os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(second))))
    if first_key == second_key:
        raise Phase4ScreeningCLIError("output, checkpoint, and protocol paths must be distinct")
    if _path_exists(first) and _path_exists(second):
        try:
            if os.path.samefile(first, second):
                raise Phase4ScreeningCLIError(
                    "output, checkpoint, and protocol paths must not alias"
                )
        except OSError as error:
            raise Phase4ScreeningCLIError("path identity could not be verified") from error


def _print_summary(report: Phase4ScreeningReport, *, checkpoint: Path, output: Path) -> None:
    shortlist = report.shortlist
    print(
        f"{report.track} screening complete | candidates={len(report.candidates)} | "
        f"rf={','.join(shortlist.random_forest_candidate_ids)} | "
        f"gbr={','.join(shortlist.gradient_boosting_candidate_ids)} | "
        f"report={output} | checkpoint={checkpoint}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
