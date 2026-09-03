"""Run AutoValue's two Phase 3 baselines from verified split artifacts only.

The command deliberately has no generic data-path option.  Every source file is
resolved from a reviewed, project-relative location and opened through its
source-specific split preparation gate.  Its only durable output is the
aggregate-only canonical experiment report; fitted estimators and row-level
values are never persisted.
"""

from __future__ import annotations

import argparse
import math
import os
import stat
import sys
import tempfile
from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from autovalue_ml.acquisition.sources.kaggle_us_sales_cars import KaggleUSSalesCarsError
from autovalue_ml.acquisition.sources.kaggle_us_sales_cars_split import (
    prepare_kaggle_us_sales_cars_split_training_rows,
)
from autovalue_ml.acquisition.sources.kaggle_vehicle_sales import KaggleVehicleSalesError
from autovalue_ml.acquisition.sources.kaggle_vehicle_sales_split import (
    prepare_kaggle_vehicle_sales_training_rows,
)

from .contracts import FeatureContractError, TrackName
from .experiment import (
    BaselineExperimentResult,
    ExperimentValidationError,
    canonical_experiment_json,
    run_retail_baseline_experiment,
    run_wholesale_baseline_experiment,
)

FeatureValue: TypeAlias = str | int | float | None
RetailTrainingRow: TypeAlias = tuple[Mapping[str, object], object]
WholesaleTrainingRow: TypeAlias = tuple[str, str | None, Mapping[str, object], object]
Partition: TypeAlias = Literal["train", "test"]

_RETAIL_FEATURES: Final = ("year", "make", "model", "vehicle_status", "mileage")
_RETAIL_REQUIRED: Final = frozenset(("year", "make", "model", "vehicle_status"))
_WHOLESALE_FEATURES: Final = (
    "year",
    "make",
    "model",
    "trim",
    "mileage",
    "condition",
    "vehicle_type",
)
_WHOLESALE_REQUIRED: Final = frozenset(("year", "make", "model"))
_RETAIL_STATUSES: Final = frozenset(("certified", "new", "used"))
_WHOLESALE_BUCKET_ORDER: Final = (
    "warmup",
    "2015_01",
    "2015_02",
    "2015_03_04",
    "2015_05",
)

_RETAIL_CANDIDATE: Final = PurePosixPath(
    "data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.csv"
)
_RETAIL_CANDIDATE_MANIFEST: Final = PurePosixPath(
    "data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.manifest.json"
)
_RETAIL_CANDIDATE_READINESS: Final = PurePosixPath(
    "data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.ready.json"
)
_RETAIL_SPLIT_MANIFEST: Final = PurePosixPath(
    "data/processed/kaggle_us_sales_cars_v2/split/split_assignments.manifest.json"
)
_RETAIL_SPLIT_READINESS: Final = PurePosixPath(
    "data/processed/kaggle_us_sales_cars_v2/split/split_assignments.ready.json"
)
_RETAIL_REVIEW: Final = PurePosixPath("docs/data-reviews/kaggle-us-sales-cars-v2.review.json")

_WHOLESALE_SPLIT_MANIFEST: Final = PurePosixPath(
    "data/processed/kaggle_vehicle_sales_v1/split_assignments.manifest.json"
)
_WHOLESALE_RAW_SOURCE: Final = PurePosixPath("data/raw/kaggle_vehicle_sales_v1/car_prices.csv")
_WHOLESALE_CANDIDATE: Final = PurePosixPath("data/interim/kaggle_vehicle_sales_v1.csv")
_WHOLESALE_CANDIDATE_MANIFEST: Final = PurePosixPath(
    "data/interim/kaggle_vehicle_sales_v1.manifest.json"
)
_WHOLESALE_CANDIDATE_READINESS: Final = PurePosixPath(
    "data/interim/kaggle_vehicle_sales_v1.ready.json"
)
_WHOLESALE_REVIEW: Final = PurePosixPath(
    "docs/data-reviews/kaggle-vehicle-sales-data-v1.review.json"
)
_WHOLESALE_SPLIT_POLICY: Final = PurePosixPath(
    "docs/data-reviews/kaggle-vehicle-sales-v1.split.json"
)
_WHOLESALE_SPLIT_READINESS: Final = PurePosixPath(
    "data/processed/kaggle_vehicle_sales_v1/split_assignments.ready.json"
)
_PROTECTED_INPUTS: Final = (
    _RETAIL_CANDIDATE,
    _RETAIL_CANDIDATE_MANIFEST,
    _RETAIL_CANDIDATE_READINESS,
    _RETAIL_SPLIT_MANIFEST,
    _RETAIL_SPLIT_READINESS,
    _RETAIL_REVIEW,
    _WHOLESALE_SPLIT_MANIFEST,
    _WHOLESALE_RAW_SOURCE,
    _WHOLESALE_CANDIDATE,
    _WHOLESALE_CANDIDATE_MANIFEST,
    _WHOLESALE_CANDIDATE_READINESS,
    _WHOLESALE_REVIEW,
    _WHOLESALE_SPLIT_POLICY,
    _WHOLESALE_SPLIT_READINESS,
)


class BaselineCLIError(RuntimeError):
    """Raised when a CLI boundary check fails without exposing source rows."""


@dataclass(frozen=True, slots=True)
class _CollectedPartition:
    features: pd.DataFrame
    target: NDArray[np.float64]


def main(argv: Sequence[str] | None = None) -> int:
    """Run one approved baseline track and return zero on success."""

    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    track = cast(TrackName, arguments.track)
    project_root_argument = Path(cast(str, arguments.project_root))
    output_argument = Path(cast(str, arguments.output))
    force = cast(bool, arguments.force)

    try:
        project_root = _validate_project_root(project_root_argument)
        output = _validate_output_path(output_argument, project_root=project_root, force=force)
        result = _run_retail(project_root) if track == "retail" else _run_wholesale(project_root)
        serialized = canonical_experiment_json(result)
        _write_atomic(output, serialized, force=force)
    except (
        BaselineCLIError,
        ExperimentValidationError,
        FeatureContractError,
        KaggleUSSalesCarsError,
        KaggleVehicleSalesError,
        OSError,
        ValueError,
    ) as error:
        parser.error(str(error))

    _print_summary(result)
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autovalue_ml.modeling.baseline_cli",
        description=(
            "Run aggregate-only baselines from AutoValue's reviewed, split-aware artifacts."
        ),
    )
    parser.add_argument("track", choices=("retail", "wholesale"))
    parser.add_argument("--project-root", required=True, metavar="PATH")
    parser.add_argument("--output", required=True, metavar="PATH")
    parser.add_argument(
        "--force",
        action="store_true",
        help="atomically replace an existing regular JSON report",
    )
    return parser


def _run_retail(project_root: Path) -> BaselineExperimentResult:
    paths = tuple(
        _project_path(project_root, relative)
        for relative in (
            _RETAIL_CANDIDATE,
            _RETAIL_CANDIDATE_MANIFEST,
            _RETAIL_SPLIT_MANIFEST,
            _RETAIL_REVIEW,
        )
    )
    candidate, candidate_manifest, split_manifest, review = paths
    train_stream = prepare_kaggle_us_sales_cars_split_training_rows(
        candidate,
        candidate_manifest,
        split_manifest,
        review,
        partition="train",
    )
    test_stream = prepare_kaggle_us_sales_cars_split_training_rows(
        candidate,
        candidate_manifest,
        split_manifest,
        review,
        partition="test",
    )
    train = _collect_retail_partition(
        cast(Iterable[RetailTrainingRow], train_stream),
        expected_rows=_expected_count(train_stream, "expected_rows"),
        label="retail train",
    )
    test = _collect_retail_partition(
        cast(Iterable[RetailTrainingRow], test_stream),
        expected_rows=_expected_count(test_stream, "expected_rows"),
        label="retail test",
    )
    return run_retail_baseline_experiment(
        outer_train_features=train.features,
        outer_train_target=train.target,
        outer_test_features=test.features,
        outer_test_target=test.target,
    )


def _run_wholesale(project_root: Path) -> BaselineExperimentResult:
    split_manifest, raw_source, candidate, candidate_manifest, review, split_policy = tuple(
        _project_path(project_root, relative)
        for relative in (
            _WHOLESALE_SPLIT_MANIFEST,
            _WHOLESALE_RAW_SOURCE,
            _WHOLESALE_CANDIDATE,
            _WHOLESALE_CANDIDATE_MANIFEST,
            _WHOLESALE_REVIEW,
            _WHOLESALE_SPLIT_POLICY,
        )
    )
    stream = prepare_kaggle_vehicle_sales_training_rows(
        split_manifest,
        raw_source,
        candidate,
        candidate_manifest,
        review,
        split_policy,
    )
    train, test, buckets = _collect_wholesale_partitions(
        cast(Iterable[WholesaleTrainingRow], stream),
        expected_train_rows=_expected_count(stream, "train_rows"),
        expected_test_rows=_expected_count(stream, "test_rows"),
    )
    return run_wholesale_baseline_experiment(
        outer_train_features=train.features,
        outer_train_target=train.target,
        outer_test_features=test.features,
        outer_test_target=test.target,
        train_cv_buckets=buckets,
        bucket_order=_WHOLESALE_BUCKET_ORDER,
    )


def _collect_retail_partition(
    rows: Iterable[RetailTrainingRow],
    *,
    expected_rows: int | None,
    label: str,
) -> _CollectedPartition:
    years = array("i")
    mileages = array("d")
    targets = array("d")
    makes: list[str] = []
    models: list[str] = []
    statuses: list[str] = []

    for item in rows:
        if not isinstance(item, tuple) or len(item) != 2:
            raise BaselineCLIError(f"{label} stream row has an invalid shape")
        raw_features, raw_target = item
        features = _validate_feature_mapping(
            raw_features,
            required=_RETAIL_REQUIRED,
            allowed=frozenset(_RETAIL_FEATURES),
            label=label,
        )
        years.append(_year(features["year"], label=label, maximum=2025))
        makes.append(_required_text(features["make"], label=label, maximum=100))
        models.append(_required_text(features["model"], label=label, maximum=200))
        status = _required_text(features["vehicle_status"], label=label, maximum=9)
        if status not in _RETAIL_STATUSES:
            raise BaselineCLIError(f"{label} row has an unsupported vehicle status")
        statuses.append(status)
        mileages.append(_optional_nonnegative_number(features.get("mileage"), label=label))
        targets.append(_positive_target(raw_target, label=label))

    _validate_collected_count(len(targets), expected_rows=expected_rows, label=label)
    frame = pd.DataFrame(
        {
            "year": np.frombuffer(years, dtype=np.int32),
            "make": pd.Series(makes, dtype=object),
            "model": pd.Series(models, dtype=object),
            "vehicle_status": pd.Series(statuses, dtype=object),
            "mileage": np.frombuffer(mileages, dtype=np.float64),
        },
        columns=list(_RETAIL_FEATURES),
    )
    return _CollectedPartition(frame, np.frombuffer(targets, dtype=np.float64))


def _collect_wholesale_partitions(
    rows: Iterable[WholesaleTrainingRow],
    *,
    expected_train_rows: int | None,
    expected_test_rows: int | None,
) -> tuple[_CollectedPartition, _CollectedPartition, pd.Series[str]]:
    columns: dict[Partition, dict[str, list[FeatureValue]]] = {
        "train": {name: [] for name in _WHOLESALE_FEATURES},
        "test": {name: [] for name in _WHOLESALE_FEATURES},
    }
    targets: dict[Partition, array[float]] = {"train": array("d"), "test": array("d")}
    buckets: list[str] = []

    for item in rows:
        if not isinstance(item, tuple) or len(item) != 4:
            raise BaselineCLIError("wholesale stream row has an invalid shape")
        raw_partition, raw_bucket, raw_features, raw_target = item
        if raw_partition not in {"train", "test"}:
            raise BaselineCLIError("wholesale stream contains an invalid partition")
        partition = cast(Partition, raw_partition)
        if partition == "train":
            if raw_bucket not in _WHOLESALE_BUCKET_ORDER:
                raise BaselineCLIError("wholesale train row has an invalid CV bucket")
            buckets.append(raw_bucket)
        elif raw_bucket is not None:
            raise BaselineCLIError("wholesale test rows must not have a CV bucket")

        features = _validate_feature_mapping(
            raw_features,
            required=_WHOLESALE_REQUIRED,
            allowed=frozenset(_WHOLESALE_FEATURES),
            label="wholesale",
        )
        normalized: dict[str, FeatureValue] = {
            "year": _year(features["year"], label="wholesale", maximum=2015),
            "make": _required_text(features["make"], label="wholesale", maximum=100),
            "model": _required_text(features["model"], label="wholesale", maximum=150),
            "trim": _optional_text(features.get("trim"), label="wholesale", maximum=150),
            "mileage": _optional_nonnegative_number(features.get("mileage"), label="wholesale"),
            "condition": _optional_condition(features.get("condition")),
            "vehicle_type": _optional_text(
                features.get("vehicle_type"), label="wholesale", maximum=100
            ),
        }
        for name in _WHOLESALE_FEATURES:
            columns[partition][name].append(normalized[name])
        targets[partition].append(_positive_target(raw_target, label="wholesale"))

    _validate_collected_count(
        len(targets["train"]), expected_rows=expected_train_rows, label="wholesale train"
    )
    _validate_collected_count(
        len(targets["test"]), expected_rows=expected_test_rows, label="wholesale test"
    )
    train = _wholesale_partition(columns["train"], targets["train"])
    test = _wholesale_partition(columns["test"], targets["test"])
    bucket_series = pd.Series(
        pd.Categorical(buckets, categories=list(_WHOLESALE_BUCKET_ORDER), ordered=True),
        index=train.features.index,
        name="cv_bucket",
    )
    return train, test, bucket_series


def _wholesale_partition(
    columns: Mapping[str, Sequence[FeatureValue]], targets: array[float]
) -> _CollectedPartition:
    frame = pd.DataFrame(
        {
            "year": np.asarray(columns["year"], dtype=np.int32),
            "make": pd.Series(columns["make"], dtype=object),
            "model": pd.Series(columns["model"], dtype=object),
            "trim": pd.Series(columns["trim"], dtype=object),
            "mileage": np.asarray(columns["mileage"], dtype=np.float64),
            "condition": np.asarray(columns["condition"], dtype=np.float64),
            "vehicle_type": pd.Series(columns["vehicle_type"], dtype=object),
        },
        columns=list(_WHOLESALE_FEATURES),
    )
    return _CollectedPartition(frame, np.frombuffer(targets, dtype=np.float64))


def _validate_feature_mapping(
    value: object,
    *,
    required: frozenset[str],
    allowed: frozenset[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise BaselineCLIError(f"{label} row features must be a string-keyed mapping")
    keys = set(value)
    if not required.issubset(keys) or not keys.issubset(allowed):
        raise BaselineCLIError(f"{label} row violates the feature allowlist")
    return cast(Mapping[str, object], value)


def _year(value: object, *, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1886 <= value <= maximum:
        raise BaselineCLIError(f"{label} row has an invalid model year")
    return value


def _required_text(value: object, *, label: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(character in value for character in "\x00\r\n")
    ):
        raise BaselineCLIError(f"{label} row has invalid categorical text")
    return sys.intern(value)


def _optional_text(value: object, *, label: str, maximum: int) -> str | None:
    if value is None:
        return None
    return _required_text(value, label=label, maximum=maximum)


def _optional_nonnegative_number(value: object, *, label: str) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineCLIError(f"{label} row has invalid mileage")
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise BaselineCLIError(f"{label} row has invalid mileage")
    return number


def _optional_condition(value: object) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineCLIError("wholesale row has invalid condition")
    number = float(value)
    if not math.isfinite(number) or not 1.0 <= number <= 5.0:
        raise BaselineCLIError("wholesale row has invalid condition")
    return number


def _positive_target(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineCLIError(f"{label} row has an invalid target")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise BaselineCLIError(f"{label} row has an invalid target")
    return number


def _expected_count(stream: object, attribute: str) -> int | None:
    value = getattr(stream, attribute, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BaselineCLIError("verified stream exposes an invalid expected row count")
    return value


def _validate_collected_count(count: int, *, expected_rows: int | None, label: str) -> None:
    if count <= 0:
        raise BaselineCLIError(f"{label} partition must not be empty")
    if expected_rows is not None and count != expected_rows:
        raise BaselineCLIError(f"{label} row count differs from the verified split")


def _validate_project_root(value: Path) -> Path:
    path = _absolute(value)
    if path.is_symlink() or not path.is_dir():
        raise BaselineCLIError("project root must be an existing non-symlink directory")
    return path


def _validate_output_path(value: Path, *, project_root: Path, force: bool) -> Path:
    path = _absolute(value)
    if path.suffix != ".json" or not path.name or path.name in {".", ".."}:
        raise BaselineCLIError("output must have the lowercase .json extension")
    _reject_symlink_components(path)
    _protect_training_inputs(path, project_root=project_root)
    if _path_exists(path):
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise BaselineCLIError("output must be a regular file")
        if not force:
            raise BaselineCLIError("output already exists; pass --force to replace it")
    return path


def _protect_training_inputs(output: Path, *, project_root: Path) -> None:
    output_key = os.path.normcase(os.path.normpath(os.fspath(output)))
    for relative in _PROTECTED_INPUTS:
        protected = _project_path(project_root, relative)
        protected_key = os.path.normcase(os.path.normpath(os.fspath(protected)))
        if output_key == protected_key:
            raise BaselineCLIError("output must not replace a reviewed training artifact")
        if _path_exists(output) and _path_exists(protected):
            try:
                if os.path.samefile(output, protected):
                    raise BaselineCLIError("output must not alias a reviewed training artifact")
            except OSError as error:
                raise BaselineCLIError("output identity could not be verified") from error


def _write_atomic(path: Path, serialized: str, *, force: bool) -> None:
    if not isinstance(serialized, str) or not serialized.endswith("\n"):
        raise BaselineCLIError("canonical experiment serialization is invalid")
    try:
        payload = serialized.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise BaselineCLIError("experiment report is not valid UTF-8") from error

    _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path.parent)
    if not path.parent.is_dir():
        raise BaselineCLIError("output parent must be a directory")

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
        _reject_symlink_components(path)
        if force:
            if _path_exists(path) and not stat.S_ISREG(path.lstat().st_mode):
                raise BaselineCLIError("output must remain a regular file")
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise BaselineCLIError(
                    "output was created concurrently; pass --force to replace it"
                ) from error
            temporary.unlink()
        committed = True
        _fsync_directory(path.parent)
    finally:
        if not committed and temporary.exists():
            temporary.unlink()


def _reject_symlink_components(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise BaselineCLIError("output path must not contain symlinks")
        if current == current.parent:
            return
        current = current.parent


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _project_path(root: Path, relative: PurePosixPath) -> Path:
    return root.joinpath(*relative.parts)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _print_summary(result: BaselineExperimentResult) -> None:
    metrics = result.holdout.overall
    r_squared = "null" if metrics.r2 is None else f"{metrics.r2:.6f}"
    print(
        f"{result.track} baseline complete | selected={result.selection.selected_model} | "
        f"train_rows={result.outer_train_sample_count} | "
        f"test_rows={result.outer_test_sample_count} | "
        f"holdout_mae_usd={metrics.mae:.2f} | holdout_rmse_usd={metrics.rmse:.2f} | "
        f"holdout_r2={r_squared}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
