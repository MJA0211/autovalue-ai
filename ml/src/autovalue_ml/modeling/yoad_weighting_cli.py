"""CLI for the resumable Yoad22 training-weight confirmation."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from .baseline_cli import _validate_output_path, _validate_project_root, _write_atomic
from .yoad_experiment import load_controlled_experiment_data
from .yoad_weighting import (
    YoadWeightingError,
    canonical_weighting_json,
    load_confirmation_report,
    make_weighting_checkpoint,
    parse_weighting_checkpoint_json,
    run_weighting_experiment,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fold-local source, mileage, and segment weights on the frozen "
            "moderate Yoad22 composition."
        )
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args(argv)

    root = _validate_project_root(cast(Path, arguments.project_root))
    output = _validate_output_path(
        cast(Path, arguments.output),
        project_root=root,
        force=cast(bool, arguments.force),
    )
    checkpoint_argument = cast(Path | None, arguments.checkpoint)
    checkpoint = _validate_output_path(
        checkpoint_argument
        if checkpoint_argument is not None
        else output.with_suffix(".checkpoint.json"),
        project_root=root,
        force=True,
    )
    if _path_key(output) == _path_key(checkpoint):
        parser.error("output and checkpoint paths must be distinct")

    confirmation_path = (
        root / "docs" / "experiments" / "yoad22-source-composition-confirmation-v1.json"
    )
    print("Verifying frozen confirmation and source artifacts...", flush=True)
    confirmation = load_confirmation_report(confirmation_path)
    completed: tuple[Mapping[str, object], ...] = ()
    if checkpoint.exists():
        completed = parse_weighting_checkpoint_json(checkpoint.read_bytes())
        print(f"Weighting resume | completed={len(completed)}/15", flush=True)
    data = load_controlled_experiment_data(root)
    print(
        "Prepared fixed moderate composition | Cars=98552 | Yoad=150000 | "
        "validation=341218 | weighted_fits=15",
        flush=True,
    )

    def persist(progress: tuple[Mapping[str, object], ...]) -> None:
        latest = progress[-1]
        _write_atomic(
            checkpoint,
            canonical_weighting_json(make_weighting_checkpoint(progress)),
            force=True,
        )
        print(
            "Weighted fit complete | "
            f"treatment={latest['treatment']} | fold={latest['fold']}/5 | "
            f"completed={len(progress)}/15",
            flush=True,
        )

    try:
        report = run_weighting_experiment(
            data=data,
            confirmation_report=confirmation,
            completed_fits=completed,
            on_progress=persist,
        )
        _write_atomic(
            output,
            canonical_weighting_json(report),
            force=cast(bool, arguments.force),
        )
    except YoadWeightingError as error:
        parser.error(str(error))
    decision = cast(Mapping[str, object], report["decision"])
    print(f"Weighting report written: {output}", flush=True)
    print(
        f"Decision: {decision['classification']} | preferred={decision['preferred_treatment']}",
        flush=True,
    )
    return 0


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(path))))


if __name__ == "__main__":
    raise SystemExit(main())
