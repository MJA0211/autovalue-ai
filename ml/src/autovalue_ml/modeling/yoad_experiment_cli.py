"""CLI for the isolated Yoad22 controlled batch experiment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .baseline_cli import _validate_output_path, _validate_project_root, _write_atomic
from .yoad_experiment import (
    canonical_experiment_json,
    load_controlled_experiment_data,
    run_controlled_experiment,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Cars.com development data with Cars.com plus approved Yoad22 rows."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args(argv)
    root = _validate_project_root(arguments.project_root)
    output = _validate_output_path(arguments.output, project_root=root, force=arguments.force)

    print("Loading and verifying approved source artifacts...", flush=True)
    data = load_controlled_experiment_data(root)
    counts = data.row_accounting
    print(
        "Prepared controlled data | "
        f"Cars development={counts['cars_development_rows']} | "
        f"Yoad approved={counts['yoad_approved_rows']} | "
        f"combined={counts['combined_training_rows']}",
        flush=True,
    )

    def progress(fold: int, total: int) -> None:
        print(f"Paired grouped fold complete | {fold}/{total}", flush=True)

    report = run_controlled_experiment(data, on_progress=progress)
    _write_atomic(output, canonical_experiment_json(report), force=arguments.force)
    decision = report["decision"]
    print(f"Controlled report written: {output}", flush=True)
    print(f"Decision: {decision}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
