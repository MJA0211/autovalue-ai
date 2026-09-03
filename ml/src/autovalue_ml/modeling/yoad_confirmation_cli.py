"""CLI for the Yoad22 source-composition confirmation experiment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .baseline_cli import _validate_output_path, _validate_project_root, _write_atomic
from .yoad_confirmation import (
    canonical_confirmation_json,
    load_controlled_report,
    run_yoad_confirmation,
)
from .yoad_experiment import load_controlled_experiment_data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Confirm balanced, moderate, and full Yoad22 source compositions."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args(argv)
    root = _validate_project_root(arguments.project_root)
    output = _validate_output_path(arguments.output, project_root=root, force=arguments.force)
    controlled_path = root / "docs" / "experiments" / "yoad22-controlled-batch-v1.json"

    print("Verifying controlled evidence and source artifacts...", flush=True)
    controlled = load_controlled_report(controlled_path)
    data = load_controlled_experiment_data(root)
    print(
        "Prepared confirmation population | Cars=98552 | Yoad=242666 | "
        "balanced=98552 | moderate=150000",
        flush=True,
    )

    def progress(arm: str, fold: int, total: int) -> None:
        print(f"Confirmation fit complete | arm={arm} | fold={fold}/{total}", flush=True)

    report = run_yoad_confirmation(
        data=data,
        controlled_report=controlled,
        on_progress=progress,
    )
    _write_atomic(output, canonical_confirmation_json(report), force=arguments.force)
    print(f"Confirmation report written: {output}", flush=True)
    print(f"Decision: {report['decision']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
