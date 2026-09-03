"""Command-line entry point for aggregate-only River shadow simulations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from autovalue_ml.online.simulator import SimulationConfig, run_simulation_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events-per-scenario", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--rolling-window-size", type=int, default=100)
    parser.add_argument("--drift-delta", type=float, default=0.002)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_simulation_suite(
        SimulationConfig(
            events_per_scenario=args.events_per_scenario,
            seed=args.seed,
            rolling_window_size=args.rolling_window_size,
            drift_delta=args.drift_delta,
        )
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "classification": report["classification"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
