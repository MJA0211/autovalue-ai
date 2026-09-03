"""Command-line entry point for reproducible Hugging Face candidate audits."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Final

from autovalue_ml.acquisition.huggingface_dataset import acquire_huggingface_artifact
from autovalue_ml.acquisition.huggingface_quality import (
    profile_huggingface_candidate,
    write_quality_report,
)
from autovalue_ml.acquisition.sources.huggingface_candidates import (
    CARSON_SHIVELY_SPEC,
    YOAD22_CRAIGSLIST_SPEC,
)

_SPECS: Final = {
    "yoad22-craigslist": YOAD22_CRAIGSLIST_SPEC,
    "carson-shively": CARSON_SHIVELY_SPEC,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Acquire and profile one reviewed Hugging Face candidate."
    )
    parser.add_argument("candidate", choices=tuple(_SPECS))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--generated-at",
        type=_utc_datetime,
        default=None,
        help="Optional reproducible UTC timestamp, for example 2026-09-01T12:00:00+00:00.",
    )
    arguments = parser.parse_args(argv)
    root = arguments.project_root.resolve()
    artifact = acquire_huggingface_artifact(
        _SPECS[arguments.candidate],
        root / "data" / "raw",
        acquired_at=arguments.generated_at,
    )
    report = profile_huggingface_candidate(artifact, generated_at=arguments.generated_at)
    output = arguments.output
    if not output.is_absolute():
        output = root / output
    write_quality_report(report, output)
    return 0


def _utc_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("generated-at must be an ISO-8601 datetime") from error
    offset = parsed.utcoffset()
    if parsed.tzinfo is None or offset is None or offset.total_seconds() != 0:
        raise argparse.ArgumentTypeError("generated-at must include a UTC offset")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
