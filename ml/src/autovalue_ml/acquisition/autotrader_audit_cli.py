"""CLI for the governed Rebrowser AutoTrader free-preview audit."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from autovalue_ml.acquisition.autotrader_audit import (
    acquire_autotrader_preview,
    audit_autotrader_preview,
    write_autotrader_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acquire and audit the pinned AutoTrader preview")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-at", type=_utc_datetime, default=None)
    arguments = parser.parse_args(argv)
    root = arguments.project_root.resolve()
    artifacts = acquire_autotrader_preview(root / "data" / "raw")
    report = audit_autotrader_preview(artifacts, generated_at=arguments.generated_at)
    output = arguments.output if arguments.output.is_absolute() else root / arguments.output
    write_autotrader_audit(report, output)
    return 0


def _utc_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("generated-at must be ISO-8601") from error
    offset = parsed.utcoffset()
    if parsed.tzinfo is None or offset is None or offset.total_seconds() != 0:
        raise argparse.ArgumentTypeError("generated-at must include a UTC offset")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
