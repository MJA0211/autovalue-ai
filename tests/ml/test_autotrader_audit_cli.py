from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import autovalue_ml.acquisition.autotrader_audit_cli as cli
import pytest


def test_cli_uses_acquisition_audit_and_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acquired = (object(),)
    report = {"decision": "reference/analytics only"}
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "acquire_autotrader_preview", lambda path: acquired)
    monkeypatch.setattr(
        cli,
        "audit_autotrader_preview",
        lambda artifacts, generated_at: report,
    )
    monkeypatch.setattr(
        cli,
        "write_autotrader_audit",
        lambda value, path: captured.update(report=value, path=path),
    )

    result = cli.main(
        [
            "--project-root",
            str(tmp_path),
            "--output",
            "audit.json",
            "--generated-at",
            "2026-09-02T00:00:00+00:00",
        ]
    )

    assert result == 0
    assert captured == {"report": report, "path": tmp_path / "audit.json"}


@pytest.mark.parametrize("value", ["not-a-date", "2026-09-02", "2026-09-02T00:00:00-04:00"])
def test_cli_requires_utc_generated_at(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        cli._utc_datetime(value)


def test_cli_parses_utc_generated_at() -> None:
    assert cli._utc_datetime("2026-09-02T00:00:00Z") == datetime(2026, 9, 2, tzinfo=UTC)
