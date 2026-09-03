"""CLI boundary tests for candidate quality reports."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from autovalue_ml.acquisition import huggingface_quality_cli


def test_cli_passes_reproducible_timestamp_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    artifact = object()

    def fake_acquire(*_args: object, **kwargs: object) -> object:
        captured["acquired_at"] = kwargs["acquired_at"]
        return artifact

    def fake_profile(value: object, **kwargs: object) -> dict[str, object]:
        assert value is artifact
        captured["generated_at"] = kwargs["generated_at"]
        return {"ok": True}

    def fake_write(report: object, destination: Path) -> None:
        captured["report"] = report
        captured["destination"] = destination

    monkeypatch.setattr(huggingface_quality_cli, "acquire_huggingface_artifact", fake_acquire)
    monkeypatch.setattr(huggingface_quality_cli, "profile_huggingface_candidate", fake_profile)
    monkeypatch.setattr(huggingface_quality_cli, "write_quality_report", fake_write)

    status = huggingface_quality_cli.main(
        [
            "yoad22-craigslist",
            "--project-root",
            str(tmp_path),
            "--output",
            "report.json",
            "--generated-at",
            "2026-09-01T12:00:00+00:00",
        ]
    )

    assert status == 0
    assert captured["acquired_at"] == datetime(2026, 9, 1, 12, tzinfo=UTC)
    assert captured["generated_at"] == captured["acquired_at"]
    assert captured["destination"] == tmp_path / "report.json"


@pytest.mark.parametrize(
    "value",
    ["not-a-date", "2026-09-01T12:00:00", "2026-09-01T12:00:00-04:00"],
)
def test_cli_rejects_non_utc_timestamps(value: str) -> None:
    with pytest.raises((ValueError, pytest.UsageError, SystemExit)):
        huggingface_quality_cli.main(
            ["carson-shively", "--output", "report.json", "--generated-at", value]
        )
