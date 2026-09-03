from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from autovalue_ml.modeling import phase4_confirmation_cli


@dataclass(frozen=True)
class _FakeCandidate:
    candidate_id: str
    mae: float

    @property
    def spec(self) -> object:
        return SimpleNamespace(candidate_id=self.candidate_id)

    @property
    def overall(self) -> object:
        return SimpleNamespace(mae=self.mae)


@dataclass(frozen=True)
class _FakeReport:
    track: str
    candidates: tuple[_FakeCandidate, ...]

    @property
    def metric_ranking(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.candidates)


def _setup_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    experiments = project / "docs" / "experiments"
    experiments.mkdir(parents=True)
    (experiments / "phase4-retail-screening-v1.json").write_text(
        '{"screening":true}\n', encoding="utf-8"
    )
    return project, tmp_path / "confirmation.json"


def _install_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase4_confirmation_cli, "load_phase4_protocol", lambda path: object())
    monkeypatch.setattr(
        phase4_confirmation_cli, "parse_phase4_screening_json", lambda payload: object()
    )
    monkeypatch.setattr(
        phase4_confirmation_cli,
        "make_phase4_confirmation_checkpoint",
        lambda track, results: {"track": track, "count": len(results)},
    )
    monkeypatch.setattr(
        phase4_confirmation_cli,
        "canonical_phase4_confirmation_checkpoint_json",
        lambda value: json.dumps(value, sort_keys=True) + "\n",
    )
    monkeypatch.setattr(
        phase4_confirmation_cli,
        "canonical_phase4_confirmation_json",
        lambda value: '{"report_type":"phase4_full_development_confirmation"}\n',
    )


def test_confirmation_cli_checkpoints_progress_and_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project, output = _setup_project(tmp_path)
    _install_mocks(monkeypatch)
    candidate = _FakeCandidate("linear", 1234.0)

    def run_retail(
        project_root: Path,
        protocol: object,
        screening_report: object,
        screening_hash: str,
        completed: tuple[object, ...],
        on_progress: Any,
    ) -> _FakeReport:
        assert project_root == project
        assert protocol is not None and screening_report is not None
        assert len(screening_hash) == 64
        assert completed == ()
        on_progress((candidate,))
        return _FakeReport("retail", (candidate,))

    monkeypatch.setattr(phase4_confirmation_cli, "_run_retail", run_retail)

    assert (
        phase4_confirmation_cli.main(
            ["retail", "--project-root", str(project), "--output", str(output)]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["report_type"] == (
        "phase4_full_development_confirmation"
    )
    assert json.loads(output.with_suffix(".checkpoint.json").read_text(encoding="utf-8")) == {
        "count": 1,
        "track": "retail",
    }
    console = capsys.readouterr().out
    assert "candidate complete | 1/5" in console
    assert "cv_mae_usd=1234.00" in console


def test_confirmation_cli_rejects_output_aliasing_screening_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _ = _setup_project(tmp_path)
    _install_mocks(monkeypatch)
    screening = project / "docs" / "experiments" / "phase4-retail-screening-v1.json"

    with pytest.raises(SystemExit) as raised:
        phase4_confirmation_cli.main(
            ["retail", "--project-root", str(project), "--output", str(screening), "--force"]
        )

    assert raised.value.code == 2
