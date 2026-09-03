from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from autovalue_ml.modeling import phase4_screening_cli


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
class _FakeCheckpoint:
    track: str
    completed_candidates: tuple[_FakeCandidate, ...]


@dataclass(frozen=True)
class _FakeShortlist:
    random_forest_candidate_ids: tuple[str, str] = ("rf-00", "rf-01")
    gradient_boosting_candidate_ids: tuple[str, str] = ("gbr-00", "gbr-01")


@dataclass(frozen=True)
class _FakeReport:
    track: str
    candidates: tuple[int, ...] = tuple(range(13))
    shortlist: _FakeShortlist = _FakeShortlist()


def _install_common_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase4_screening_cli, "load_phase4_protocol", lambda path: object())
    monkeypatch.setattr(
        phase4_screening_cli,
        "make_phase4_screening_checkpoint",
        lambda track, results: {"track": track, "count": len(results)},
    )
    monkeypatch.setattr(
        phase4_screening_cli,
        "canonical_phase4_checkpoint_json",
        lambda checkpoint: json.dumps(checkpoint, sort_keys=True) + "\n",
    )
    monkeypatch.setattr(
        phase4_screening_cli,
        "canonical_phase4_screening_json",
        lambda report: '{"report_type":"phase4_screening","safe":true}\n',
    )


def test_cli_persists_aggregate_progress_then_final_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "retail-screening.json"
    _install_common_mocks(monkeypatch)

    def run_retail(
        project_root: Path,
        protocol: object,
        completed: tuple[object, ...],
        on_progress: Any,
    ) -> _FakeReport:
        assert project_root == project
        assert protocol is not None
        assert completed == ()
        on_progress((_FakeCandidate("linear", 1234.5),))
        return _FakeReport("retail")

    monkeypatch.setattr(phase4_screening_cli, "_run_retail", run_retail)

    assert (
        phase4_screening_cli.main(
            ["retail", "--project-root", str(project), "--output", str(output)]
        )
        == 0
    )

    checkpoint = output.with_suffix(".checkpoint.json")
    assert json.loads(checkpoint.read_text(encoding="utf-8")) == {
        "count": 1,
        "track": "retail",
    }
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "report_type": "phase4_screening",
        "safe": True,
    }
    console = capsys.readouterr().out
    assert "candidate complete | 1/13" in console
    assert "screening complete | candidates=13" in console


def test_cli_resumes_existing_track_bound_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "screening.json"
    checkpoint_path = output.with_suffix(".checkpoint.json")
    checkpoint_path.write_text('{"placeholder":true}\n', encoding="utf-8")
    completed = (_FakeCandidate("linear", 1000.0), _FakeCandidate("rf-00", 900.0))
    _install_common_mocks(monkeypatch)
    monkeypatch.setattr(
        phase4_screening_cli,
        "parse_phase4_checkpoint_json",
        lambda payload: _FakeCheckpoint("retail", completed),
    )

    def run_retail(
        project_root: Path,
        protocol: object,
        received: tuple[object, ...],
        on_progress: Any,
    ) -> _FakeReport:
        del project_root, protocol, on_progress
        assert received == completed
        return _FakeReport("retail")

    monkeypatch.setattr(phase4_screening_cli, "_run_retail", run_retail)

    assert (
        phase4_screening_cli.main(
            ["retail", "--project-root", str(project), "--output", str(output)]
        )
        == 0
    )
    assert "screening resume | completed=2/13 | next=3" in capsys.readouterr().out


def test_cli_rejects_checkpoint_that_aliases_final_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "same.json"
    _install_common_mocks(monkeypatch)

    with pytest.raises(SystemExit) as raised:
        phase4_screening_cli.main(
            [
                "retail",
                "--project-root",
                str(project),
                "--output",
                str(output),
                "--checkpoint",
                str(output),
            ]
        )

    assert raised.value.code == 2
