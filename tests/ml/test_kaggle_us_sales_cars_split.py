from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from autovalue_ml.acquisition.sources import kaggle_us_sales_cars_split as split_module
from autovalue_ml.acquisition.sources.kaggle_us_sales_cars_split import (
    KaggleUSSalesCarsSplitError,
    build_kaggle_us_sales_cars_group_split,
    prepare_kaggle_us_sales_cars_split_training_rows,
    verify_kaggle_us_sales_cars_group_split,
)

_CANDIDATE_COLUMNS = (
    "source_id",
    "source_listing_id",
    "canonical_url",
    "observed_at",
    "market_country",
    "year",
    "make",
    "model",
    "trim",
    "mileage",
    "mileage_unit",
    "condition",
    "vehicle_status",
    "engine",
    "drivetrain",
    "accident_status",
    "accident_count",
    "owner_count",
    "vehicle_type",
    "price_cents",
    "currency",
    "price_kind",
    "sale_status",
    "raw_content_sha256",
    "parser_version",
    "normalization_version",
    "ingestion_run_id",
    "authorization_policy_id",
)


def _candidate_row(
    number: int,
    *,
    year: int | None = None,
    make: str | None = None,
    model: str | None = None,
    mileage: int | None = 10_000,
    status: str = "used",
    price_cents: int | None = None,
) -> dict[str, str]:
    return {
        "source_id": "kaggle_us_sales_cars_v2",
        "source_listing_id": f"row-{number:024x}",
        "canonical_url": f"https://www.kaggle.com/datasets/example?record={number}",
        "observed_at": "2023-12-31T23:59:59+00:00",
        "market_country": "US",
        "year": str(2000 + number % 24 if year is None else year),
        "make": make if make is not None else f"Make {number % 7}",
        "model": model if model is not None else f"Model {number}",
        "trim": "",
        "mileage": "" if mileage is None else str(mileage),
        "mileage_unit": "miles",
        "condition": "",
        "vehicle_status": status,
        "engine": "",
        "drivetrain": "",
        "accident_status": "",
        "accident_count": "",
        "owner_count": "",
        "vehicle_type": "",
        "price_cents": str(price_cents if price_cents is not None else 1_000_000 + number),
        "currency": "USD",
        "price_kind": "asking",
        "sale_status": "active",
        "raw_content_sha256": hashlib.sha256(str(number).encode()).hexdigest(),
        "parser_version": "test",
        "normalization_version": "test",
        "ingestion_run_id": "test-run",
        "authorization_policy_id": "test-policy",
    }


def _representative_rows() -> list[dict[str, str]]:
    rows = [
        _candidate_row(
            number,
            mileage=None if number % 5 == 0 else number * 1_111,
            status=("new", "used", "certified")[number % 3],
        )
        for number in range(1, 61)
    ]
    # These rows differ in opaque ID and target, but have an identical predictor tuple.
    rows.extend(
        [
            _candidate_row(
                1001,
                year=2020,
                make="Private Fixture Make",
                model="Grouped Model",
                mileage=42_000,
                status="used",
                price_cents=2_000_000,
            ),
            _candidate_row(
                1002,
                year=2020,
                make="Private Fixture Make",
                model="Grouped Model",
                mileage=42_000,
                status="used",
                price_cents=9_999_900,
            ),
        ]
    )
    return rows


def _write_candidate(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CANDIDATE_COLUMNS, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_verified_source(
    monkeypatch: pytest.MonkeyPatch,
    candidate: Path,
    *,
    approved: bool = True,
) -> Any:
    source_manifest = candidate.with_suffix(".manifest.json")
    source_manifest.write_text('{"fixture":"source manifest"}\n', encoding="utf-8")
    context = split_module._VerifiedSourceContext(
        candidate_path=candidate.resolve(),
        candidate_sha256=_sha256(candidate),
        candidate_size_bytes=candidate.stat().st_size,
        candidate_manifest_path=source_manifest.resolve(),
        candidate_manifest_sha256=_sha256(source_manifest),
        candidate_artifact_set_id="a" * 64,
        review_id="fixture-review",
        review_sha256="b" * 64,
        approved_for_ml_training=approved,
    )

    def verified_context(*args: object, **kwargs: object) -> Any:
        return context

    monkeypatch.setattr(split_module, "_verify_source_context", verified_context)
    return context


def _install_source_api_stubs(
    monkeypatch: pytest.MonkeyPatch,
    candidate: Path,
    source_manifest: Path,
    *,
    approved: bool = True,
) -> SimpleNamespace:
    review = SimpleNamespace(
        review_id="fixture-review",
        review_sha256="b" * 64,
        approved_for_ml_training=approved,
    )
    candidate_hash = _sha256(candidate)
    source_manifest.write_text(
        json.dumps(
            {
                "candidate_file": candidate.name,
                "candidate_sha256": candidate_hash,
                "candidate_size_bytes": candidate.stat().st_size,
                "source_id": "kaggle_us_sales_cars_v2",
                "target_track": "historical_us_retail_asking_price",
                "review_id": review.review_id,
                "review_sha256": review.review_sha256,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        split_module,
        "load_kaggle_us_sales_cars_review",
        lambda *args, **kwargs: review,
    )
    monkeypatch.setattr(
        split_module,
        "require_kaggle_us_sales_cars_ml_training_approval",
        lambda value: value,
    )
    monkeypatch.setattr(
        split_module,
        "verify_kaggle_us_sales_cars_artifact_set",
        lambda *args, **kwargs: {
            "candidate_sha256": candidate_hash,
            "artifact_set_id": "a" * 64,
        },
    )
    return review


def _read_assignments(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {row["source_listing_id"]: row["split"] for row in csv.DictReader(stream)}


def _rewrite_readiness(output_dir: Path) -> None:
    manifest_path = output_dir / "split_assignments.manifest.json"
    assignments_path = output_dir / "split_assignments.csv"
    readiness_path = output_dir / "split_assignments.ready.json"
    manifest_sha = _sha256(manifest_path)
    assignments_sha = _sha256(assignments_path)
    readiness = cast(dict[str, Any], json.loads(readiness_path.read_text(encoding="utf-8")))
    readiness["manifest_sha256"] = manifest_sha
    readiness["assignments_sha256"] = assignments_sha
    readiness["artifact_set_id"] = hashlib.sha256(
        "|".join((manifest_sha, assignments_sha, readiness["candidate_sha256"])).encode("ascii")
    ).hexdigest()
    readiness_path.write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def built_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Any, list[dict[str, str]]]:
    rows = _representative_rows()
    candidate = tmp_path / "asking_price_candidate.csv"
    _write_candidate(candidate, rows)
    context = _install_verified_source(monkeypatch, candidate)
    output_dir = tmp_path / "split"
    build_kaggle_us_sales_cars_group_split(
        candidate,
        context.candidate_manifest_path,
        tmp_path / "review.json",
        output_dir,
    )
    return candidate, output_dir, context, rows


def test_split_is_deterministic_target_independent_and_group_aware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _representative_rows()
    candidate = tmp_path / "first" / "asking_price_candidate.csv"
    _write_candidate(candidate, rows)
    context = _install_verified_source(monkeypatch, candidate)
    first = build_kaggle_us_sales_cars_group_split(
        candidate,
        context.candidate_manifest_path,
        tmp_path / "review.json",
        tmp_path / "first-split",
    )
    second = build_kaggle_us_sales_cars_group_split(
        candidate,
        context.candidate_manifest_path,
        tmp_path / "review.json",
        tmp_path / "second-split",
    )
    assert first.assignments_path.read_bytes() == second.assignments_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.readiness_path.read_bytes() == second.readiness_path.read_bytes()

    assignments = _read_assignments(first.assignments_path)
    assert assignments[f"row-{1001:024x}"] == assignments[f"row-{1002:024x}"]
    assert first.metrics.total_rows == len(rows)
    assert first.metrics.total_groups == len(rows) - 1
    assert first.metrics.train_rows + first.metrics.test_rows == len(rows)
    assert first.metrics.train_groups + first.metrics.test_groups == len(rows) - 1
    assert set(first.metrics.status_slices) == {"certified", "new", "used"}
    for counts in first.metrics.status_slices.values():
        assert counts["total"] == counts["train"] + counts["test"]

    target_changed = [dict(row) for row in rows]
    for number, row in enumerate(target_changed, start=1):
        row["price_cents"] = str(50_000_000 + number)
    second_candidate = tmp_path / "target-changed" / "asking_price_candidate.csv"
    _write_candidate(second_candidate, target_changed)
    second_context = _install_verified_source(monkeypatch, second_candidate)
    target_independent = build_kaggle_us_sales_cars_group_split(
        second_candidate,
        second_context.candidate_manifest_path,
        tmp_path / "review.json",
        tmp_path / "target-independent-split",
    )
    assert first.assignments_path.read_bytes() == target_independent.assignments_path.read_bytes()


def test_real_source_context_boundary_is_hash_and_review_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "asking_price_candidate.csv"
    _write_candidate(candidate, _representative_rows())
    source_manifest = tmp_path / "asking_price_candidate.manifest.json"
    review = _install_source_api_stubs(monkeypatch, candidate, source_manifest)

    context = split_module._verify_source_context(
        candidate,
        source_manifest,
        tmp_path / "review.json",
        today=None,
        require_ml=True,
    )
    assert context.candidate_path == candidate.resolve()
    assert context.candidate_sha256 == _sha256(candidate)
    assert context.review_id == review.review_id
    assert context.approved_for_ml_training is True

    artifacts = build_kaggle_us_sales_cars_group_split(
        candidate,
        source_manifest,
        tmp_path / "review.json",
        tmp_path / "split",
    )
    verified = verify_kaggle_us_sales_cars_group_split(
        artifacts.manifest_path,
        candidate,
        source_manifest,
        tmp_path / "review.json",
    )
    assert verified.candidate_sha256 == context.candidate_sha256


def test_verifier_recomputes_semantics_accounting_and_lineage(
    built_split: tuple[Path, Path, Any, list[dict[str, str]]],
) -> None:
    candidate, output_dir, context, rows = built_split
    verified = verify_kaggle_us_sales_cars_group_split(
        output_dir / "split_assignments.manifest.json",
        candidate,
        context.candidate_manifest_path,
        Path("unused-review.json"),
    )
    assert verified.metrics.total_rows == len(rows)
    assert verified.metrics.total_groups == len(rows) - 1
    assert verified.metrics.realized_test_fraction == pytest.approx(
        verified.metrics.test_rows / len(rows)
    )
    manifest = json.loads(verified.manifest_path.read_text(encoding="utf-8"))
    assert manifest["non_temporal_split"] is True
    assert "not forward-in-time" in manifest["non_temporal_limitation"]
    assert manifest["grouping"]["fields"] == [
        "year",
        "make",
        "model",
        "mileage",
        "vehicle_status",
    ]
    assert manifest["grouping"]["target_fields_excluded"] == ["price_cents"]


def test_private_assignment_artifact_contains_only_opaque_id_and_partition(
    built_split: tuple[Path, Path, Any, list[dict[str, str]]],
) -> None:
    _, output_dir, _, rows = built_split
    assignments = output_dir / "split_assignments.csv"
    with assignments.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        assert next(reader) == ["source_listing_id", "split"]
        assert all(len(row) == 2 for row in reader)
    published = assignments.read_text(encoding="utf-8")
    manifest = (output_dir / "split_assignments.manifest.json").read_text(encoding="utf-8")
    assert "Private Fixture Make" not in published + manifest
    assert all(row["model"] not in published + manifest for row in rows)
    assert all(row["price_cents"] not in published + manifest for row in rows)
    assert "raw_source_values_in_assignments" in manifest


def test_assignment_byte_tampering_fails_hash_verification(
    built_split: tuple[Path, Path, Any, list[dict[str, str]]],
) -> None:
    candidate, output_dir, context, _ = built_split
    assignments = output_dir / "split_assignments.csv"
    assignments.write_bytes(assignments.read_bytes() + b"\r\n")
    with pytest.raises(KaggleUSSalesCarsSplitError, match="assignments hash"):
        verify_kaggle_us_sales_cars_group_split(
            output_dir / "split_assignments.manifest.json",
            candidate,
            context.candidate_manifest_path,
            Path("unused-review.json"),
        )


def test_semantic_partition_tampering_fails_even_with_recomputed_hashes(
    built_split: tuple[Path, Path, Any, list[dict[str, str]]],
) -> None:
    candidate, output_dir, context, _ = built_split
    assignments_path = output_dir / "split_assignments.csv"
    with assignments_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["split"] = "test" if rows[0]["split"] == "train" else "train"
    with assignments_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("source_listing_id", "split"),
            lineterminator="\r\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = output_dir / "split_assignments.manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    manifest["assignments"]["sha256"] = _sha256(assignments_path)
    manifest["assignments"]["size_bytes"] = assignments_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_readiness(output_dir)
    with pytest.raises(KaggleUSSalesCarsSplitError, match="deterministic group policy"):
        verify_kaggle_us_sales_cars_group_split(
            manifest_path,
            candidate,
            context.candidate_manifest_path,
            Path("unused-review.json"),
        )


def test_extra_assignment_fails_accounting_even_with_recomputed_hashes(
    built_split: tuple[Path, Path, Any, list[dict[str, str]]],
) -> None:
    candidate, output_dir, context, _ = built_split
    assignments_path = output_dir / "split_assignments.csv"
    with assignments_path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(f"row-{'f' * 24},train\r\n")
    manifest_path = output_dir / "split_assignments.manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    manifest["assignments"]["sha256"] = _sha256(assignments_path)
    manifest["assignments"]["size_bytes"] = assignments_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_readiness(output_dir)
    with pytest.raises(KaggleUSSalesCarsSplitError, match="IDs absent"):
        verify_kaggle_us_sales_cars_group_split(
            manifest_path,
            candidate,
            context.candidate_manifest_path,
            Path("unused-review.json"),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["algorithm"].update({"seed": "different"}), "algorithm policy"),
        (
            lambda value: value["grouping"].update({"fields": ["year", "price_cents"]}),
            "grouping policy",
        ),
        (
            lambda value: value["privacy"].update({"target_in_assignments": True}),
            "privacy policy",
        ),
        (
            lambda value: value["source_lineage"].update({"review_sha256": "c" * 64}),
            "source lineage",
        ),
    ],
)
def test_manifest_policy_tampering_fails_after_attacker_rehashes_marker(
    built_split: tuple[Path, Path, Any, list[dict[str, str]]],
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    candidate, output_dir, context, _ = built_split
    manifest_path = output_dir / "split_assignments.manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    mutation(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_readiness(output_dir)
    with pytest.raises(KaggleUSSalesCarsSplitError, match=message):
        verify_kaggle_us_sales_cars_group_split(
            manifest_path,
            candidate,
            context.candidate_manifest_path,
            Path("unused-review.json"),
        )


def test_lazy_partition_iterators_are_disjoint_complete_and_feature_safe(
    built_split: tuple[Path, Path, Any, list[dict[str, str]]],
) -> None:
    candidate, output_dir, context, rows = built_split
    args = (
        candidate,
        context.candidate_manifest_path,
        output_dir / "split_assignments.manifest.json",
        Path("unused-review.json"),
    )
    train_stream = prepare_kaggle_us_sales_cars_split_training_rows(
        *args,
        partition="train",
    )
    test_stream = prepare_kaggle_us_sales_cars_split_training_rows(
        *args,
        partition="test",
    )
    train_rows = list(train_stream)
    test_rows = list(test_stream)
    assert len(train_rows) + len(test_rows) == len(rows)
    assert len(train_rows) == train_stream.expected_rows
    assert len(test_rows) == test_stream.expected_rows
    for features, target in train_rows + test_rows:
        assert set(features) <= {"year", "make", "model", "mileage", "vehicle_status"}
        assert "price_cents" not in features
        assert isinstance(target, float)


def test_iterator_rechecks_hashes_after_preparation(
    built_split: tuple[Path, Path, Any, list[dict[str, str]]],
) -> None:
    candidate, output_dir, context, _ = built_split
    stream = prepare_kaggle_us_sales_cars_split_training_rows(
        candidate,
        context.candidate_manifest_path,
        output_dir / "split_assignments.manifest.json",
        Path("unused-review.json"),
        partition="train",
    )
    with stream.assignments_path.open("a", encoding="utf-8") as destination:
        destination.write("tamper")
    with pytest.raises(KaggleUSSalesCarsSplitError, match="assignments hash"):
        list(stream)


def test_training_gate_rejects_unapproved_source(
    built_split: tuple[Path, Path, Any, list[dict[str, str]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, output_dir, _, _ = built_split
    context = _install_verified_source(monkeypatch, candidate, approved=False)
    with pytest.raises(KaggleUSSalesCarsSplitError, match="does not approve ML training"):
        prepare_kaggle_us_sales_cars_split_training_rows(
            candidate,
            context.candidate_manifest_path,
            output_dir / "split_assignments.manifest.json",
            Path("unused-review.json"),
            partition="train",
        )


def test_duplicate_candidate_ids_and_invalid_headers_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = _candidate_row(1)
    candidate = tmp_path / "duplicate" / "asking_price_candidate.csv"
    _write_candidate(candidate, [duplicate, dict(duplicate), _candidate_row(2)])
    context = _install_verified_source(monkeypatch, candidate)
    with pytest.raises(KaggleUSSalesCarsSplitError, match="duplicate opaque listing IDs"):
        build_kaggle_us_sales_cars_group_split(
            candidate,
            context.candidate_manifest_path,
            tmp_path / "review.json",
            tmp_path / "duplicate-split",
        )

    candidate = tmp_path / "bad-header" / "asking_price_candidate.csv"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("source_listing_id,price_cents\nrow-000000000000000000000001,1\n")
    context = _install_verified_source(monkeypatch, candidate)
    with pytest.raises(KaggleUSSalesCarsSplitError, match="header"):
        build_kaggle_us_sales_cars_group_split(
            candidate,
            context.candidate_manifest_path,
            tmp_path / "review.json",
            tmp_path / "bad-header-split",
        )


def test_invalid_partition_and_strict_json_are_rejected(
    built_split: tuple[Path, Path, Any, list[dict[str, str]]],
) -> None:
    candidate, output_dir, context, _ = built_split
    with pytest.raises(KaggleUSSalesCarsSplitError, match="partition"):
        prepare_kaggle_us_sales_cars_split_training_rows(
            candidate,
            context.candidate_manifest_path,
            output_dir / "split_assignments.manifest.json",
            Path("unused-review.json"),
            partition=cast(Any, "validation"),
        )

    manifest_path = output_dir / "split_assignments.manifest.json"
    manifest_path.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    _rewrite_readiness(output_dir)
    with pytest.raises(KaggleUSSalesCarsSplitError, match="strict UTF-8 JSON"):
        verify_kaggle_us_sales_cars_group_split(
            manifest_path,
            candidate,
            context.candidate_manifest_path,
            Path("unused-review.json"),
        )
