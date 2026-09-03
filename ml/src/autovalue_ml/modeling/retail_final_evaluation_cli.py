"""Execute and publish the one-time frozen retail RF05 holdout evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

from autovalue_ml.acquisition.sources.kaggle_us_sales_cars import KaggleUSSalesCarsError
from autovalue_ml.acquisition.sources.kaggle_us_sales_cars_split import (
    KaggleUSSalesCarsSplitError,
    prepare_kaggle_us_sales_cars_split_training_rows,
)

from .baseline_cli import (
    BaselineCLIError,
    RetailTrainingRow,
    _collect_retail_partition,
    _expected_count,
    _project_path,
    _validate_project_root,
)
from .calibration import retail_calibration_partition
from .calibration_artifact import (
    CALIBRATION_ASSIGNMENT_SHA256,
    RetailCalibrationArtifact,
    load_calibration_artifact,
)
from .final_evaluation_policy import (
    FINAL_EVALUATION_POLICY_SHA256,
    FinalEvaluationPolicy,
    FinalEvaluationPolicyError,
    load_final_evaluation_policy_file,
)
from .phase4_screening_cli import _RETAIL_PATHS
from .phase4_screening_experiment import _partition_hash
from .retail_calibration_experiment import CALIBRATION_SEED
from .retail_final_evaluation import (
    FINAL_GENERATED_AT,
    RF05_IDENTITY_SHA256,
    FinalEvaluationError,
    FinalEvaluationResult,
    _validate_prior_report,
    _validate_runtime_policy,
    canonical_final_report_json,
    evaluate_final_holdout,
    fit_frozen_rf05_for_final,
)

_POLICY: Final = PurePosixPath("docs/experiments/retail-rf05-final-evaluation-policy-v1.json")
_REPORT: Final = PurePosixPath("docs/experiments/retail-rf05-final-holdout-v1.report.json")
_MARKDOWN: Final = PurePosixPath("docs/experiments/retail-rf05-final-holdout-v1.md")
_MODEL_CARD: Final = PurePosixPath("docs/model-cards/autovalue-retail-rf05-v1.md")
_MANIFEST: Final = PurePosixPath("docs/experiments/retail-rf05-final-evaluation-v1.manifest.json")
_CALIBRATION_ARTIFACT: Final = PurePosixPath(
    "docs/experiments/retail-rf05-calibration-v1.artifact.json"
)
_PRIOR_REPORT: Final = PurePosixPath(
    "docs/experiments/retail-rf05-uncertainty-sharpness-v1.report.json"
)

_BOUND_FILES: Final[dict[str, tuple[PurePosixPath, str]]] = {
    "candidate": (
        PurePosixPath("data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.csv"),
        "12880cfbb2cb7f600f291c077adfa247afb9774b400b21bb7eb7409d72f7fb92",
    ),
    "candidate_manifest": (
        PurePosixPath(
            "data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.manifest.json"
        ),
        "16a707481684894e3136223985d23cb8b01e113526c7c36ed674e51685f7146d",
    ),
    "candidate_readiness": (
        PurePosixPath("data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.ready.json"),
        "0a3ab941d3145b36c6439c36c27759c1ec6273ab0dcde8e57d16d1bb5be818c7",
    ),
    "split_assignments": (
        PurePosixPath("data/processed/kaggle_us_sales_cars_v2/split/split_assignments.csv"),
        "5b3e39d0ef418c07b0c4d08ecc18700fc9f387518a21dbd604f515463cb5ebe5",
    ),
    "split_manifest": (
        PurePosixPath(
            "data/processed/kaggle_us_sales_cars_v2/split/split_assignments.manifest.json"
        ),
        "c60bf010fb47dff44d03b5da80b191ddb4b748661cb5cf02397422fdbaaf3466",
    ),
    "split_readiness": (
        PurePosixPath("data/processed/kaggle_us_sales_cars_v2/split/split_assignments.ready.json"),
        "a6e27179fbfc10c2d6e1a7ff89d21569711a0853859208ac83ee464c144fc248",
    ),
    "source_review": (
        PurePosixPath("docs/data-reviews/kaggle-us-sales-cars-v2.review.json"),
        "316f02c398339022c5667ec66a93f4e209f6386d3578c75a18efbf40d774cf76",
    ),
    "phase4_protocol": (
        PurePosixPath("docs/experiments/phase4-model-selection-v1.json"),
        "6e517acb29634d676155c80fb73f4f126db492eba12a4281e9216dc568b1d384",
    ),
    "phase4_partition_audit": (
        PurePosixPath("docs/experiments/phase4-partition-audit-v1.json"),
        "7a272ce7e99078195bf08846eaa4edc7587d376f59a9b14563a507f0b92a7a10",
    ),
    "phase4_retail_confirmation": (
        PurePosixPath("docs/experiments/phase4-retail-full-development-v1.json"),
        "07cf667e2e325f0bbb9b0fca1d62f4f3cdb54db4d607a03ff603142ee5fbc54f",
    ),
    "calibration_policy": (
        PurePosixPath("docs/experiments/retail-rf05-calibration-policy-v1.json"),
        "1398519c699bd129ef4fbb552813c064839c6c1e1c4ecd35c7f5d42bcf8e1ca2",
    ),
    "calibration_artifact": (
        _CALIBRATION_ARTIFACT,
        "b7eb5970b164ec68fb76cf8314f36080d085cda02968d3570d11f724490a6da0",
    ),
    "calibration_report": (
        PurePosixPath("docs/experiments/retail-rf05-calibration-v1.report.json"),
        "e7fafff505603669e73cfbff2fe1cf5e04f9c5d896666470fe212411aa1b3084",
    ),
    "development_diagnostics": (
        PurePosixPath("docs/experiments/retail-rf05-development-residual-diagnostics-v1.json"),
        "8f79ac027a72fff2512ab0b168d91a3a7b46677d72374dc00571a4646aac925d",
    ),
    "sharpness_policy": (
        PurePosixPath("docs/experiments/retail-rf05-uncertainty-sharpness-policy-v1.json"),
        "ec1787be963a907bbae2d1d521aeaef4239b8a5bf7816ced844dcd16902f1058",
    ),
    "sharpness_report": (
        _PRIOR_REPORT,
        "8614bad1ccd5345c64925c11e6172a7b4ef000ed6f16856aa45b48c3e4a741dd",
    ),
    "phase3_retail_baseline": (
        PurePosixPath("docs/results/retail-baseline-v1.json"),
        "b5cae941ebb01d9766716d01a24acc75ad7d0432b05e8dde44a6200caffad28a",
    ),
    "candidate_definitions_code": (
        PurePosixPath("ml/src/autovalue_ml/modeling/candidates.py"),
        "480140a97af8a230a38d56d7b037b3f393944e1f7166d3f45b1584179516d18d",
    ),
    "tree_preprocessing_code": (
        PurePosixPath("ml/src/autovalue_ml/modeling/tree_preprocessing.py"),
        "fab8894876c0eb82e9dbf8fa353d7f322d4aae2a9e73f32feb76be663473986d",
    ),
    "feature_engineering_code": (
        PurePosixPath("ml/src/autovalue_ml/modeling/feature_engineering.py"),
        "5471abae3057273733203718072cf6595c9a178f4159c4b58bb1ee9145e8c1a2",
    ),
    "feature_contract_code": (
        PurePosixPath("ml/src/autovalue_ml/modeling/contracts.py"),
        "312f8a3aaa31e3b02681611f681c22929785225c904d28bd295fb324652f7316",
    ),
    "calibration_artifact_code": (
        PurePosixPath("ml/src/autovalue_ml/modeling/calibration_artifact.py"),
        "73e6d0ae0d0ecf1a14e6b31408dd41a3ebba0ffa5e316ebcaac9d2952be839a5",
    ),
    "metrics_code": (
        PurePosixPath("ml/src/autovalue_ml/modeling/metrics.py"),
        "a498473c100719e06e8562a78ec374e86f57e51c344c6c91033fad2e64535dee",
    ),
    "calibration_experiment_code": (
        PurePosixPath("ml/src/autovalue_ml/modeling/retail_calibration_experiment.py"),
        "c29db265dc580645190b64241cd2476c82d370bd94ede955f3b06db0bb346a8c",
    ),
    "split_gate_code": (
        PurePosixPath("ml/src/autovalue_ml/acquisition/sources/kaggle_us_sales_cars_split.py"),
        "744146a22d4e26c18d049d6bdbd52cc164bbdf18f05bf6263ad7b357a8cd8a1f",
    ),
}

_IMPLEMENTATION_FILES: Final = (
    PurePosixPath("ml/src/autovalue_ml/modeling/final_evaluation_policy.py"),
    PurePosixPath("ml/src/autovalue_ml/modeling/retail_final_evaluation.py"),
    PurePosixPath("ml/src/autovalue_ml/modeling/retail_final_evaluation_cli.py"),
)


class FinalEvaluationCLIError(RuntimeError):
    """A one-time evaluation input, boundary, or publication is invalid."""


@dataclass(frozen=True, slots=True)
class _VerifiedEvidence:
    policy: FinalEvaluationPolicy
    artifact: RetailCalibrationArtifact
    prior_report: Mapping[str, object]
    file_entries: tuple[dict[str, object], ...]
    implementation_entries: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _CreatedOutput:
    path: Path
    sha256: str
    identity: tuple[int, int, int, int]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m autovalue_ml.modeling.retail_final_evaluation_cli",
        description="Run the one-time frozen RF05 final holdout evaluation.",
    )
    parser.add_argument("--project-root", required=True, metavar="PATH")
    arguments = parser.parse_args(argv)
    try:
        project_root = _validate_project_root(Path(cast(str, arguments.project_root)))
        outputs = _validated_output_paths(project_root)
        evidence = _verify_frozen_evidence(project_root)

        print("All frozen policy, source, model, and calibration bindings verified", flush=True)
        source_paths = tuple(_project_path(project_root, relative) for relative in _RETAIL_PATHS)
        training_stream = prepare_kaggle_us_sales_cars_split_training_rows(
            *source_paths,
            partition="train",
        )
        phase3_train = _collect_retail_partition(
            cast(Iterable[RetailTrainingRow], training_stream),
            expected_rows=_expected_count(training_stream, "expected_rows"),
            label="retail train",
        )
        partition = retail_calibration_partition(phase3_train.features, seed=CALIBRATION_SEED)
        assignment_hash = _partition_hash(
            partition.calibration_indices,
            population_count=len(phase3_train.features),
            selected_label="calibration",
            unselected_label="development",
        )
        if assignment_hash != CALIBRATION_ASSIGNMENT_SHA256:
            raise FinalEvaluationCLIError("calibration boundary differs from frozen evidence")
        development_features = phase3_train.features.iloc[
            partition.development_indices
        ].reset_index(drop=True)
        development_target = phase3_train.target[partition.development_indices]

        print("Opening the governed final holdout for its sole evaluation use", flush=True)
        holdout_stream = prepare_kaggle_us_sales_cars_split_training_rows(
            *source_paths,
            partition="test",
        )
        holdout = _collect_retail_partition(
            cast(Iterable[RetailTrainingRow], holdout_stream),
            expected_rows=_expected_count(holdout_stream, "expected_rows"),
            label="retail final holdout",
        )
        print("Fitting frozen RF05 on development rows only and scoring holdout", flush=True)
        predictions = fit_frozen_rf05_for_final(
            development_features=development_features,
            development_target=development_target,
            holdout_features=holdout.features,
        )
        result = evaluate_final_holdout(
            policy=evidence.policy,
            holdout_features=holdout.features,
            holdout_target=holdout.target,
            holdout_predictions=predictions,
            calibration_artifact=evidence.artifact,
            prior_sharpness_report=evidence.prior_report,
        )
        report_json = canonical_final_report_json(result.report)
        report_sha256 = _payload_sha256(report_json)
        markdown = render_final_report(result, report_sha256=report_sha256)
        markdown_sha256 = _payload_sha256(markdown)
        model_card = render_model_card(result, report_sha256=report_sha256)
        model_card_sha256 = _payload_sha256(model_card)
        _revalidate_entries(
            project_root,
            evidence.file_entries + evidence.implementation_entries,
        )
        manifest = _build_manifest(
            result=result,
            frozen_entries=evidence.file_entries,
            implementation_entries=evidence.implementation_entries,
            report_sha256=report_sha256,
            markdown_sha256=markdown_sha256,
            model_card_sha256=model_card_sha256,
        )
        manifest_json = _canonical_json(manifest)
        manifest_sha256 = _payload_sha256(manifest_json)
        _publish_outputs(
            outputs,
            report_json=report_json,
            markdown=markdown,
            model_card=model_card,
            manifest_json=manifest_json,
        )
    except (
        BaselineCLIError,
        FinalEvaluationCLIError,
        FinalEvaluationError,
        FinalEvaluationPolicyError,
        KaggleUSSalesCarsError,
        KaggleUSSalesCarsSplitError,
        OSError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(
        "retail RF05 final evaluation complete | "
        f"classification={result.classification} | "
        f"report_sha256={report_sha256} | manifest_sha256={manifest_sha256}",
        flush=True,
    )
    return 0


def _verify_frozen_evidence(project_root: Path) -> _VerifiedEvidence:
    policy_path = _project_path(project_root, _POLICY)
    policy = load_final_evaluation_policy_file(policy_path)
    if policy.policy_sha256 != FINAL_EVALUATION_POLICY_SHA256:
        raise FinalEvaluationCLIError("loaded final policy identity differs")
    _validate_policy_file_bindings(policy)
    entries = [
        _evidence_entry(
            project_root,
            "final_evaluation_policy",
            _POLICY,
            expected=FINAL_EVALUATION_POLICY_SHA256,
        )
    ]
    payloads: dict[str, bytes] = {}
    for role, (relative, expected) in _BOUND_FILES.items():
        entry, payload = _verified_evidence_file(
            project_root,
            role,
            relative,
            expected=expected,
        )
        entries.append(entry)
        if relative in {_CALIBRATION_ARTIFACT, _PRIOR_REPORT}:
            payloads[role] = payload
    artifact = load_calibration_artifact(
        payloads["calibration_artifact"],
        active_model_identity_sha256=RF05_IDENTITY_SHA256,
    )
    prior_report = _json_mapping(payloads["sharpness_report"], "sharpness report")
    _validate_runtime_policy(policy, artifact)
    _validate_prior_report(prior_report)
    implementation_entries = tuple(
        _evidence_entry(
            project_root,
            f"final_implementation_{index}",
            relative,
            expected=None,
        )
        for index, relative in enumerate(_IMPLEMENTATION_FILES, start=1)
    )
    return _VerifiedEvidence(
        policy=policy,
        artifact=artifact,
        prior_report=prior_report,
        file_entries=tuple(entries),
        implementation_entries=implementation_entries,
    )


def _validate_policy_file_bindings(policy: FinalEvaluationPolicy) -> None:
    boundary = policy.section("one_time_boundary")
    system = policy.section("frozen_system")
    uncertainty = cast(Mapping[str, object], system["uncertainty"])
    implementation = policy.section("implementation_bindings")
    upstream = policy.section("frozen_upstream_evidence")
    expected = {
        "candidate": boundary["candidate_sha256"],
        "candidate_manifest": boundary["candidate_manifest_sha256"],
        "candidate_readiness": boundary["candidate_readiness_sha256"],
        "split_assignments": boundary["split_assignments_sha256"],
        "split_manifest": boundary["split_manifest_sha256"],
        "split_readiness": boundary["split_readiness_sha256"],
        "source_review": boundary["source_review_sha256"],
        "phase4_protocol": upstream["phase4_protocol_sha256"],
        "phase4_partition_audit": upstream["phase4_partition_audit_sha256"],
        "phase4_retail_confirmation": upstream["phase4_retail_confirmation_sha256"],
        "calibration_policy": uncertainty["policy_sha256"],
        "calibration_artifact": uncertainty["artifact_sha256"],
        "calibration_report": uncertainty["report_sha256"],
        "development_diagnostics": upstream["development_diagnostics_sha256"],
        "sharpness_policy": upstream["uncertainty_sharpness_policy_sha256"],
        "sharpness_report": upstream["uncertainty_sharpness_report_sha256"],
        "phase3_retail_baseline": upstream["phase3_retail_baseline_report_sha256"],
        "candidate_definitions_code": implementation["candidate_definitions_sha256"],
        "tree_preprocessing_code": implementation["tree_preprocessing_sha256"],
        "feature_engineering_code": implementation["feature_engineering_sha256"],
        "feature_contract_code": implementation["feature_contract_sha256"],
        "calibration_artifact_code": implementation["calibration_artifact_logic_sha256"],
        "metrics_code": implementation["metric_logic_sha256"],
        "calibration_experiment_code": implementation["calibration_experiment_logic_sha256"],
        "split_gate_code": implementation["split_gate_logic_sha256"],
    }
    declared = {role: sha256 for role, (_, sha256) in _BOUND_FILES.items()}
    if declared != expected:
        raise FinalEvaluationCLIError("runner file bindings differ from frozen policy")


def _verified_evidence_file(
    project_root: Path,
    role: str,
    relative: PurePosixPath,
    *,
    expected: str,
) -> tuple[dict[str, object], bytes]:
    path = _project_path(project_root, relative)
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise FinalEvaluationCLIError(f"{role} must be a regular non-symlink file")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise FinalEvaluationCLIError(f"{role} could not be verified") from error
    if _file_identity(before) != _file_identity(after):
        raise FinalEvaluationCLIError(f"{role} changed while it was verified")
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise FinalEvaluationCLIError(f"{role} checksum differs from frozen evidence")
    return (
        {
            "role": role,
            "path": relative.as_posix(),
            "sha256": observed,
            "size_bytes": len(payload),
        },
        payload,
    )


def _evidence_entry(
    project_root: Path,
    role: str,
    relative: PurePosixPath,
    *,
    expected: str | None,
) -> dict[str, object]:
    path = _project_path(project_root, relative)
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise FinalEvaluationCLIError(f"{role} must be a regular non-symlink file")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        after = path.lstat()
    except OSError as error:
        raise FinalEvaluationCLIError(f"{role} could not be verified") from error
    if _file_identity(before) != _file_identity(after):
        raise FinalEvaluationCLIError(f"{role} changed while it was verified")
    observed = digest.hexdigest()
    if expected is not None and observed != expected:
        raise FinalEvaluationCLIError(f"{role} checksum differs from frozen evidence")
    return {
        "role": role,
        "path": relative.as_posix(),
        "sha256": observed,
        "size_bytes": size,
    }


def _json_mapping(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalEvaluationCLIError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise FinalEvaluationCLIError(f"{label} root must be a string-keyed object")
    return cast(Mapping[str, object], value)


def _revalidate_entries(
    project_root: Path,
    entries: tuple[dict[str, object], ...],
) -> None:
    for entry in entries:
        role = cast(str, entry["role"])
        relative = PurePosixPath(cast(str, entry["path"]))
        expected = cast(str, entry["sha256"])
        observed = _evidence_entry(project_root, role, relative, expected=expected)
        if observed != entry:
            raise FinalEvaluationCLIError(f"{role} evidence changed during final evaluation")


def _build_manifest(
    *,
    result: FinalEvaluationResult,
    frozen_entries: tuple[dict[str, object], ...],
    implementation_entries: tuple[dict[str, object], ...],
    report_sha256: str,
    markdown_sha256: str,
    model_card_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "manifest_type": "retail_rf05_final_evaluation_evidence",
        "generated_at": FINAL_GENERATED_AT,
        "classification": result.classification,
        "holdout": {
            "partition": "test",
            "rows": 27_589,
            "opened_once_for_final_evaluation": True,
            "future_role": "permanently_evaluation_only",
            "raw_or_row_level_evidence_persisted": False,
        },
        "frozen_inputs": list(frozen_entries),
        "final_implementation": list(implementation_entries),
        "outputs": [
            {
                "role": "machine_report",
                "path": _REPORT.as_posix(),
                "sha256": report_sha256,
            },
            {
                "role": "human_report",
                "path": _MARKDOWN.as_posix(),
                "sha256": markdown_sha256,
            },
            {
                "role": "model_card",
                "path": _MODEL_CARD.as_posix(),
                "sha256": model_card_sha256,
            },
        ],
        "governance": {
            "policy_sha256": FINAL_EVALUATION_POLICY_SHA256,
            "rf05_retuned_or_replaced": False,
            "calibration_changed": False,
            "post_holdout_optimization_performed": False,
            "model_binary_persisted": False,
            "manifest_published_last": True,
        },
    }


def _validated_output_paths(project_root: Path) -> dict[str, Path]:
    specifications = {
        "report": (_REPORT, project_root / "docs" / "experiments", ".json"),
        "markdown": (_MARKDOWN, project_root / "docs" / "experiments", ".md"),
        "model_card": (_MODEL_CARD, project_root / "docs" / "model-cards", ".md"),
        "manifest": (_MANIFEST, project_root / "docs" / "experiments", ".json"),
    }
    outputs: dict[str, Path] = {}
    for role, (relative, parent, suffix) in specifications.items():
        path = _project_path(project_root, relative)
        if path.parent != parent or path.suffix != suffix:
            raise FinalEvaluationCLIError(f"{role} output path differs from frozen policy")
        _reject_symlink_components(path)
        try:
            path.lstat()
        except FileNotFoundError:
            outputs[role] = path
            continue
        raise FinalEvaluationCLIError(f"{role} output already exists; holdout cannot be reopened")
    if len({os.path.normcase(os.path.normpath(os.fspath(path))) for path in outputs.values()}) != (
        len(outputs)
    ):
        raise FinalEvaluationCLIError("final output paths must be distinct")
    return outputs


def _publish_outputs(
    outputs: Mapping[str, Path],
    *,
    report_json: str,
    markdown: str,
    model_card: str,
    manifest_json: str,
) -> None:
    publications = (
        (outputs["report"], report_json),
        (outputs["markdown"], markdown),
        (outputs["model_card"], model_card),
        (outputs["manifest"], manifest_json),
    )
    created: list[_CreatedOutput] = []
    try:
        for path, payload in publications:
            expected = _payload_sha256(payload)
            _write_atomic(path, payload)
            output = _created_output(path, expected_sha256=expected)
            created.append(output)
            _validate_created_output(output)
    except Exception as error:
        refused = _rollback_created_outputs(created)
        if refused:
            joined = ", ".join(os.fspath(path) for path in refused)
            raise FinalEvaluationCLIError(
                f"publication failed and safe rollback refused changed outputs: {joined}"
            ) from error
        raise


def _write_atomic(path: Path, payload: str) -> None:
    encoded = payload.encode("utf-8", errors="strict")
    if not encoded:
        raise FinalEvaluationCLIError("final output payload must not be empty")
    _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    committed = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FinalEvaluationCLIError("final output was created concurrently") from error
        temporary.unlink()
        committed = True
    finally:
        if not committed and temporary.exists():
            temporary.unlink()


def _created_output(path: Path, *, expected_sha256: str) -> _CreatedOutput:
    current = path.lstat()
    if not stat.S_ISREG(current.st_mode) or stat.S_ISLNK(current.st_mode):
        raise FinalEvaluationCLIError("published output must be a regular non-symlink file")
    return _CreatedOutput(path, expected_sha256, _file_identity(current))


def _validate_created_output(output: _CreatedOutput) -> None:
    payload = output.path.read_bytes()
    after = output.path.lstat()
    if _file_identity(after) != output.identity:
        raise FinalEvaluationCLIError("published output changed while it was verified")
    if hashlib.sha256(payload).hexdigest() != output.sha256:
        raise FinalEvaluationCLIError("published output checksum differs from generated payload")


def _rollback_created_outputs(created: Sequence[_CreatedOutput]) -> tuple[Path, ...]:
    refused: list[Path] = []
    for output in reversed(created):
        try:
            before = output.path.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or _file_identity(before) != output.identity
            ):
                refused.append(output.path)
                continue
            payload = output.path.read_bytes()
            after = output.path.lstat()
            if (
                _file_identity(after) != output.identity
                or hashlib.sha256(payload).hexdigest() != output.sha256
            ):
                refused.append(output.path)
                continue
            output.path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            refused.append(output.path)
    return tuple(reversed(refused))


def _reject_symlink_components(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise FinalEvaluationCLIError("output path must not contain symlinks")
        if current == current.parent:
            return
        current = current.parent


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _payload_sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8", errors="strict")).hexdigest()


def _canonical_json(value: Mapping[str, object]) -> str:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise FinalEvaluationCLIError("evidence manifest is not JSON-safe") from error


def render_final_report(result: FinalEvaluationResult, *, report_sha256: str) -> str:
    report = result.report
    point = cast(Mapping[str, object], report["point_prediction"])
    uncertainty = cast(Mapping[str, object], report["uncertainty"])
    coverages = cast(Mapping[str, object], uncertainty["coverages"])
    comparison = cast(Mapping[str, object], report["generalization_comparison"])
    point_comparison = cast(Mapping[str, object], comparison["point"])
    confidence = cast(Mapping[str, object], report["confidence_labels"])
    manufacturer = cast(Mapping[str, object], report["manufacturer_summary"])
    lines = [
        "# Retail RF05 one-time final holdout evaluation",
        "",
        "## Decision",
        "",
        f"**{result.classification}.**",
        "",
        (
            "This is the sole final evaluation of frozen Phase 4 retail RF05 with the "
            "unchanged status-conditional conformal v1 intervals. The 27,589-row grouped "
            "holdout is now permanently evaluation-only. No model, preprocessing, quantile, "
            "bucket, confidence threshold, or source composition was changed."
        ),
        "",
        "## Point performance",
        "",
        "All dollar values are USD asking-price errors.",
        "",
        "| Rows | MAE | RMSE | R² | Median AE | p90 AE | p95 AE | Mean signed error |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {int(cast(int, point['sample_count'])):,} | {_usd(point['mae'])} | "
            f"{_usd(point['rmse'])} | {float(cast(float, point['r2'])):.4f} | "
            f"{_usd(point['median_absolute_error_usd'])} | "
            f"{_usd(_nested(point, 'absolute_error_usd', 'p90'))} | "
            f"{_usd(_nested(point, 'absolute_error_usd', 'p95'))} | "
            f"{_usd(point['mean_signed_error_usd'])} |"
        ),
        "",
        (
            f"Underpredictions: {_pct(point['underprediction_rate'])}; overpredictions: "
            f"{_pct(point['overprediction_rate'])}. MAPE remains intentionally omitted because "
            "low-dollar targets make percentage errors unstable."
        ),
        "",
        "## Frozen uncertainty performance",
        "",
        (
            "| Nominal | Empirical | Gap | Mean width | Median width | p90 width | "
            "Clipped | Fallback |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("0.8", "0.9", "0.95"):
        item = cast(Mapping[str, object], coverages[key])
        lines.append(
            f"| {_pct(item['nominal_coverage'])} | {_pct(item['empirical_coverage'])} | "
            f"{_pp(item['coverage_gap'])} | {_usd(_nested(item, 'displayed_width_usd', 'mean'))} | "
            f"{_usd(_nested(item, 'displayed_width_usd', 'median'))} | "
            f"{_usd(_nested(item, 'displayed_width_usd', 'p90'))} | "
            f"{_pct(item['lower_bound_clipping_rate'])} | {_pct(item['fallback_rate'])} |"
        )
    lines.extend(
        [
            "",
            "Coverage is the primary uncertainty criterion. Confidence is a precision/support "
            "label, not a probability that the estimate is correct.",
            "",
            "## Generalization gaps",
            "",
            "| Reference | MAE reference | MAE ratio | RMSE ratio | R² reference | R² change |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in (("Development OOF", "development_oof"), ("Calibration", "calibration")):
        item = cast(Mapping[str, object], point_comparison[key])
        lines.append(
            f"| {label} | {_usd(item['prior_mae_usd'])} | "
            f"{float(cast(float, item['mae_ratio'])):.3f}× | "
            f"{float(cast(float, item['rmse_ratio'])):.3f}× | "
            f"{float(cast(float, item['prior_r2'])):.4f} | "
            f"{float(cast(float, item['r2_difference'])):+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Confidence-label diagnostics",
            "",
            "| Label | Rows | MAE | Median AE | 90% coverage | Median 90% width |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in cast(list[Mapping[str, object]], confidence["labels"]):
        if int(cast(int, item["sample_count"])) == 0:
            lines.append(f"| {item['label']} | 0 | — | — | — | — |")
        else:
            lines.append(
                f"| {item['label']} | {int(cast(int, item['sample_count'])):,} | "
                f"{_usd(item['mae_usd'])} | {_usd(item['median_absolute_error_usd'])} | "
                f"{_pct(item['empirical_coverage_90pct'])} | "
                f"{_usd(item['median_displayed_width_usd_90pct'])} |"
            )
    lines.extend(
        [
            "",
            f"Expected High ≤ Moderate ≤ Low ordering passed: "
            f"**{str(bool(confidence['all_expected_metrics_ordered'])).lower()}**.",
            "",
            "## Manufacturer summary",
            "",
            "Only manufacturers with at least 200 holdout records are ranked.",
            "",
            "| Group | Manufacturer | Rows | MAE | RMSE | Bias | 90% coverage |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for group, key in (("Strongest", "strongest_five"), ("Weakest", "weakest_five")):
        for item in cast(list[Mapping[str, object]], manufacturer[key]):
            lines.append(
                f"| {group} | {item['manufacturer']} | "
                f"{int(cast(int, item['sample_count'])):,} | {_usd(item['mae_usd'])} | "
                f"{_usd(item['rmse_usd'])} | {_usd(item['mean_signed_error_usd'])} | "
                f"{_pct(item['coverage_90pct'])} |"
            )
    slices = cast(Mapping[str, object], report["slices"])
    for title, key in (
        ("Vehicle status", "vehicle_status"),
        ("Mileage bands", "mileage_band"),
        ("Vehicle age bands", "vehicle_age_band"),
        ("Actual asking-price bands", "actual_price_band"),
        ("Predicted-value bands", "predicted_value_band"),
        ("Mileage presence", "mileage_presence"),
    ):
        lines.extend(_slice_table(title, cast(list[Mapping[str, object]], slices[key])))
    lines.extend(
        [
            "",
            "## Interpretation and restrictions",
            "",
            "- This evaluates historical U.S. advertised asking prices in USD; it does not "
            "estimate a guaranteed sale, trade-in, auction, or KBB value.",
            "- The split is grouped but non-temporal, so future-market drift is not measured.",
            "- Trim, engine, transmission, drivetrain, condition, and vehicle history are not "
            "available to this frozen model.",
            "- Slice results are diagnostic and were not used to tune the evaluated system.",
            "- Yoad remains a separate unpromoted experiment; River remains shadow-only; "
            "AutoTrader remains separate; Carson-Shively remains excluded.",
            "- No post-holdout tuning, recalibration, promotion experiment, or model persistence "
            "is authorized by this report.",
            "",
            "## Reproducibility",
            "",
            f"Policy SHA-256: `{FINAL_EVALUATION_POLICY_SHA256}`  ",
            f"Aggregate report SHA-256: `{report_sha256}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_model_card(result: FinalEvaluationResult, *, report_sha256: str) -> str:
    report = result.report
    point = cast(Mapping[str, object], report["point_prediction"])
    uncertainty = cast(Mapping[str, object], report["uncertainty"])
    coverages = cast(Mapping[str, object], uncertainty["coverages"])
    manufacturer = cast(Mapping[str, object], report["manufacturer_summary"])
    return "\n".join(
        [
            "# Model card: AutoValue retail RF05 v1",
            "",
            "## Status",
            "",
            f"Final classification: **{result.classification}.** This card documents the frozen "
            "portfolio reference; it is not a production-readiness claim.",
            "",
            "## Model and intended use",
            "",
            "RF05 is a scikit-learn Random Forest regression pipeline for educational and "
            "portfolio demonstrations of historical U.S. advertised vehicle asking-price "
            "estimation in USD. Intended inputs are year, make, exact source model string, "
            "mileage when present, and vehicle status. It must not be presented as an appraisal, "
            "offer, guaranteed transaction price, financial advice, or current-market quote.",
            "",
            "The estimator uses 96 trees, 1,024 maximum leaf nodes, minimum leaf size 5, all "
            "transformed features per split, a 60% bootstrap sample, and random state 1254777149. "
            "Numeric imputation and categorical encoding are learned from training data only.",
            "",
            "## Data boundaries",
            "",
            "RF05 was fit on exactly 98,552 Cars.com-derived development rows. The separate "
            "10,958-row calibration partition did not fit RF05 and only created the frozen v1 "
            "conformal radii. The 27,589-row grouped final holdout was opened once, did not fit or "
            "calibrate any component, and is now permanently evaluation-only. The split is "
            "non-temporal.",
            "",
            "Yoad/Craigslist data is not part of this model. AutoTrader/KBB is governed as a "
            "different target. River is shadow-only. Carson-Shively is excluded.",
            "",
            "## Final performance",
            "",
            "| MAE | RMSE | R² | Median absolute error | Mean signed error |",
            "|---:|---:|---:|---:|---:|",
            f"| {_usd(point['mae'])} | {_usd(point['rmse'])} | "
            f"{float(cast(float, point['r2'])):.4f} | "
            f"{_usd(point['median_absolute_error_usd'])} | "
            f"{_usd(point['mean_signed_error_usd'])} |",
            "",
            "| Interval | Empirical coverage | Mean displayed width | Median displayed width |",
            "|---:|---:|---:|---:|",
            *[
                f"| {int(float(key) * 100)}% | "
                f"{_pct(cast(Mapping[str, object], coverages[key])['empirical_coverage'])} | "
                f"{_usd(_coverage_width(coverages, key, 'mean'))} | "
                f"{_usd(_coverage_width(coverages, key, 'median'))} |"
                for key in ("0.8", "0.9", "0.95")
            ],
            "",
            "Confidence labels communicate relative interval precision and calibration support; "
            "they are not probabilities of correctness. Data-quality warnings are separate.",
            "",
            "## Subgroup evaluation and risks",
            "",
            f"{int(cast(int, manufacturer['supported_manufacturer_count']))} manufacturers met the "
            "200-row reporting threshold. Full manufacturer, vehicle-status, mileage, age, actual "
            "price, predicted-value, and mileage-missingness results are in the final report. "
            "Subgroup metrics with lower support are omitted rather than treated as reliable.",
            "",
            "Known limitations include omitted trim and mechanical/history attributes, "
            "asking-price "
            "rather than completed-sale labels, historical data, non-temporal validation, broad "
            "intervals for some vehicles, uneven manufacturer support, and possible "
            "source-specific "
            "selection bias. Users must see the interval, confidence semantics, and limitations "
            "alongside a point estimate.",
            "",
            "## Governance and reproducibility",
            "",
            "The policy, RF05 definition, training boundary, preprocessing, calibration artifact, "
            "coverage levels, confidence thresholds, metrics, slices, and decision gates were "
            "frozen before holdout access. The holdout result cannot authorize tuning or "
            "recalibration. No trained model binary or row-level holdout evidence was persisted.",
            "",
            f"Policy SHA-256: `{FINAL_EVALUATION_POLICY_SHA256}`  ",
            f"Aggregate final report SHA-256: `{report_sha256}`",
            "",
        ]
    )


def _slice_table(title: str, items: list[Mapping[str, object]]) -> list[str]:
    lines = [
        "",
        f"### {title}",
        "",
        "| Slice | Rows | MAE | RMSE | R² | Bias | 90% coverage | Mean 90% width |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in items:
        interval = cast(Mapping[str, object], item["interval_90pct"])
        lines.append(
            f"| {item['label']} | {int(cast(int, item['sample_count'])):,} | "
            f"{_usd(item['mae_usd'])} | {_usd(item['rmse_usd'])} | "
            f"{float(cast(float, item['r2'])):.4f} | "
            f"{_usd(item['mean_signed_error_usd'])} | "
            f"{_pct(interval['empirical_coverage'])} | "
            f"{_usd(interval['mean_displayed_width_usd'])} |"
        )
    return lines


def _nested(parent: Mapping[str, object], outer: str, inner: str) -> object:
    return cast(Mapping[str, object], parent[outer])[inner]


def _coverage_width(coverages: Mapping[str, object], key: str, statistic: str) -> object:
    item = cast(Mapping[str, object], coverages[key])
    return _nested(item, "displayed_width_usd", statistic)


def _usd(value: object) -> str:
    number = float(cast(float, value))
    sign = "-" if number < 0.0 else ""
    return f"{sign}${abs(number):,.2f}"


def _pct(value: object) -> str:
    return f"{float(cast(float, value)) * 100:.2f}%"


def _pp(value: object) -> str:
    return f"{float(cast(float, value)) * 100:+.2f} pp"


if __name__ == "__main__":
    raise SystemExit(main())
