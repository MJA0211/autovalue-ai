"""Strict loader for the frozen retail RF05 uncertainty-sharpness policy."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

UNCERTAINTY_SHARPNESS_POLICY_SHA256: Final = (
    "ec1787be963a907bbae2d1d521aeaef4239b8a5bf7816ced844dcd16902f1058"
)
_MAX_POLICY_BYTES: Final = 100_000
_DIGEST_PATTERN: Final = re.compile(r"^[a-f0-9]{64}$")


class UncertaintySharpnessPolicyError(ValueError):
    """The policy is malformed or differs from its frozen preregistration."""


@dataclass(frozen=True, slots=True)
class FrozenInputs:
    """Immutable identities and population sizes bound by the experiment."""

    phase4_protocol_sha256: str
    phase4_retail_confirmation_sha256: str
    rf05_identity_sha256: str
    rf05_candidate_id: str
    rf05_parameters: tuple[int, int, int, float, float]
    rf05_random_state: int
    feature_contract_version: str
    calibration_v1_policy_sha256: str
    calibration_v1_artifact_sha256: str
    calibration_v1_report_sha256: str
    development_residual_diagnostics_sha256: str
    calibration_assignment_sha256: str
    phase3_train_rows: int
    development_rows: int
    calibration_rows: int
    legacy_holdout_rows: int


@dataclass(frozen=True, slots=True)
class BoundaryInterpretation:
    """Allowed reconstruction and prohibited data-use behavior."""

    no_persisted_rf05_or_row_level_predictions_exist: bool
    authorized_reconstruction: str
    reconstruction_is_not: tuple[str, ...]
    outer_train_loader: str
    legacy_holdout_definition: str
    calibration_targets_fit_rf05: bool
    calibration_targets_fit_residual_scale_model: bool
    calibration_targets_select_scale_model_hyperparameters: bool
    raw_rows_predictions_or_residuals_persisted: bool


@dataclass(frozen=True, slots=True)
class DevelopmentDiagnosticsPolicy:
    """Training-side residual-diagnostic boundary used before preregistration."""

    population: str
    folds: str
    actual_price_role: str
    minimum_manufacturer_support: int
    minimum_model_support: int
    minimum_combination_support: int
    maximum_reported_manufacturers_or_models: int
    heteroscedasticity_gate: str
    observed_ratio_before_preregistration: float


@dataclass(frozen=True, slots=True)
class BaselineMethodPolicy:
    """Frozen absolute-residual conformal baseline."""

    method_id: str
    role: str
    scale: str
    score: str
    quantile_hierarchy: tuple[str, ...]
    minimum_status_support: int


@dataclass(frozen=True, slots=True)
class GammaHyperparameters:
    """Exact, untuned Gamma scale-model parameters."""

    alpha: float
    fit_intercept: bool
    solver: str
    max_iter: int
    tol: float
    warm_start: bool


@dataclass(frozen=True, slots=True)
class GammaScaleMethodPolicy:
    """Development-only learned residual-scale candidate."""

    method_id: str
    role: str
    scale_target: str
    scale_inputs: tuple[str, ...]
    preprocessing: str
    estimator: str
    hyperparameters: GammaHyperparameters
    scale_floor_usd: float
    scale_cap_usd: float
    score: str
    quantile_hierarchy: tuple[str, ...]
    minimum_status_support: int
    fit_boundary: str


@dataclass(frozen=True, slots=True)
class SmoothValueScaleMethodPolicy:
    """Unfitted smooth predicted-value scale candidate."""

    method_id: str
    role: str
    scale_formula: str
    scale_fit: str
    score: str
    quantile_hierarchy: tuple[str, ...]
    minimum_status_support: int


@dataclass(frozen=True, slots=True)
class BootstrapPolicy:
    """Paired predictor-group bootstrap configuration."""

    unit: str
    replicates: int
    random_state: int
    confidence_level: float
    paired_across_methods: bool


@dataclass(frozen=True, slots=True)
class CalibrationComparisonPolicy:
    """Common calibration folds, metrics, slices, and physical bounds."""

    coverage_levels: tuple[float, float, float]
    finite_sample_order: str
    scheme: str
    same_folds_and_point_predictions_for_every_method: bool
    fold_quantiles_use_other_four_calibration_folds_only: bool
    lower_bound: str
    upper_bound: str
    sharpness_primary_width: str
    displayed_width: str
    minimum_reported_slice_support: int
    price_bands_usd: tuple[int, int, int]
    mileage_bands_miles: tuple[int, int, int]
    vehicle_age_reference_year: int
    vehicle_age_bands_years: tuple[int, int, int]
    manufacturer_support: int
    bootstrap: BootstrapPolicy


@dataclass(frozen=True, slots=True)
class ValidityGates:
    invalid_or_nonfinite_intervals: int
    reversed_or_point_excluding_intervals: int
    negative_displayed_lower_bounds: int
    clipped_and_unclipped_coverage_must_match: bool
    gamma_scale_floor_hit_maximum_rate: float
    gamma_scale_cap_hit_maximum_rate: float


@dataclass(frozen=True, slots=True)
class OverallCoverageGates:
    minimum_gap_from_target_each_level: float
    maximum_regression_vs_baseline_each_level: float
    minimum_cluster_bootstrap_95pct_lower_delta_vs_baseline: float


@dataclass(frozen=True, slots=True)
class SharpnessGates:
    minimum_unclipped_mean_width_reduction_80pct: float
    minimum_unclipped_mean_width_reduction_90pct: float
    minimum_unclipped_mean_width_reduction_95pct: float
    minimum_displayed_median_width_reduction_each_level: float
    maximum_bootstrap_95pct_upper_mean_width_ratio_at_90pct: float
    maximum_p95_width_ratio_vs_baseline_each_level: float


@dataclass(frozen=True, slots=True)
class ConditionalCoverageGates:
    minimum_status_gap_from_target_each_level: float
    maximum_status_regression_vs_baseline_each_level: float
    maximum_broad_slice_regression_vs_baseline_each_level: float
    new_broad_slice_undercoverage_boundary: float
    maximum_manufacturer_regression_vs_baseline_at_90pct: float
    manufacturer_count_below_80pct_at_90pct_may_increase: bool
    focus_90pct_maximum_regression_vs_baseline: float
    focus_slices: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StabilityGates:
    maximum_fold_coverage_regression_vs_baseline_each_level: float
    maximum_fold_coverage_sd_increase_vs_baseline_each_level: float
    maximum_fallback_rate_increase_vs_baseline: float
    maximum_p95_to_median_width_ratio: float
    maximum_interval_width_usd: float


@dataclass(frozen=True, slots=True)
class AcceptanceGates:
    validity: ValidityGates
    overall_coverage: OverallCoverageGates
    sharpness: SharpnessGates
    conditional_coverage: ConditionalCoverageGates
    stability: StabilityGates


@dataclass(frozen=True, slots=True)
class SelectionRule:
    candidate_must_pass_every_gate: bool
    if_neither_passes: str
    if_execution_or_baseline_reproduction_fails: str
    if_both_pass: str
    coverage_is_primary: bool
    do_not_force_a_winner: bool


@dataclass(frozen=True, slots=True)
class ConfidencePolicy:
    coverage_level: float
    relative_width: str
    high_max_relative_width: float
    moderate_max_relative_width: float
    high_minimum_support: int
    moderate_minimum_support: int
    thresholds_reused_from_v1_for_direct_comparability: bool
    confidence_is_not_a_probability: bool
    data_quality_warnings_remain_separate: bool


@dataclass(frozen=True, slots=True)
class PublicationPolicy:
    reports_are_aggregate_only: bool
    persist_new_serving_artifact_only_if_candidate_passes_every_gate: bool
    preserve_calibration_v1_artifacts: bool
    preserve_phase4_yoad_river_autotrader_and_carson_artifacts: bool
    legacy_holdout_access: bool


@dataclass(frozen=True, slots=True)
class UncertaintySharpnessPolicy:
    """Fully typed view of the immutable preregistered policy."""

    schema_version: int
    policy_id: str
    preregistered_on: str
    objective: str
    scope: str
    frozen_inputs: FrozenInputs
    boundary_interpretation: BoundaryInterpretation
    development_diagnostics: DevelopmentDiagnosticsPolicy
    baseline_method: BaselineMethodPolicy
    gamma_method: GammaScaleMethodPolicy
    smooth_value_method: SmoothValueScaleMethodPolicy
    calibration_comparison: CalibrationComparisonPolicy
    acceptance_gates: AcceptanceGates
    selection_rule: SelectionRule
    confidence_policy: ConfidencePolicy
    publication: PublicationPolicy
    policy_sha256: str = UNCERTAINTY_SHARPNESS_POLICY_SHA256

    @property
    def candidate_ids(self) -> tuple[str, str, str]:
        """Return the exact candidate order used by the governed comparison."""

        return (
            self.baseline_method.method_id,
            self.gamma_method.method_id,
            self.smooth_value_method.method_id,
        )


def load_uncertainty_sharpness_policy(
    serialized: str | bytes,
) -> UncertaintySharpnessPolicy:
    """Parse only the byte-exact frozen policy and return immutable typed state."""

    payload_bytes, text = _bounded_utf8(serialized)
    try:
        raw = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise UncertaintySharpnessPolicyError("policy is not valid JSON") from error
    policy = _parse_policy(raw)
    observed = hashlib.sha256(payload_bytes).hexdigest()
    if observed != UNCERTAINTY_SHARPNESS_POLICY_SHA256:
        raise UncertaintySharpnessPolicyError(
            "policy bytes differ from the immutable preregistered policy"
        )
    return policy


def load_uncertainty_sharpness_policy_file(path: str | Path) -> UncertaintySharpnessPolicy:
    """Read a stable regular file and validate it against the frozen checksum."""

    resolved = Path(path)
    try:
        before = resolved.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise UncertaintySharpnessPolicyError("policy path must be a regular non-symlink file")
        payload = resolved.read_bytes()
        after = resolved.lstat()
    except OSError as error:
        raise UncertaintySharpnessPolicyError("policy file could not be read") from error
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise UncertaintySharpnessPolicyError("policy file changed while it was read")
    return load_uncertainty_sharpness_policy(payload)


def _parse_policy(value: object) -> UncertaintySharpnessPolicy:
    root = _exact_mapping(
        value,
        {
            "schema_version",
            "policy_id",
            "preregistered_on",
            "objective",
            "scope",
            "frozen_inputs",
            "boundary_interpretation",
            "development_diagnostics",
            "candidate_methods",
            "calibration_comparison",
            "acceptance_gates",
            "selection_rule",
            "confidence_policy",
            "publication",
        },
        label="policy",
    )
    methods = root["candidate_methods"]
    if not isinstance(methods, list) or len(methods) != 3:
        raise UncertaintySharpnessPolicyError("candidate_methods must contain exactly three items")
    policy = UncertaintySharpnessPolicy(
        schema_version=_integer(root["schema_version"], label="schema_version"),
        policy_id=_text(root["policy_id"], label="policy_id"),
        preregistered_on=_text(root["preregistered_on"], label="preregistered_on"),
        objective=_text(root["objective"], label="objective"),
        scope=_text(root["scope"], label="scope"),
        frozen_inputs=_parse_frozen_inputs(root["frozen_inputs"]),
        boundary_interpretation=_parse_boundary(root["boundary_interpretation"]),
        development_diagnostics=_parse_diagnostics(root["development_diagnostics"]),
        baseline_method=_parse_baseline(methods[0]),
        gamma_method=_parse_gamma(methods[1]),
        smooth_value_method=_parse_smooth(methods[2]),
        calibration_comparison=_parse_comparison(root["calibration_comparison"]),
        acceptance_gates=_parse_gates(root["acceptance_gates"]),
        selection_rule=_parse_selection(root["selection_rule"]),
        confidence_policy=_parse_confidence(root["confidence_policy"]),
        publication=_parse_publication(root["publication"]),
    )
    _validate_fixed_policy(policy)
    return policy


def _parse_frozen_inputs(value: object) -> FrozenInputs:
    keys = {
        "phase4_protocol_sha256",
        "phase4_retail_confirmation_sha256",
        "rf05_identity_sha256",
        "rf05_candidate_id",
        "rf05_parameters",
        "rf05_random_state",
        "feature_contract_version",
        "calibration_v1_policy_sha256",
        "calibration_v1_artifact_sha256",
        "calibration_v1_report_sha256",
        "development_residual_diagnostics_sha256",
        "calibration_assignment_sha256",
        "phase3_train_rows",
        "development_rows",
        "calibration_rows",
        "legacy_holdout_rows",
    }
    item = _exact_mapping(value, keys, label="frozen_inputs")
    parameters = item["rf05_parameters"]
    if not isinstance(parameters, list) or len(parameters) != 5:
        raise UncertaintySharpnessPolicyError("rf05_parameters must contain five values")
    return FrozenInputs(
        phase4_protocol_sha256=_digest(item["phase4_protocol_sha256"], label="phase4 protocol"),
        phase4_retail_confirmation_sha256=_digest(
            item["phase4_retail_confirmation_sha256"], label="Phase 4 confirmation"
        ),
        rf05_identity_sha256=_digest(item["rf05_identity_sha256"], label="RF05 identity"),
        rf05_candidate_id=_text(item["rf05_candidate_id"], label="RF05 candidate"),
        rf05_parameters=(
            _integer(parameters[0], label="n_estimators"),
            _integer(parameters[1], label="max_leaf_nodes"),
            _integer(parameters[2], label="min_samples_leaf"),
            _number(parameters[3], label="max_features"),
            _number(parameters[4], label="max_samples"),
        ),
        rf05_random_state=_integer(item["rf05_random_state"], label="RF05 random state"),
        feature_contract_version=_text(
            item["feature_contract_version"], label="feature contract version"
        ),
        calibration_v1_policy_sha256=_digest(
            item["calibration_v1_policy_sha256"], label="calibration v1 policy"
        ),
        calibration_v1_artifact_sha256=_digest(
            item["calibration_v1_artifact_sha256"], label="calibration v1 artifact"
        ),
        calibration_v1_report_sha256=_digest(
            item["calibration_v1_report_sha256"], label="calibration v1 report"
        ),
        development_residual_diagnostics_sha256=_digest(
            item["development_residual_diagnostics_sha256"], label="residual diagnostics"
        ),
        calibration_assignment_sha256=_digest(
            item["calibration_assignment_sha256"], label="calibration assignment"
        ),
        phase3_train_rows=_integer(item["phase3_train_rows"], label="Phase 3 rows"),
        development_rows=_integer(item["development_rows"], label="development rows"),
        calibration_rows=_integer(item["calibration_rows"], label="calibration rows"),
        legacy_holdout_rows=_integer(item["legacy_holdout_rows"], label="legacy holdout rows"),
    )


def _parse_boundary(value: object) -> BoundaryInterpretation:
    keys = {
        "no_persisted_rf05_or_row_level_predictions_exist",
        "authorized_reconstruction",
        "reconstruction_is_not",
        "outer_train_loader",
        "legacy_holdout_definition",
        "calibration_targets_fit_rf05",
        "calibration_targets_fit_residual_scale_model",
        "calibration_targets_select_scale_model_hyperparameters",
        "raw_rows_predictions_or_residuals_persisted",
    }
    item = _exact_mapping(value, keys, label="boundary_interpretation")
    return BoundaryInterpretation(
        no_persisted_rf05_or_row_level_predictions_exist=_boolean(
            item["no_persisted_rf05_or_row_level_predictions_exist"], label="persisted RF05 flag"
        ),
        authorized_reconstruction=_text(
            item["authorized_reconstruction"], label="authorized reconstruction"
        ),
        reconstruction_is_not=_text_tuple(
            item["reconstruction_is_not"], label="reconstruction exclusions"
        ),
        outer_train_loader=_text(item["outer_train_loader"], label="outer train loader"),
        legacy_holdout_definition=_text(
            item["legacy_holdout_definition"], label="legacy holdout definition"
        ),
        calibration_targets_fit_rf05=_boolean(
            item["calibration_targets_fit_rf05"], label="calibration RF05 fit flag"
        ),
        calibration_targets_fit_residual_scale_model=_boolean(
            item["calibration_targets_fit_residual_scale_model"],
            label="calibration scale fit flag",
        ),
        calibration_targets_select_scale_model_hyperparameters=_boolean(
            item["calibration_targets_select_scale_model_hyperparameters"],
            label="calibration scale selection flag",
        ),
        raw_rows_predictions_or_residuals_persisted=_boolean(
            item["raw_rows_predictions_or_residuals_persisted"], label="raw persistence flag"
        ),
    )


def _parse_diagnostics(value: object) -> DevelopmentDiagnosticsPolicy:
    keys = {
        "population",
        "folds",
        "actual_price_role",
        "minimum_manufacturer_support",
        "minimum_model_support",
        "minimum_combination_support",
        "maximum_reported_manufacturers_or_models",
        "heteroscedasticity_gate",
        "observed_ratio_before_preregistration",
    }
    item = _exact_mapping(value, keys, label="development_diagnostics")
    return DevelopmentDiagnosticsPolicy(
        population=_text(item["population"], label="diagnostic population"),
        folds=_text(item["folds"], label="diagnostic folds"),
        actual_price_role=_text(item["actual_price_role"], label="actual price role"),
        minimum_manufacturer_support=_integer(
            item["minimum_manufacturer_support"], label="manufacturer support"
        ),
        minimum_model_support=_integer(item["minimum_model_support"], label="model support"),
        minimum_combination_support=_integer(
            item["minimum_combination_support"], label="combination support"
        ),
        maximum_reported_manufacturers_or_models=_integer(
            item["maximum_reported_manufacturers_or_models"], label="maximum reported groups"
        ),
        heteroscedasticity_gate=_text(
            item["heteroscedasticity_gate"], label="heteroscedasticity gate"
        ),
        observed_ratio_before_preregistration=_number(
            item["observed_ratio_before_preregistration"], label="observed ratio"
        ),
    )


def _parse_baseline(value: object) -> BaselineMethodPolicy:
    item = _exact_mapping(
        value,
        {"id", "role", "scale", "score", "quantile_hierarchy", "minimum_status_support"},
        label="baseline method",
    )
    return BaselineMethodPolicy(
        method_id=_text(item["id"], label="baseline id"),
        role=_text(item["role"], label="baseline role"),
        scale=_text(item["scale"], label="baseline scale"),
        score=_text(item["score"], label="baseline score"),
        quantile_hierarchy=_text_tuple(
            item["quantile_hierarchy"], label="baseline quantile hierarchy"
        ),
        minimum_status_support=_integer(
            item["minimum_status_support"], label="baseline status support"
        ),
    )


def _parse_gamma(value: object) -> GammaScaleMethodPolicy:
    keys = {
        "id",
        "role",
        "scale_target",
        "scale_inputs",
        "preprocessing",
        "estimator",
        "hyperparameters",
        "scale_floor_usd",
        "scale_cap_usd",
        "score",
        "quantile_hierarchy",
        "minimum_status_support",
        "fit_boundary",
    }
    item = _exact_mapping(value, keys, label="Gamma method")
    hyper = _exact_mapping(
        item["hyperparameters"],
        {"alpha", "fit_intercept", "solver", "max_iter", "tol", "warm_start"},
        label="Gamma hyperparameters",
    )
    return GammaScaleMethodPolicy(
        method_id=_text(item["id"], label="Gamma id"),
        role=_text(item["role"], label="Gamma role"),
        scale_target=_text(item["scale_target"], label="Gamma scale target"),
        scale_inputs=_text_tuple(item["scale_inputs"], label="Gamma scale inputs"),
        preprocessing=_text(item["preprocessing"], label="Gamma preprocessing"),
        estimator=_text(item["estimator"], label="Gamma estimator"),
        hyperparameters=GammaHyperparameters(
            alpha=_number(hyper["alpha"], label="Gamma alpha"),
            fit_intercept=_boolean(hyper["fit_intercept"], label="Gamma fit_intercept"),
            solver=_text(hyper["solver"], label="Gamma solver"),
            max_iter=_integer(hyper["max_iter"], label="Gamma max_iter"),
            tol=_number(hyper["tol"], label="Gamma tolerance"),
            warm_start=_boolean(hyper["warm_start"], label="Gamma warm_start"),
        ),
        scale_floor_usd=_number(item["scale_floor_usd"], label="Gamma scale floor"),
        scale_cap_usd=_number(item["scale_cap_usd"], label="Gamma scale cap"),
        score=_text(item["score"], label="Gamma score"),
        quantile_hierarchy=_text_tuple(
            item["quantile_hierarchy"], label="Gamma quantile hierarchy"
        ),
        minimum_status_support=_integer(
            item["minimum_status_support"], label="Gamma status support"
        ),
        fit_boundary=_text(item["fit_boundary"], label="Gamma fit boundary"),
    )


def _parse_smooth(value: object) -> SmoothValueScaleMethodPolicy:
    item = _exact_mapping(
        value,
        {
            "id",
            "role",
            "scale_formula",
            "scale_fit",
            "score",
            "quantile_hierarchy",
            "minimum_status_support",
        },
        label="smooth-value method",
    )
    return SmoothValueScaleMethodPolicy(
        method_id=_text(item["id"], label="smooth-value id"),
        role=_text(item["role"], label="smooth-value role"),
        scale_formula=_text(item["scale_formula"], label="smooth-value formula"),
        scale_fit=_text(item["scale_fit"], label="smooth-value fit"),
        score=_text(item["score"], label="smooth-value score"),
        quantile_hierarchy=_text_tuple(
            item["quantile_hierarchy"], label="smooth-value quantile hierarchy"
        ),
        minimum_status_support=_integer(
            item["minimum_status_support"], label="smooth-value status support"
        ),
    )


def _parse_comparison(value: object) -> CalibrationComparisonPolicy:
    keys = {
        "coverage_levels",
        "finite_sample_order",
        "scheme",
        "same_folds_and_point_predictions_for_every_method",
        "fold_quantiles_use_other_four_calibration_folds_only",
        "lower_bound",
        "upper_bound",
        "sharpness_primary_width",
        "displayed_width",
        "minimum_reported_slice_support",
        "price_bands_usd",
        "mileage_bands_miles",
        "vehicle_age_reference_year",
        "vehicle_age_bands_years",
        "manufacturer_support",
        "bootstrap",
    }
    item = _exact_mapping(value, keys, label="calibration_comparison")
    bootstrap = _exact_mapping(
        item["bootstrap"],
        {"unit", "replicates", "random_state", "confidence_level", "paired_across_methods"},
        label="bootstrap",
    )
    return CalibrationComparisonPolicy(
        coverage_levels=_float_triplet(item["coverage_levels"], label="coverage levels"),
        finite_sample_order=_text(item["finite_sample_order"], label="finite sample order"),
        scheme=_text(item["scheme"], label="calibration scheme"),
        same_folds_and_point_predictions_for_every_method=_boolean(
            item["same_folds_and_point_predictions_for_every_method"],
            label="same folds and predictions",
        ),
        fold_quantiles_use_other_four_calibration_folds_only=_boolean(
            item["fold_quantiles_use_other_four_calibration_folds_only"],
            label="fold-local quantile flag",
        ),
        lower_bound=_text(item["lower_bound"], label="lower bound"),
        upper_bound=_text(item["upper_bound"], label="upper bound"),
        sharpness_primary_width=_text(
            item["sharpness_primary_width"], label="sharpness primary width"
        ),
        displayed_width=_text(item["displayed_width"], label="displayed width"),
        minimum_reported_slice_support=_integer(
            item["minimum_reported_slice_support"], label="reported slice support"
        ),
        price_bands_usd=_integer_triplet(item["price_bands_usd"], label="price bands"),
        mileage_bands_miles=_integer_triplet(item["mileage_bands_miles"], label="mileage bands"),
        vehicle_age_reference_year=_integer(
            item["vehicle_age_reference_year"], label="age reference year"
        ),
        vehicle_age_bands_years=_integer_triplet(
            item["vehicle_age_bands_years"], label="age bands"
        ),
        manufacturer_support=_integer(item["manufacturer_support"], label="manufacturer support"),
        bootstrap=BootstrapPolicy(
            unit=_text(bootstrap["unit"], label="bootstrap unit"),
            replicates=_integer(bootstrap["replicates"], label="bootstrap replicates"),
            random_state=_integer(bootstrap["random_state"], label="bootstrap random state"),
            confidence_level=_number(
                bootstrap["confidence_level"], label="bootstrap confidence level"
            ),
            paired_across_methods=_boolean(
                bootstrap["paired_across_methods"], label="paired bootstrap flag"
            ),
        ),
    )


def _parse_gates(value: object) -> AcceptanceGates:
    item = _exact_mapping(
        value,
        {"validity", "overall_coverage", "sharpness", "conditional_coverage", "stability"},
        label="acceptance_gates",
    )
    validity = _exact_mapping(
        item["validity"],
        {
            "invalid_or_nonfinite_intervals",
            "reversed_or_point_excluding_intervals",
            "negative_displayed_lower_bounds",
            "clipped_and_unclipped_coverage_must_match",
            "gamma_scale_floor_hit_maximum_rate",
            "gamma_scale_cap_hit_maximum_rate",
        },
        label="validity gates",
    )
    overall = _exact_mapping(
        item["overall_coverage"],
        {
            "minimum_gap_from_target_each_level",
            "maximum_regression_vs_baseline_each_level",
            "minimum_cluster_bootstrap_95pct_lower_delta_vs_baseline",
        },
        label="overall coverage gates",
    )
    sharpness = _exact_mapping(
        item["sharpness"],
        {
            "minimum_unclipped_mean_width_reduction_80pct",
            "minimum_unclipped_mean_width_reduction_90pct",
            "minimum_unclipped_mean_width_reduction_95pct",
            "minimum_displayed_median_width_reduction_each_level",
            "maximum_bootstrap_95pct_upper_mean_width_ratio_at_90pct",
            "maximum_p95_width_ratio_vs_baseline_each_level",
        },
        label="sharpness gates",
    )
    conditional = _exact_mapping(
        item["conditional_coverage"],
        {
            "minimum_status_gap_from_target_each_level",
            "maximum_status_regression_vs_baseline_each_level",
            "maximum_broad_slice_regression_vs_baseline_each_level",
            "new_broad_slice_undercoverage_boundary",
            "maximum_manufacturer_regression_vs_baseline_at_90pct",
            "manufacturer_count_below_80pct_at_90pct_may_increase",
            "focus_90pct_maximum_regression_vs_baseline",
            "focus_slices",
        },
        label="conditional coverage gates",
    )
    stability = _exact_mapping(
        item["stability"],
        {
            "maximum_fold_coverage_regression_vs_baseline_each_level",
            "maximum_fold_coverage_sd_increase_vs_baseline_each_level",
            "maximum_fallback_rate_increase_vs_baseline",
            "maximum_p95_to_median_width_ratio",
            "maximum_interval_width_usd",
        },
        label="stability gates",
    )
    return AcceptanceGates(
        validity=ValidityGates(
            invalid_or_nonfinite_intervals=_integer(
                validity["invalid_or_nonfinite_intervals"], label="invalid interval maximum"
            ),
            reversed_or_point_excluding_intervals=_integer(
                validity["reversed_or_point_excluding_intervals"],
                label="reversed interval maximum",
            ),
            negative_displayed_lower_bounds=_integer(
                validity["negative_displayed_lower_bounds"], label="negative lower maximum"
            ),
            clipped_and_unclipped_coverage_must_match=_boolean(
                validity["clipped_and_unclipped_coverage_must_match"],
                label="clipped coverage match flag",
            ),
            gamma_scale_floor_hit_maximum_rate=_number(
                validity["gamma_scale_floor_hit_maximum_rate"], label="scale floor rate"
            ),
            gamma_scale_cap_hit_maximum_rate=_number(
                validity["gamma_scale_cap_hit_maximum_rate"], label="scale cap rate"
            ),
        ),
        overall_coverage=OverallCoverageGates(
            minimum_gap_from_target_each_level=_number(
                overall["minimum_gap_from_target_each_level"], label="coverage gap"
            ),
            maximum_regression_vs_baseline_each_level=_number(
                overall["maximum_regression_vs_baseline_each_level"],
                label="coverage regression",
            ),
            minimum_cluster_bootstrap_95pct_lower_delta_vs_baseline=_number(
                overall["minimum_cluster_bootstrap_95pct_lower_delta_vs_baseline"],
                label="bootstrap coverage delta",
            ),
        ),
        sharpness=SharpnessGates(
            minimum_unclipped_mean_width_reduction_80pct=_number(
                sharpness["minimum_unclipped_mean_width_reduction_80pct"],
                label="80% width reduction",
            ),
            minimum_unclipped_mean_width_reduction_90pct=_number(
                sharpness["minimum_unclipped_mean_width_reduction_90pct"],
                label="90% width reduction",
            ),
            minimum_unclipped_mean_width_reduction_95pct=_number(
                sharpness["minimum_unclipped_mean_width_reduction_95pct"],
                label="95% width reduction",
            ),
            minimum_displayed_median_width_reduction_each_level=_number(
                sharpness["minimum_displayed_median_width_reduction_each_level"],
                label="median width reduction",
            ),
            maximum_bootstrap_95pct_upper_mean_width_ratio_at_90pct=_number(
                sharpness["maximum_bootstrap_95pct_upper_mean_width_ratio_at_90pct"],
                label="bootstrap width ratio",
            ),
            maximum_p95_width_ratio_vs_baseline_each_level=_number(
                sharpness["maximum_p95_width_ratio_vs_baseline_each_level"],
                label="p95 width ratio",
            ),
        ),
        conditional_coverage=ConditionalCoverageGates(
            minimum_status_gap_from_target_each_level=_number(
                conditional["minimum_status_gap_from_target_each_level"],
                label="status coverage gap",
            ),
            maximum_status_regression_vs_baseline_each_level=_number(
                conditional["maximum_status_regression_vs_baseline_each_level"],
                label="status regression",
            ),
            maximum_broad_slice_regression_vs_baseline_each_level=_number(
                conditional["maximum_broad_slice_regression_vs_baseline_each_level"],
                label="broad slice regression",
            ),
            new_broad_slice_undercoverage_boundary=_number(
                conditional["new_broad_slice_undercoverage_boundary"],
                label="broad slice undercoverage boundary",
            ),
            maximum_manufacturer_regression_vs_baseline_at_90pct=_number(
                conditional["maximum_manufacturer_regression_vs_baseline_at_90pct"],
                label="manufacturer regression",
            ),
            manufacturer_count_below_80pct_at_90pct_may_increase=_boolean(
                conditional["manufacturer_count_below_80pct_at_90pct_may_increase"],
                label="manufacturer count increase flag",
            ),
            focus_90pct_maximum_regression_vs_baseline=_number(
                conditional["focus_90pct_maximum_regression_vs_baseline"],
                label="focus regression",
            ),
            focus_slices=_text_tuple(conditional["focus_slices"], label="focus slices"),
        ),
        stability=StabilityGates(
            maximum_fold_coverage_regression_vs_baseline_each_level=_number(
                stability["maximum_fold_coverage_regression_vs_baseline_each_level"],
                label="fold coverage regression",
            ),
            maximum_fold_coverage_sd_increase_vs_baseline_each_level=_number(
                stability["maximum_fold_coverage_sd_increase_vs_baseline_each_level"],
                label="fold coverage SD increase",
            ),
            maximum_fallback_rate_increase_vs_baseline=_number(
                stability["maximum_fallback_rate_increase_vs_baseline"],
                label="fallback rate increase",
            ),
            maximum_p95_to_median_width_ratio=_number(
                stability["maximum_p95_to_median_width_ratio"], label="width tail ratio"
            ),
            maximum_interval_width_usd=_number(
                stability["maximum_interval_width_usd"], label="maximum interval width"
            ),
        ),
    )


def _parse_selection(value: object) -> SelectionRule:
    keys = {
        "candidate_must_pass_every_gate",
        "if_neither_passes",
        "if_execution_or_baseline_reproduction_fails",
        "if_both_pass",
        "coverage_is_primary",
        "do_not_force_a_winner",
    }
    item = _exact_mapping(value, keys, label="selection_rule")
    return SelectionRule(
        candidate_must_pass_every_gate=_boolean(
            item["candidate_must_pass_every_gate"], label="all-gates flag"
        ),
        if_neither_passes=_text(item["if_neither_passes"], label="neither-passes action"),
        if_execution_or_baseline_reproduction_fails=_text(
            item["if_execution_or_baseline_reproduction_fails"], label="execution-failure action"
        ),
        if_both_pass=_text(item["if_both_pass"], label="both-pass action"),
        coverage_is_primary=_boolean(item["coverage_is_primary"], label="coverage-primary flag"),
        do_not_force_a_winner=_boolean(
            item["do_not_force_a_winner"], label="no-forced-winner flag"
        ),
    )


def _parse_confidence(value: object) -> ConfidencePolicy:
    keys = {
        "coverage_level",
        "relative_width",
        "high_max_relative_width",
        "moderate_max_relative_width",
        "high_minimum_support",
        "moderate_minimum_support",
        "thresholds_reused_from_v1_for_direct_comparability",
        "confidence_is_not_a_probability",
        "data_quality_warnings_remain_separate",
    }
    item = _exact_mapping(value, keys, label="confidence_policy")
    return ConfidencePolicy(
        coverage_level=_number(item["coverage_level"], label="confidence coverage"),
        relative_width=_text(item["relative_width"], label="confidence relative width"),
        high_max_relative_width=_number(
            item["high_max_relative_width"], label="high-confidence threshold"
        ),
        moderate_max_relative_width=_number(
            item["moderate_max_relative_width"], label="moderate-confidence threshold"
        ),
        high_minimum_support=_integer(
            item["high_minimum_support"], label="high-confidence support"
        ),
        moderate_minimum_support=_integer(
            item["moderate_minimum_support"], label="moderate-confidence support"
        ),
        thresholds_reused_from_v1_for_direct_comparability=_boolean(
            item["thresholds_reused_from_v1_for_direct_comparability"],
            label="reused-threshold flag",
        ),
        confidence_is_not_a_probability=_boolean(
            item["confidence_is_not_a_probability"], label="confidence semantics flag"
        ),
        data_quality_warnings_remain_separate=_boolean(
            item["data_quality_warnings_remain_separate"], label="warning separation flag"
        ),
    )


def _parse_publication(value: object) -> PublicationPolicy:
    keys = {
        "reports_are_aggregate_only",
        "persist_new_serving_artifact_only_if_candidate_passes_every_gate",
        "preserve_calibration_v1_artifacts",
        "preserve_phase4_yoad_river_autotrader_and_carson_artifacts",
        "legacy_holdout_access",
    }
    item = _exact_mapping(value, keys, label="publication")
    return PublicationPolicy(
        reports_are_aggregate_only=_boolean(
            item["reports_are_aggregate_only"], label="aggregate-only flag"
        ),
        persist_new_serving_artifact_only_if_candidate_passes_every_gate=_boolean(
            item["persist_new_serving_artifact_only_if_candidate_passes_every_gate"],
            label="conditional persistence flag",
        ),
        preserve_calibration_v1_artifacts=_boolean(
            item["preserve_calibration_v1_artifacts"], label="preserve calibration v1 flag"
        ),
        preserve_phase4_yoad_river_autotrader_and_carson_artifacts=_boolean(
            item["preserve_phase4_yoad_river_autotrader_and_carson_artifacts"],
            label="preserve frozen artifacts flag",
        ),
        legacy_holdout_access=_boolean(
            item["legacy_holdout_access"], label="legacy holdout access flag"
        ),
    )


def _validate_fixed_policy(policy: UncertaintySharpnessPolicy) -> None:
    """Reject semantically invalid state even if parser constants are changed later."""

    if policy.schema_version != 1:
        raise UncertaintySharpnessPolicyError("schema_version differs from policy")
    if policy.policy_id != "autovalue-retail-rf05-uncertainty-sharpness-v1":
        raise UncertaintySharpnessPolicyError("policy_id differs from policy")
    if policy.candidate_ids != (
        "vehicle_status_absolute_residual_v1",
        "normalized_gamma_scale_v1",
        "normalized_smooth_value_scale_v1",
    ):
        raise UncertaintySharpnessPolicyError("candidate order differs from policy")
    frozen = policy.frozen_inputs
    if (
        frozen.rf05_parameters != (96, 1024, 5, 1.0, 0.6)
        or frozen.rf05_random_state != 1_254_777_149
        or frozen.phase3_train_rows != 109_510
        or frozen.development_rows != 98_552
        or frozen.calibration_rows != 10_958
        or frozen.legacy_holdout_rows != 27_589
    ):
        raise UncertaintySharpnessPolicyError("frozen model or population values differ")
    if policy.calibration_comparison.coverage_levels != (0.8, 0.9, 0.95):
        raise UncertaintySharpnessPolicyError("coverage levels differ from policy")
    if policy.gamma_method.hyperparameters != GammaHyperparameters(
        alpha=1.0,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=2000,
        tol=1e-7,
        warm_start=False,
    ):
        raise UncertaintySharpnessPolicyError("Gamma hyperparameters differ from policy")
    if policy.gamma_method.scale_floor_usd != 500.0 or (
        policy.gamma_method.scale_cap_usd != 250_000.0
    ):
        raise UncertaintySharpnessPolicyError("Gamma scale bounds differ from policy")
    boundary = policy.boundary_interpretation
    if (
        not boundary.no_persisted_rf05_or_row_level_predictions_exist
        or boundary.calibration_targets_fit_rf05
        or boundary.calibration_targets_fit_residual_scale_model
        or boundary.calibration_targets_select_scale_model_hyperparameters
        or boundary.raw_rows_predictions_or_residuals_persisted
        or policy.publication.legacy_holdout_access
    ):
        raise UncertaintySharpnessPolicyError("protected data boundary differs from policy")
    if not (
        policy.selection_rule.candidate_must_pass_every_gate
        and policy.selection_rule.coverage_is_primary
        and policy.selection_rule.do_not_force_a_winner
        and policy.publication.reports_are_aggregate_only
        and policy.publication.persist_new_serving_artifact_only_if_candidate_passes_every_gate
    ):
        raise UncertaintySharpnessPolicyError("selection or publication safeguards differ")


def _bounded_utf8(serialized: str | bytes) -> tuple[bytes, str]:
    if isinstance(serialized, str):
        try:
            payload = serialized.encode("utf-8")
        except UnicodeEncodeError as error:
            raise UncertaintySharpnessPolicyError("policy must be valid UTF-8") from error
        text = serialized
    elif isinstance(serialized, bytes):
        payload = serialized
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise UncertaintySharpnessPolicyError("policy must be valid UTF-8") from error
    else:
        raise UncertaintySharpnessPolicyError("policy must be text or bytes")
    if len(payload) > _MAX_POLICY_BYTES:
        raise UncertaintySharpnessPolicyError("policy exceeds maximum size")
    return payload, text


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise UncertaintySharpnessPolicyError(f"policy has duplicate field: {key}")
        result[key] = value
    return result


def _exact_mapping(value: object, keys: set[str], *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise UncertaintySharpnessPolicyError(f"{label} must be an object")
    if set(value) != keys:
        raise UncertaintySharpnessPolicyError(f"{label} fields are invalid")
    return cast(Mapping[str, object], value)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UncertaintySharpnessPolicyError(f"{label} must be non-empty text")
    return value


def _digest(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if not _DIGEST_PATTERN.fullmatch(text):
        raise UncertaintySharpnessPolicyError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise UncertaintySharpnessPolicyError(f"{label} must be a boolean")
    return value


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise UncertaintySharpnessPolicyError(f"{label} must be an integer")
    return value


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UncertaintySharpnessPolicyError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise UncertaintySharpnessPolicyError(f"{label} must be finite")
    return number


def _text_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise UncertaintySharpnessPolicyError(f"{label} must be a non-empty array")
    return tuple(_text(item, label=f"{label} item") for item in value)


def _float_triplet(value: object, *, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise UncertaintySharpnessPolicyError(f"{label} must contain exactly three values")
    return (
        _number(value[0], label=label),
        _number(value[1], label=label),
        _number(value[2], label=label),
    )


def _integer_triplet(value: object, *, label: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise UncertaintySharpnessPolicyError(f"{label} must contain exactly three values")
    return (
        _integer(value[0], label=label),
        _integer(value[1], label=label),
        _integer(value[2], label=label),
    )


__all__ = [
    "AcceptanceGates",
    "BaselineMethodPolicy",
    "BoundaryInterpretation",
    "BootstrapPolicy",
    "CalibrationComparisonPolicy",
    "ConditionalCoverageGates",
    "ConfidencePolicy",
    "DevelopmentDiagnosticsPolicy",
    "FrozenInputs",
    "GammaHyperparameters",
    "GammaScaleMethodPolicy",
    "OverallCoverageGates",
    "PublicationPolicy",
    "SelectionRule",
    "SharpnessGates",
    "SmoothValueScaleMethodPolicy",
    "StabilityGates",
    "UNCERTAINTY_SHARPNESS_POLICY_SHA256",
    "UncertaintySharpnessPolicy",
    "UncertaintySharpnessPolicyError",
    "ValidityGates",
    "load_uncertainty_sharpness_policy",
    "load_uncertainty_sharpness_policy_file",
]
