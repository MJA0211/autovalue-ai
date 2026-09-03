# Retail RF05 uncertainty sharpness experiment

## Decision

**Classification: `retain_current_calibration_baseline`.** Selected method: `vehicle_status_absolute_residual_v1`.
This is uncertainty calibration around the unchanged Phase 4 RF05 point estimator; 
it does not promote a production-final system or authorize opening the legacy holdout.

## Protected design

- Development OOF residual rows: 98,552
- Protected calibration rows: 10,958
- Gamma residual-scale fit used calibration targets: no
- RF05 retuned or replaced: no
- This experiment requested, loaded, or evaluated legacy-holdout rows: no
- Yoad, River, AutoTrader, and Carson-Shively were outside this experiment.
- Raw rows, row-level predictions, or residuals persisted: no
- Displayed lower bounds are explicitly clipped at $0; coverage equivalence is audited.

## Why heteroscedastic methods were evaluated

Development OOF absolute residuals have a median of $6,325.78 and a mean of $10,269.78.
Mean error rises from $5,648.98 in the lowest predicted-value quartile to $18,500.38 in the highest, a 3.27x ratio. This is the preregistered basis for testing scaled conformal scores.

Actual price was used only to evaluate development residual behavior, never as a residual-scale predictor.

## Direct comparison

All widths are USD. Sharpness gates use unclipped symmetric mean width; displayed 
widths reflect the physical $0 lower bound.

| Method | Nominal | Coverage | Gap | Mean width | Median | p75 | p90 | p95 | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current status-conditional baseline | 80% | 79.80% | -0.20% | $25,813.15 | $28,122.10 | $30,550.35 | $32,362.10 | $32,362.10 | 0.00% |
| Current status-conditional baseline | 90% | 89.81% | -0.19% | $38,697.46 | $40,562.24 | $46,089.60 | $51,268.94 | $51,268.94 | 0.00% |
| Current status-conditional baseline | 95% | 94.26% | -0.74% | $61,836.28 | $56,141.06 | $80,731.06 | $84,822.81 | $90,871.06 | 0.00% |
| Normalized Gamma residual scale | 80% | 79.83% | -0.17% | $26,078.29 | $24,666.93 | $32,393.45 | $36,673.00 | $41,257.01 | 0.00% |
| Normalized Gamma residual scale | 90% | 89.93% | -0.07% | $37,427.42 | $35,084.30 | $46,621.87 | $53,681.57 | $59,155.42 | 0.00% |
| Normalized Gamma residual scale | 95% | 94.34% | -0.66% | $54,480.80 | $54,425.36 | $73,735.92 | $79,966.19 | $88,789.21 | 0.00% |
| Simple smooth predicted-value scale | 80% | 79.86% | -0.14% | $25,804.26 | $26,688.36 | $30,670.72 | $34,917.98 | $36,788.84 | 0.00% |
| Simple smooth predicted-value scale | 90% | 89.95% | -0.05% | $37,343.42 | $36,567.95 | $45,214.72 | $51,322.63 | $53,575.92 | 0.00% |
| Simple smooth predicted-value scale | 95% | 94.25% | -0.75% | $57,012.20 | $59,783.06 | $77,236.01 | $83,104.53 | $84,735.85 | 0.00% |

## Preregistered gate result

| Candidate | Passed every gate | Failed gates |
|---|---:|---:|
| Normalized Gamma residual scale | no | 9 |
| Simple smooth predicted-value scale | no | 7 |

Failed gate names (full observed values and thresholds are in the JSON report):

- `normalized_gamma_scale_v1`: bootstrap_coverage_delta_lower_0.8, mean_unclipped_width_reduction_0.8, worst_fold_coverage_regression_0.8, mean_unclipped_width_reduction_0.9, median_displayed_width_reduction_0.95, bootstrap_90pct_mean_width_ratio_upper, worst_broad_slice_regression_0.8, worst_broad_slice_regression_0.9, worst_manufacturer_regression_0.9
- `normalized_smooth_value_scale_v1`: mean_unclipped_width_reduction_0.8, mean_unclipped_width_reduction_0.9, median_displayed_width_reduction_0.95, bootstrap_90pct_mean_width_ratio_upper, worst_broad_slice_regression_0.8, worst_broad_slice_regression_0.9, worst_manufacturer_regression_0.9

## Focus-slice 90% coverage

These prespecified diagnostics were not individually tuned.

| Slice | Baseline | Gamma scale | Smooth scale |
|---|---:|---:|---:|
| Highest price band | 87.27% | 88.60% | 88.01% |
| GMC | 46.69% | 46.99% | 46.99% |
| Genesis | 65.65% | 71.74% | 68.26% |
| BMW | 70.98% | 74.72% | 72.23% |
| Audi | 75.07% | 78.12% | 77.01% |
| Mercedes | 79.61% | 82.57% | 80.59% |

Coverage, fold stability, status coverage, all supported broad slices and 
manufacturers, focus slices, clipping, interval validity, relative widths, and 
paired predictor-group bootstrap uncertainty are retained in the JSON report.
Confidence labels remain precision/support labels, not probabilities; data-quality 
warnings remain separate.

## Reproducibility

- Sharpness policy SHA-256: `ec1787be963a907bbae2d1d521aeaef4239b8a5bf7816ced844dcd16902f1058`
- Development diagnostics SHA-256: `8f79ac027a72fff2512ab0b168d91a3a7b46677d72374dc00571a4646aac925d`
- Frozen RF05 identity SHA-256: `3bbd73d6442387496b05253dd20bc749db24aa482d56fa6ba73ec2702de8b513`
- Comparison report SHA-256: `8614bad1ccd5345c64925c11e6172a7b4ef000ed6f16856aa45b48c3e4a741dd`
- Candidate serving artifact: not created
- Gamma residual-scale model: not persisted
