# Retail RF05 calibrated prediction intervals

## Decision

**Classification: `validated_for_calibrated_prediction_intervals`.** The selected calibration method is 
`vehicle_status`. This calibrates the frozen Phase 4 RF05 point predictor; it does 
not promote, replace, retrain, or retune it. The legacy final holdout remains unopened.

These are empirical intervals around a historical U.S. advertised asking-price model. 
They are not a probability for one vehicle, a guaranteed sale price, or a Kelley Blue 
Book/third-party valuation.

## Protected boundary

- Development-only RF05 fit: 98,552 rows
- Reserved calibration population: 10,958 rows
- Calibration rows used for RF fitting, model choice, or tuning: no
- Legacy holdout, Yoad, and River data accessed: no
- Raw rows, row-level predictions, or residuals persisted: no

## Calibration point performance

All price errors are USD.

| Rows | MAE | RMSE | R-squared |
|---:|---:|---:|---:|
| 10,958 | $9,926.84 | $20,566.28 | 0.6695 |

## Cross-fitted interval results

Each diagnostic row is scored with radii derived from the other four predictor-group 
folds. All three preregistered levels are reported.

| Nominal | Empirical | Gap | Average width | Median width | Fold coverage SD |
|---:|---:|---:|---:|---:|---:|
| 80% | 79.80% | -0.20% | $25,813.15 | $28,122.10 | 4.87% |
| 90% | 89.81% | -0.19% | $38,697.46 | $40,562.24 | 4.43% |
| 95% | 94.26% | -0.74% | $61,836.28 | $56,141.06 | 3.41% |

At 90% nominal coverage, the global method reached 90.07% with $38,775.32
average width. Vehicle-status calibration reached 89.81% with $38,697.46
average width. The status-plus-predicted-value hierarchy reached 89.03% but
widened the average interval to $40,661.80. It failed the preregistered overall
coverage-gap gate by 0.10 percentage point and exceeded the status method's
average width by 13.19% at its worst coverage level. The two remaining
conditional gates passed. The policy therefore selected vehicle status with a
global fallback; no post-result bucket or threshold was tuned.

### 90% coverage by vehicle status

| Status | Rows | Empirical coverage | Average width | Median width |
|---|---:|---:|---:|---:|
| Certified | 607 | 89.95% | $34,345.10 | $33,475.29 |
| New | 6,493 | 89.65% | $45,710.87 | $46,089.60 |
| Used | 3,858 | 90.05% | $27,578.70 | $27,704.85 |

The full-calibration 90% radii are $16,737.64 for Certified, $22,625.25 for
New, and $13,906.15 for Used vehicles. Unknown status values fall back to the
$19,442.61 global radius. The minimum conditional support is 400 rows.

### Stability and segment audit

The five 90% cross-calibration folds covered 94.39%, 84.49%, 90.56%, 84.89%,
and 94.71%, so the method is well calibrated in aggregate but has meaningful
fold heterogeneity. At the required 200-row reporting threshold, 90% coverage
was 85.24% for 0-38,282 miles, 92.66% for 38,282-86,204 miles, 96.74% for
86,204-135,803 miles, 98.78% above 135,803 miles, and 89.65% when mileage was
missing.

The highest reported asking-price band (above $36,590; 7,447 rows) received the
widest average 90% interval at $41,581.37, compared with $34,646.60 for
$19,995-$36,590 and $27,710.61 for $8,995-$19,995. This is directionally
appropriate because its point-prediction MAE was also highest at $11,862.91,
but its 87.27% coverage remains below nominal. The lowest price band had only
170 calibration rows and is intentionally not reported as a slice.

Manufacturer coverage is diagnostic, not a claim of conditional validity.
Seventeen manufacturers met the 200-row threshold. Some were materially below
90%, including GMC (46.69%), Genesis (65.65%), BMW (70.98%), Audi (75.07%), and
Mercedes (79.61%). These weak slices and the fold spread must remain visible in
any integration; the aggregate interval must not be described as guaranteeing
90% coverage for every make or vehicle segment. Complete price, mileage, age,
manufacturer, and fold aggregates are retained in the machine-readable report.

## Confidence labels

Confidence labels use the preregistered empirical 90% relative-width thresholds 
and applicable calibration support. They are not probabilities. Data-quality warnings 
remain separate and do not silently widen the interval.

High: 4,111; moderate: 3,410; low: 3,437.

## Reproducibility

- Calibration policy SHA-256: `1398519c699bd129ef4fbb552813c064839c6c1e1c4ecd35c7f5d42bcf8e1ca2`
- Calibration assignment SHA-256: `caa743681158c4eaccb2ec75ce17a1c5e20327a311f66c5e8e0d0c630c48e992`
- Serving artifact: `docs/experiments/retail-rf05-calibration-v1.artifact.json`
- Serving artifact SHA-256: `b7eb5970b164ec68fb76cf8314f36080d085cda02968d3570d11f724490a6da0`
- Aggregate report SHA-256: `e7fafff505603669e73cfbff2fe1cf5e04f9c5d896666470fe212411aa1b3084`

The JSON report contains fold-level coverage, interval widths, under/overcoverage, 
vehicle-status diagnostics, and aggregate price/mileage/age/manufacturer slices.
