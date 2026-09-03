# Model card: AutoValue retail RF05 v1

## Status

Final classification: **final evaluation passed with material limitations.** This card documents the frozen portfolio reference; it is not a production-readiness claim.

## Model and intended use

RF05 is a scikit-learn Random Forest regression pipeline for educational and portfolio demonstrations of historical U.S. advertised vehicle asking-price estimation in USD. Intended inputs are year, make, exact source model string, mileage when present, and vehicle status. It must not be presented as an appraisal, offer, guaranteed transaction price, financial advice, or current-market quote.

The estimator uses 96 trees, 1,024 maximum leaf nodes, minimum leaf size 5, all transformed features per split, a 60% bootstrap sample, and random state 1254777149. Numeric imputation and categorical encoding are learned from training data only.

## Data boundaries

RF05 was fit on exactly 98,552 Cars.com-derived development rows. The separate 10,958-row calibration partition did not fit RF05 and only created the frozen v1 conformal radii. The 27,589-row grouped final holdout was opened once, did not fit or calibrate any component, and is now permanently evaluation-only. The split is non-temporal.

Yoad/Craigslist data is not part of this model. AutoTrader/KBB is governed as a different target. River is shadow-only. Carson-Shively is excluded.

## Final performance

| MAE | RMSE | R² | Median absolute error | Mean signed error |
|---:|---:|---:|---:|---:|
| $10,575.36 | $34,118.14 | 0.4176 | $6,678.93 | -$1,863.46 |

| Interval | Empirical coverage | Mean displayed width | Median displayed width |
|---:|---:|---:|---:|
| 80% | 76.32% | $25,885.04 | $30,669.62 |
| 90% | 89.10% | $38,434.98 | $45,250.50 |
| 95% | 95.64% | $64,028.15 | $78,772.89 |

Confidence labels communicate relative interval precision and calibration support; they are not probabilities of correctness. Data-quality warnings are separate.

## Subgroup evaluation and risks

27 manufacturers met the 200-row reporting threshold. Full manufacturer, vehicle-status, mileage, age, actual price, predicted-value, and mileage-missingness results are in the final report. Subgroup metrics with lower support are omitted rather than treated as reliable.

Known limitations include omitted trim and mechanical/history attributes, asking-price rather than completed-sale labels, historical data, non-temporal validation, broad intervals for some vehicles, uneven manufacturer support, and possible source-specific selection bias. Users must see the interval, confidence semantics, and limitations alongside a point estimate.

## Governance and reproducibility

The policy, RF05 definition, training boundary, preprocessing, calibration artifact, coverage levels, confidence thresholds, metrics, slices, and decision gates were frozen before holdout access. The holdout result cannot authorize tuning or recalibration. No trained model binary or row-level holdout evidence was persisted.

Policy SHA-256: `2be880be315f39a727bd8f1c6545b9410ea855bee63a3e72336f4da8cd7d5c33`  
Aggregate final report SHA-256: `017ab1824b1ddf4248959ecf8bb4a7d87991b513526a5567f29af0dd6e191e86`
