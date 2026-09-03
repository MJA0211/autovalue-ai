# Retail RF05 one-time final holdout evaluation

## Decision

**final evaluation passed with material limitations.**

This is the sole final evaluation of frozen Phase 4 retail RF05 with the unchanged status-conditional conformal v1 intervals. The 27,589-row grouped holdout is now permanently evaluation-only. No model, preprocessing, quantile, bucket, confidence threshold, or source composition was changed.

## Point performance

All dollar values are USD asking-price errors.

| Rows | MAE | RMSE | R² | Median AE | p90 AE | p95 AE | Mean signed error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 27,589 | $10,575.36 | $34,118.14 | 0.4176 | $6,678.93 | $20,913.04 | $27,903.84 | -$1,863.46 |

Underpredictions: 49.50%; overpredictions: 50.50%. MAPE remains intentionally omitted because low-dollar targets make percentage errors unstable.

## Frozen uncertainty performance

| Nominal | Empirical | Gap | Mean width | Median width | p90 width | Clipped | Fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 80.00% | 76.32% | -3.68 pp | $25,885.04 | $30,669.62 | $30,669.62 | 0.43% | 0.00% |
| 90.00% | 89.10% | -0.90 pp | $38,434.98 | $45,250.50 | $45,250.50 | 2.85% | 0.00% |
| 95.00% | 95.64% | +0.64 pp | $64,028.15 | $78,772.89 | $81,217.06 | 22.63% | 0.00% |

Coverage is the primary uncertainty criterion. Confidence is a precision/support label, not a probability that the estimate is correct.

## Generalization gaps

| Reference | MAE reference | MAE ratio | RMSE ratio | R² reference | R² change |
|---|---:|---:|---:|---:|---:|
| Development OOF | $10,269.78 | 1.030× | 1.115× | 0.4857 | -0.0681 |
| Calibration | $9,926.84 | 1.065× | 1.659× | 0.6695 | -0.2520 |

## Confidence-label diagnostics

| Label | Rows | MAE | Median AE | 90% coverage | Median 90% width |
|---|---:|---:|---:|---:|---:|
| High confidence | 9,839 | $14,296.67 | $9,229.10 | 81.38% | $45,250.50 |
| Moderate confidence | 10,015 | $9,234.78 | $5,830.45 | 91.66% | $45,250.50 |
| Low confidence | 7,735 | $7,577.55 | $5,638.78 | 95.62% | $45,250.50 |

Expected High ≤ Moderate ≤ Low ordering passed: **false**.

## Manufacturer summary

Only manufacturers with at least 200 holdout records are ranked.

| Group | Manufacturer | Rows | MAE | RMSE | Bias | 90% coverage |
|---|---|---:|---:|---:|---:|---:|
| Strongest | subaru | 209 | $3,674.74 | $4,947.90 | $2,098.14 | 98.56% |
| Strongest | infiniti | 203 | $3,749.92 | $4,856.84 | -$36.63 | 100.00% |
| Strongest | mazda | 528 | $4,044.61 | $4,906.13 | $2,874.88 | 99.81% |
| Strongest | lincoln | 213 | $4,723.29 | $6,429.00 | $93.83 | 98.59% |
| Strongest | honda | 1,030 | $4,835.81 | $5,833.50 | $2,088.22 | 99.32% |
| Weakest | mercedes | 701 | $24,373.31 | $49,647.74 | -$16,229.02 | 68.33% |
| Weakest | porsche | 344 | $23,013.34 | $69,684.25 | -$3,988.29 | 59.01% |
| Weakest | bmw | 1,412 | $15,037.80 | $19,940.86 | -$4,455.53 | 70.47% |
| Weakest | cadillac | 560 | $13,396.24 | $19,856.29 | -$2,511.74 | 77.50% |
| Weakest | land rover | 236 | $12,683.46 | $19,094.74 | -$1,435.41 | 66.95% |

### Vehicle status

| Slice | Rows | MAE | RMSE | R² | Bias | 90% coverage | Mean 90% width |
|---|---:|---:|---:|---:|---:|---:|---:|
| certified | 1,518 | $8,275.83 | $16,233.80 | 0.8157 | -$58.14 | 90.12% | $33,474.72 |
| new | 16,425 | $12,441.72 | $32,554.42 | 0.3184 | -$2,902.64 | 88.99% | $45,250.50 |
| used | 9,646 | $7,759.23 | $38,513.48 | 0.3858 | -$378.09 | 89.14% | $27,610.27 |

### Mileage bands

| Slice | Rows | MAE | RMSE | R² | Bias | 90% coverage | Mean 90% width |
|---|---:|---:|---:|---:|---:|---:|---:|
| mileage_1 | 5,158 | $10,526.80 | $52,611.74 | 0.3500 | -$812.98 | 83.60% | $29,054.03 |
| mileage_2 | 3,877 | $6,147.08 | $9,159.63 | 0.6385 | $17.37 | 92.55% | $28,340.69 |
| mileage_3 | 1,533 | $4,459.25 | $7,529.27 | 0.4546 | $121.18 | 97.00% | $27,550.59 |
| mileage_4 | 596 | $4,098.58 | $5,548.50 | 0.4626 | $343.90 | 97.15% | $25,454.12 |
| mileage_missing | 16,425 | $12,441.72 | $32,554.42 | 0.3184 | -$2,902.64 | 88.99% | $45,250.50 |

### Vehicle age bands

| Slice | Rows | MAE | RMSE | R² | Bias | 90% coverage | Mean 90% width |
|---|---:|---:|---:|---:|---:|---:|---:|
| age_1 | 22,114 | $11,399.94 | $29,350.18 | 0.4463 | -$2,202.20 | 88.53% | $41,103.12 |
| age_2 | 3,621 | $7,464.20 | $56,317.82 | 0.1458 | -$696.74 | 91.19% | $28,109.72 |
| age_3 | 1,195 | $5,453.88 | $9,661.72 | 0.6835 | -$242.40 | 93.81% | $27,086.22 |
| age_4 | 659 | $9,286.88 | $47,212.30 | 0.2529 | $153.01 | 88.47% | $26,213.88 |

### Actual asking-price bands

| Slice | Rows | MAE | RMSE | R² | Bias | 90% coverage | Mean 90% width |
|---|---:|---:|---:|---:|---:|---:|---:|
| price_1 | 451 | $5,567.56 | $6,518.15 | -15.7380 | $5,562.94 | 98.00% | $25,544.38 |
| price_2 | 2,156 | $4,828.87 | $6,495.98 | -3.2420 | $3,732.54 | 96.71% | $27,546.03 |
| price_3 | 5,894 | $6,343.84 | $8,505.18 | -2.6227 | $4,184.32 | 95.40% | $34,113.17 |
| price_4 | 19,088 | $12,649.36 | $40,673.81 | 0.2753 | -$4,538.44 | 86.09% | $41,303.96 |

### Predicted-value bands

| Slice | Rows | MAE | RMSE | R² | Bias | 90% coverage | Mean 90% width |
|---|---:|---:|---:|---:|---:|---:|---:|
| predicted_value_1 | 6,887 | $5,726.48 | $8,627.00 | 0.4899 | -$477.02 | 94.51% | $29,929.24 |
| predicted_value_2 | 5,146 | $8,715.79 | $12,362.06 | 0.0415 | -$1,970.53 | 95.03% | $39,769.58 |
| predicted_value_3 | 10,874 | $11,205.79 | $35,621.46 | 0.0290 | -$1,554.35 | 87.73% | $41,499.45 |
| predicted_value_4 | 4,682 | $18,287.51 | $60,289.13 | 0.2728 | -$4,503.11 | 77.83% | $42,362.41 |

### Mileage presence

| Slice | Rows | MAE | RMSE | R² | Bias | 90% coverage | Mean 90% width |
|---|---:|---:|---:|---:|---:|---:|---:|
| mileage_missing | 16,425 | $12,441.72 | $32,554.42 | 0.3184 | -$2,902.64 | 88.99% | $45,250.50 |
| mileage_present | 11,164 | $7,829.48 | $36,296.49 | 0.4280 | -$334.58 | 89.27% | $28,407.67 |

## Interpretation and restrictions

- This evaluates historical U.S. advertised asking prices in USD; it does not estimate a guaranteed sale, trade-in, auction, or KBB value.
- The split is grouped but non-temporal, so future-market drift is not measured.
- Trim, engine, transmission, drivetrain, condition, and vehicle history are not available to this frozen model.
- Slice results are diagnostic and were not used to tune the evaluated system.
- Yoad remains a separate unpromoted experiment; River remains shadow-only; AutoTrader remains separate; Carson-Shively remains excluded.
- No post-holdout tuning, recalibration, promotion experiment, or model persistence is authorized by this report.

## Reproducibility

Policy SHA-256: `2be880be315f39a727bd8f1c6545b9410ea855bee63a3e72336f4da8cd7d5c33`  
Aggregate report SHA-256: `017ab1824b1ddf4248959ecf8bb4a7d87991b513526a5567f29af0dd6e191e86`
