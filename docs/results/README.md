# Baseline results

Phase 3 compares a median-price dummy regressor with Linear Regression on two
separate historical U.S. price targets. The retail target is an advertised
asking price from 2023 listing snapshots; the wholesale target is a completed
auction sale price from 2014-2015. These rows, labels, models, and metrics are
never combined, and neither result is a current-market valuation.

MAE is the selection metric. RMSE shows sensitivity to large misses, and R²
summarizes variance explained. Dollar metrics below are rounded to cents and R²
to four decimals; the linked canonical JSON retains full precision.

## Retail asking-price baseline

Model selection used five-fold predictor-group cross-validation on the 109,510
outer-training rows. Every equal year/make/model/mileage/status group stays in
one fold. Linear Regression had the lower out-of-fold MAE and was therefore fit
on all outer-training rows and evaluated once on the untouched 27,589-row
holdout.

| CV model | Out-of-fold rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Median dummy | 109,510 | $21,694.48 | $42,464.18 | -0.0206 |
| Linear Regression | 109,510 | $11,552.82 | $31,408.86 | 0.4417 |

Linear Regression reduced retail cross-validation MAE by 46.75% relative to
the median dummy.

| Selected Linear Regression holdout slice | Rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Overall | 27,589 | $12,040.29 | $35,452.63 | 0.3711 |
| Certified | 1,518 | $9,428.31 | $19,805.13 | 0.7257 |
| New | 16,425 | $12,641.75 | $32,691.63 | 0.3127 |
| Used | 9,646 | $11,427.17 | $41,392.40 | 0.2906 |

The source has no row-level observation dates or stable upstream listing IDs,
so this is a deterministic, non-temporal grouped evaluation. It does not measure
forward-in-time performance. The accepted target extends to $8,078,160 and the
baseline deliberately applies no target clipping, outlier removal, or log
transform. Those extreme prices make retail RMSE and R² especially sensitive;
the status slices also show that one overall metric hides material variation.

The v2 feature contract carries raw `model_year` through preprocessing. Clipped
age is used only as the denominator for mileage-per-year, avoiding the earlier
risk that adjacent or future model years collapsed to the same engineered year
value. The successful experiment recomputed the transformed retail grouping key
and verified zero outer train/holdout group overlap.

Full report: [`retail-baseline-v1.json`](retail-baseline-v1.json)

## Wholesale completed-sale baseline

Model selection used forward-only validation within the 442,130-row outer
training period. Each fold trains on earlier, VIN-isolated buckets and validates
on the next bucket. The 51,586-row warmup bucket is used for training but never
as validation, so the aggregate comparison covers 390,544 out-of-fold rows.
Linear Regression then fit all outer-training rows and was evaluated once on the
untouched 98,634-row June/July holdout.

| CV model | Forward-CV rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Median dummy | 390,544 | $7,009.35 | $9,876.08 | -0.0822 |
| Linear Regression | 390,544 | $2,382.13 | $4,028.35 | 0.8200 |

Linear Regression reduced wholesale forward-validation MAE by 66.01% relative
to the median dummy.

| Selected Linear Regression holdout | Rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Overall | 98,634 | $2,256.02 | $4,014.77 | 0.8502 |

The outer split seeds test at 2015-06-01 and moves every earlier row sharing a
VIN with that period into test. Verification found zero VIN overlap and no
post-cutoff training row. VIN is used transiently for isolation only and never
enters model features or durable assignments. This baseline also fits the raw
dollar target without target clipping, outlier removal, or a log transform.

Full report: [`wholesale-baseline-v1.json`](wholesale-baseline-v1.json)

## Reproduce the reports

After placing the two exact reviewed source artifacts in their documented local
paths and generating their verified candidates and split artifacts, run from the
repository root:

```powershell
$env:PYTHONPATH = "ml/src"
New-Item -ItemType Directory -Force models/repro | Out-Null
python -m autovalue_ml.modeling.baseline_cli retail --project-root . --output models/repro/retail-baseline-v1.json
python -m autovalue_ml.modeling.baseline_cli wholesale --project-root . --output models/repro/wholesale-baseline-v1.json
Get-FileHash -Algorithm SHA256 models/repro/*.json
```

Each command reopens data only through its source-specific split gate, fits
fold-local preprocessing, selects by CV MAE, scores the reserved holdout once,
and writes deterministic aggregate-only JSON. It does not persist an estimator,
prediction rows, category vocabulary, coefficients, or source data.

Expected SHA-256 values are
`b5cae941ebb01d9766716d01a24acc75ad7d0432b05e8dde44a6200caffad28a`
for retail and
`b0be8b30367f7b7adca904d80610dd161b9b33dffd9e116d5030bd34403a3030`
for wholesale. Independent second runs matched the canonical reports
byte-for-byte.

## What these results do not provide yet

Phase 3 establishes reproducible baselines, not a production model. There is no
persisted model bundle, calibrated prediction range, model-info route,
prediction endpoint, SQLite prediction history, or model-backed dashboard yet.
Those belong to later phases.

Linear-model coefficients are also not presented as feature importance. Their
magnitudes depend on scaling and one-hot representation and do not form a safe
product explanation. The future UI should use held-out permutation importance
and aggregate related one-hot columns into understandable groups such as make,
model, mileage, and status; local explanations require their own validation.
