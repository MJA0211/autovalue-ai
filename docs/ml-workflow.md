# Machine-learning workflow

## Objective

AutoValue AI evaluates two separate supervised-learning tracks:

1. a **historical U.S. advertised asking-price model** from 2023 New, Used, and
   Certified listing snapshots; and
2. a **historical U.S. wholesale-auction completed-sale model** from transactions
   dated 2014-01-01 through 2015-07-21.

The first is the closer consumer-product fit, but neither target establishes a
current live appraisal. The targets, splits, metrics, and persisted artifacts
remain separate. Their feature contracts follow the completed source and split
audit and may include only the documented allowlists.

The wholesale raw source does not document currency. USD is an explicit,
unresolved semantic assumption applied only after a row maps to one of the 50
U.S. states or Washington, D.C. Canada and Puerto Rico are excluded. Phase 3 has
fit reproducible baseline estimators transiently and published aggregate
evaluation reports; it has not promoted or persisted a model.

MAE is the primary selection metric because it translates directly into typical
absolute pricing error. RMSE exposes costly large misses, and R² provides a
scale-relative fit summary. All three are reported on cross-validation
predictions and on the untouched test set.

## 1. Data contract and validation

- Pin dataset source, version, retrieval date, license, and file checksum.
- Require explicit ML-training approval independently of permission to acquire
  or store the source data.
- Establish target meaning: listing price versus completed sale price.
- Normalize column names and explicitly document units and currency.
- Validate types, plausible ranges, missingness, cardinality, and duplicates.
- Remove identifiers, URLs, post-outcome fields, and other leakage candidates.
- Produce a validation report before any model comparison.
- Verify each reviewed CSV's exact encoding, size, SHA-256, header, and row count
  before reading it as an approved artifact.
- Remove `vin` and `seller`; use VIN only in a private split-grouping step.
- Forbid `mmr` from features, preprocessing, tuning, explanation, inference, and
  any automated feature-selection input.
- Preserve New, Used, and Certified as an explicit categorical feature, remove
  `Dealer`, and report status-specific performance for the asking-price track.

## 2. Cleaning and feature engineering

Transformations are represented in tested scikit-learn pipelines. The Phase 3
baseline uses numeric median imputation, explicit numeric missingness flags,
fold-local scaling, explicit missing categorical values, bounded rare-category
one-hot encoding, and unknown-category-safe sparse matrices. It engineers
`mileage_per_year` with a minimum one-year denominator.

The v2 contracts preserve raw `model_year` as a numeric feature. Clipped vehicle
age is used only for the mileage-per-year denominator, so adjacent and future
model years cannot collapse to the same engineered year value. The retail
grouping key is recomputed after this transform; the real baseline run passed
with zero transformed predictor-group overlap between outer train and holdout.

Phase 3 fits raw dollar targets. It performs no target clipping, outlier removal,
or log transformation. A later experiment may compare a transformation or
training-fold-only outlier policy, but it must preserve valid rare and luxury
vehicles and disclose the effect rather than silently changing acquisition.

## 3. Leakage-safe splitting

Duplicate and near-duplicate records must be handled before splitting. For the
wholesale source, the verified primary holdout is chronological and VIN-aware.
Rows dated on or after 2015-06-01 seed test; if any row in a private VIN group
reaches that period, the complete group goes to test. This produces 442,130
train rows and 98,634 test rows. The grouping step promoted 1,066 earlier rows
from 1,039 VIN groups, and verification found zero VIN overlap and zero
post-cutoff train rows. VIN values and their transient hashes are never written
to the durable assignments.

The asking-price CSV has no row-level observation date or stable listing ID, so
it cannot honestly claim a temporal holdout. Its verified deterministic,
status-stratified holdout groups year, make, model, mileage (including null), and
vehicle status without using price. It assigns 109,510 rows to train and 27,589
to test with zero overlap across 56,529 predictor groups. New, Used, and
Certified remain explicit features and slices. This design measures grouped
generalization, not forward-in-time performance. Each track reserves its test
set once and leaves it untouched during tuning.

Cross-validation runs only on the training partition. Every imputer, encoder,
scaler, and learned feature transform is fitted independently inside each fold.
Retail folds must retain predictor-group isolation. Wholesale uses five ordered,
VIN-isolated train buckets: `warmup` (51,586 rows), `2015_01` (134,449),
`2015_02` (158,432), `2015_03_04` (47,174), and `2015_05` (50,489). Forward
folds train on earlier buckets and validate on the next eligible bucket; the
final June/July holdout is not a CV fold.

The durable assignment SHA-256 values are
`5b3e39d0ef418c07b0c4d08ecc18700fc9f387518a21dbd604f515463cb5ebe5`
for retail and
`a96909345612f5ddc5665c4d6817d2c8f0dd6d59c3a84fc523cb82b6adeeb5f2`
for wholesale. Modeling code may obtain real rows only through the corresponding
split-aware training gate, which revalidates lineage, hashes, row accounting,
and isolation before streaming allowlisted features and targets. Unsplit
candidate iterators are not accepted as modeling inputs.

## 4. Baselines and results

The comparison begins with a median-price dummy regressor, followed by Linear
Regression using the same fold-local preprocessing. Selection uses the lowest
aggregate out-of-fold MAE; the target-free model-name order breaks a theoretical
tie. The selected estimator is then refit on all outer-training rows and scored
once on the reserved holdout.

Retail model selection uses five predictor-group folds. Every one of the 109,510
outer-training rows receives exactly one out-of-fold prediction:

| Retail CV model | Rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Median dummy | 109,510 | $21,694.48 | $42,464.18 | -0.0206 |
| Linear Regression | 109,510 | $11,552.82 | $31,408.86 | 0.4417 |

Linear Regression won by CV MAE. Its untouched retail holdout results are:

| Holdout slice | Rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Overall | 27,589 | $12,040.29 | $35,452.63 | 0.3711 |
| Certified | 1,518 | $9,428.31 | $19,805.13 | 0.7257 |
| New | 16,425 | $12,641.75 | $32,691.63 | 0.3127 |
| Used | 9,646 | $11,427.17 | $41,392.40 | 0.2906 |

Wholesale selection uses forward validation. The 51,586-row warmup bucket
trains the first fold but is not itself validation data, so the aggregate covers
390,544 out-of-fold predictions:

| Wholesale forward-CV model | Rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Median dummy | 390,544 | $7,009.35 | $9,876.08 | -0.0822 |
| Linear Regression | 390,544 | $2,382.13 | $4,028.35 | 0.8200 |

| Selected wholesale holdout model | Rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 98,634 | $2,256.02 | $4,014.77 | 0.8502 |

Dollar metrics are shown to cents and R² to four decimals. Full-precision,
aggregate-only reports are committed as
[`retail-baseline-v1.json`](results/retail-baseline-v1.json) and
[`wholesale-baseline-v1.json`](results/wholesale-baseline-v1.json); see the
[result interpretation](results/README.md). Retail CV and holdout are grouped
but non-temporal. Wholesale CV and holdout preserve chronology and private VIN
isolation. CV chooses the model; the untouched holdout estimates performance
after that choice and is not another tuning fold.

The retail target includes extreme advertised prices up to $8,078,160. Because
the baseline neither clips nor log-transforms the target, its RMSE and R² are
especially sensitive to a small number of large errors. The status slices show
that overall performance is not uniform. The wholesale figures apply only to
the historical completed-sale auction population and are not interchangeable
with retail asking-price performance.

Reproduce both reports from the repository root after preparing the exact
reviewed local artifacts:

```powershell
$env:PYTHONPATH = "ml/src"
New-Item -ItemType Directory -Force models/repro | Out-Null
python -m autovalue_ml.modeling.baseline_cli retail --project-root . --output models/repro/retail-baseline-v1.json
python -m autovalue_ml.modeling.baseline_cli wholesale --project-root . --output models/repro/wholesale-baseline-v1.json
Get-FileHash -Algorithm SHA256 models/repro/*.json
```

Expected SHA-256 values are
`b5cae941ebb01d9766716d01a24acc75ad7d0432b05e8dde44a6200caffad28a`
for retail and
`b0be8b30367f7b7adca904d80610dd161b9b33dffd9e116d5030bd34403a3030`
for wholesale. Independent second runs matched the canonical reports
byte-for-byte.

Each command revalidates the reviewed source lineage, candidate, split manifest,
row accounting, and isolation before fitting. It writes only deterministic
aggregate JSON; no source rows, row-level predictions, estimator, coefficients,
or category vocabulary are persisted.

## 5. Candidate selection and error analysis

Phase 4 compares Random Forest and Gradient Boosting under the frozen
[model-selection protocol](experiments/phase4-model-selection-v1.json). XGBoost
is deferred unless neither scikit-learn challenger clears the material-gain
gate and a separate dependency review justifies its deployment footprint.
Hyperparameter search uses only the new development subsets, explicit candidate
sets, leakage-safe folds, and SHA-256-derived fixed seeds. The calibration
subsets and already opened Phase 3 outer holdouts cannot choose parameters.

The protocol is loaded through an exact SHA-256 and strict-schema gate. Its
sparse tree preprocessing always ends at CSR float32 without numeric scaling,
and its six Random Forest plus six Gradient Boosting tuples per track are fixed
before fitting. The aggregate
[partition audit](experiments/phase4-partition-audit-v1.json) verifies 98,552
retail development / 10,958 calibration rows and 391,641 wholesale development /
50,489 calibration rows. Target-free screening retains 29,619 retail and 97,909
wholesale development rows while preserving the required group and time
boundaries.

The frozen screening run is complete for both tracks. These figures are model
errors, not vehicle prices: Linear's predictions had a mean absolute error
(MAE) of **$13,666.84 USD** on the retail screening sample and **$2,711.94 USD**
on the wholesale forward screening sample. The
retail challenger shortlist is Random Forest 05/00 plus Gradient Boosting 05/02;
the wholesale shortlist is Random Forest 05/00 plus Gradient Boosting 03/04.
The aggregate-only reports and resumable checkpoints live in
[`docs/experiments`](experiments/README.md).

Full-development confirmation is also complete. Retail Random Forest 05 reduced
average absolute prediction error from Linear's **$11,654.99 USD MAE** to
**$10,269.78 USD MAE** and passed the frozen relative, absolute, RMSE, Certified,
New, and Used accuracy guardrails. Wholesale Linear remained best at
**$2,380.59 USD MAE**; every wholesale challenger was worse. Promotion remains
pending measured artifact-size, memory, startup, and latency gates. Calibration
and final holdout data remained excluded from every candidate fit and selection
metric. The retail calibration population was used only afterward for the
separately preregistered uncertainty layer below; the holdout was used later
only for the terminal frozen-system evaluation in Section 7.

Error analysis will cover vehicle age, mileage band, price band, popular versus
rare makes/models, missing-data patterns, and other supported attributes. It will
record latency, artifact size, serving memory, residual distributions, and weak
segments using aggregate, non-row-revealing outputs. Reports must retain target
semantics, historical period, wholesale-auction selection bias, missing retail
observation dates, and the wholesale source's unresolved USD assumption.

## 6. Prediction intervals

The first governed retail calibration is complete. Its policy was checksum
pinned before residuals were opened. Exact Phase 4 RF05 was fitted once on only
the 98,552 development rows and then predicted the 10,958 reserved calibration
rows. Calibration observations did not fit preprocessing or the estimator,
choose a model, retune RF05, or affect Yoad or River. The legacy holdout remained
unopened. No row, prediction, or residual was persisted.

Absolute residuals use the exact finite-sample order statistic
`ceil((n + 1) * coverage)` at all three preregistered levels. Five-fold
predictor-group cross-calibration derives each diagnostic radius from the other
four folds. The selected vehicle-status method with global fallback produced:

| Nominal coverage | Empirical coverage | Average width | Median width |
|---:|---:|---:|---:|
| 80% | 79.80% | $25,813.15 | $28,122.10 |
| 90% | 89.81% | $38,697.46 | $40,562.24 |
| 95% | 94.26% | $61,836.28 | $56,141.06 |

The more granular status-plus-predicted-value hierarchy was evaluated exactly
as preregistered and rejected: its worst overall gap was -2.10 percentage
points and its worst average-width ratio versus status calibration was 1.1319,
failing the -2-point and 1.05 gates. The runner did not tune another partition
after seeing those results.

The final row-free artifact stores global, status, predicted-value-band, and
status/value support and radii for 80%, 90%, and 95%, although serving selects
only status with global fallback. Its 90% radii are $16,737.64 for Certified,
$22,625.25 for New, $13,906.15 for Used, and $19,442.61 globally. Buckets below
400 rows cannot carry their own radius. Confidence labels use the 33rd and 67th
percentiles of cross-fitted 90% relative width plus minimum support of 1,000 and
400 rows; they are not probabilities. Missing mileage, rare/unseen categories,
and unsupported combinations remain separate warnings and do not silently add
an invented interval penalty.

The calibration is validated for integration with the frozen RF05 reference as
a marginal empirical interval, not as guaranteed conditional coverage.
Important limitations remain visible: the five 90% fold coverages range from
84.49% to 94.71%, the high-price slice covers 87.27%, and several adequately
supported manufacturers are materially below nominal. The UI and API must not
call this a KBB range, a guaranteed sale price, or a per-vehicle probability.
See the [human report](experiments/retail-rf05-calibration-v1.md) and
[aggregate JSON](experiments/retail-rf05-calibration-v1.report.json).

The separately preregistered sharpness follow-up is also complete. Leakage-safe
development OOF diagnostics confirmed heteroscedasticity: mean absolute residual
rose from $5,648.98 in the lowest predicted-value quartile to $18,500.38 in the
highest, a 3.27x ratio. The frozen comparison evaluated the current status-
conditional method, a development-only Gamma residual-scale model, and an
unfitted smooth predicted-value scale on the same calibration predictions and
predictor-group folds.

Both candidates remained close to nominal aggregate coverage, but neither met
the complete coverage-first gate set. At 90%, Gamma covered 89.93% with a
$37,427.42 mean displayed width and the smooth method covered 89.95% with a
$37,343.42 mean displayed width, versus 89.81% and $38,697.46 for the baseline.
Those reductions were only 3.46% and 3.67% on the preregistered unclipped-width
measure, below the required 10%, and their paired-bootstrap upper width-ratio
bounds did not clear 0.95. Both also introduced excessive broad-slice and
manufacturer coverage regressions. Gamma failed nine gates; the smooth method
failed seven.

The decision is therefore `retain_current_calibration_baseline`. No candidate
serving artifact or Gamma estimator was persisted, and the API/frontend remain
on their prior non-live integration state. At that point the final holdout was
still reserved.
See the frozen [sharpness policy](experiments/retail-rf05-uncertainty-sharpness-policy-v1.json),
[human report](experiments/retail-rf05-uncertainty-sharpness-v1.md), and
[aggregate comparison](experiments/retail-rf05-uncertainty-sharpness-v1.report.json).

## 7. Final frozen-system evaluation

Before final target access, the project froze the exact RF05 definition,
98,552-row development corpus, feature/preprocessing contract, calibration v1
artifact, confidence logic, metrics, slices, support thresholds, holdout
identity, decision gates, and upstream checksums in the
[final-evaluation policy](experiments/retail-rf05-final-evaluation-policy-v1.json).
All bindings passed before the test partition was requested. The 10,958
calibration rows did not fit RF05, and none of the 27,589 holdout rows fit or
calibrated any component.

The one-time final point results for historical U.S. advertised asking price in
USD are:

| Rows | MAE | RMSE | R2 | Median AE | p90 AE | p95 AE | Mean bias |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 27,589 | $10,575.36 | $34,118.14 | 0.4176 | $6,678.93 | $20,913.04 | $27,903.84 | -$1,863.46 |

MAE was 2.98% above development OOF and 6.53% above calibration. RMSE was
11.49% above development OOF and 65.89% above calibration, reflecting the long
asking-price tail and a $3.32 million maximum absolute miss. R2 was 0.0681 below
development OOF. Underprediction and overprediction rates were balanced at
49.50% and 50.50%, though the negative mean bias shows the largest errors skew
toward underprediction.

The unchanged status-conditional calibration produced:

| Nominal | Final coverage | Gap | Mean width | Median width | Lower clipping | Fallback |
|---:|---:|---:|---:|---:|---:|---:|
| 80% | 76.32% | -3.68 pp | $25,885.04 | $30,669.62 | 0.43% | 0.00% |
| 90% | 89.10% | -0.90 pp | $38,434.98 | $45,250.50 | 2.85% | 0.00% |
| 95% | 95.64% | +0.64 pp | $64,028.15 | $78,772.89 | 22.63% | 0.00% |

All intervals were finite, ordered, and contained their point prediction. The
system passed every preregistered significant-generalization gate but failed
four stricter portfolio gates: the 80% coverage gap was below -3 points, its
coverage regression versus calibration exceeded 3 points, minimum supported
manufacturer 90% coverage was 59.01% rather than at least 60%, and confidence
labels did not rank realized error or median width High <= Moderate <= Low.
High-confidence rows in fact had $14,296.67 MAE versus $9,234.78 Moderate and
$7,577.55 Low, so the labels must not be interpreted as accuracy probabilities.

Among manufacturers with at least 200 rows, Subaru and Infiniti had the lowest
MAE at $3,674.74 and $3,749.92. Mercedes and Porsche had the highest at
$24,373.31 and $23,013.34; Porsche's 59.01% 90%-interval coverage was the
minimum. The top predicted-value band also remained weak at $18,287.51 MAE and
77.83% 90% coverage. These are terminal diagnostics, not new tuning targets.

The final classification is **final evaluation passed with material
limitations**. The complete [human report](experiments/retail-rf05-final-holdout-v1.md),
[aggregate JSON](experiments/retail-rf05-final-holdout-v1.report.json),
[model card](model-cards/autovalue-retail-rf05-v1.md), and
[evidence manifest](experiments/retail-rf05-final-evaluation-v1.manifest.json)
are immutable final evidence. No post-holdout optimization occurred. Future ML
work requires a new development/evaluation cycle and a newly established future
evaluation boundary.

## 8. Explainability

Global importance is generated offline using permutation importance on held-out
data, or a model-native equivalent with its limitations documented. Local
explanations must be stable enough to describe the direction and relative effect
of inputs without implying causality.

Raw Linear Regression coefficients are not the product's feature-importance
measure: magnitude depends on scaling, category representation, and which
one-hot level is encoded. The future UI should use held-out permutation
importance and aggregate related transformed columns back to understandable
feature families such as make, model, mileage, and vehicle status.

## 9. Persistence and promotion

Persist the complete preprocessing/model pipeline with joblib only from trusted
training code. Produce a JSON manifest beside it and verify checksums and schema
compatibility at API startup. Exact dependency versions are part of the release.
The approved hosted service may load a private model artifact, but that artifact
must not be offered for download until publication permission receives a new
review.

There is still no persisted RF05 estimator bundle, prediction endpoint, or
model-info route. A checksum-bound, aggregate-only calibration artifact now
defines the future interval response, and the frontend contains a clearly
labeled static example of that presentation. Neither is a live model-backed
prediction workflow.

A model is promotable only when:

- data and license gates pass;
- development cross-validation and the labeled legacy-holdout audit are reproducible;
- leakage and slice checks pass;
- interval coverage is measured;
- serialization round-trip predictions match;
- unknown categories are handled;
- memory, startup, artifact-size, and latency budgets pass; and
- the model card documents intended use and limitations.

## 10. Future online-learning boundary

Online learning is a later, separately validated capability. The acquisition
layer currently prepares immutable `TrainingRecordEvent` objects only after the
ML-reuse gate rechecks permission, policy lineage, price semantics, and currency.
Events carry deterministic per-run IDs, stable content-deduplication keys, and
separate acquisition-policy and ML-grant hashes. They can expose the
`(features, target)` pair used by River's `learn_one(x, y)` interface.

River is isolated as a project dependency for the completed synthetic shadow
simulation, and acquisition never updates it or another model automatically.
No real-world outcome source is approved. A future consumer must add a governed
live source, durable append-only storage, idempotent replay, delayed-label
validation, drift and performance monitoring, challenger evaluation, promotion
criteria, and rollback before online updates can influence production
predictions.

## 11. Governed candidate expansion

Phase 4 screening and full-development confirmation are complete and were not
restarted for the Hugging Face work. Their frozen definitions, checkpoints, and
reports remain unchanged.

Yoad22's revision-pinned Austin Reese/Craigslist derivative is approved only for
controlled offline batch experimentation. Online/River learning remains blocked.
The approval excludes 7,695 unknown-manufacturer rows, preserves attribution,
uses a no-model common feature contract, and requires pooled predictor-group
folds. Carson-Shively's reviewed bronze layer has 4,009 rows—not 7,970
independent observations—and remains blocked from batch and online learning
until its upstream origin, U.S. scope, USD target semantics, and license
metadata are resolved.

The controlled comparison used the 98,552-row Cars development partition and
242,666 approved Yoad rows. It did not use 10,958 calibration rows or the legacy
holdout. With paired five-fold grouped CV, the Cars-only broad model scored
$10,121.05 pooled MAE versus $7,920.62 for the combined model. Yoad-domain MAE
improved from $8,491.11 to $5,388.76, while Cars-domain MAE changed from
$14,134.49 to $14,154.84, a 0.14% degradation. The Cars fold-MAE standard
deviation improved, but the highest-mileage Cars band degraded 5.62% and several
manufacturer and age slices also regressed.

This is evidence that Yoad improves generalization to the evaluated mixed
historical population, not evidence that it improves every Cars.com segment or
represents today's market. That result triggered the separately defined
source-composition confirmation below; it was not promoted, persisted, or
substituted for the Phase 4 retail RF05 result. The aggregate report is
[`yoad22-controlled-batch-v1.json`](experiments/yoad22-controlled-batch-v1.json).
No Phase 4 definition, checkpoint, challenger, result, or model-selection
decision was modified.

The separate source-composition confirmation then compared exact nested Yoad
samples of 98,552 rows (balanced), 150,000 rows (moderate), and 242,666 rows
(full) against Cars-only. Manufacturer, exact-year, and mileage-decile sampling
was deterministic, target-free, and kept maximum share drift below 0.035
percentage points. All four treatments used the same validation population and
fold assignments; the checksum-bound Cars-only and full endpoints were reused
without refitting.

Moderate augmentation is the preferred composition. Compared with Cars-only,
it improves Cars MAE from $14,134.49 to $14,012.21 (0.87%), Yoad MAE from
$8,491.11 to $5,445.65 (35.87%), and pooled MAE from $10,121.05 to $7,919.88.
It retains 98.17% of the full augmentation's Yoad gain and lowers Cars fold-MAE
standard deviation from $1,453.34 to $1,250.03. This choice is based first on
Cars accuracy, critical Cars slices, and fold stability—not pooled MAE alone.

Moderate remains a separate experimental model and is not eligible for final
promotion evaluation. Its highest-mileage Cars band degrades 4.79%; Jaguar,
Hyundai, Chevrolet, low-mileage Cars, and ages 3–8 also regress. Sixteen of 36
reported Cars manufacturers worsen even though all 37 reported Yoad
manufacturers improve. The complete aggregate evidence is in the
[confirmation report](experiments/yoad22-source-composition-confirmation-v1.json)
and [interpretation](experiments/yoad22-source-composition-confirmation-v1.md).
Yoad online/River learning remains blocked, Carson remains excluded, and Phase
4 RF05 remains unchanged.

A subsequent weighting confirmation kept that exact moderate composition and
reused its verified result without refitting. Three RF05 treatments derived
training weights independently inside each fold: equal source totals, equal
source totals plus mileage-distribution alignment, and equal source totals plus
broad mileage/age/manufacturer-support alignment. Price never determined a
weight, validation observations remained unweighted, and the complete protected
fold and feature methodology stayed fixed.

All three treatments improved aggregate Cars MAE and retained more than 97% of
moderate's Yoad-domain gain. The broad segment treatment improved eight of nine
focus slices, reduced highest-mileage Cars degradation from 4.79% to 3.41%, and
lowered Cars MAE to $13,889.84. It nevertheless produced a 7.43% Genesis
regression, exceeding the preregistered 5% all-slice ceiling. Source balancing
also caused an 8.24% Hyundai regression, while mileage weighting caused a 14.00%
Genesis regression. Weighting is therefore rejected and the unweighted moderate
branch remains unchanged. The complete [weighting report](experiments/yoad22-training-weight-confirmation-v1.md)
and [aggregate JSON](experiments/yoad22-training-weight-confirmation-v1.json)
preserve the evidence and decision.
