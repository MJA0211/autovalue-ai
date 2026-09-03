# Machine-learning package

`ml/src/autovalue_ml` owns reusable, deterministic code for dataset
validation, cleaning, feature engineering, training, evaluation, interval
calibration, and safe inference contracts.

The package currently exposes its version and a tested, permission-gated
acquisition slice. It includes strict local licensed-dataset loading, reviewed
adapter registration, policy and HTTP boundaries, pure parsing, a shared
vehicle-listing contract, schema validation, normalization, deduplication,
malformed-record quarantine, response caching, provenance writing, and ingestion
metrics. Its only enabled scraping adapter is a project-owned synthetic
dealership served on loopback; it contains no commercial marketplace adapter.

The common contract is U.S.-only: every normalized listing must use
`market_country="US"` and `currency="USD"`. Non-U.S. manifests, policies, and
records fail closed before training-event creation.

Two checksum-pinned Kaggle artifacts are integrated behind source-specific
adapters and verified split gates. Their labels remain separate: historical U.S.
retail asking price for New, Used, and Certified listings, and historical U.S.
wholesale completed-sale price. Dealer, seller, VIN, and MMR do not enter model
features.

The retail split assigns 109,510 rows to train and 27,589 to test while keeping
each year/make/model/mileage/status predictor group in one partition. It is
deterministic and status-stratified, but non-temporal because the source supplies
neither row-level dates nor stable upstream listing IDs. The wholesale split
assigns 442,130 rows to train and 98,634 to test at a 2015-06-01 boundary while
keeping each private VIN group in one partition. Its train stream also exposes
ordered, VIN-isolated CV buckets for forward validation.

Modeling callers must use
`prepare_kaggle_us_sales_cars_split_training_rows(...)` or
`prepare_kaggle_vehicle_sales_training_rows(...)`. Both functions reverify the
candidate, review, manifest, readiness marker, checksums, row accounting, and
split isolation before yielding allowlisted features and a target. The unsplit
normalized candidates are not training interfaces.

External crawling is disabled pending address-pinned transport. Acquisition and
ML training use separate permission checks and fingerprints, and no scraped or
loaded record trains a model automatically. An explicit bridge revalidates every
nested record and can produce immutable, content-deduplicated events and
`(features, target)` pairs compatible with River's `learn_one` interface. River
is not currently imported or installed, and the bridge performs no model update.

Run it from the repository root:

```powershell
$env:PYTHONPATH = "ml/src"
python -m autovalue_ml.acquisition.demo
```

The three-page demo recovers from an injected 503 and 429, skips one exact
duplicate, accepts four listings (including one with missing optional fields),
and quarantines two malformed cards. It writes normalized, quarantine, and
manifest artifacts beneath `data/interim/`. Full design and artifact details are
in `docs/data-acquisition.md`.

The modeling package now implements exact feature allowlists, stateless feature
engineering, sparse fold-local preprocessing, grouped retail CV, forward-only
wholesale CV, median-dummy and Linear Regression pipelines, MAE/RMSE/R²
evaluation, retail status slices, strict aggregate report validation, and a
source-pinned CLI. Training dependencies remain outside the backend-only
requirement set where practical.

The v2 feature contracts preserve raw `model_year`, while clipped age is used
only to calculate mileage-per-year. This prevents adjacent and future model
years from colliding after transformation. The retail experiment recomputes its
grouping key through that feature contract and rejects any outer train/holdout
overlap; the completed real-data run had zero transformed predictor-group
overlap.

Phase 3 selected Linear Regression by lowest cross-validation MAE on both
independent targets:

| Track and evaluation | Model | Rows | MAE (USD) | RMSE (USD) | R² |
|---|---|---:|---:|---:|---:|
| Retail grouped CV | Median dummy | 109,510 | $21,694.48 | $42,464.18 | -0.0206 |
| Retail grouped CV | Linear Regression | 109,510 | $11,552.82 | $31,408.86 | 0.4417 |
| Retail untouched holdout | Linear Regression | 27,589 | $12,040.29 | $35,452.63 | 0.3711 |
| Wholesale forward CV | Median dummy | 390,544 | $7,009.35 | $9,876.08 | -0.0822 |
| Wholesale forward CV | Linear Regression | 390,544 | $2,382.13 | $4,028.35 | 0.8200 |
| Wholesale untouched holdout | Linear Regression | 98,634 | $2,256.02 | $4,014.77 | 0.8502 |

The wholesale CV count excludes the 51,586-row warmup bucket, which trains the
first forward fold but is never itself validation data. Retail CV is grouped but
non-temporal; wholesale CV and holdout are chronological and VIN-isolated. Both
experiments use raw dollar targets without target clipping, outlier removal, or
log transformation. The retail target's extreme tail makes its RMSE and R²
particularly sensitive to large errors. Asking prices and completed auction
sales remain different labels and are never pooled.

Reproduce the canonical reports from the repository root after preparing the
reviewed private artifacts:

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

See the [full result tables and limitations](../docs/results/README.md),
[`retail-baseline-v1.json`](../docs/results/retail-baseline-v1.json), and
[`wholesale-baseline-v1.json`](../docs/results/wholesale-baseline-v1.json).
The commands persist only deterministic aggregate JSON, not fitted estimators or
row-level predictions.

There is no promoted or persisted point-estimator bundle or prediction endpoint.
Phase 4 is executing its frozen development/calibration protocol. A checksum-
bound, row-free vehicle-status conformal artifact now defines the validated
interval behavior for eventual RF05 integration, but it is not yet connected to
a live estimator. The
strict protocol loader and sparse float32 tree
preprocessor, 12 explicit nonlinear candidates per track, deterministic
calibration/screening partitions, split-conformal primitives, and pure promotion
gate are implemented. The real retail boundary is 98,552 development / 10,958
calibration rows with a 29,619-row screening sample; wholesale is 391,641 /
50,489 with a 97,909-row screening sample. It will produce a checksummed private
bundle only after every promotion criterion passes. The already published Phase
3 holdouts are legacy audits, not Phase 4 tuning inputs. Raw Linear
Regression coefficients are not feature importance; offline held-out permutation
importance should be grouped across related one-hot columns before a future UI
uses it.

All 13 candidates per track have completed development-only screening. These
are average prediction errors, not vehicle prices: Linear led with retail MAE
of **$13,666.84 USD** and wholesale MAE of **$2,711.94 USD**. Retail shortlisted
Random Forest 05/00 and Gradient Boosting 05/02; wholesale shortlisted Random
Forest 05/00 and Gradient Boosting 03/04. The reports and hash-bound resumable
checkpoints are in [`docs/experiments`](../docs/experiments/README.md). These
numbers are screening evidence, not final performance claims.

Full-development confirmation evaluated only Linear and those four challengers.
Retail Random Forest 05 led at **$10,269.78 USD MAE** versus Linear at
**$11,654.99 USD MAE**, passing every frozen accuracy guardrail. Wholesale
Linear remained best at **$2,380.59 USD MAE**, so no wholesale challenger can be
promoted. Deployment measurements are still required before the retail model is
selected and persisted; calibration and legacy holdout rows remain isolated.

The first retail RF05 calibration and its separately preregistered sharpness
follow-up are complete. The baseline status-conditional intervals achieved
79.80%, 89.81%, and 94.26% cross-fitted coverage at 80%, 90%, and 95%, with
mean displayed widths of $25,813.15, $38,697.46, and $61,836.28. Development
OOF diagnostics then showed a 3.27x lowest-to-highest predicted-value-quartile
residual ratio, motivating two heteroscedastic candidates.

Neither candidate passed the complete frozen gate set. The normalized Gamma
scale failed nine gates and the simple smooth value scale failed seven,
including the required 80%/90% sharpness, paired-bootstrap, and conditional-
coverage checks. The current vehicle-status calibration is retained, and the
runner correctly persisted neither a v2 serving artifact nor a Gamma joblib.
The aggregate policy, comparison, and interpretation are in
[`docs/experiments`](../docs/experiments/README.md). This experiment did not
emit or evaluate a legacy-holdout row and did not touch Yoad, River,
AutoTrader, or Carson-Shively.

Reproduce or resume screening from the repository root:

```powershell
$env:PYTHONPATH = "ml/src"
python -m autovalue_ml.modeling.phase4_screening_cli retail --project-root . --output docs/experiments/phase4-retail-screening-v1.json --force
python -m autovalue_ml.modeling.phase4_screening_cli wholesale --project-root . --output docs/experiments/phase4-wholesale-screening-v1.json --force
python -m autovalue_ml.modeling.phase4_confirmation_cli retail --project-root . --output docs/experiments/phase4-retail-full-development-v1.json --force
python -m autovalue_ml.modeling.phase4_confirmation_cli wholesale --project-root . --output docs/experiments/phase4-wholesale-full-development-v1.json --force
```

Each command revalidates the reviewed source and split lineage, recomputes the
target-free assignment hashes, and atomically updates a neighboring
`.checkpoint.json` after each candidate. A later run accepts only an exact,
policy-bound candidate prefix and resumes at the next candidate.
