# Dataset strategy

## U.S. and USD invariant

AutoValue AI is scoped exclusively to the 50 U.S. states and Washington, D.C.
Every accepted target, prediction, prediction range, MAE, and RMSE is represented
in U.S. dollars (`USD`). Prices are normalized as integer USD cents before they
cross the acquisition boundary and converted to USD values for model training.
Non-U.S. rows and records labeled with a non-USD currency cannot enter training.

For the wholesale source, `sellingprice` is operationally treated as USD only
after the U.S./D.C. geographic gate. The source file does not independently
declare its currency, so that fact remains documented as a source-semantic
assumption even though the AutoValue product contract and all outputs are USD.

## Decision status

[US Sales Cars Dataset v2](https://www.kaggle.com/datasets/juanmerinobermejo/us-sales-cars-dataset)
is the historical retail asking-price candidate. Its version-pinned CSV contains
144,867 New, Used, and Certified U.S. listings; 140,956 have a valid positive
price before exact deduplication and 137,099 remain after removing 3,857 exact
duplicates. Status remains an explicit feature and evaluation slice. Missing
mileage is retained for pipeline-based handling, including 85,054 target-valid
New rows and two target-valid Used rows.

[Kaggle Vehicle Sales Data v1](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data/data)
is the separate wholesale completed-sale candidate. It contains 558,837
historical auction records with vehicle year, make, model, trim, body,
transmission, condition, odometer, state,
sale date, MMR, and selling price.

These sources support two different honest product claims. The listing candidate
can support a **historical U.S. advertised asking-price model for 2023 source
snapshots**. The auction candidate can support a **historical U.S.
wholesale-auction sale-price model for the 2014–2015 source period**. Neither
establishes current live consumer market value, and their rows must not be merged
under one label. Phase 3 has now fit and evaluated separate reproducible
baselines; neither result is a current-market value claim.

The raw data mixes the United States, Canada, and Puerto Rico. Only rows mapped
to one of the 50 U.S. states or Washington, D.C. may cross the geographic gate:

| Stage | Rows |
|---|---:|
| Source CSV | 558,837 |
| 50 U.S. states plus Washington, D.C. | 550,410 |
| U.S./D.C. rows with valid target and sale date | 550,398 |
| U.S./D.C. rows complete for the initial core fields | 529,245 |

The observed dates run from 2014-01-01 through 2015-07-21. The source does not
explicitly state its currency. AutoValue AI will treat `sellingprice` as USD only
for validated U.S./D.C. rows and record this as an open semantic assumption until
the uploader or upstream source confirms it.

## Permission and provenance decision

Kaggle displays an MIT license label for v1, initially released 2024-02-21. The
project owner attests that they spoke directly with the uploader and were told
the data is legal to use. This is meaningful project evidence, but the project's
review has not independently verified upstream ownership or the uploader's chain
of title.

Approval is therefore bounded:

| Use | Decision |
|---|---|
| Official Kaggle download and private local storage | Approved |
| Private cleaning, transformation, training, and evaluation | Approved |
| Public aggregate metrics and non-row-revealing charts | Approved |
| Hosted inference | Approved |
| Redistribution of raw or processed rows | Pending |
| Publication of a downloadable trained model | Pending |
| Sublicensing or commercial use | Pending |

The machine-readable
[review record](data-reviews/kaggle-vehicle-sales-data-v1.review.json) records
the evidence, exact artifacts, allowed uses, unresolved provenance, and required
gates. It is not a legal conclusion or a general sublicense. The project's MIT
code license never covers third-party data or model artifacts derived from it.

### Asking-price source decision

Kaggle labels US Sales Cars v2 as Apache-2.0 and links the author's
[upstream cleaning repository](https://github.com/juanmerino89/cars-data-cleaning),
which identifies historical Cars.com extraction as the origin. The project owner
attests that they were in contact with the relevant source and that both the
historical collection and this noncommercial portfolio ML use were authorized.
That attestation separately approves acquisition and ML reuse of this exact
artifact; it does not authorize AutoValue AI to scrape Cars.com.

Its scoped publication restrictions match the wholesale source: private local
transformation, training/evaluation, aggregate results, and hosted inference are
approved, while row redistribution, downloadable-model publication,
sublicensing, and commercial use remain pending. The complete decision is in the
[US Sales Cars v2 review record](data-reviews/kaggle-us-sales-cars-v2.review.json).

The exact official KaggleHub version-2 download is
`data/raw/kaggle_us_sales_cars_v2/cars.csv`: 17,171,976 bytes, 144,867 data rows,
UTF-16 with a BOM, and SHA-256
`25854afc3ef8b6c6a0349bf7f422c40dacb9bec60a8b318462737ebf9edcc5ea`.
It must expose exactly `Brand`, `Model`, `Year`, `Status`, `Mileage`, `Dealer`,
and `Price`. Dealer is dropped, exact duplicates are removed, `Price` becomes a
USD asking-price target, and `Status` becomes explicit `vehicle_status` values
of New, Used, or Certified.
Mileage remains optional and is imputed only inside training folds.

## Immutable artifact gate

Use Kaggle's official download and preserve the exact source files. The reviewed
artifacts are:

| Artifact | Local path | Bytes | SHA-256 |
|---|---|---:|---|
| Original archive | `data/raw/kaggle_syedanwarafridi_vehicle-sales-data_v1/vehicle-sales-data-v1.zip` | 19,753,181 | `8eb8e42023ee7255818b31ac1a716b438e7bd9298116b98c250937754418c8b1` |
| CSV | `data/raw/kaggle_vehicle_sales_v1/car_prices.csv` | 88,047,552 | `32ba3ce51664e6a12c0c927ed193b41e3c4743fdf18bc0317389892aed27f556` |

The CSV has 558,837 data rows. A byte-size, checksum, or row-count mismatch
means the artifact is not the reviewed version and ingestion must fail pending a
new review. Raw files are immutable, ignored by Git, and never redistributed by
this repository.

## Mandatory processing gates

The published private candidates and split sets have passed the following
reproducible acquisition and split requirements. The same checks run again when
a modeling stream is opened:

1. Verify the exact artifact size and SHA-256 before parsing.
2. Normalize state values against an explicit allowlist containing the 50 states
   and Washington, D.C.; reject Canada, Puerto Rico, unknown, and malformed
   regions.
3. Require a parseable sale date and a positive `sellingprice` target.
4. Remove `vin` and `seller` from feature tables, reports, examples, logs, and
   published artifacts.
5. Forbid `mmr` from preprocessing, features, feature selection, tuning,
   explanations, and inference. MMR is an existing valuation estimate and would
   create severe target leakage.
6. Use a chronological test boundary and ensure each normalized VIN belongs to
   only one split. Cross-validation must also respect time and VIN groups.
7. Fit every imputer, encoder, category grouper, and learned transformation only
   on the relevant training fold.
8. Keep the USD interpretation visible as an unresolved semantic assumption.
9. Label all metrics and predictions as historical wholesale-auction results for
   the source period; do not present them as current retail appraisals.
10. Publish aggregate evidence only. Do not commit source rows, processed rows,
    row-level examples, or downloadable model artifacts without a new review.

Passing the legal-use decision does not waive these privacy, leakage, geography,
quality, and reproducibility gates.

## Verified holdout decisions

The two targets use separate private assignments, manifests, readiness markers,
and training streams. Neither an acquisition `.ready.json` file nor a general
ML-reuse approval is enough to bypass the source-specific split verifier.

| Track | Candidate rows | Train | Test | Split design |
|---|---:|---:|---:|---|
| Historical retail asking price | 137,099 | 109,510 | 27,589 | Deterministic, status-stratified predictor-group holdout |
| Historical wholesale completed sale | 540,764 | 442,130 | 98,634 | 2015-06-01 chronological holdout with private VIN isolation |

The retail grouping key is year, make, model, mileage (including null), and
vehicle status. It isolates all 56,529 predictor groups and retains New, Used,
and Certified in both partitions: test contains 16,425 New, 9,646 Used, and
1,518 Certified rows. Because the source has no row-level observation dates or
stable upstream listing IDs, this evaluates reproducible grouped generalization,
not forward-in-time performance. Its private assignment SHA-256 is
`5b3e39d0ef418c07b0c4d08ecc18700fc9f387518a21dbd604f515463cb5ebe5`;
the manifest SHA-256 is
`c60bf010fb47dff44d03b5da80b191ddb4b748661cb5cf02397422fdbaaf3466`.

For wholesale, rows dated on or after 2015-06-01 seed test. If a private VIN
group contains any such row, every row in that group goes to test. This promoted
1,066 pre-cutoff rows from 1,039 groups. Verification found zero VIN overlap,
zero post-cutoff train rows, and zero train VIN groups crossing ordered CV
buckets. Train bucket row counts are 51,586 `warmup`, 134,449 `2015_01`,
158,432 `2015_02`, 47,174 `2015_03_04`, and 50,489 `2015_05`. Its committed
[split policy](data-reviews/kaggle-vehicle-sales-v1.split.json) is separately
pinned. The private assignment SHA-256 is
`a96909345612f5ddc5665c4d6817d2c8f0dd6d59c3a84fc523cb82b6adeeb5f2`;
the manifest SHA-256 is
`d0dd0c24f342a8a45c1f89419780f470d0f152d61cf5dd54b2cb786df9525bd3`.

These counts and hashes establish artifact and split integrity. Phase 3 then
consumed the candidates only through these split-aware gates, selected each model
from training-partition CV, and scored each final holdout once.

## Baseline modeling evidence

The two targets retain separate feature contracts, CV schemes, selection
decisions, and reports. Linear Regression beat the median dummy by CV MAE for
both tracks:

| Track and evaluation | Model | Rows | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| Retail grouped CV | Median dummy | 109,510 | $21,694.48 | $42,464.18 | -0.0206 |
| Retail grouped CV | Linear Regression | 109,510 | $11,552.82 | $31,408.86 | 0.4417 |
| Retail untouched holdout | Linear Regression | 27,589 | $12,040.29 | $35,452.63 | 0.3711 |
| Wholesale forward CV | Median dummy | 390,544 | $7,009.35 | $9,876.08 | -0.0822 |
| Wholesale forward CV | Linear Regression | 390,544 | $2,382.13 | $4,028.35 | 0.8200 |
| Wholesale untouched holdout | Linear Regression | 98,634 | $2,256.02 | $4,014.77 | 0.8502 |

Retail holdout slices are Certified: 1,518 rows, $9,428.31 MAE, $19,805.13
RMSE, and 0.7257 R²; New: 16,425 rows, $12,641.75 MAE, $32,691.63 RMSE, and
0.3127 R²; Used: 9,646 rows, $11,427.17 MAE, $41,392.40 RMSE, and 0.2906 R².
See the [interpretation and reproduction guide](results/README.md) and canonical
[retail](results/retail-baseline-v1.json) and
[wholesale](results/wholesale-baseline-v1.json) reports.

CV performs model selection inside the outer training data; the untouched
holdout is not another tuning fold. Retail evaluation is predictor-grouped but
non-temporal because the source lacks row dates and stable listing IDs.
Wholesale validation and holdout preserve chronology and private VIN isolation.
The v2 feature contracts preserve raw `model_year`, preventing adjacent or
future years from colliding after transformation; the retail baseline passed
with zero transformed predictor-group overlap.

No target clipping, outlier removal, or log transformation was applied. Retail's
extreme advertised prices make RMSE and R² sensitive to large errors. The
results remain historical asking-price and completed-auction-sale benchmarks,
not current consumer valuations. No model artifact or prediction endpoint exists
yet, and raw linear coefficients must not be presented as feature importance;
future product explanations should use held-out, grouped permutation importance.

## Product-schema fit

The asking-price candidate supports `Year`, `Brand`, `Model`, optional `Mileage`,
and explicit `Status` (`New`, `Used`, or `Certified`). It does not contain engine,
drivetrain, accident history, owner count, vehicle type, stable listing IDs,
row-level geography, or row-level observation dates. `Dealer` is deliberately
removed. The retail model and UI cannot fabricate unsupported inputs.

The wholesale feature contract adds the fields shown below:

| Requested input | Candidate support | v1 decision |
|---|---|---|
| Year | `year` | Candidate feature after range validation |
| Make | `make` | Candidate categorical feature |
| Model | `model` | Candidate categorical feature |
| Mileage | `odometer` | Candidate numeric feature after unit/range audit |
| Condition | `condition` | Candidate feature only after scale semantics are documented |
| Engine | Missing | Omit from v1 |
| Drivetrain | Missing | Omit from v1 |
| Accident history | Missing | Omit from v1 |
| Number of owners | Missing | Omit from v1 |
| Vehicle type | `body` | Candidate categorical feature after normalization |
| Transmission | `transmission` | Candidate categorical feature |
| Target price | `sellingprice` | Historical wholesale auction sale price; USD inferred for U.S./D.C. rows |
| Existing valuation | `mmr` | Permanently forbidden from model inputs |

Unsupported inputs will not be fabricated. The React form and API schema should
be generated from the final audited feature contract, not from the aspirational
original list.

## Quality and representativeness audit

The source-specific adapters currently report:

| Track | Raw rows | Accepted | Quarantined | Exact duplicates removed |
|---|---:|---:|---:|---:|
| Retail asking price | 144,867 | 137,099 | 3,911 | 3,857 |
| Wholesale completed sale | 558,837 | 540,764 | 18,073 | 0 |

The retail accepted rows comprise 81,278 New, 48,229 Used, and 7,592 Certified
listings. The wholesale quarantine includes 8,401 non-U.S. rows; its accepted
artifact exposes none of VIN, seller, MMR, or transmission. These are ingestion
facts, not model-quality results.

In the accepted retail candidate, 81,280 mileage values are missing after
deduplication. Its median asking price is $46,984, while 41 records exceed
$500,000 and the maximum is $8,078,160. The accepted wholesale candidate has
155 missing mileages, 11,496 missing condition values, and a $12,200 median
completed-sale price. Both tracks contain $1 targets. These tails stay visible
through splitting; any outlier rule must be learned from training data only and
reported rather than silently baked into acquisition.

For the asking-price source, modeling and evaluation must emphasize status imbalance, the
structural absence of mileage for New listings, missing observation dates and
stable IDs, exact duplicates across historical snapshots, extreme luxury prices,
and the difference between an advertised price and a completed transaction.
Metrics must be reported separately for New, Used, and Certified slices.

Across both sources, model reports must quantify types, units, parsing failures,
missingness, category cardinality, exact and near duplicates, VIN recurrence,
implausible years, odometer and price outliers, target skew, and coverage across
makes, models, vehicle ages, U.S. regions, and price bands. It must also document
wholesale auction selection bias, the short historical time window, unknown
currency metadata, and any unexplained condition coding.

The 529,245 core-complete rows are a profiling fact, not an instruction to drop
all incomplete records. Missing-value strategies must be learned inside the
pipeline and compared with a clearly documented complete-case baseline.

## Reproducibility and publication policy

Every transformation must be deterministic code. The audit will record source
version, retrieval/review date (2026-08-28), hashes, filtering counts, column
semantics, USD assumption, split boundaries, fixed seeds, and dependency
versions. Notebooks may investigate but cannot be the only implementation.

Aggregate metrics, residual charts, feature importance, and slice results may be
published when they cannot reconstruct individual source rows. Example API and
UI records must be synthetic. A deployed service may return hosted inference,
but neither the source data nor model artifact may be exposed for download under
the current decision.

Approved future records may eventually cross the existing explicit gate into
append-only, River-compatible `learn_one(x, y)` events. Acquisition never trains
or updates a model automatically.

## Other reviewed sources

### Hugging Face candidates

The pinned Yoad22 Craigslist and Carson-Shively sources have been acquired and
profiled independently. Yoad contains 250,361 cleaned Austin Reese/Craigslist
rows and has no known upstream overlap with the current Cars.com-derived retail
source. After excluding 7,695 unknown manufacturers, 242,666 rows are approved
only for controlled offline batch experimentation. The controlled run, separate
source-composition confirmation, and fold-local weighting confirmation use a
common no-model feature contract and pooled predictor-group folds; online reuse
remains blocked.

Carson's authoritative bronze layer contains 4,009 raw-format rows. The
repository's additional 3,961-row silver layer is a transformation of that
source, not an independent corpus to concatenate. Carson has valuable
model/engine/accident/title coverage, but no documented upstream origin,
row-level geography, observation time, or confirmed U.S./USD semantic contract.
Batch and online reuse therefore fail closed.

The full [candidate review](data-reviews/hugging-face-candidates.md) compares
prices, years, mileage, makes/models, geography, missingness, category coverage,
outliers, and coarse cross-source collision keys against current retail. The
confirmation prefers the 150,000-row moderate augmentation as a separate
experimental model: it improves aggregate Cars and Yoad accuracy but retains
material Cars slice regressions, so it is not eligible for final promotion
evaluation. Conservative training weights improved aggregate Cars error but
introduced new severe manufacturer regressions, so weighting was rejected and
the unweighted moderate branch remains unchanged. Carson and the broader
multi-source experiment remain blocked.

The City of Seattle
[Sold Fleet Equipment](https://cos-data.seattle.gov/City-Administration/Sold-Fleet-Equipment/y6ef-jf2w)
dataset remains useful as a public-domain licensed-ingestion smoke test. Its 267
municipal-fleet records lack mileage and cannot establish national model quality.

A future [GSA fleet-sales](https://www.gsa.gov/buy-through-us/products-and-services/transportation-and-logistics-services/fleet-management/vehicle-leasing/sales-of-gsa-fleet-vehicles)
corpus should come from an official bulk release or direct request, not scraping.
UCI Automobile remains too old and lacks the needed model year, model, and
mileage coverage. MUCars-2024 remains rejected because it represents Morocco.
The synthetic dealership remains test plumbing, never model evidence.

FuelEconomy.gov [web services](https://www.fueleconomy.gov/feg/ws/index.shtml)
or [downloadable data](https://www.fueleconomy.gov/feg/download.shtml) may later
enrich an independently approved U.S. target dataset after a separate license,
join-quality, and leakage review. They cannot supply the sale-price target.
