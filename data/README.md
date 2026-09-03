# Data

## Status

**Two Kaggle artifacts are conditionally approved as separate training
candidates.** AutoValue AI is a United States-only product: every accepted source,
normalized listing, and future training event must declare
`market_country="US"`. For this source, USD is an explicit modeling assumption
for validated U.S. rows because the source does not state a currency.

The wholesale artifact contains 558,837 rows. Processing may retain only the 50
U.S. states and Washington, D.C.; Canada and Puerto Rico must be rejected. The
resulting target is a 2014–2015 historical wholesale-auction sale price, not a
current consumer retail price. Its chronological, VIN-isolated holdout is now
verified. Phase 3 trained and evaluated a transient Linear Regression baseline;
no estimator or downloadable model artifact was persisted.

Kaggle labels the dataset MIT. The project owner also attests that the uploader
directly confirmed legal use, although underlying ownership has not been
independently verified. The scoped approval covers official download, private
local storage and transformation, ML training/evaluation, aggregate public
results, and hosted inference. Do not publish raw rows, transformed rows, or a
downloadable model, and do not sublicense or use the data commercially, without
a new permission review. The complete decision is in the
[dataset review record](../docs/data-reviews/kaggle-vehicle-sales-data-v1.review.json).

The additional
[US Sales Cars v2 review](../docs/data-reviews/kaggle-us-sales-cars-v2.review.json)
covers a version-pinned, UTF-16 CSV of historical U.S. asking-price listings.
New, Used, and Certified status values are retained explicitly to
broaden coverage; 140,956 rows have a valid target before exact deduplication.
Dealer is removed, exact duplicates are deduplicated, and missing or invalid
core fields are quarantined. AutoValue AI does not scrape Cars.com: the project
owner's attestation covers the historical
source collection and noncommercial portfolio ML reuse of this fixed artifact.

MUCars-2024 remains rejected because it represents Morocco. The City of Seattle
[Sold Fleet Equipment](https://cos-data.seattle.gov/City-Administration/Sold-Fleet-Equipment/y6ef-jf2w)
dataset remains a public-domain ingestion smoke-test candidate only; it lacks
mileage and cannot support national valuation claims.

A future
[GSA fleet-sales dataset](https://www.gsa.gov/buy-through-us/products-and-services/transportation-and-logistics-services/fleet-management/vehicle-leasing/sales-of-gsa-fleet-vehicles)
should be obtained by an official bulk release, direct data request, or other
expressly permitted method; the project will not scrape the GSA auction site.
The historical
[UCI Automobile dataset](https://archive.ics.uci.edu/dataset/10/automobile) is
legally clear but lacks the model year, model, mileage, and modern market coverage
needed for AutoValue AI.

## Local layout

```text
data/
|-- README.md
|-- manifest.example.json
|-- raw/          immutable source files; ignored by Git
|-- interim/      generated intermediate files; ignored by Git
|-- processed/    model-ready generated files; ignored by Git
`-- fixtures/     tiny, clearly synthetic test data only
```

Never edit files under `raw/`. Acquisition and transformations must be
repeatable in code. Real listings must not be added to `fixtures/`.

## Place the Kaggle files

Download through Kaggle's official dataset page and put the extracted CSV at
this exact Git-ignored path:

```text
data/raw/kaggle_vehicle_sales_v1/car_prices.csv
```

If retaining the original download archive for provenance, store it separately
at:

```text
data/raw/kaggle_syedanwarafridi_vehicle-sales-data_v1/vehicle-sales-data-v1.zip
```

Do not rename, edit, open-and-resave, or commit either source artifact. Before
processing, the files must match the reviewed values:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `vehicle-sales-data-v1.zip` | 19,753,181 | `8eb8e42023ee7255818b31ac1a716b438e7bd9298116b98c250937754418c8b1` |
| `car_prices.csv` | 88,047,552 | `32ba3ce51664e6a12c0c927ed193b41e3c4743fdf18bc0317389892aed27f556` |

The reviewed CSV has 558,837 data rows. A mismatch means the artifact is a
different version and must stop ingestion pending a new review.

The second artifact was downloaded through KaggleHub version 1.0.2 and is stored
at this exact Git-ignored path:

```text
data/raw/kaggle_us_sales_cars_v2/cars.csv
```

It must remain 17,171,976 bytes with SHA-256
`25854afc3ef8b6c6a0349bf7f422c40dacb9bec60a8b318462737ebf9edcc5ea`,
contain 144,867 data rows, use UTF-16 with a BOM, and expose exactly `Brand`,
`Model`, `Year`, `Status`, `Mileage`, `Dealer`, and `Price`.

To reproduce that official version-pinned download in a clean checkout:

```powershell
python -m pip install -r requirements-acquisition.txt
python -c "import kagglehub; kagglehub.dataset_download('juanmerinobermejo/us-sales-cars-dataset/versions/2', output_dir='data/raw/kaggle_us_sales_cars_v2')"
```

The adapter still rejects the result unless its complete bytes, hash, encoding,
header, row count, and committed review all match. Kaggle credentials, if a
future client requires them, belong outside the repository.

## Verified local candidates

The source adapters have published these private, Git-ignored artifact sets:

```text
data/interim/kaggle_vehicle_sales_v1.csv
data/interim/kaggle_vehicle_sales_v1.quarantine.jsonl
data/interim/kaggle_vehicle_sales_v1.manifest.json
data/interim/kaggle_vehicle_sales_v1.ready.json

data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.csv
data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.quarantine.jsonl
data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.manifest.json
data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.ready.json
```

The wholesale set contains 540,764 accepted U.S. rows. The asking-price set
contains 137,099 rows after exact deduplication and retains explicit
`vehicle_status` for New, Used, and Certified listings. A `.ready.json` marker
proves an artifact set is complete and checksum-consistent; it never replaces
the separate ML-reuse or split gate.

## Verified private holdouts

Both source-specific split layers have published and reverified private,
Git-ignored assignment sets:

```text
data/processed/kaggle_us_sales_cars_v2/split/split_assignments.csv
data/processed/kaggle_us_sales_cars_v2/split/split_assignments.manifest.json
data/processed/kaggle_us_sales_cars_v2/split/split_assignments.ready.json

data/processed/kaggle_vehicle_sales_v1/split_assignments.csv
data/processed/kaggle_vehicle_sales_v1/split_assignments.manifest.json
data/processed/kaggle_vehicle_sales_v1/split_assignments.ready.json
```

| Track | Train rows | Test rows | Assignment SHA-256 | Manifest SHA-256 |
|---|---:|---:|---|---|
| Retail asking price | 109,510 | 27,589 | `5b3e39d0ef418c07b0c4d08ecc18700fc9f387518a21dbd604f515463cb5ebe5` | `c60bf010fb47dff44d03b5da80b191ddb4b748661cb5cf02397422fdbaaf3466` |
| Wholesale completed sale | 442,130 | 98,634 | `a96909345612f5ddc5665c4d6817d2c8f0dd6d59c3a84fc523cb82b6adeeb5f2` | `d0dd0c24f342a8a45c1f89419780f470d0f152d61cf5dd54b2cb786df9525bd3` |

The retail artifact-set ID is
`8ab1e31f08aab9cefe3293e9a1e4bfe6ddf544da5020e5ac64b6b5fff7625edf`.
Its status-stratified test rows comprise 16,425 New, 9,646 Used, and 1,518
Certified listings. The five-field grouping key is year, make, model, mileage
(including null), and vehicle status. All 56,529 predictor groups are isolated
between partitions. Because the source has no row-level dates or stable upstream
listing IDs, this holdout is deliberately non-temporal.

The wholesale artifact-set ID is
`75fe4bbe4b1d77c48e2e8804dfdde86a2ef037580fdf58c257e762fa53ee6d37`.
Rows dated on or after 2015-06-01 seed its test set. VIN isolation promoted
1,066 earlier rows from 1,039 VIN groups into test; verification found zero VIN
overlap, no post-cutoff train rows, and no train VIN group crossing its ordered
CV buckets. The
[committed split decision](../docs/data-reviews/kaggle-vehicle-sales-v1.split.json)
records the reviewed policy.

Training code must open these datasets through
`prepare_kaggle_us_sales_cars_split_training_rows(...)` or
`prepare_kaggle_vehicle_sales_training_rows(...)`. It must not treat either
unsplit normalized candidate as model input. The asking-price and completed-sale
labels also remain separate; rows cannot be concatenated into one regression
target. These are split-integrity results, not model-accuracy results.

`fixtures/scraper_site/` is an invented, project-owned three-page U.S.
dealership used to prove pagination, retry/backoff, rate-limit handling,
normalization, deduplication, quarantine, provenance, and ingestion metrics
without accessing a third party. It contains an exact duplicate, missing fields,
and a malformed monthly-payment card. The demo server also injects one HTTP 503
and one HTTP 429 with `Retry-After`. Its local terms explicitly cover the demo,
and its permission digest is pinned in code.

The demo writes accepted records to `interim/synthetic_listings.jsonl`, rejected
record lineage to `interim/synthetic_listings.quarantine.jsonl`, and permissions,
provenance, metrics, and checksums to
`interim/synthetic_listings.manifest.json`. It publishes
`interim/synthetic_listings.ready.json` last. Consumers must call
`verify_scrape_artifact_set(...)` and must not read an artifact set without that
final marker. Raw HTML is not copied there. These files are synthetic plumbing
artifacts, not model-quality evidence.

After the official download, use the dataset-specific manifest and review record
created for this artifact. The strict manifest records immutable
source/version/license details, the artifact filename/format/SHA-256,
acquisition approval evidence, and a separate ML-training approval decision.
Its reviewed SHA-256 must be pinned outside the supplied manifest so a file
cannot approve itself. `manifest.example.json` remains a non-approved template
for other sources.

A dataset may be loaded for audit while ML approval remains false;
`require_ml_training_approval(...)` must pass before training code may consume
it. Record retrieval time, byte size, row count, columns, encoding, delimiter,
units, USD target semantics, and quality findings in the source review.
See `docs/data-acquisition.md` for the exact workflow.

The MIT project license applies to AutoValue AI code only. Third-party datasets
retain their original licenses and terms.

## Hugging Face candidate artifacts

Two Git-ignored, immutable files are used only for governed quality review:

```text
data/raw/hf_yoad22_craigslist_used_cars/vehicles_clean.csv
data/raw/hf_carson_shively_used_car_price/bronze.parquet
data/raw/hf_carson_shively_used_car_price/silver.parquet  # layer audit only
```

They are revision- and checksum-pinned in
`autovalue_ml.acquisition.sources.huggingface_candidates`. Acquisition approval
does not grant training approval. Yoad is approved only for its completed,
controlled offline batch experiments; Carson batch reuse remains blocked. Both
are blocked from online/River learning. Do not move either source blindly into
`processed/` or combine it with an existing split. See the
[candidate review](../docs/data-reviews/hugging-face-candidates.md).

## Public-repository and Rebrowser boundaries

The public repository contains no real source rows. `.gitignore` blocks raw,
interim, processed, private, and local data directories plus common CSV,
Parquet, Arrow, Feather, JSONL, and stateful-checkpoint formats. Clearly
project-owned synthetic fixtures are the only row-level publication exception.
Do not remove local artifacts merely to prepare a commit; verify the ignore rules
instead.

The 30-file Rebrowser AutoTrader free preview remains local at immutable revision
`a6cd0c8addded3591ccdfcd6ee4249b454f99792`. It is allowed only for controlled
non-commercial research/educational aggregate analysis. Raw Parquet/CSV data,
VINs, listing IDs, and dealer records must not be committed. Premium fields and
endpoints must not be accessed. Batch training, hosted inference from its KBB
targets, redistribution, and River updates remain blocked because third-party
AutoTrader/KBB rights are unresolved.

The complete publication matrix is in [`DATA_SOURCES.md`](../DATA_SOURCES.md),
with required credit in the [attribution document](../docs/data-attribution.md).
