# Data acquisition layer

AutoValue AI has a lightweight, offline acquisition subsystem for auditable
public datasets and reusable source adapters. It uses ordinary Python objects,
local files, and an in-memory response cache; it does not require Kafka, Spark,
Kubernetes, a message broker, or a paid service.

The acquisition layer is implemented, but no commercial marketplace adapter is
enabled. External crawling is deliberately disabled until the HTTP transport can
pin a reviewed hostname to the exact validated network address for the complete
request. This avoids a DNS time-of-check/time-of-use gap. The only runnable
scraper targets the repository-owned fixture over numeric loopback.

The acquisition contract is U.S.-only. A licensed manifest, scraping policy,
normalized listing, and downstream training event must all declare
`market_country="US"`; price-bearing records must use `currency="USD"`. This
layer now has two conditionally approved, checksum-pinned local-file candidates.
Both have verified source-specific holdouts, but no acquisition action or
general-purpose `.ready.json` marker trains a model automatically.

## Trust boundaries

```text
Licensed local file                       Reviewed scraping adapter
        |                                          |
externally pinned manifest SHA-256        source policy + robots rules
        |                                          |
bounded CSV/JSONL load                   rate-limited paginated retrieval
        |                                          |
        +-------------------+----------------------+
                            |
                  common listing schema
                            |
              validation and normalization
                            |
               deduplication + quarantine
                            |
          normalized data + lineage manifest + ready marker
                            |
                 explicit ML-reuse approval
                     /                 \
            future batch ML       River-shaped events
                                  (no model update here)
```

Acquisition and ML reuse are independent permissions. Successfully downloading,
loading, parsing, or storing records never authorizes training. No acquisition
function calls a training function, and no accepted record enters a model unless
the caller explicitly invokes the ML-reuse gate.

## Common vehicle-listing contract

Every adapter normalizes its source into `VehicleListingSnapshot`, represented
by the machine-readable JSON Schema at
`ml/schemas/vehicle-listing-v1.schema.json`. The contract includes:

- source ID, source listing ID, canonical listing URL, and observation time;
- `market_country="US"` as required lineage metadata;
- year, make, model, trim, mileage, physical condition, vehicle status, engine,
  and drivetrain;
- accident status/count, owner count, and vehicle type;
- price in integer cents, `currency="USD"`, price kind, and sale status; and
- raw-content hash, parser and normalizer versions, acquisition run ID, and
  source-policy ID. The surrounding manifest records the immutable acquisition
  policy fingerprint.

Price meaning is explicit. Asking price, completed sale, bid, reserve, monthly
payment, and unknown values are not interchangeable targets. Missing optional
features remain null; they are not invented. Non-finite or structurally invalid
values fail schema validation.

`VehicleFieldMapping` lets a licensed dataset map its own column names into this
contract. Normalization accepts only reviewed scalar field types, applies strict
price/currency and integer grammars, derives a stable source listing ID when
necessary, skips exact duplicate records, and rejects a conflicting reuse of the
same source listing ID. Structured objects are quarantined rather than converted
into valid-looking strings or prices.

## Licensed public datasets

`load_licensed_dataset` reads only local CSV or JSONL files. It performs no
network request. Before parsing any manifest field, it hashes the complete
manifest and compares it with a digest copied from a separate, project-owned
review record. This prevents the supplied file from approving itself. It then
requires a strict versioned manifest and checks:

- source owner, immutable dataset name/version, and canonical public URL;
- `market_country="US"` in source metadata;
- SPDX license identifier, license URL, and a non-future review date;
- exact artifact filename and format;
- lowercase SHA-256 checksum of the complete artifact;
- explicit acquisition approval and its evidence; and
- an independent Boolean and evidence field for ML-training approval.

The loader also rejects symbolic links, malformed UTF-8/JSON/CSV, duplicate JSON
keys at any nesting depth, duplicate or empty CSV headers, non-finite JSON
numbers, empty datasets, and artifacts over configured byte or row limits.
Returned rows and lineage are immutable.

A reviewed manifest has this shape:

```json
{
  "manifest_schema_version": 1,
  "source": {
    "owner": "SOURCE OWNER",
    "name": "DATASET NAME",
    "version": "IMMUTABLE VERSION",
    "canonical_url": "https://example.org/datasets/version",
    "market_country": "US"
  },
  "license": {
    "spdx_id": "CC-BY-4.0",
    "url": "https://creativecommons.org/licenses/by/4.0/",
    "reviewed_on": "2026-08-27"
  },
  "artifact": {
    "file_name": "vehicles.csv",
    "format": "csv",
    "sha256": "REPLACE_WITH_LOWERCASE_SHA256"
  },
  "approvals": {
    "approved_for_acquisition": true,
    "acquisition_evidence": "REVIEW RECORD OR OWNER PERMISSION",
    "approved_for_ml_training": false,
    "ml_training_evidence": ""
  }
}
```

Generate an intentionally non-approved template with
`sample_manifest(...)`, then review and replace every placeholder. A dataset can
be loaded for an audit while `approved_for_ml_training` remains false. The
separate call below must succeed before a downstream training job can use it:

```python
from pathlib import Path

from autovalue_ml.acquisition import (
    load_licensed_dataset,
    require_ml_training_approval,
)

loaded = load_licensed_dataset(
    Path("data/raw/vehicles.csv"),
    Path("data/raw/vehicles.license.json"),
    # Copy this from the independent review record, never from the supplied file.
    trusted_manifest_sha256="<64 lowercase hexadecimal characters>",
)
training_input = require_ml_training_approval(loaded)
```

The existing `data/manifest.example.json` is an intentionally non-approved
starting template for the strict per-artifact manifest above. The external
review record and its pinned manifest digest remain separate.

### Approved Kaggle candidates

[Kaggle Vehicle Sales Data v1](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data/data)
is conditionally approved for the local-file path. Download it only through the
official Kaggle interface, then place the immutable extracted CSV at:

```text
data/raw/kaggle_vehicle_sales_v1/car_prices.csv
```

The reviewed CSV has 558,837 rows, is 88,047,552 bytes, and has SHA-256
`32ba3ce51664e6a12c0c927ed193b41e3c4743fdf18bc0317389892aed27f556`.
If preserving the original 19,753,181-byte archive, store it at
`data/raw/kaggle_syedanwarafridi_vehicle-sales-data_v1/vehicle-sales-data-v1.zip`;
its SHA-256 is
`8eb8e42023ee7255818b31ac1a716b438e7bd9298116b98c250937754418c8b1`.
Any mismatch stops ingestion as an unreviewed artifact version.

Kaggle displays an MIT label. The project owner attests that the uploader
directly confirmed legal use, but upstream ownership was not independently
verified. The scoped decision allows official download, private local storage
and transformation, ML training/evaluation, aggregate public results, and hosted
inference. Raw/processed row redistribution, a downloadable model, sublicensing,
and commercial use remain pending. See the exact
[review record](data-reviews/kaggle-vehicle-sales-data-v1.review.json).

The raw source mixes U.S., Canadian, and Puerto Rican rows. A candidate row may
reach training only after mapping to one of the 50 U.S. states or Washington,
D.C., receiving a valid date and positive target, and passing quality checks.
The current audit counts 550,410 U.S./D.C. rows, 550,398 with valid date and
target, and 529,245 complete across the initial core fields. Dates span
2014-01-01 through 2015-07-21. Currency is not explicitly documented; USD is an
open semantic assumption restricted to validated U.S./D.C. rows.

The target is historical wholesale-auction `sellingprice`, not current retail
market value. `vin` and `seller` must be removed. `mmr` is forbidden everywhere
in modeling because it is already a competing valuation and presents severe
target leakage. Splits must be both chronological and VIN-aware.

The verified normalized run accepted 540,764 rows and quarantined 18,073,
including 8,401 non-U.S. rows. Its private candidate has no VIN, seller, MMR, or
transmission column. The verified 2015-06-01 chronological, VIN-isolated split
contains 442,130 train and 98,634 test rows. It promoted 1,066 earlier rows from
1,039 VIN groups into test and has zero verified VIN overlap. Training can open
only through the split-aware verifier; the normalized candidate alone is not a
training interface.

[US Sales Cars Dataset v2](https://www.kaggle.com/datasets/juanmerinobermejo/us-sales-cars-dataset)
is independently approved as the historical retail asking-price track. It was
downloaded through KaggleHub 1.0.2 at the pinned version-2 handle and stored at:

```text
data/raw/kaggle_us_sales_cars_v2/cars.csv
```

The UTF-16 CSV has 144,867 rows, is 17,171,976 bytes, and has SHA-256
`25854afc3ef8b6c6a0349bf7f422c40dacb9bec60a8b318462737ebf9edcc5ea`.
It must have exactly seven columns: `Brand`, `Model`, `Year`, `Status`,
`Mileage`, `Dealer`, and `Price`. New, Used, and Certified status values are
retained explicitly. `Dealer` is removed; a missing mileage is valid; missing or
invalid price is quarantined; and exact source rows are deduplicated. The target
is a historical USD advertised asking price, not a completed sale or current
live appraisal.

The uploader links an upstream repository that identifies historical Cars.com
extraction. The project owner's attestation separately approves the historical
collection and noncommercial portfolio ML reuse of this fixed artifact. It does
not authorize AutoValue AI to contact or scrape Cars.com. The exact evidence,
permissions, artifact pin, profile, and gates are in the
[review record](data-reviews/kaggle-us-sales-cars-v2.review.json).

The verified normalized run accepted 137,099 rows after removing 3,857 exact
duplicates and quarantining 3,911 missing-price rows. Accepted status counts are
81,278 New, 48,229 Used, and 7,592 Certified. Missing mileage remains null; Dealer
is absent from candidate and training features. The verified predictor-group
holdout contains 109,510 train and 27,589 test rows. No one year/make/model/
mileage/status group crosses partitions. Because the source has no row-level
dates or stable upstream listing IDs, this is explicitly a non-temporal split.

### Split publication and consumption

Each split publishes assignments, a manifest, and a readiness marker under its
Git-ignored processed-data directory. Verification rechecks source-review and
candidate lineage, complete file hashes, artifact-set identity, row accounting,
assignment semantics, privacy restrictions, and group isolation.

The retail files are under
`data/processed/kaggle_us_sales_cars_v2/split/`; their assignment SHA-256 is
`5b3e39d0ef418c07b0c4d08ecc18700fc9f387518a21dbd604f515463cb5ebe5`
and manifest SHA-256 is
`c60bf010fb47dff44d03b5da80b191ddb4b748661cb5cf02397422fdbaaf3466`.
The wholesale files are under `data/processed/kaggle_vehicle_sales_v1/`;
their assignment SHA-256 is
`a96909345612f5ddc5665c4d6817d2c8f0dd6d59c3a84fc523cb82b6adeeb5f2`
and manifest SHA-256 is
`d0dd0c24f342a8a45c1f89419780f470d0f152d61cf5dd54b2cb786df9525bd3`.

Only `prepare_kaggle_us_sales_cars_split_training_rows(...)` and
`prepare_kaggle_vehicle_sales_training_rows(...)` expose these real candidates
to modeling. Retail returns a selected train or test stream. Wholesale returns
the partition plus an ordered CV-bucket label for train rows; its five buckets
support forward validation without inspecting the final test period. The target
meanings remain separate, so these two streams are never concatenated into one
training target. Split verification is not a model result: no real estimator has
been fitted or scored yet.

MUCars-2024 remains rejected for geographic mismatch. City of Seattle Sold
Fleet Equipment remains a public-domain ingestion smoke candidate because it is
small municipal-auction data without mileage. A future GSA corpus must come from
an official bulk release or direct request rather than scraping. UCI Automobile
is also inadequate for production training.

## Reviewed scraping adapters

A `ReviewedScrapingAdapter` bundles a fixed adapter ID/version, a reviewed
`SourcePolicy`, one start path, and a pure page parser. `AdapterRegistry`
revalidates permission when registering and prevents duplicate adapter or source
registrations. The policy pins one exact origin, allowed paths and query keys,
`market_country="US"`, permitted output fields, terms evidence, data retention,
cache limits, and hard crawl budgets.

The sequential client enforces:

- `robots.txt` restrictions in addition to, never instead of, permission;
- source request delay and robots crawl-delay/request-rate limits;
- bounded requests, pages, records, bytes per response, total bytes, retries,
  retry delay, and total runtime;
- exponential backoff and safe `Retry-After` handling for temporary failures and
  HTTP 429 responses;
- deterministic pagination constrained to the reviewed origin and URL policy;
- an optional process-memory response cache with TTL, LRU eviction, byte limit,
  and SHA-256 integrity checking; and
- rejection of redirects, authentication, cookies, unapproved query parameters,
  pagination loops, parser drift, and conflicting listing IDs.

The response cache is a performance optimization, not a durable data store. It
does not cache `robots.txt`, disappears when the process exits, and stores no raw
HTML artifact in `data/interim/`.

Scrapy is BSD-licensed and remains compatible with the adapter/parser design, but
it is intentionally not a dependency for this three-page controlled demo. The
current sequential client makes the crawl and byte budgets explicit with less
runtime machinery. A future permitted source may justify a Scrapy-backed adapter
only after the same policy, address-pinning, permission, provenance, and test
requirements are met; choosing Scrapy never supplies legal permission by itself.

## Controlled synthetic dealership

`data/fixtures/scraper_site/` is a three-page, project-owned dealership fixture.
It is deliberately imperfect so the demo exercises production-style behavior:

- page 2 repeats one listing from page 1, proving exact deduplication;
- page 3 has a valid listing with missing optional fields;
- page 3 also has a listing missing its required make, which is quarantined;
- a monthly-payment card contains a malformed amount and is quarantined rather
  than interpreted as a vehicle price;
- the local server returns one HTTP 503 for page 2; and
- the local server returns one HTTP 429 with `Retry-After` for page 3.

The retry policy recovers from both injected temporary responses. A normal run
fetches three pages, accepts four unique listings, skips one duplicate, and
quarantines two malformed cards.

Run it from the repository root after installing the development requirements:

```powershell
$env:PYTHONPATH = "ml/src"
python -m autovalue_ml.acquisition.demo
```

The command writes Git-ignored artifacts to:

- `data/interim/synthetic_listings.jsonl` — accepted normalized snapshots;
- `data/interim/synthetic_listings.quarantine.jsonl` — rejected-record lineage
  and reason, without raw HTML; and
- `data/interim/synthetic_listings.manifest.json` — terms and policy hashes,
  separate permissions, counts, timings, requests, retries, HTTP status counts,
  bytes, cache activity, duplicates, quarantine count, versions, and artifact
  checksums; and
- `data/interim/synthetic_listings.ready.json` — the final publication marker,
  artifact-set ID, filenames, and hashes.

The writer publishes `.ready.json` last. A consumer must call
`verify_scrape_artifact_set(...)` before reading the set; the verifier fails
closed if the marker is absent or malformed, references unsafe names, or if any
declared file, checksum, or artifact-set ID no longer matches. The readiness
marker prevents a partial run from looking complete. It is an integrity and
publication boundary, not ML-training approval.

## Revision-pinned Hugging Face candidates

`acquire_huggingface_artifact(...)` extends the licensed-data boundary for
reviewed Hugging Face repository files. It accepts only an exact owner/repo,
full 40-character commit, allowlisted repository path, size, and SHA-256. The
direct downloader is atomic, bounded, retry/backoff aware, and locally cached;
it does not execute dataset repository code or require the full `datasets`
package. `pyarrow` is an acquisition-only dependency for Parquet profiling.

Each `HuggingFaceArtifactSpec` carries declared license, attribution,
restrictions, upstream source, config/split, U.S./USD product requirements,
schema-mapping version, and three separate decisions: acquisition, batch
training, and online learning. `build_huggingface_provenance(...)` binds those
decisions and aggregate accepted/rejected/duplicate counts to the exact local
bytes and acquisition timestamp.

Yoad22/Austin Reese Craigslist is acquisition-approved for controlled offline
batch experimentation and remains online-blocked. Its first controlled run and
the separate source-composition confirmation did not authorize production
training or promotion. Carson-Shively is acquisition-approved for private audit
but blocked from both training paths. The normalization layer preserves raw
source values in local audit metadata while `feature_values()` excludes source
IDs, hashes, and provenance. Exact schema checks, parser quarantine, aggregate
quality reports, and source-family overlap checks happen before any training
iterator could be designed.

See the [candidate review](data-reviews/hugging-face-candidates.md) for artifact
pins, quality metrics, the Carson bronze/silver layer finding, and current merge
decisions.

## Explicit ML-reuse and River boundary

For scraped results, `build_training_event_batch(...)` revalidates the separate,
current ML-training/public-portfolio permission, verifies the result's source
policy fingerprint, and filters target semantics and currency. The default gate
admits only USD asking prices; every exclusion retains a reason.

Approved records become immutable `TrainingRecordEvent` values with deterministic
event IDs, a stable content-deduplication key, schema version, source/listing/run
IDs, observation and emission times, `market_country="US"`, separate
acquisition-policy and ML-grant hashes, parser/normalizer versions, features, and
USD target metadata. The gate
validates every nested record against its containing result before emitting
anything. Each event can return the `(x, y)` pair expected by River's
`learn_one(x, y)` interface:

```python
batch = build_training_event_batch(result, policy, today=date.today())
for features, target in iter_river_examples(batch):
    # A future online-learning service may call model.learn_one(features, target).
    pass
```

River is not currently a dependency, and this bridge never imports River or
updates a model. This keeps acquisition testable and lets a future append-only
stream consumer introduce validation, replay, drift monitoring, promotion, and
rollback without changing the source adapters.

## Verification

Acquisition tests are deterministic and never contact an external source. Run
them with:

```powershell
python -m pytest tests/ml
```

The suite covers permission expiry and separation, manifests and checksums,
schema parity, normalization, malformed-record quarantine, duplicates, bounded
retries/backoff, 429 handling, pagination and URL loops, cache integrity,
provenance, U.S./USD enforcement, artifact-set readiness and checksums, and the
explicit training-event gate.
