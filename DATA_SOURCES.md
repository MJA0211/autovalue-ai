# AutoValue AI data sources

AutoValue AI is limited to United States vehicle records and U.S. dollars. Each
source has independent permissions for acquisition, offline model use,
publication, redistribution, and online learning. Permission in one column never
implies permission in another. Unknown sources fail closed.

This register records project governance, not a legal opinion or a sublicense.
Third-party material remains subject to its own terms and rights.

## Current register

| Source | Current role | Offline batch use | River/online use | Raw redistribution |
|---|---|---|---|---|
| Juan Merino Bermejo US Sales Cars v2; historical Cars.com listings | Primary historical U.S. retail asking-price corpus | Approved within the pinned, split-aware workflow | Blocked | Blocked by project policy |
| Syed Anwar Afridi Vehicle Sales Data v1 | Separate 2014–2015 U.S. wholesale benchmark | Approved within the pinned, VIN-isolated workflow | Blocked | Blocked by project policy |
| Yoad22 Craigslist Used Cars EDA | Separate unweighted moderate augmentation reference | Controlled offline experimentation only | Blocked | Not published by AutoValue |
| Rebrowser AutoTrader free preview | Private aggregate reference/analytics for KBB-range feasibility | Blocked pending third-party-rights evidence | Blocked | Blocked |
| Carson-Shively Used Car Price | Private compatibility review only | Blocked | Blocked | Blocked |
| AutoValue synthetic dealership and shadow simulator | Test plumbing and architecture validation | Approved for tests only | Approved for synthetic simulation only | Project-owned fixtures may be published |

## Cars.com-derived retail corpus

- Dataset: [US Sales Cars Dataset, version 2](https://www.kaggle.com/datasets/juanmerinobermejo/us-sales-cars-dataset), published by Juan Merino Bermejo.
- Provenance: the dataset author identifies a historical Cars.com listing
  extraction. AutoValue AI does not scrape Cars.com.
- License/permission evidence: Kaggle metadata displays Apache 2.0. The project
  owner separately attests that the historical acquisition and this
  non-commercial portfolio ML reuse were authorized.
- Scope: historical U.S. advertised asking prices in USD, not completed sales or
  current guaranteed values.
- Boundary: only the pinned version-2 artifact and verified split-aware loaders
  are approved. Dealer data is removed. Calibration and legacy holdout boundaries
  remain protected.
- Publication: aggregate metrics and hosted inference may be published; raw or
  processed rows and downloadable derivatives are not published without a new
  review.

The current Phase 4 RF05 decision is frozen. This register does not alter or
promote it.

## Yoad22 / Craigslist

- Dataset: [Yoad22/craigslist-used-cars-eda](https://huggingface.co/datasets/Yoad22/craigslist-used-cars-eda).
- Provenance: a derivative of the Austin Reese Craigslist Cars and Trucks
  dataset. The Hugging Face card declares CC BY 4.0; the documented upstream
  Kaggle page displays CC0.
- Current use: controlled offline batch experimentation only. The frozen
  reference is the unweighted moderate composition of 98,552 Cars.com
  development rows and the deterministic 150,000-row Yoad subset.
- Prohibited path: no production merge, automatic promotion, or River/online
  update. Missing stable listing IDs, observation timestamps, and delayed-label
  semantics make the existing artifact unsuitable for replay-safe online use.
- Publication: AutoValue publishes aggregate reports and attribution, not source
  rows. If upstream terms ever narrow redistribution, the stricter term applies.

The rejected weighting experiments remain rejected and are not reopened here.

## Rebrowser AutoTrader free preview

- Dataset: [rebrowser/autotrader-dataset](https://huggingface.co/datasets/rebrowser/autotrader-dataset), audited only at immutable revision
  `a6cd0c8addded3591ccdfcd6ee4249b454f99792`.
- Current use: controlled non-commercial research/educational aggregate analysis
  only. The audit classification remains reference/analytics only; neither KBB
  targets nor listing rows are approved for model training.
- Access boundary: use only the 30 free files already audited. Do not access or
  bypass premium fields or endpoints.
- Attribution: “Rebrowser, AutoTrader Vehicle Listings Dataset (2026),
  https://rebrowser.net/products/datasets/autotrader”.
- Rights caveat: the card declares CC BY-NC 4.0, while Rebrowser's terms state
  that its license does not grant rights in third-party source intellectual
  property. AutoTrader and Kelley Blue Book rights are not supplied by that
  dataset license.
- Publication boundary: never commit raw Parquet/CSV files or row-level VIN,
  listing, or dealer data. Do not present AutoValue output as an official,
  affiliated, sponsored, or endorsed KBB, AutoTrader, or Rebrowser product.
- Prohibited path: batch training, model publication, hosted inference from KBB
  targets, raw redistribution, and River/online learning remain blocked.

See the [aggregate audit](docs/data-reviews/rebrowser-autotrader-preview-v1.md).

## Carson-Shively

The source is excluded from batch and online learning. Its upstream origin,
U.S. scope, USD semantics, and repository license metadata remain unresolved.
Its bronze and silver layers are related transformations, not independent rows.

## Public repository boundary

Only code, synthetic fixtures, permission decisions, checksums, and
aggregate-only reports belong in GitHub. Real source rows, normalized row-level
artifacts, raw archives, local databases, caches, credentials, model binaries,
and stateful checkpoints remain local and are covered by `.gitignore`.

Before adding a source, record its immutable revision/checksum, provenance,
license evidence, attribution, U.S./USD semantics, allowed uses, redistribution
status, stable identifiers, and separate batch and River decisions. Acquisition
or offline approval never enables River automatically.
