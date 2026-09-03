# Dataset review records

This directory stores project-owned decisions about specific third-party data
artifacts. A review record is evidence for AutoValue AI's gate; it does not
replace the source's terms, prove upstream ownership, or expand a license.

The two approved candidates are documented separately:

- [`kaggle-vehicle-sales-data-v1.review.json`](kaggle-vehicle-sales-data-v1.review.json)
  covers completed 2014-2015 wholesale-auction sales.
- [`kaggle-us-sales-cars-v2.review.json`](kaggle-us-sales-cars-v2.review.json)
  covers historical 2023 U.S. retail asking-price listings.

Their labels are not interchangeable and the rows must not be concatenated into
one target without a separately reviewed domain or multi-task design. Both
approvals cover official download, local analysis, transformation, ML
training/evaluation, public aggregate results, and hosted inference. Raw or
processed row redistribution, downloadable model publication, sublicensing,
and commercial use remain pending.

The second review records separate project-owner attestations for the historical
source acquisition and ML reuse. It does not authorize AutoValue AI to scrape
Cars.com; only the already-created, version-pinned Kaggle artifact is approved.

## Verified split evidence

Both reviewed candidates now have private, Git-ignored split artifact sets. The
retail asking-price holdout lives under
`data/processed/kaggle_us_sales_cars_v2/split/`. It contains 109,510 train and
27,589 test rows, with no overlap among the 56,529 year/make/model/mileage/status
groups. Its assignment SHA-256 is
`5b3e39d0ef418c07b0c4d08ecc18700fc9f387518a21dbd604f515463cb5ebe5`
and its manifest SHA-256 is
`c60bf010fb47dff44d03b5da80b191ddb4b748661cb5cf02397422fdbaaf3466`.
The test partition includes 16,425 New, 9,646 Used, and 1,518 Certified rows.
This is a deterministic grouped holdout, not a temporal one, because the source
does not provide row-level dates or stable upstream listing IDs.

The wholesale completed-sale holdout lives under
`data/processed/kaggle_vehicle_sales_v1/`. Its committed
[split decision](kaggle-vehicle-sales-v1.split.json) fixes 2015-06-01 as the
chronological boundary and defines five ordered train CV buckets. The verified
artifact contains 442,130 train and 98,634 test rows, zero VIN overlap, and no
post-cutoff train row. Its assignment SHA-256 is
`a96909345612f5ddc5665c4d6817d2c8f0dd6d59c3a84fc523cb82b6adeeb5f2`
and its manifest SHA-256 is
`d0dd0c24f342a8a45c1f89419780f470d0f152d61cf5dd54b2cb786df9525bd3`.

These records establish review scope and split integrity, not legal certainty,
current-market relevance, or model accuracy. Training code must use the
split-aware gates and cannot consume either unsplit candidate directly. Retail
asking-price and wholesale completed-sale rows remain separate targets.

## Baseline reports

Phase 3 opened both real datasets only through those split-aware gates. Linear
Regression was selected over the median dummy by cross-validation MAE, then
evaluated once on each untouched holdout:

| Track | CV scheme | Linear CV MAE | Holdout rows | Linear holdout MAE |
|---|---|---:|---:|---:|
| Historical retail asking price | Non-temporal predictor-group CV | $11,552.82 | 27,589 | $12,040.29 |
| Historical wholesale completed sale | Forward, VIN-isolated CV | $2,382.13 | 98,634 | $2,256.02 |

The committed aggregate-only reports are
[`retail-baseline-v1.json`](../results/retail-baseline-v1.json) and
[`wholesale-baseline-v1.json`](../results/wholesale-baseline-v1.json). The
[results guide](../results/README.md) documents full MAE, RMSE, R², retail status
slices, reproduction, and limitations. These evaluations do not widen the
permission scope above: data acquisition/storage approval remains distinct from
ML training/evaluation approval, and neither implies redistribution or
downloadable-model rights. The commands persist no estimator or row-level data.

Review JSON is committed because it contains decisions, checksums, and aggregate
facts rather than dataset rows or private correspondence. Raw files belong under
`data/raw/` and remain ignored by Git.

## Governed Hugging Face candidates

Two additional sources have completed acquisition and aggregate compatibility
review. Yoad has entered an isolated controlled experiment, a separate
source-composition confirmation, and a fold-local weighting confirmation;
neither source has entered the production corpus. See the
[Hugging Face candidate review](hugging-face-candidates.md) and its linked
machine-readable reports.

- Yoad22/Austin Reese Craigslist: acquisition and controlled offline batch
  experimentation approved, production training not approved, online learning
  blocked.
- Carson-Shively used-car-price: acquisition approved for private review, batch
  and online learning blocked because upstream origin, U.S. scope, USD semantics,
  and license metadata remain unresolved.

The current retail source is Cars.com-derived, not Austin Reese/Craigslist, so
Yoad has no known shared upstream family with it. Any other Austin Reese or
Craigslist derivative is a confirmed source-level overlap and cannot be blindly
concatenated. Carson's repository has 4,009 bronze rows and 3,961 transformed
silver rows; their apparent 7,970 total is not 7,970 independent observations.

Yoad's machine-readable
[controlled approval](yoad22-controlled-batch-approval-v1.json) and
[aggregate experiment report](../experiments/yoad22-controlled-batch-v1.json)
preserve the 98,552-row Cars development boundary, exclude calibration and
legacy holdout rows, and keep online/River learning blocked. The combined model
was not promoted. The separate
[source-composition confirmation](../experiments/yoad22-source-composition-confirmation-v1.md)
prefers the 150,000-row moderate augmentation only as a separate experimental
model; it is not eligible for final promotion evaluation.
The subsequent
[weighting confirmation](../experiments/yoad22-training-weight-confirmation-v1.md)
rejects all three tested weighting formulas because each introduces at least one
unacceptable Cars manufacturer regression. The unweighted moderate branch
remains unchanged and unpromoted.

## Rebrowser AutoTrader/KBB preview

The separate
[AutoTrader free-preview audit](rebrowser-autotrader-preview-v1.md) verifies the
30-file, 8,019-row artifact at immutable Hugging Face revision
`a6cd0c8addded3591ccdfcd6ee4249b454f99792`. KBB low/high are real numeric
fields and 6,770 rows have a valid positive range, but the source's terms do not
grant rights in third-party AutoTrader/KBB intellectual property. AutoValue
therefore classifies the source as **reference/analytics only**. Batch training,
model publication, merging, premium access, and online/River learning remain
blocked; no model was trained.
