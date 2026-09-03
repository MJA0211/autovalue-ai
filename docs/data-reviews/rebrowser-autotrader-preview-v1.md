# Rebrowser AutoTrader free-preview acquisition audit

## Decision

**Classification: reference/analytics only.** The free artifact is technically
useful and its KBB low/high fields are genuinely populated, but it does not pass
AutoValue AI's independent batch-training permission gate. No model was trained,
no data was merged with Cars.com or Yoad, and online/River use remains blocked.

The Hugging Face card labels the repository `CC-BY-NC-4.0` and permits
noncommercial research with attribution. Rebrowser's separate terms say that it
does not grant rights in third-party source intellectual property and place the
responsibility for obtaining necessary permissions on the user. Because the
proposed labels reproduce Kelley Blue Book values obtained through AutoTrader,
the repository label alone is not sufficient evidence that a public portfolio
project may train and publish a derivative KBB valuation model. This review
therefore fails closed on training while allowing private aggregate analysis.

## Frozen acquisition

| Item | Verified value |
|---|---|
| Repository | `rebrowser/autotrader-dataset` |
| Immutable revision | `a6cd0c8addded3591ccdfcd6ee4249b454f99792` |
| Free files acquired | 30 Parquet files only |
| File dates | 2026-07-20 through 2026-08-18 |
| Parquet bytes | 8,198,109 |
| Rows | 8,019 |
| Actual columns | 71 |
| Manifest bundle SHA-256 | `768920f599e7fc0edab66f6784170c891531c001a9ba4a36da8d1283d7bf15b3` |
| Pinned README SHA-256 | `385be8e2dd01fd106e0a096c7073fc4bc9fb7cc3fa691963682c99436268d90e` |
| Pinned schema SHA-256 | `51ea8985e4d2e26fdddbd2619e639db197c2782c59838c56122ed74690cf6002` |

Every Parquet file matched its expected Hugging Face LFS SHA-256, byte size,
row count, and ordered 71-column schema. The complete per-file manifest is in
[`rebrowser-autotrader-preview-v1.audit.json`](rebrowser-autotrader-preview-v1.audit.json).
The local bytes remain under Git-ignored `data/raw/`; neither raw nor normalized
rows are published.

The card says the rotating preview can contain up to 30,000 rows and calls the
entity a 68-field dataset. The pinned artifact actually has 8,019 rows and 71
columns. Those are not audit failures—the card uses a maximum and appears to lag
the artifact schema—but downstream work must use the verified values, not the
marketing summary.

## Free and restricted fields

The artifact contains 62 usable free columns and nine premium columns. All
8,019 values in every premium column are exactly `[PREMIUM]`; zero premium
values were exposed and no premium endpoint was accessed.

| Premium/restricted column | Free-artifact result |
|---|---|
| `vin` | 8,019 sentinels; no usable VIN |
| `salePrice` | 8,019 sentinels |
| `kbbFairPurchasePrice` | 8,019 sentinels |
| `sellerName` | 8,019 sentinels |
| `sellerPhone` | 8,019 sentinels |
| `sellerRating` | 8,019 sentinels |
| `sellerWebsite` | 8,019 sentinels |
| `images` | 8,019 sentinels |
| `listingUrl` | 8,019 sentinels |

The free inventory includes record/listing IDs and timestamps; year, make,
exact model, trim, body style and listing type; mileage; KBB low/high; colors;
engine, drivetrain, transmission, fuel and efficiency fields; selected listing
flags; seller ID and location; KBB consumer metadata; safety recalls; vehicle
history preview flags; options; and description. The machine report lists all
71 columns and explicitly marks the nine restricted ones.

## KBB target audit

The pinned schema describes `kbbFairPriceLow` and `kbbFairPriceHigh` as the low
and high ends of a **KBB Fair Purchase Price range in USD**. They are values
reproduced in an AutoTrader listing dataset. They are not completed sale prices,
not proof of current cash value, and not an empirical uncertainty interval from
an AutoValue model.

Both fields have 7,495 numeric values and 524 missing values. There are no
nonnumeric populated values, no one-sided missing ranges, no negatives, and no
rows where low exceeds high. However, 725 rows have both values equal to zero;
they are invalid valuation targets. After requiring finite positive values and
`low <= high`, 6,770 rows (84.42%) remain.

| USD statistic | KBB low | Midpoint | KBB high | Range width |
|---|---:|---:|---:|---:|
| Count | 7,495 numeric | 6,770 valid | 7,495 numeric | 6,770 valid |
| Minimum | $0 | $1,197.50 | $0 | $300 |
| 25th percentile | $13,775 | $18,305 | $15,703 | $1,875 |
| Median | $23,610 | $27,040 | $26,550 | $2,400 |
| Mean | $25,109.31 | $29,217.88 | $27,673.89 | $2,839.21 |
| 75th percentile | $33,062 | $35,819 | $36,048.50 | $3,300 |
| 95th percentile | $56,960.90 | $61,185.50 | $62,293 | $6,055 |
| 99th percentile | $81,906 | $85,359.22 | $86,727.80 | $8,931 |
| Maximum | $117,140 | $120,640 | $124,140 | $13,300 |

The mechanical three-IQR diagnostic flags 47 low values, 46 high values, 56
midpoints, and 157 widths as extreme relative to their distributions. These are
review flags, not evidence of invalidity and were not deleted or winsorized.
There are no values above $500,000 and no widths above $100,000.

The source has 6,494 Used, 744 Certified, 696 New, and 85 Third-Party Certified
records. Positive valid-range coverage is 85.56%, 94.35%, 61.93%, and 95.29%
respectively. Any later used-vehicle experiment would require an explicit
listing-type boundary rather than treating all 8,019 rows as interchangeable.

## U.S., USD, timestamp, and feature scope

The schema explicitly labels the KBB values as USD; there is no row-level
currency field. Seller state supports the U.S. scope: 8,000 rows contain valid
50-state-or-D.C. codes, 19 are missing, and none contain a non-U.S. or invalid
populated code. There are 49 represented state codes and 8,000 five-digit ZIP
codes.

| Field | Present | Coverage | Notes |
|---|---:|---:|---|
| Year | 8,019 | 100% | 39 distinct years |
| Make | 8,019 | 100% | 49 makes |
| Exact model | 8,019 | 100% | 633 values |
| Trim | 7,584 | 94.58% | 645 values |
| Mileage | 8,019 | 100% | miles per source schema |
| Engine | 8,019 | 100% | 18 descriptions |
| Transmission | 7,826 | 97.59% | 18 descriptions |
| Drivetrain | 8,014 | 99.94% | five values |
| Physical condition | 0 | 0% | `listingType` is inventory status, not condition |
| Vehicle-history preview | 7,457 | 92.99% | preview flags, not a full report |
| City/state/ZIP | 8,000 | 99.76% | detailed row geography |
| First/last-seen timestamps | 8,019 | 100% | valid UTC timestamps |

The first-seen span is 2026-07-20 through 2026-08-18; last-seen values extend
through 2026-08-30. No row has first-seen after last-seen. Every row's
first-seen date matches its file name, so these files behave as first-seen date
cohorts rather than repeated full daily snapshots.

All 7,457 populated vehicle-history values are valid JSON arrays. They provide
flags such as no/known accidents, one owner, salvage title, frame damage and
flood damage. These flags are a preview, may contain paired negated statuses,
and must not be described as a complete vehicle-history report.

## Identity, duplicates, and snapshots

All 8,019 `listingId` and `_primaryKey` values are populated and unique. There
are zero exact duplicate rows, zero repeated listing IDs, zero IDs spanning more
than one file, and zero repeated seller-ID/stock-number groups among the 7,955
usable pairs. VIN repetition cannot be evaluated because every VIN is a premium
sentinel. Because no listing repeats, KBB changes between snapshots cannot be
evaluated in this preview; there is no evidence either way.

If a future authorized artifact contains repetitions, validation must group all
rows with the same `listingId` together, also use stable seller-ID/stock-number
links where available, and keep every snapshot of the group in one partition.
Timestamps should define a forward evaluation boundary. Identifiers are group
keys, never model inputs.

## Target-leakage denylist

A future KBB model must exclude:

- direct and derived targets: KBB low, high, midpoint and width;
- KBB Fair Purchase Price, KBB vehicle ID, KBB deal indicator, and all KBB
  metadata;
- sale price, MSRP, reduced-price and no-haggle signals;
- days on market, hot-listing status, and marketplace priority;
- record, listing, stock and seller identifiers;
- first/last-seen timestamps as predictors (they are split/audit fields);
- unsanitized title and description text, which may mention prices or KBB.

Source identity is also forbidden as a predictor. The complete field-to-reason
denylist is stored in the JSON. Preprocessing, imputation, category grouping,
outlier handling and any text sanitation would have to fit inside training
folds only.

## Coverage compared with existing sources

This is a feature-availability comparison, not a proposal to concatenate rows.

| Capability | Cars.com development | Yoad22 approved | AutoTrader preview |
|---|---:|---:|---:|
| Rows in current governed boundary | 98,552 | 242,666 | 8,019 raw |
| Exact model | 100% | 0% | 100% |
| Trim | 0% | 0% | 94.58% |
| Engine | 0% | cylinders proxy only | 100% |
| Row timestamps | 0% | 0% | 100% |
| Vehicle history | 0% | title status only | 92.99% preview flags |
| Row geography | 0% | 100% state | 99.76% city/state/ZIP |
| Mileage | 40.78% | 100% | 100% |
| KBB range | none | none | 84.42% valid positive range |

AutoTrader therefore adds trim, full engine descriptions, timestamps, preview
history, detailed geography, and a distinct KBB valuation-range target. Cars.com
already supplies exact model; Yoad supplies older Craigslist breadth and a true
physical-condition field. AutoTrader remains a separate target/domain and is
not merged into either corpus.

## License and provenance assessment

Three layers must remain distinct:

1. The [pinned Hugging Face dataset card](https://huggingface.co/datasets/rebrowser/autotrader-dataset/blob/a6cd0c8addded3591ccdfcd6ee4249b454f99792/README.md)
   declares CC BY-NC 4.0, noncommercial research use, and required Rebrowser
   attribution.
2. The [Rebrowser terms](https://rebrowser.net/terms-of-use) say the data comes
   from public sources, Rebrowser is not affiliated with or licensed by the
   source platform, and its dataset license does not grant third-party source-IP
   rights or raw redistribution rights.
3. The targets are represented as Kelley Blue Book valuation ranges surfaced
   through AutoTrader. No reviewed artifact supplies an AutoTrader/KBB license
   or authorization covering ML training, derivative model publication, or
   hosted inference from those targets.

This is not a legal conclusion. It is the project's evidence-based permission
gate. The result is acquisition for private aggregate audit approved; batch
training, model publication, hosted inference, raw redistribution, and
online/River learning blocked.

## What would be required to reconsider

Before moving to “suitable for a separate controlled valuation-range
experiment,” the project would need written or otherwise authoritative evidence
that the KBB-derived targets may be used for noncommercial ML training and that
aggregate results and a trained derivative may be published. That approval
would then be pinned to this exact artifact and followed by a preregistered
design with KBB low/midpoint/high and width as separate targets, forward time
evaluation, grouped listings/vehicles, the denylist above, interval-order
checks, and no Cars.com/Yoad merge.

Because that permission evidence is absent, this audit stops here and no
experiment design is activated or model trained.

## Reproduction

With the verified files in the Git-ignored raw cache:

```powershell
$env:PYTHONPATH = "ml/src;backend/src"
.\.venv\Scripts\python.exe -m autovalue_ml.acquisition.autotrader_audit_cli `
  --project-root . `
  --output docs/data-reviews/rebrowser-autotrader-preview-v1.audit.json `
  --generated-at 2026-09-02T04:20:00+00:00
```

The aggregate JSON SHA-256 is
`a4eb30c62ce63059b0dcd73313ed171340dbb1924c9e56e9c8fc8cd0fae10a25`.
