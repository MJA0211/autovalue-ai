# Hugging Face candidate review

## Decision

The two pinned Hugging Face sources remain governed additions. Yoad22 has now
passed the narrow gate for one controlled offline batch experiment; it has not
been merged into the production corpus or approved for deployment. Carson has
not entered any experiment.

| Candidate | Acquisition | Batch training | Online/River | Current decision |
|---|---|---|---|---|
| `Yoad22/craigslist-used-cars-eda` | Approved | Controlled experiment only | Blocked | Composition and weighting confirmations complete; no promotion |
| `Carson-Shively/used-car-price` | Approved | Blocked | Blocked | Resolve upstream origin, U.S. scope, USD semantics, and license metadata |

“Accepted row” in the quality reports means only that a row passes the current
schema/parser checks. It is not ML approval. The training and future online
gates fail closed independently.

## Reproducible acquisition choice

AutoValue uses revision-pinned direct file retrieval rather than the full
Hugging Face `datasets` package. Each allowlisted file is downloaded atomically,
bounded to 100 MB, retried with exponential backoff for HTTP 429/temporary 5xx,
and accepted only when its byte count and SHA-256 match the reviewed values.
Matching local bytes act as an immutable cache; conflicting existing bytes are
never overwritten silently.

This keeps the API/model runtime free of Hugging Face dependencies. The optional
acquisition environment adds only `pyarrow` for the Carson Parquet audit. CSV
and Parquet files are parsed locally; no repository code is executed and no
changing `main` revision is fetched silently.

| Source artifact | Commit | Bytes | SHA-256 |
|---|---|---:|---|
| Yoad `vehicles_clean.csv` | `912f968086868effb8523537015fb6a107c8eb3a` | 20,749,648 | `f702408dee80181f1d003e9e7d2173340f26eb6eb013e76e9a47af6296833791` |
| Carson `data/bronze/bronze.parquet` | `4f58418cafab4dff1bd273aae8c5da66cd2ed3f5` | 114,074 | `b5530c96732db26d05e59d4d02c868a1facb1e1612a7eb1c8ee5d204d497962e` |
| Carson auxiliary `data/silver/silver.parquet` | `4f58418cafab4dff1bd273aae8c5da66cd2ed3f5` | 82,060 | `26ed9d0d159ece7ab68b152e1355503ecd6bba46604523d21fe59fe506a7ffa7` |

The raw files remain under `data/raw/`, are ignored by Git, and must not be
redistributed from this repository.

## Quality findings

| Measure | Current Cars.com-derived retail | Yoad Craigslist | Carson bronze |
|---|---:|---:|---:|
| Rows | 137,099 | 250,361 | 4,009 |
| Exact duplicates in candidate artifact | already deduplicated | 0 | 0 |
| Median asking price | $46,984 USD | $13,900 USD | $31,000 USD |
| Price range | $1–$8,078,160 | $500–$54,241 | $2,000–$2,954,083 |
| Median model year | 2023 | 2013 | 2017 |
| Median mileage when present | 41,418 mi | 96,000 mi | 52,775 mi |
| Mileage present | 40.71% | 100% | 100% |
| Distinct makes | 62 | 43 | 57 |
| Model coverage | 100% | 0% | 100% (1,898 values) |
| Row-level geography | none | 100% (50 states + D.C.) | none |
| Accident history | none | none | 97.18% |
| Title status | none | 100% | 85.13% |

Yoad is not a simple row-count extension of the current corpus. It is an older,
lower-price Craigslist used-vehicle population: its median year is ten years
older, median asking price is $33,084 lower, and it contains no model, VIN,
listing ID, URL, or row timestamp. The source card also describes filtering
price/year/odometer values and applying IQR outlier removal. Those upstream
choices are retained as lineage; AutoValue's audit applies no extra silent
target filter.

Carson supplies the richer model, engine, transmission, fuel, accident, and
clean-title fields requested for a possible rich-feature model. It is also
small, category-heavy (1,898 raw model values across 4,009 rows), and has 450
vehicles above $75,000. Forty-eight prices exceed $250,000 and three exceed
$1 million. These values remain visible rather than being blindly removed.
Make aliases such as `Mercedes`/`Mercedes-Benz`, `Land`/`Land Rover`, and
`Alfa`/`Alfa Romeo` require a reviewed normalization map before comparisons.

### Carson's 7,970-row discrepancy

The repository contains 4,009 original-format bronze rows and a 3,961-row
cleaned silver layer. Their sum is 7,970, but they are processing layers, not
7,970 independent vehicle observations. The bronze file is therefore the
pinned audit source because it preserves strings such as `51,000 mi.` and
`$10,300`. The silver layer is not concatenated, and its transformed values do
not replace raw audit metadata.

## Provenance and overlap

The current retail corpus comes from a historical Cars.com extraction. Yoad is
explicitly derived from the Austin Reese Craigslist Cars and Trucks dataset.
No shared upstream family is known between those two sources, so Yoad is not
classified as a confirmed source duplicate of the current corpus. It must,
however, be treated as fully overlapping with any other Austin Reese/Craigslist
derivative. The source-overlap gate detects that family and blocks a merge.

The cross-source coarse-key audit found one shared unique
year/make/mileage/price key between Yoad and current retail, and zero shared
year/make/model/mileage/price keys between Carson and current retail. These are
only possible-duplicate signals—not proof of identity—because the sources lack
a common stable ID. Any approved combined experiment must group/deduplicate
before splitting and cannot use these counts to claim zero leakage.

Carson does not identify its original row source. Its overlap classification is
therefore `indeterminate`, which blocks merging even though the coarse-key audit
found no exact collision. Dollar-formatted targets also do not independently
prove a U.S. market or USD semantic contract.

## Licensing

The [Yoad dataset card](https://huggingface.co/datasets/Yoad22/craigslist-used-cars-eda)
declares CC BY 4.0 and identifies the Austin Reese/Craigslist lineage. AutoValue
retains attribution to both the derivative publisher and documented upstream
source. The upstream [Austin Reese Kaggle page](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
displays CC0. These labels do not authorize representing the rows as current
market observations or combining another derivative without leakage controls.

The [Carson dataset card](https://huggingface.co/datasets/Carson-Shively/used-car-price)
states MIT in its text, while its repository metadata is incomplete and the
upstream data source is not named. Private acquisition for compatibility review
is approved; training, redistribution, and online learning remain blocked
pending clarification. This project review records evidence and restrictions;
it is not a legal conclusion or sublicense.

## Schema and feature strategy

Both adapters preserve source identity and immutable raw values in audit
metadata. `source_id`, raw values, content hashes, and repository lineage are
explicitly excluded from `feature_values()`.

- Yoad maps `manufacturer`, `year`, `odometer`, `condition`, `cylinders`,
  `fuel`, `title_status`, `transmission`, `drive`, `type`, and `state`. It does
  not fabricate a model or accident field.
- Carson strictly parses `milage` strings into miles and `price` strings into
  integer USD cents, maps `brand` to make and `model_year` to year, and
  normalizes accident/clean-title semantics while retaining original strings.
  Its candidate market remains unresolved until U.S. scope is verified.

The Yoad experiment uses the common `year`, `make`, `mileage`, and
`vehicle_status` contract. `model` is excluded from both arms instead of being
fabricated for Craigslist rows. Make values are case-normalized, but aliases
such as `Mercedes`/`Mercedes-Benz` are not silently merged. Five shared
GroupKFold assignments are derived from pooled predictor groups without using
price or source identity, so identical common predictors cannot cross a fold.

If Carson is later approved, Experiment B should compare a broad model
(`year`, `make`, `mileage`, and only genuinely shared fields) with a separate
rich model containing model/engine/transmission/fuel/accident/title. Missing
rich fields must not cause most current-retail rows to be discarded. Yoad's
Experiment C needs a used-only target/domain decision and grouped split design.
Experiment D remains blocked until each input independently passes its gate.

## Yoad controlled batch result

All Yoad batch requirements are satisfied for controlled experimentation only:
the artifact is revision- and checksum-pinned; U.S./USD asking-price scope,
schema quality, lineage, CC BY 4.0 attribution, and source overlap were reviewed;
the unknown-manufacturer filter and common feature contract are explicit; and
the split keeps pooled predictor groups intact. The machine-readable
[approval](yoad22-controlled-batch-approval-v1.json) records the exact limits.
Online/River learning remains blocked because the data has no stable listing ID,
row timestamp, delayed-label contract, or replay-safe append-only semantics.

The experiment used only the 98,552-row Cars.com development partition. It left
10,958 calibration rows and the legacy holdout untouched. From 250,361 Yoad
rows, 7,695 `unknown` manufacturers were excluded; there were zero raw exact
duplicates and zero exact year/make/mileage/price collisions with this Cars
development subset. The final combined input contains 341,218 rows: 28.88%
Cars.com and 71.12% Yoad.

| Paired out-of-fold evaluation | Rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Cars-only model, pooled validation | 341,218 | $10,121.05 | $21,949.41 | 0.4555 |
| Combined model, pooled validation | 341,218 | $7,920.62 | $19,620.95 | 0.5649 |
| Cars-only model, Cars slice | 98,552 | $14,134.49 | $34,528.21 | 0.3452 |
| Combined model, Cars slice | 98,552 | $14,154.84 | $34,493.00 | 0.3466 |
| Cars-only model, Yoad slice | 242,666 | $8,491.11 | $13,901.77 | -0.3407 |
| Combined model, Yoad slice | 242,666 | $5,388.76 | $7,624.98 | 0.5967 |

Pooled MAE improves 21.74%, and Yoad-domain MAE improves 36.54%. Cars-domain
MAE worsens 0.14%, although Cars RMSE and R² improve slightly. The Cars fold-MAE
standard deviation falls from $1,453.34 to $1,329.28; the worst Cars fold
degradation is 1.67%. The combined model still shows local Cars regressions:
the highest-mileage band worsens 5.62%, ages 3–8 worsen 3.10%, and several makes
worsen, led by Alfa Romeo at 4.07%, Hyundai at 2.43%, and Chevrolet at 2.24%.
These shifts are material audit findings even though source-level guardrails pass.

Yoad is ten model years older at the median, $32,510 cheaper at the median, and
54,468.5 miles higher at the median than Cars development data. It also has
59.22 percentage points more mileage coverage and no model field or row
timestamps. The gain therefore demonstrates better generalization across this
specific mixed historical population, not a universally better Cars.com model.
That result triggered the separately defined source-composition confirmation
below, but the full combined model is **not promoted** and does not replace
Phase 4 RF05.

## Source-composition confirmation

The confirmation compares Cars-only with exact nested Yoad samples of 98,552
rows (balanced), 150,000 rows (moderate), and all 242,666 rows (full). Sampling
is target-free and proportional within normalized manufacturer, exact model
year, and Yoad mileage-decile strata. Maximum distribution drift is below 0.035
percentage points for every audited make/year/mileage dimension.

| Composition | Cars MAE | Yoad MAE | Pooled MAE | Cars MAE change | Yoad MAE improvement |
|---|---:|---:|---:|---:|---:|
| Cars only | $14,134.49 | $8,491.11 | $10,121.05 | — | — |
| Balanced | $14,029.85 | $5,522.58 | $7,979.69 | -0.74% | 34.96% |
| Moderate | **$14,012.21** | $5,445.65 | **$7,919.88** | **-0.87%** | 35.87% |
| Full | $14,154.84 | **$5,388.76** | $7,920.62 | +0.14% | **36.54%** |

Moderate is preferred and retains 98.17% of the full Yoad-domain improvement.
It also lowers Cars fold-MAE standard deviation from $1,453.34 to $1,250.03.
The treatment is retained only as a **separate experimental model**: the
highest-mileage Cars band still worsens 4.79%, Jaguar 3.06%, Hyundai 1.89%,
Chevrolet 1.72%, low-mileage Cars 1.62%, and ages 3–8 worsen 1.53%. Sixteen of
36 reported Cars manufacturers regress, while all 37 reported Yoad
manufacturers improve. This does not clear final promotion-evaluation gates.

The confirmation reuses the checksum-bound Cars-only and full endpoints, fits
only the two new compositions, and preserves the same protected Cars boundary,
folds, common feature contract, preprocessing, RF05 parameters, and random
state. Calibration, legacy holdout, Phase 4, Carson, model persistence, and
River remain outside its scope.

## Fold-local weighting confirmation

The follow-up weighting experiment fixes the selected moderate composition at
98,552 Cars development rows plus the same deterministic 150,000 Yoad rows. It
reuses the unweighted moderate result and newly compares source-balanced,
source-plus-mileage, and broad source-plus-segment weighting. Each fold derives
weights only from its training predictors and source distribution; validation
remains unweighted and price never influences observation weights.

| Treatment | Cars MAE | Yoad MAE | Yoad gain retained | Worst focus regression | Cars makes regressing |
|---|---:|---:|---:|---:|---:|
| Moderate, unweighted | $14,012.21 | $5,445.65 | 100.00% | 4.96% | 16/36 |
| Source balanced | $13,954.26 | $5,498.02 | 98.28% | 8.24% | 18/36 |
| Source + mileage | **$13,874.78** | $5,511.04 | 97.85% | 3.91% | 17/36 |
| Source + segments | $13,889.84 | $5,522.97 | 97.46% | **3.41%** | **13/36** |

The broad segment formula improves eight of nine focus slices and reduces the
highest-mileage regression to 3.41%, but it creates a 7.43% Genesis regression.
Mileage weighting produces a 14.00% Genesis regression, and source balancing
produces an 8.24% Hyundai regression. Those failures exceed the preregistered
5% all-slice ceiling. **Weighting is rejected; the unweighted moderate branch
remains the separate experimental reference.** No formula is promoted or made
eligible for final promotion evaluation.

## Reports and reproduction

- [Yoad quality report](hf-yoad22-craigslist-used-cars-eda-v1.quality.json)
- [Carson quality report](hf-carson-shively-used-car-price-v1.quality.json)
- [Comparison with current retail](hf-candidates-vs-current-retail-v1.comparison.json)
- [Yoad controlled batch approval](yoad22-controlled-batch-approval-v1.json)
- [Yoad controlled experiment](../experiments/yoad22-controlled-batch-v1.json)
- [Yoad composition confirmation](../experiments/yoad22-source-composition-confirmation-v1.json)
- [Confirmation interpretation](../experiments/yoad22-source-composition-confirmation-v1.md)
- [Yoad weighting confirmation](../experiments/yoad22-training-weight-confirmation-v1.json)
- [Weighting interpretation](../experiments/yoad22-training-weight-confirmation-v1.md)

The Yoad quality and candidate-comparison JSON files are immutable pre-approval
snapshots, so their embedded Yoad status remains `pending`. The later approval
and experiment reports above are the current decision evidence and do not
rewrite those earlier audit artifacts.

Reproduce an independent candidate report from the repository root:

```powershell
python -m pip install -r requirements-acquisition.txt
$env:PYTHONPATH = "ml/src"
python -m autovalue_ml.acquisition.huggingface_quality_cli yoad22-craigslist --project-root . --output docs/data-reviews/yoad-repro.json
python -m autovalue_ml.acquisition.huggingface_quality_cli carson-shively --project-root . --output docs/data-reviews/carson-repro.json
```

The committed reports use `2026-09-01T12:00:00+00:00` as a reproducible report
timestamp. Pass the same value through `--generated-at` for byte-stable report
metadata. Raw rows are never written into the reports.

Reproduce the controlled experiment only after the pinned Cars.com and Yoad
artifacts are present locally:

```powershell
$env:PYTHONPATH = "ml/src"
python -m autovalue_ml.modeling.yoad_experiment_cli --project-root . --output docs/experiments/yoad22-repro.json
```

The canonical experiment report SHA-256 is
`30d1f6011b7f2d5e611bbae6197be4780eeabcda3daca501c0b683807cf12ec5`.

Reproduce the separate confirmation without replacing the controlled report:

```powershell
$env:PYTHONPATH = "ml/src"
python -m autovalue_ml.modeling.yoad_confirmation_cli --project-root . --output docs/experiments/yoad22-confirmation-repro.json
```

The canonical confirmation JSON SHA-256 is
`6ca3dd25cfb24bb0734497e4703cc516b3152e42f319286fcdd73374a6b2e5f5`.

Reproduce or resume the separate weighting confirmation without overwriting
earlier reports:

```powershell
$env:PYTHONPATH = "ml/src"
python -m autovalue_ml.modeling.yoad_weighting_cli --project-root . --output docs/experiments/yoad22-weighting-repro.json --checkpoint docs/experiments/yoad22-weighting-repro.checkpoint.json
```

The canonical weighting JSON SHA-256 is
`ceddd3dd530487ef57ee3d24390d5f0ef8e26db9c04f5d5b4f0ba56e84fb11a2`.
