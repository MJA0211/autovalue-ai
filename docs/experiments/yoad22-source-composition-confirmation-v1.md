# Yoad22 source-composition confirmation

## Decision

**Recommendation: retain moderate augmentation as a separate experimental
model.** It is not promoted, does not replace Phase 4 retail RF05, and is not
eligible for online/River learning.

Moderate augmentation uses all 98,552 Cars.com development rows plus 150,000
deterministically sampled Yoad rows. It improves Cars-domain MAE by 0.87% and
retains 98.17% of the full augmentation's Yoad-domain MAE gain. It is preferred
over the balanced and full treatments without using pooled MAE as the selection
key. Promotion evaluation is not recommended yet because important Cars slices
still regress, especially the highest-mileage band (+4.79%), Jaguar (+3.06%),
Hyundai (+1.89%), Chevrolet (+1.72%), low mileage (+1.62%), and ages 3–8
(+1.53%).

## Protected design

This is a separate confirmation of the immutable
[`yoad22-controlled-batch-v1.json`](yoad22-controlled-batch-v1.json), whose
SHA-256 is
`30d1f6011b7f2d5e611bbae6197be4780eeabcda3daca501c0b683807cf12ec5`.
The Cars-only and full-augmentation endpoints are reused byte-for-byte from that
report. Only balanced and moderate augmentation were newly fitted.

- Cars development remains exactly 98,552 rows.
- All 10,958 calibration rows and the legacy holdout remain excluded.
- The common no-model feature contract, RF05 tuple, preprocessing, random state,
  and five pooled predictor-group folds are unchanged.
- Every arm is scored on all 98,552 Cars and 242,666 Yoad validation records.
- Source identity and target price are excluded from sampling and fold assignment.
- Carson-Shively is not loaded or evaluated.

## Deterministic sampling

Balanced and moderate Yoad subsets are exact, nested samples. Allocation is
proportional within normalized-manufacturer, exact-model-year, and full-Yoad
mileage-decile strata. Rows are selected inside each stratum by a
domain-separated SHA-256 rank over predictors and stable pinned-artifact row
position. Price is never used.

| Sample | Yoad rows | Median year | Median mileage | Max make-share drift | Max year-share drift | Max mileage-decile drift |
|---|---:|---:|---:|---:|---:|---:|
| Balanced | 98,552 | 2013 | 95,858 mi | 0.0184 pp | 0.0252 pp | 0.0342 pp |
| Moderate | 150,000 | 2013 | 95,724 mi | 0.0145 pp | 0.0135 pp | 0.0183 pp |
| Full | 242,666 | 2013 | 95,756 mi | 0 pp | 0 pp | 0 pp |

## Aggregate paired results

All dollar metrics are USD.

| Composition | Pooled MAE | Pooled RMSE | Pooled R² | Cars MAE | Cars RMSE | Cars R² | Yoad MAE | Yoad RMSE | Yoad R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cars only | $10,121.05 | $21,949.41 | 0.4555 | $14,134.49 | $34,528.21 | 0.3452 | $8,491.11 | $13,901.77 | -0.3407 |
| Balanced | $7,979.69 | $19,486.84 | 0.5708 | $14,029.85 | $34,123.04 | 0.3605 | $5,522.58 | $7,815.08 | 0.5763 |
| Moderate | **$7,919.88** | **$19,407.16** | **0.5743** | **$14,012.21** | **$34,029.35** | **0.3640** | $5,445.65 | $7,701.34 | 0.5885 |
| Full | $7,920.62 | $19,620.95 | 0.5649 | $14,154.84 | $34,493.00 | 0.3466 | **$5,388.76** | **$7,624.98** | **0.5967** |

Relative to Cars-only, balanced improves Cars MAE 0.74% and Yoad MAE 34.96%.
Moderate improves Cars MAE 0.87% and Yoad MAE 35.87%. Full improves Yoad MAE
36.54% but worsens Cars MAE 0.14%.

## Fold stability

### Pooled fold MAE

| Fold | Cars only | Balanced | Moderate | Full |
|---:|---:|---:|---:|---:|
| 1 | $10,673.69 | $8,498.57 | $8,426.16 | $8,421.56 |
| 2 | $9,974.97 | $7,947.37 | $7,907.59 | $7,869.01 |
| 3 | $10,494.02 | $8,292.66 | $8,191.49 | $8,261.50 |
| 4 | $9,882.50 | $7,665.71 | $7,626.35 | $7,623.28 |
| 5 | $9,580.06 | $7,494.12 | $7,447.81 | $7,427.73 |

### Cars fold MAE

| Fold | Cars only | Balanced | Moderate | Full |
|---:|---:|---:|---:|---:|
| 1 | $15,808.07 | $15,860.78 | $15,795.50 | $15,930.84 |
| 2 | $13,967.35 | $13,957.51 | $13,990.43 | $13,974.62 |
| 3 | $15,694.38 | $15,001.68 | $14,874.27 | $15,305.27 |
| 4 | $13,133.60 | $13,113.89 | $13,154.08 | $13,292.34 |
| 5 | $12,061.54 | $12,205.56 | $12,236.60 | $12,262.51 |

Cars fold-MAE standard deviation improves from $1,453.34 for Cars-only to
$1,301.46 balanced and $1,250.03 moderate. Moderate's worst Cars fold regression
is 1.45%, versus 1.19% balanced and 1.67% full. Yoad fold-MAE standard deviation
is $50.65 moderate, $55.97 balanced, $54.03 full, and $102.51 Cars-only.

## Required Cars segment audit

Values are MAE changes versus the Cars-only model; negative is better.

| Cars segment | Balanced | Moderate | Full |
|---|---:|---:|---:|
| Highest mileage, 135,803–405,187 mi | +4.89% | **+4.79%** | +5.62% |
| Low mileage, 0–38,282 mi | **+0.92%** | +1.62% | +2.75% |
| Age 3–8 years | **+0.60%** | +1.53% | +3.10% |
| Age 8–13 years | -1.89% | **-1.18%** | +1.53% |
| Highest price band, above $36,590 | +0.01% | **-0.09%** | +1.22% |
| Alfa Romeo | -1.59% | **+0.41%** | +4.07% |
| Hyundai | **+0.72%** | +1.89% | +2.43% |
| Chevrolet | +2.03% | **+1.72%** | +2.24% |
| Audi | +0.75% | **-1.12%** | +2.14% |
| Jaguar | **+2.52%** | +3.06% | +2.04% |

Moderate improves every Cars price band, including the highest band, and three
of five mileage bands. It still degrades low- and highest-mileage Cars. It
improves ages 0–3, 8–13, and 13+, but degrades ages 3–8. Sixteen of 36 reported
Cars manufacturers regress under moderate augmentation. The largest are
Cadillac +4.96%, Jaguar +3.06%, Hyundai +1.89%, Toyota +1.89%, Chevrolet +1.72%,
and Acura +1.60%.

## Yoad segment audit

Moderate augmentation improves MAE for all four price bands, all four age bands,
all four mileage bands, and all 37 reported Yoad manufacturers. Improvements
range from 20.40% for GMC to more than 60% for several small makes. Weak absolute
segments remain: Harley-Davidson has $11,854.54 MAE and negative R², Porsche has
$11,533.97 MAE and negative R², and the high-price Yoad band has $12,543.47 MAE.
These results prevent a blanket claim that every Craigslist segment is reliable.

## Recommendation rationale

Moderate is preferred because it produces the best Cars MAE/RMSE/R², the best
pooled RMSE/R², stable folds, slightly lower worst focus-segment regression than
full, and 98.17% retention of the full Yoad MAE gain. Balanced is a credible
lower-data alternative but gives up additional Yoad accuracy and slightly worse
Cars aggregate metrics.

Moderate is not eligible for final promotion evaluation yet. The high-mileage
Cars regression remains above 4%, 16 Cars manufacturers regress, and the Yoad
population still lacks model and timestamps. No estimator is persisted and no
production, online, River, calibration, or holdout action follows this report.

The complete aggregate JSON, including all fold-level MAE/RMSE/R² and every
manufacturer, price, age, and mileage slice, is
[`yoad22-source-composition-confirmation-v1.json`](yoad22-source-composition-confirmation-v1.json).
Its SHA-256 is
`6ca3dd25cfb24bb0734497e4703cc516b3152e42f319286fcdd73374a6b2e5f5`.
