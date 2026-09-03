# Yoad22 fold-local training-weight confirmation

## Decision

**Recommendation: weighting rejected; retain the unweighted moderate
augmentation baseline.** None of the three conservative weighting treatments
passes every preregistered Cars-slice, manufacturer, Yoad-retention, and
fold-stability gate. No result is promoted, persisted as a model, or made
eligible for final promotion evaluation.

The broad source-plus-segment treatment comes closest. It improves Cars MAE from
$14,012.21 to $13,889.84, retains 97.46% of the moderate model's Yoad-domain
gain, improves eight of nine focus slices, and reduces the highest-mileage Cars
regression from 4.79% to 3.41%. It nevertheless creates a 7.43% Genesis
regression versus Cars-only, above the preregistered 5% ceiling. Changing the
formula after observing that manufacturer would turn a diagnostic slice into a
tuning target, so no such adjustment is made.

## Protected design

This is a separate experiment based on the immutable
[`yoad22-source-composition-confirmation-v1.json`](yoad22-source-composition-confirmation-v1.json),
SHA-256
`6ca3dd25cfb24bb0734497e4703cc516b3152e42f319286fcdd73374a6b2e5f5`.
The verified moderate result is reused without refitting.

- Composition remains exactly 98,552 Cars.com development rows plus the same
  deterministic 150,000-row Yoad subset.
- All 10,958 calibration rows and the legacy holdout remain excluded.
- The five pooled predictor-group folds, common no-model feature contract,
  preprocessing, RF05 tuple, and random state remain unchanged.
- All validation observations are unweighted and all 341,218 records receive a
  validation prediction in each treatment.
- Each fold derives weights only from that fold's training predictors and source
  labels. Target price and validation statistics are never inputs to a weight.
- Source identity affects training weight only; it is not a model feature.
- Phase 4 artifacts are unchanged, River/online learning remains blocked, and
  Carson-Shively is not loaded.

## Preregistered weighting formulas

All formulas first assign each source half of the fold's total training weight:

`base_weight(source) = fold_training_rows / (2 × source_training_rows)`

The source-balanced treatment stops there. The mileage treatment multiplies
that base by a fold-local category-alignment factor:

`clip((mean_source_band_share / source_band_share)^0.5, 0.85, 1.15)`

The segment treatment applies the same distribution principle to mileage band,
vehicle-age band, and manufacturer with exponent 0.25 and per-dimension bounds
of 0.90–1.10. Manufacturer adjustment is shrunk toward one according to
training support, using the feature contract's 25-row rare-category threshold.
Their combined adjustment is bounded at 0.80–1.25. Every treatment is finally
renormalized within source; absolute observation weights must remain within
0.50–2.00.

These constants were fixed before the weighted validation results were opened.

## Aggregate unweighted results

All dollar metrics are USD. Pooled MAE is reported but is not a selection key.

| Treatment | Pooled MAE | Pooled RMSE | Pooled R² | Cars MAE | Cars RMSE | Cars R² | Yoad MAE | Yoad RMSE | Yoad R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cars only | $10,121.05 | $21,949.41 | 0.4555 | $14,134.49 | $34,528.21 | 0.3452 | $8,491.11 | $13,901.77 | -0.3407 |
| Moderate, unweighted | **$7,919.88** | $19,407.16 | 0.5743 | $14,012.21 | $34,029.35 | 0.3640 | **$5,445.65** | **$7,701.34** | **0.5885** |
| Source balanced | $7,940.39 | $19,316.53 | 0.5783 | $13,954.26 | $33,810.31 | 0.3722 | $5,498.02 | $7,772.44 | 0.5809 |
| Source + mileage | $7,926.69 | **$19,254.39** | **0.5810** | **$13,874.78** | **$33,681.43** | **0.3770** | $5,511.04 | $7,782.89 | 0.5798 |
| Source + segments | $7,939.53 | $19,333.53 | 0.5776 | $13,889.84 | $33,824.28 | 0.3717 | $5,522.97 | $7,807.13 | 0.5772 |

Relative to unweighted moderate, Cars MAE improves 0.41% with source balancing,
0.98% with mileage weighting, and 0.87% with segment weighting. The treatments
retain 98.28%, 97.85%, and 97.46% of moderate's Yoad improvement respectively.

## Weight diagnostics

| Treatment | Weight range across folds | Median of fold medians | Mean | Minimum effective-sample fraction |
|---|---:|---:|---:|---:|
| Source balanced | 0.8279–1.2625 | 0.8284 | 1.0000 | 95.68% |
| Source + mileage | 0.7937–1.5292 | 0.9080 | 1.0000 | 94.56% |
| Source + segments | 0.7793–1.7493 | 0.8735 | 1.0000 | 94.02% |

Every fold assigns the two sources identical total weight. Per-source totals
range from 99,393 to 99,458 because fold training sizes differ. No formula
approaches the absolute 0.50–2.00 limits, and no effective-sample fraction falls
below 94%. The machine-readable report includes per-fold minimum, maximum,
median, mean, effective sample size, source row counts/totals, and the largest
and smallest mileage, age, and manufacturer adjustment factors.

## Fold stability

Each cell is `Cars MAE / Yoad MAE`; all observations are unweighted.

| Fold | Cars only | Moderate | Source balanced | Source + mileage | Source + segments |
|---:|---:|---:|---:|---:|---:|
| 1 | $15,808.07 / $8,579.35 | $15,795.50 / $5,420.16 | $15,887.44 / $5,470.23 | $15,776.06 / $5,481.99 | $15,766.21 / $5,489.76 |
| 2 | $13,967.35 / $8,340.53 | $13,990.43 / $5,417.35 | $13,926.84 / $5,465.80 | $13,915.59 / $5,477.42 | $13,923.95 / $5,490.33 |
| 3 | $15,694.38 / $8,394.46 | $14,874.27 / $5,493.42 | $14,647.45 / $5,560.54 | $14,491.82 / $5,570.32 | $14,501.54 / $5,589.80 |
| 4 | $13,133.60 / $8,561.99 | $13,154.08 / $5,381.14 | $12,987.01 / $5,421.10 | $12,979.92 / $5,441.11 | $13,084.62 / $5,451.87 |
| 5 | $12,061.54 / $8,578.96 | $12,236.60 / $5,515.89 | $12,311.89 / $5,572.08 | $12,199.03 / $5,583.99 | $12,161.27 / $5,592.75 |

Cars fold-MAE standard deviation is $1,250.03 unweighted, $1,253.11 source
balanced, $1,233.38 source-plus-mileage, and $1,227.39 source-plus-segments.
Worst Cars fold degradation versus Cars-only is 1.45%, 2.08%, 1.14%, and 0.83%
respectively. Full fold MAE/RMSE/R² values are retained in the JSON report.

## Cars focus slices

Values are MAE changes versus Cars-only; negative is better.

| Cars focus slice | Moderate | Source balanced | Source + mileage | Source + segments |
|---|---:|---:|---:|---:|
| Highest mileage | +4.79% | +4.14% | +3.91% | **+3.41%** |
| Below 38,282 miles | +1.62% | **+0.33%** | +0.30% | +0.44% |
| Age 3–8 years | +1.53% | -0.12% | **-0.35%** | -0.30% |
| Cadillac | +4.96% | +0.88% | **+0.07%** | +0.42% |
| Jaguar | +3.06% | +2.93% | **+2.28%** | +2.56% |
| Hyundai | +1.89% | +8.24% | +2.25% | **+0.80%** |
| Toyota | **+1.89%** | +2.04% | +2.62% | +2.33% |
| Chevrolet | +1.72% | +1.97% | +0.78% | **+0.02%** |
| Acura | +1.60% | +0.26% | +0.01% | **-0.28%** |

Source balancing improves six of nine focus slices but makes Hyundai much worse.
Mileage weighting improves seven of nine; segment weighting improves eight of
nine. Toyota remains worse under every weighting treatment.

## All Cars mileage, age, and price bands

Values are MAE changes versus Cars-only.

| Mileage band | Moderate | Source balanced | Source + mileage | Source + segments |
|---|---:|---:|---:|---:|
| 0–38,282 mi | +1.62% | +0.33% | +0.30% | +0.44% |
| 38,282–86,204 mi | -3.69% | -4.19% | -4.04% | -4.18% |
| 86,204–135,803 mi | -1.22% | -1.80% | -1.97% | -2.07% |
| 135,803–405,187 mi | +4.79% | +4.14% | +3.91% | +3.41% |
| Missing mileage | -1.27% | -1.42% | -2.22% | -2.09% |

| Vehicle-age band | Moderate | Source balanced | Source + mileage | Source + segments |
|---|---:|---:|---:|---:|
| 0–3 years | -1.04% | -1.26% | -1.87% | -1.74% |
| 3–8 years | +1.53% | -0.12% | -0.35% | -0.30% |
| 8–13 years | -1.18% | -2.59% | -2.97% | -3.45% |
| 13+ years | -3.16% | -7.29% | -7.01% | -7.58% |

| Price band | Moderate | Source balanced | Source + mileage | Source + segments |
|---|---:|---:|---:|---:|
| $1–$8,995 | -31.08% | -29.77% | -29.02% | -29.02% |
| $8,995–$19,995 | -10.88% | -10.25% | -10.00% | -10.21% |
| $19,995–$36,590 | -1.30% | -1.32% | -1.44% | -1.58% |
| Above $36,590 | -0.09% | -0.65% | -1.36% | -1.19% |

No weighted treatment introduces an aggregate age- or price-band regression.
The rejection is driven by manufacturer instability, not by these broad bands.

## Cars manufacturer audit

Rows and Cars-only MAE establish support and reference error. Remaining columns
are MAE changes versus Cars-only.

| Manufacturer | Rows | Cars-only MAE | Moderate | Balanced | Mileage | Segments |
|---|---:|---:|---:|---:|---:|---:|
| Acura | 1,512 | $6,990.10 | +1.60% | +0.26% | +0.01% | -0.28% |
| Alfa Romeo | 329 | $8,670.14 | +0.41% | -4.97% | +2.03% | -0.32% |
| Aston Martin | 100 | $133,461.96 | -21.63% | -40.37% | -46.32% | -40.10% |
| Audi | 3,306 | $22,087.47 | -1.12% | +0.98% | -3.32% | -2.05% |
| Bentley | 192 | $135,944.98 | -31.04% | -32.73% | -29.04% | -23.70% |
| BMW | 4,838 | $19,472.60 | -5.99% | -7.43% | -8.71% | -9.30% |
| Buick | 851 | $11,773.70 | -5.65% | -6.44% | -5.71% | -5.60% |
| Cadillac | 1,712 | $13,383.13 | +4.96% | +0.88% | +0.07% | +0.42% |
| Chevrolet | 10,176 | $17,114.78 | +1.72% | +1.97% | +0.78% | +0.02% |
| Chrysler | 828 | $9,029.84 | -4.37% | +1.15% | +2.23% | -0.81% |
| Dodge | 2,552 | $14,113.81 | -0.92% | +0.15% | +0.45% | +0.95% |
| Ford | 9,735 | $13,025.06 | +0.18% | +0.07% | +0.05% | -0.03% |
| Genesis | 865 | $8,481.99 | +1.27% | +6.25% | +14.00% | +7.43% |
| GMC | 4,710 | $16,866.59 | -0.23% | -0.74% | -0.05% | +0.00% |
| Honda | 4,175 | $5,599.93 | -1.69% | -2.14% | -2.45% | -2.23% |
| Hyundai | 2,500 | $9,793.69 | +1.89% | +8.24% | +2.25% | +0.80% |
| Infiniti | 1,087 | $7,538.60 | +0.74% | +1.38% | +0.89% | +0.18% |
| Jaguar | 473 | $14,083.64 | +3.06% | +2.93% | +2.28% | +2.56% |
| Jeep | 5,589 | $11,288.39 | -0.67% | -1.06% | -0.07% | -0.83% |
| Kia | 3,492 | $8,102.51 | -1.85% | -2.16% | -2.17% | -1.86% |
| Land Rover | 1,709 | $19,382.46 | +0.21% | +0.55% | -0.99% | -1.78% |
| Lexus | 3,892 | $13,915.39 | -0.53% | -0.85% | -1.94% | -1.42% |
| Lincoln | 1,141 | $14,577.40 | +0.19% | +1.10% | +0.11% | +0.22% |
| Maserati | 339 | $35,709.43 | +0.22% | -0.37% | +0.44% | +0.24% |
| Mazda | 1,814 | $8,097.91 | -4.93% | -3.38% | -4.10% | -3.70% |
| Mercedes | 4,605 | $26,539.62 | -0.12% | +0.21% | +0.72% | +0.23% |
| Mini | 247 | $10,642.00 | -4.28% | +6.88% | -2.76% | -6.47% |
| Mitsubishi | 616 | $9,674.80 | -7.28% | -6.40% | -5.82% | -6.37% |
| Nissan | 3,269 | $10,967.28 | +0.15% | +0.18% | +0.25% | +0.73% |
| Porsche | 1,211 | $28,743.52 | -0.86% | -1.04% | -0.75% | -1.27% |
| Ram | 3,918 | $13,240.86 | +0.92% | +0.21% | +0.43% | -0.18% |
| Subaru | 1,866 | $5,663.55 | -2.47% | -2.86% | -3.75% | -4.11% |
| Tesla | 724 | $10,658.47 | -1.57% | -2.57% | -2.11% | -1.99% |
| Toyota | 9,909 | $9,484.00 | +1.89% | +2.04% | +2.62% | +2.33% |
| Volkswagen | 2,352 | $7,266.94 | -3.51% | -4.06% | -4.41% | -4.64% |
| Volvo | 1,279 | $6,476.66 | +1.41% | -2.69% | -4.03% | -1.67% |

Manufacturer regression counts are 16 of 36 for moderate, 18 for source
balanced, 17 for mileage weighting, and 13 for segment weighting. Despite the
lower count, segment weighting fails because Genesis exceeds the severity gate.

## Yoad manufacturer audit

All values are unweighted MAE in USD. Every treatment remains substantially
better than Cars-only on every reported Yoad manufacturer, but every weighted
treatment gives back some accuracy on many makes relative to moderate.

| Manufacturer | Rows | Cars only | Moderate | Balanced | Mileage | Segments |
|---|---:|---:|---:|---:|---:|---:|
| Acura | 3,751 | $6,676.85 | $3,625.55 | $3,652.56 | $3,662.46 | $3,678.56 |
| Alfa-Romeo | 495 | $12,772.84 | $2,563.36 | $2,669.33 | $2,631.18 | $2,636.83 |
| Audi | 4,777 | $8,755.60 | $4,595.29 | $4,783.73 | $4,812.38 | $4,771.09 |
| BMW | 8,786 | $6,780.68 | $4,503.80 | $4,550.64 | $4,540.37 | $4,534.01 |
| Buick | 3,354 | $9,607.04 | $3,991.07 | $4,091.02 | $4,155.95 | $4,178.74 |
| Cadillac | 4,155 | $8,311.92 | $5,341.84 | $5,318.50 | $5,386.29 | $5,422.97 |
| Chevrolet | 30,264 | $10,797.69 | $7,497.86 | $7,556.49 | $7,556.59 | $7,589.39 |
| Chrysler | 3,980 | $7,979.38 | $3,311.12 | $3,481.02 | $3,540.24 | $3,568.36 |
| Dodge | 7,726 | $9,596.93 | $4,899.97 | $5,044.19 | $5,082.37 | $5,102.29 |
| Fiat | 510 | $14,569.42 | $5,310.78 | $5,942.12 | $5,871.01 | $6,146.60 |
| Ford | 39,779 | $9,472.29 | $7,416.18 | $7,417.22 | $7,412.63 | $7,419.49 |
| GMC | 8,931 | $8,346.26 | $6,643.96 | $6,712.32 | $6,701.50 | $6,701.57 |
| Harley-Davidson | 102 | $31,840.04 | $11,854.54 | $12,197.32 | $11,620.62 | $12,229.52 |
| Honda | 14,554 | $6,051.69 | $2,744.80 | $2,776.92 | $2,790.98 | $2,797.76 |
| Hyundai | 6,430 | $6,862.64 | $2,977.74 | $3,028.09 | $3,049.59 | $3,046.41 |
| Infiniti | 3,067 | $5,769.72 | $3,694.60 | $3,709.80 | $3,670.46 | $3,698.44 |
| Jaguar | 1,244 | $8,587.62 | $4,910.78 | $4,968.55 | $4,810.89 | $4,926.37 |
| Jeep | 11,077 | $7,586.70 | $5,843.65 | $5,835.33 | $5,845.88 | $5,852.43 |
| Kia | 5,158 | $8,706.49 | $3,454.56 | $3,519.12 | $3,520.31 | $3,499.81 |
| Lexus | 5,226 | $6,006.26 | $4,161.23 | $4,200.21 | $4,194.49 | $4,198.86 |
| Lincoln | 2,547 | $7,843.49 | $4,013.87 | $4,117.97 | $4,144.61 | $4,121.19 |
| Mazda | 3,479 | $7,045.09 | $2,917.86 | $3,083.90 | $3,139.29 | $3,113.88 |
| Mercedes-Benz | 6,468 | $7,312.21 | $5,013.88 | $5,033.53 | $5,026.02 | $5,061.15 |
| Mercury | 733 | $9,224.48 | $2,825.10 | $3,063.83 | $3,209.99 | $3,224.59 |
| Mini | 1,601 | $8,471.94 | $3,637.58 | $3,777.37 | $3,919.45 | $4,002.90 |
| Mitsubishi | 2,058 | $11,715.81 | $5,101.06 | $5,239.97 | $5,263.33 | $5,279.99 |
| Nissan | 12,148 | $8,344.21 | $4,227.08 | $4,314.46 | $4,348.85 | $4,370.30 |
| Pontiac | 1,376 | $7,666.88 | $2,965.21 | $3,105.26 | $3,204.81 | $3,216.05 |
| Porsche | 708 | $16,759.53 | $11,533.97 | $12,413.72 | $12,612.75 | $12,523.47 |
| Ram | 9,160 | $9,768.39 | $7,739.51 | $7,713.81 | $7,726.07 | $7,730.04 |
| Rover | 1,291 | $10,096.62 | $6,642.34 | $6,767.78 | $6,795.05 | $6,764.20 |
| Saturn | 790 | $8,654.13 | $2,904.89 | $3,034.40 | $3,148.62 | $3,213.87 |
| Subaru | 6,659 | $6,459.67 | $3,043.94 | $3,107.53 | $3,135.61 | $3,155.99 |
| Tesla | 412 | $9,783.23 | $6,364.06 | $6,377.90 | $6,495.52 | $6,358.13 |
| Toyota | 21,845 | $7,735.91 | $5,729.36 | $5,742.08 | $5,742.43 | $5,751.38 |
| Volkswagen | 5,806 | $7,927.04 | $3,409.13 | $3,511.59 | $3,575.58 | $3,577.32 |
| Volvo | 2,192 | $6,030.16 | $3,410.59 | $3,477.97 | $3,561.24 | $3,570.27 |

## Gate outcome and interpretation

| Treatment | Focus slices improved | Worst focus regression | Cars manufacturers regressing | Worst reported Cars regression | Result |
|---|---:|---:|---:|---:|---|
| Source balanced | 6/9 | 8.24% | 18/36 | 8.24% Hyundai | Rejected |
| Source + mileage | 7/9 | 3.91% | 17/36 | 14.00% Genesis | Rejected |
| Source + segments | 8/9 | 3.41% | 13/36 | 7.43% Genesis | Rejected |

Source balancing fails the focus-reduction, manufacturer-count, and 5% severity
gates. Mileage weighting fails the manufacturer-count and severity gates.
Segment weighting passes every gate except the all-slice 5% ceiling. That near
miss is useful evidence, but not permission to tune specifically for Genesis or
reopen the same validation results.

The correct current action is to retain the existing moderate augmentation
model as a separate, unpromoted experimental branch. If weighting is revisited,
it needs a separately preregistered methodology and fresh evaluation boundary,
not another adjustment against these validation slices.

## Reproducibility and artifacts

The aggregate-only, resumable checkpoint contains 15 completed fold-treatment
fits, with no raw records or row-level predictions. Reassembling from it produces
the final report byte-for-byte without refitting.

- Machine-readable report:
  [`yoad22-training-weight-confirmation-v1.json`](yoad22-training-weight-confirmation-v1.json),
  SHA-256
  `ceddd3dd530487ef57ee3d24390d5f0ef8e26db9c04f5d5b4f0ba56e84fb11a2`.
- Aggregate checkpoint:
  [`yoad22-training-weight-confirmation-v1.checkpoint.json`](yoad22-training-weight-confirmation-v1.checkpoint.json),
  SHA-256
  `52fb1f1c57c358fdee9339e76b5cc84e4573c7ad32851b903cf30df452c8360e`.
- Weighting-policy SHA-256:
  `66a7d7e4e25bb3ce9e85b4ac5a37833b58a1c6b2647000ebe3c9aec0be112641`.

The JSON contains complete fold MAE/RMSE/R², source metrics, all mileage/age/
price/manufacturer slices, fold-local formula diagnostics, source-weight totals,
effective sample sizes, gate outcomes, and governance assertions.
