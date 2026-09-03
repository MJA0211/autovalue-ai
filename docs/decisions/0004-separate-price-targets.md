# ADR 0004: Keep asking-price and completed-sale targets separate

- **Status:** accepted
- **Date:** 2026-08-28

## Context

AutoValue AI has two approved U.S./USD artifacts. US Sales Cars v2 contains
historical advertised retail asking prices for New, Used, and Certified
listings. Vehicle Sales Data v1 contains completed wholesale-auction sale prices.
Both are useful, but they measure different outcomes, market channels, and source
periods.

The retail source is broader when all three status values are retained. Almost
all New listings lack mileage, so a naive merge or complete-case filter would
either confuse target semantics or discard most of that useful coverage.

## Decision

Maintain two independently versioned modeling tracks:

1. a retail asking-price track with New, Used, and Certified as an explicit
   categorical feature and evaluation slice; and
2. a wholesale completed-sale track with chronological, VIN-isolated splitting.

Do not concatenate their rows under one regression target. Each track receives
its own validation report, split, cross-validation results, test metrics,
interval calibration, model card, artifact, and `PriceKind` lineage.

For the retail track, mileage remains nullable. Imputation and any missingness
indicator are fitted inside training folds, and MAE, RMSE, and R² are reported by
status so the large New segment cannot conceal weaker Used performance.

## Consequences

- AutoValue AI gains broader vehicle and price coverage without claiming that an
  asking price is a completed transaction.
- The product may select the retail model for its primary historical estimate
  while presenting wholesale results only as a clearly labeled benchmark.
- Model comparison happens within a track. Cross-track metric differences are
  not interpreted as a fair estimator comparison.
- A future multi-task or domain-adaptation experiment requires a new reviewed
  design; it is not an implicit consequence of having both datasets.
