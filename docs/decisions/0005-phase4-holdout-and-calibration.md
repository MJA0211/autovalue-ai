# ADR 0005 — Freeze Phase 4 development and calibration before tuning

## Status

Accepted on 2026-08-29.

## Context

Phase 3 selected a transparent baseline inside each training partition and then
published one evaluation on each outer holdout. Those holdout aggregates are now
known. Reusing them as tuning feedback would gradually turn them into validation
sets, while creating a new random slice of the same historical files would not
restore an unbiased external test.

Phase 4 also needs prediction ranges. Interval width must be learned from data
that did not participate in preprocessing, hyperparameter selection, or fitting
the point estimator.

## Decision

All Phase 4 preprocessing, model-family choices, hyperparameters, and promotion
decisions use only a new development subset carved from the Phase 3 training
partition without looking at targets:

- Retail reserves approximately 10% of training rows for calibration. Whole
  year/make/model/mileage/status predictor groups remain together, allocation is
  deterministic and status-stratified, and price never enters allocation.
- Wholesale reserves the complete `2015_05` bucket for calibration. Development
  CV remains forward-only over warmup, January, February, and March/April.

The selected estimator is fit without calibration rows. Calibration absolute
residuals determine a finite-sample 90% estimated prediction range. The old
outer tests may audit one frozen winner per track, but every Phase 4 report must
label them `phase3_reused_legacy_holdout`, not untouched.

The full machine-readable policy, candidate configurations, seeds, selection
thresholds, resource limits, interval rules, and artifact controls are in
[`phase4-model-selection-v1.json`](../experiments/phase4-model-selection-v1.json).
Before any nonlinear fit ran, a reproducibility audit found that the initial
draft did not fully specify screening allocation or shortlist tie-breaking.
Those rules were made exact and target-free. The resulting frozen SHA-256 is
`6e517acb29634d676155c80fb73f4f126db492eba12a4281e9216dc568b1d384`;
implementation must reject a different policy before training.

The target-free implementation was exercised against both gated training
streams. Aggregate row counts, strata/buckets, zero retail group overlap, and
assignment hashes are recorded in
[`phase4-partition-audit-v1.json`](../experiments/phase4-partition-audit-v1.json).

## Consequences

- Phase 4 cannot claim a new unbiased final estimate from these two files.
- A genuinely fresh claim requires newly authorized later-period U.S. data.
- Calibration reduces the rows available to fit the interval-producing model,
  but keeps interval construction separate from tuning and fitting.
- The historical retail and wholesale targets, models, intervals, and reports
  remain independent.
- Private model persistence is allowed for local testing and hosted inference;
  downloadable publication remains blocked pending a new permission review.
