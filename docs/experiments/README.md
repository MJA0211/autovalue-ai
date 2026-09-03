# Experiment controls

This directory contains project-owned, aggregate-only controls and audits. It
must not contain source rows, targets, row-level predictions, residuals, learned
category vocabularies, or fitted estimators.

- [`phase4-model-selection-v1.json`](phase4-model-selection-v1.json) is the
  checksum-pinned Phase 4 protocol. It fixes data lineage, target semantics,
  development/calibration boundaries, target-free screening, candidate tuples,
  seeds, cross-validation, shortlist and promotion rules, interval construction,
  resource budgets, feature-importance scope, and private-artifact controls.
- [`phase4-partition-audit-v1.json`](phase4-partition-audit-v1.json) records the
  real gated partition and screening counts plus reproducible assignment hashes.
  It contains no price values and no model results.
- [`phase4-retail-screening-v1.json`](phase4-retail-screening-v1.json) and
  [`phase4-wholesale-screening-v1.json`](phase4-wholesale-screening-v1.json)
  contain aggregate-only development CV metrics for all 13 frozen candidates
  and the deterministic two-per-family shortlists. Their SHA-256 digests are
  `62dcd2c1c41d30d49a4c98eab98e82529170aedd6f9313b46d00ffa50fdc4c9c`
  and `0b0bb79ce82138215e6e8920f7b4ba57086e0f75e60cfc2095f8ab93e6e240c7`.
- The matching `*.checkpoint.json` files contain the same aggregate candidate
  prefix evidence, bound to the frozen policy and audited assignment hashes.
  They make an interrupted local run resumable and contain no row-level values.
- [`phase4-retail-full-development-v1.json`](phase4-retail-full-development-v1.json)
  and
  [`phase4-wholesale-full-development-v1.json`](phase4-wholesale-full-development-v1.json)
  confirm Linear plus the four frozen challengers on the complete development
  partitions. Their SHA-256 digests are
  `07cf667e2e325f0bbb9b0fca1d62f4f3cdb54db4d607a03ff603142ee5fbc54f`
  and `e8b89ca4607f7afb6814560c9d053976c00cca6103590262b00a8e89c4461f3e`.
  Retail Random Forest 05 leads the accuracy comparison; wholesale retains
  Linear as its accuracy leader. Promotion remains pending measured deployment
  gates, and no calibration or legacy holdout rows were used.
- [`yoad22-controlled-batch-v1.json`](yoad22-controlled-batch-v1.json) is an
  isolated, aggregate-only dataset-composition experiment. It compares a
  Cars.com-development-only broad model with the same model trained on Cars.com
  development plus approved Yoad22 Craigslist rows, using shared pooled
  predictor-group folds. Its SHA-256 digest is
  `30d1f6011b7f2d5e611bbae6197be4780eeabcda3daca501c0b683807cf12ec5`.
  It does not modify Phase 4, use calibration or legacy holdout rows, persist a
  model, approve online learning, or promote the combined challenger.
- [`yoad22-source-composition-confirmation-v1.json`](yoad22-source-composition-confirmation-v1.json)
  and its [interpretation](yoad22-source-composition-confirmation-v1.md) compare
  Cars-only, balanced, moderate, and full Yoad source compositions on the exact
  controlled folds. Balanced and moderate use nested, target-free stratified
  samples; the immutable Cars-only and full endpoints are reused by checksum.
  Moderate augmentation is retained only as a separate experimental model. The
  JSON SHA-256 is
  `6ca3dd25cfb24bb0734497e4703cc516b3152e42f319286fcdd73374a6b2e5f5`.
- [`yoad22-training-weight-confirmation-v1.json`](yoad22-training-weight-confirmation-v1.json)
  and its [interpretation](yoad22-training-weight-confirmation-v1.md) compare
  three leakage-safe fold-local weighting formulas on the fixed moderate
  composition. All improve aggregate Cars accuracy and retain at least 97% of
  moderate's Yoad gain, but each fails at least one preregistered Cars slice
  guardrail. Weighting is rejected and the unweighted moderate branch remains
  unchanged. The JSON SHA-256 is
  `ceddd3dd530487ef57ee3d24390d5f0ef8e26db9c04f5d5b4f0ba56e84fb11a2`;
  its aggregate-only resumable checkpoint SHA-256 is
  `52fb1f1c57c358fdee9339e76b5cc84e4573c7ad32851b903cf30df452c8360e`.
- [`river-shadow-simulation-v1.json`](river-shadow-simulation-v1.json) validates
  the isolated River test-then-train lifecycle on 3,000 project-owned synthetic
  events across stable, gradual, abrupt, manufacturer-specific, and
  mileage-related target processes. It contains paired aggregate prequential
  metrics, telemetry-only ADWIN signals, a successful checkpoint/restart
  comparison, and a successful duplicate-outcome idempotency probe. It loads no
  real source, changes no frozen model, exposes no FastAPI route, and is not
  promotion evidence. Its SHA-256 is
  `7c5d8b6a3fd28bcd0f65644a5736e6785645ba1492d3c11015eb33c410528957`.
- [`retail-rf05-calibration-policy-v1.json`](retail-rf05-calibration-policy-v1.json)
  preregisters the first authorized use of the 10,958-row retail calibration
  population. The resulting [human report](retail-rf05-calibration-v1.md),
  [aggregate JSON](retail-rf05-calibration-v1.report.json), and
  [row-free serving artifact](retail-rf05-calibration-v1.artifact.json) bind to
  the unchanged Phase 4 RF05 identity. Vehicle-status calibration with global
  fallback achieved cross-fitted empirical coverage of 79.80%, 89.81%, and
  94.26% at the preregistered 80%, 90%, and 95% levels. The more granular
  status/value hierarchy failed its coverage/width gates and was rejected. The
  calibration is validated for integration with explicit marginal-coverage and
  weak-segment disclosures; it does not promote a model or open the legacy
  holdout. Artifact/report SHA-256 values are
  `b7eb5970b164ec68fb76cf8314f36080d085cda02968d3570d11f724490a6da0`
  and `e7fafff505603669e73cfbff2fe1cf5e04f9c5d896666470fe212411aa1b3084`.
- [`retail-rf05-development-residual-diagnostics-v1.json`](retail-rf05-development-residual-diagnostics-v1.json)
  establishes the training-side basis for a separately preregistered interval-
  sharpness study. Development OOF mean absolute residual rises from $5,648.98
  in the lowest predicted-value quartile to $18,500.38 in the highest, a 3.27x
  ratio. Its SHA-256 is
  `8f79ac027a72fff2512ab0b168d91a3a7b46677d72374dc00571a4646aac925d`.
- [`retail-rf05-uncertainty-sharpness-policy-v1.json`](retail-rf05-uncertainty-sharpness-policy-v1.json)
  freezes the baseline, normalized Gamma scale, simple smooth value scale, and
  every acceptance gate before calibration comparison. The resulting
  [human report](retail-rf05-uncertainty-sharpness-v1.md) and
  [aggregate JSON](retail-rf05-uncertainty-sharpness-v1.report.json) retain the
  current vehicle-status conformal baseline: Gamma failed 9 gates and the
  smooth scale failed 7. Neither candidate delivered the required 80%/90%
  sharpness reduction with acceptable conditional and bootstrap behavior, so
  no v2 serving artifact or residual-scale model was created. Policy/report
  SHA-256 values are
  `ec1787be963a907bbae2d1d521aeaef4239b8a5bf7816ced844dcd16902f1058`
  and `8614bad1ccd5345c64925c11e6172a7b4ef000ed6f16856aa45b48c3e4a741dd`.
- [`retail-rf05-final-evaluation-policy-v1.json`](retail-rf05-final-evaluation-policy-v1.json)
  freezes the RF05 identity, development-only fit boundary, retained calibration
  v1 system, confidence logic, metrics, slices, support thresholds, decision
  gates, 27,589-row final holdout identity, and upstream checksums before final
  target access. Its SHA-256 is
  `2be880be315f39a727bd8f1c6545b9410ea855bee63a3e72336f4da8cd7d5c33`.
- The resulting [human report](retail-rf05-final-holdout-v1.md),
  [aggregate JSON](retail-rf05-final-holdout-v1.report.json), and
  [model card](../model-cards/autovalue-retail-rf05-v1.md) classify the frozen
  system as **final evaluation passed with material limitations**. Final MAE is
  $10,575.36, RMSE is $34,118.14, and R2 is 0.4176. Frozen 80%/90%/95% interval
  coverage is 76.32%/89.10%/95.64%. The report SHA-256 is
  `017ab1824b1ddf4248959ecf8bb4a7d87991b513526a5567f29af0dd6e191e86`.
  The human-report and model-card SHA-256 values are
  `bc246570117734fb87787f8d05e3401d6d0be79ca7472d771334030c27cc8457`
  and `bfffefc801434d12eaecfc98be0e79387afb9a80eb511818feb52c19bfdef09a`.
- [`retail-rf05-final-evaluation-v1.manifest.json`](retail-rf05-final-evaluation-v1.manifest.json)
  binds all frozen inputs, final implementation files, and output checksums. Its
  SHA-256 is
  `40edba9a6846197c6d322b567e110c8bfa908518ee71c9da884263cb009e7c45`.
  No row-level holdout evidence or estimator was persisted, no post-holdout
  optimization occurred, and the holdout is permanently evaluation-only.
- [`retail-rf05-serving-reconstruction-policy-v1.json`](retail-rf05-serving-reconstruction-policy-v1.json)
  governs deployment-only reconstruction of the already-selected RF05 estimator.
  It pins the 98,552-row development boundary, model tuple and seed, feature
  contract, runtime and serialization policy, calibration binding, upstream
  hashes, exact OOF tolerances, and two independent full-fit checks. Its SHA-256
  is `becb895893f81cf04744786b722fc2a3c40be9e52257590989e1f9f2b44c831b`.
- The resulting [human report](retail-rf05-serving-reconstruction-v1.md) and
  [`aggregate JSON`](retail-rf05-serving-reconstruction-v1.json) prove exact
  development-boundary recreation and OOF reproduction without accessing the
  final holdout. The private Git-ignored model SHA-256 is
  `00ceb2680639a555a4705717e21ffe993a04e5731a3143e147d92d43b082e4fd`,
  its authenticated manifest SHA-256 is
  `dd31703302dce38d1a85907d3f818439e70c00f179155609be9bb93f41aaf3a2`,
  and the complete two-file bundle fingerprint is
  `ef343a2fa275c85c0c666dcbc38e4899eef773d0b6dd3c0c3a159bdc3dc616d4`.
  This packaging action did not tune, select, promote, recalibrate, or create a
  new performance claim.

The retail holdout has now received its sole frozen RF05 plus calibration v1
final-evaluation use. It cannot be reopened for tuning, calibration, candidate
selection, Yoad, or River. Any future unbiased claim requires a new development
cycle and newly authorized later-period evaluation data.
