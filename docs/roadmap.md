# Development roadmap

Each phase ends in evidence that can be reviewed before the next phase expands
the product.

## Phase 0 — Architecture and scope

**Status:** complete.

Accepted when component boundaries, ML workflow, dataset strategy, deployment
shape, persistence tradeoff, U.S.-only/USD product contract, and portfolio
narrative are agreed.

## Phase 1 — Repository foundation

**Status:** complete.

Deliver repository hygiene and documentation, a health-only FastAPI application,
a responsive React/Vite shell, local environment examples, tests, and CI.

Acceptance criteria:

- backend lint, type checks, and tests pass;
- frontend lint and production build pass;
- browser can display API liveness without prediction behavior;
- no dataset, model, database, or fake metric is committed; and
- documentation distinguishes implemented from planned behavior.

## Phase 2 — Dataset acquisition and audit

**Status:** complete for the two pinned baseline inputs. The compliant acquisition framework, licensed-local-file
gate, common schema/normalization layer, lineage and ingestion metrics, explicit
ML-reuse gate, and controlled synthetic demo are complete. Two exact Kaggle
artifacts are conditionally approved for separate historical U.S. asking-price
and wholesale completed-sale tracks. Both normalized artifact sets now pass
lineage, schema, privacy, and checksum verification. Both final holdout artifact
sets also pass their source-specific lineage, checksum, accounting, and isolation
gates. Phase 3 baseline training now consumes only these verified streams.

Verify license/provenance, acquire the exact version, produce a checksummed local
manifest, profile the schema and quality, document market limitations, and decide
the supported v1 feature contract. Every candidate must declare
`market_country="US"`, use USD targets, and pass acquisition and ML reuse as
separate decisions.

Acceptance criteria: every data gate in `docs/data-strategy.md` has evidence, and
the candidate is explicitly accepted or rejected.

Implemented acquisition evidence includes externally pinned strict manifests and
artifact checksum checks, separate acquisition/training permissions and
fingerprints, hard U.S./USD validation, reusable reviewed adapters, bounded
retry/backoff/pagination, an in-memory cache, normalization, deduplication,
quarantine, final `.ready.json` publication plus artifact-set verification, and
River-shaped append-only training events. External crawling stays disabled until
address-pinned transport is implemented and reviewed.

The wholesale CSV must match its reviewed 88,047,552-byte size, SHA-256, and
558,837-row count. Processing must retain only the 50 U.S. states plus
Washington, D.C.; remove VIN and seller; forbid MMR; document USD as an inferred
assumption; and produce a chronological, VIN-aware split manifest. The project
owner's direct-uploader permission attestation supports scoped portfolio use,
while row redistribution, downloadable models, sublicensing, and commercial use
remain pending. The asking-price CSV must independently match its reviewed
17,171,976-byte UTF-16 artifact and 144,867-row count, preserve New, Used, and
Certified status, remove Dealer, deduplicate, and preserve asking-price
semantics.

The verified retail holdout contains 109,510 train and 27,589 test rows, with
all year/make/model/mileage/status groups isolated. It is explicitly
non-temporal. The verified wholesale holdout contains 442,130 train and 98,634
test rows at a 2015-06-01 boundary, with zero VIN overlap and five ordered,
VIN-isolated training CV buckets. Phase 3 may consume only these split-aware
gates and must keep their labels separate.

MUCars-2024 remains rejected for geographic mismatch. City of Seattle Sold Fleet
Equipment remains a public-domain ingestion smoke candidate. A GSA corpus still
requires an official bulk release or direct request, and UCI Automobile remains
inadequate for production.

## Phase 3 — Reproducible ML baseline

**Status:** complete. Both real-data experiments fit the median dummy and Linear
Regression through fold-local pipelines, select only by CV MAE, and report the
selected Linear Regression once on each untouched holdout. The canonical reports
reproduce byte-for-byte from the documented commands.

Implement cleaning and candidate-supported features as tested code, freeze each
track's reviewed held-out split, train dummy and Linear Regression baselines, and
produce separate repeatable aggregate evaluation reports. Asking-price and
wholesale completed-sale labels remain explicit throughout.

Acceptance criteria: leakage-safe pipeline tests pass and baseline MAE, RMSE, and
R² reproduce from a documented command; `mmr`, `vin`, and `seller` cannot enter
the feature matrix.

Evidence: the [retail asking-price report](results/retail-baseline-v1.json)
records grouped-CV MAE of $11,552.82 and untouched-holdout MAE of $12,040.29.
The [wholesale completed-sale report](results/wholesale-baseline-v1.json) records
forward-CV MAE of $2,382.13 and untouched-holdout MAE of $2,256.02. Both beat
their median-dummy CV baselines. The [results guide](results/README.md) contains
the full MAE, RMSE, R², retail status slices, reproducibility commands, and
limitations. No target clipping, outlier removal, or log transform is part of
these baselines.

## Phase 4 — Model selection and artifact

**Status:** ML selection and final evaluation complete; serving-artifact and
deployment integration remain deferred to Phase 5. The development/calibration boundary, explicit
candidate set, deterministic seeds, selection thresholds, resource budgets,
prediction-range method, legacy-holdout label, and artifact controls are frozen
in the [Phase 4 protocol](experiments/phase4-model-selection-v1.json) before any
nonlinear tuning. The protocol loader, sparse tree candidate factories,
calibration and screening partitions, conformal-range primitives, and promotion
gate are implemented and tested. The
[real-data partition audit](experiments/phase4-partition-audit-v1.json) confirms
the exact row accounting and target-free assignment hashes. All 13 candidates
per track have been screened with resumable aggregate-only evidence. Linear led
both screening fields; the required two Random Forest and two Gradient Boosting
challengers per track are now shortlisted. Full-development confirmation is
complete: retail Random Forest 05 leads and clears every accuracy guardrail,
while wholesale Linear remains the metric leader. Deployment measurements and
promotion gates are next. The first retail RF05 calibration use is complete:
vehicle-status split conformal intervals with global fallback are validated at
80%, 90%, and 95%, while the more granular conditional hierarchy was rejected
by its preregistered gates. A subsequent frozen sharpness experiment confirmed
strong development-side heteroscedasticity, then rejected both a normalized
Gamma residual scale and a simple smooth predicted-value scale because neither
passed all coverage-first sharpness, bootstrap, stability, and conditional-
coverage gates. The existing vehicle-status calibration remains the frozen
uncertainty reference; no v2 artifact or scale estimator was persisted. A
checksum-bound policy then governed the frozen system's sole 27,589-row final
holdout evaluation. It achieved $10,575.36 MAE, $34,118.14 RMSE, and 0.4176 R2
and is classified `final evaluation passed with material limitations`. The
holdout is permanently evaluation-only, no post-holdout tuning occurred, and
ML experimentation is stopped.

Compare Random Forest, Gradient Boosting, and optionally XGBoost; tune on training
folds; perform slice/error analysis; calibrate intervals; select against accuracy
and deployment budgets; and persist a checksummed bundle plus model card.

Acceptance criteria: development-only selection reproduces; calibration remains
fully isolated from preprocessing, tuning, and point-model fitting; the immutable
final report, model card, and evidence manifest reproduce their checksums; and
future serving-artifact load/predict/schema/resource checks pass before API
integration. Any new unbiased performance claim requires a new development
cycle and newly authorized later-period evaluation data.

## Phase 5 — Prediction API and private browser history

**Status:** complete for authenticated local inference. The versioned
point-plus-interval schema, strict request validation, model metadata/readiness,
checksum-bound loader, exact RF05 verification, calibration binding, prediction
route, sanitized failures, and API tests are implemented. Successful estimates
are stored in bounded local SQLite and isolated by a hashed anonymous browser
UUID; history contains no VIN, dealer, account, listing, source, or free text.
The frozen RF05 estimator was reconstructed from exactly 98,552 development rows
under a checksum-pinned deployment-only policy. Its OOF evidence and two-fit
determinism were verified before the authenticated private bundle was written.
The binary remains Git-ignored and a clone without it fails closed.

Acceptance evidence: the private bundle drives golden API predictions without
training at application startup, while missing or mismatched artifacts return a
sanitized unavailable state.

## Phase 6 — Product dashboard

**Status:** complete for the authenticated local bundle. The responsive interface
includes the exact feature-contract form,
80/90/95% selector, 90% default, truthful artifact state, model-backed result
contract, calibrated range, data-quality warnings, browser-isolated history,
limitations, keyboard/focus behavior, and loading/error states. It contains no
fabricated prediction.

The separate ML Engineering view shows final metrics, uncertainty coverage,
architecture paths, governed source states, the complete decision table, the
active feature contract, and replay controls for five verified aggregate River
scenarios. It explicitly avoids an importance chart because no defensible
importance artifact was persisted.

Acceptance criteria: end-to-end user flows pass with no unsupported or fake input
options, and production screenshots are captured.

## Phase 7 — Deployment and portfolio finish

**Status:** repository/productization audit complete; public deployment pending.

Provision the already-authenticated trusted estimator through a private
deployment channel, set production CORS and HTTPS, run hosted resource/smoke
tests, and capture final desktop/tablet/mobile screenshots.
The README, security/privacy audit, limitations, results, and experiment story
are complete.

Acceptance criteria: the public app survives a clean deployment, reports the
expected model version, preserves browser-scoped history, and documents all free
tier limitations honestly.
