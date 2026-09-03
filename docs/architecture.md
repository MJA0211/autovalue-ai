# Architecture

## Goals and constraints

AutoValue AI runs its own machine-learning workflow instead of delegating
valuation to a paid API. It runs locally with free, open-source software and
supports a zero-dollar public portfolio deployment.

The valuation scope is intentionally narrow: United States vehicles only, with
`market_country="US"` retained in lineage and all targets and predictions in
`currency="USD"`. Two conditionally approved, checksum-pinned local artifacts
support separate modeling tracks: a historical retail asking-price model and a
historical wholesale completed-sale benchmark. Phase 3 has trained and evaluated
transient baselines through each source's independent processing and split
gates; no estimator has been persisted or promoted for inference.
The separate-label decision is recorded in
[ADR 0004](decisions/0004-separate-price-targets.md).

The architecture separates training from serving so experiments remain flexible
while production inference stays small, deterministic, and auditable.

## System context

```text
                    Offline boundary

Licensed manifest ----> local artifact verification ---+
                                                       |
Reviewed adapter -----> bounded collection ------------+--> common schema
                                                               |
                                             normalization / dedup / quarantine
                                                               |
                                           manifest + final ready marker
                                                               |
                                                   explicit ML-reuse gate
                                                   /                  \
                                       batch training       River-shaped events
                                                   \                  /
                                                    trusted model bundle

                    Online boundary

Browser -> React dashboard -> FastAPI -> fitted pipeline -> estimate + interval
                              |    |
                              |    -> model metadata / global importance
                              v
                     prediction-history repository
                    (SQLite local, libSQL deployment)
```

The browser never receives database credentials or the model artifact. FastAPI
is the sole owner of validation, inference, explanation, and history writes.

## Components

| Component | Responsibility | Must not do |
|---|---|---|
| `frontend` | Collect inputs and present estimates, uncertainty, model facts, importance, and browser-scoped history | Reimplement model logic or contain secrets |
| `backend` | Own HTTP contracts, validation, inference orchestration, safe errors, and history access | Train models during requests |
| `ml` | Own deterministic cleaning, features, splitting, training, evaluation, interval calibration, and artifact contracts | Import the API package |
| `data` | Document provenance and hold ignored local raw/intermediate/processed files | Treat public availability as a license |
| `models` | Document immutable, checksummed model releases | Accept or load untrusted artifacts |
| `tests` | Verify component behavior and cross-boundary contracts | Depend on the full real dataset in routine CI |

Dependency direction is one-way: the API may import a small inference surface
from `autovalue_ml`; the ML package is independent of FastAPI. OpenAPI is the
browser/API contract, so a separate shared-code package is not currently needed.

## Acquisition boundary

Acquisition runs offline and is never part of an API request. Approval to collect
and store data is separate from approval for ML training and public portfolio
use, so a successful acquisition does not automatically become a training
dependency. For a licensed local file, a project-owned review record pins the
strict manifest's SHA-256 before that manifest can authorize source/version/license,
approval evidence, format, and artifact checksum for bounded CSV/JSONL parsing.
For a scraping adapter, a versioned policy pins exact origin, paths, query keys,
permitted fields, terms evidence, retention, and crawl budgets. `robots.txt` may
further restrict access but never grants permission.

Both paths fail closed unless the source declares `market_country="US"`, and
price-bearing records must use `currency="USD"`. Scrape output is published as
normalized JSONL, quarantine JSONL, a lineage/metrics manifest, and a
`.ready.json` marker written last. `verify_scrape_artifact_set(...)` must validate
the final marker, artifact-set ID, safe filenames, and every checksum before a
consumer reads the set. Readiness proves a complete artifact publication; it
does not grant ML reuse.

The runtime uses one sequential client with hard request, page, record, byte,
retry, delay, and time budgets; exponential backoff and bounded `Retry-After`
handling; deterministic pagination; and an optional integrity-checked memory
cache. Redirects, authentication, cookies, unreviewed query parameters,
pagination loops, parser drift, and conflicting listing IDs fail closed. A bad
individual card is quarantined with lineage while valid cards continue. Parsing
remains pure and source-specific; only common-schema, provenance-tagged records
cross into normalization and deduplication.

External crawling is disabled pending an address-pinned transport that closes
the DNS validation/request gap. The enabled demonstration is restricted to the
repository-owned, numeric-loopback synthetic dealership. Its three pages contain
a duplicate, missing optional and required fields, a malformed monthly-payment
value, and deterministic HTTP 503/429 failures. See
[the acquisition design](data-acquisition.md).

The future online-learning boundary is append-only `TrainingRecordEvent` data,
not a direct scraper-to-model call. A separate gate rechecks ML permission,
validates every nested record's acquisition lineage, records independent
acquisition and ML-grant fingerprints, and filters target price kind and currency
before producing the feature mapping and target accepted by River's
`learn_one(x, y)`. River is not yet a dependency and the acquisition subsystem
never updates a model.

Each future event retains `market_country="US"` and USD target metadata so the
append-only boundary remains compatible with a later River consumer without
allowing a non-U.S. record to enter online learning.

## Offline training flow

1. Acquire an explicitly licensed, versioned U.S. dataset and record its
   checksum, independent manifest approval, and USD target semantics.
2. Validate schema, units, currency, target meaning, missingness, duplicates,
   ranges, and potential leakage.
3. Freeze a held-out test set before model selection.
4. Fit preprocessing only on training folds through scikit-learn pipelines.
5. Compare baselines and tuned candidate regressors with repeated,
   reproducible cross-validation.
6. Select using MAE as the primary product metric, with RMSE, R², stability,
   latency, memory, and artifact size as additional evidence.
7. Calibrate and evaluate a prediction interval using data not used to fit the
   final estimator.
8. Persist the complete fitted pipeline and a separate metadata manifest.

No notebook is the source of truth. Notebooks may explore data, while reusable
code performs every accepted transformation and evaluation.

## Model bundle contract

A promoted model release will contain:

- the fitted preprocessing-and-regression pipeline;
- input feature names, types, units, allowed categories, and validation ranges;
- model and schema versions;
- training-data fingerprint and split seed;
- dependency versions;
- held-out MAE, RMSE, R², interval coverage, and interval width;
- global feature importance generated offline; and
- a checksum used to reject corrupted or unexpected artifacts.

The service will load and validate the bundle during application startup. It will
not download an artifact on the first prediction request.

## API resources

Planned public resources are:

- `GET /health/live`: process liveness only;
- `GET /health/ready`: model/schema/database readiness;
- `GET /model/info`: safe public model metadata and held-out metrics;
- `POST /api/v1/predictions`: validate, predict, explain, and save; and
- `GET /api/v1/predictions/recent`: browser-scoped recent history.

Only liveness exists in Phase 1. Routes will be added with their implementation
and contract tests.

## Prediction interval and explanation strategy

The point estimate and interval have different meanings. A practical first
interval is now defined by the frozen vehicle-status conditional split-conformal
artifact, with global fallback for unsupported statuses and an explicit $0
physical lower bound. Its cross-fitted calibration coverage and width are
reported at 80%, 90%, and 95%. The UI will call it an estimated range, not a
guarantee or a per-vehicle probability.

A separate preregistered sharpness experiment confirmed that RF05 residuals are
heteroscedastic, but rejected both tested normalized-conformal candidates. Their
80% width results were flat or worse and their roughly 3.5% reductions at 90%
did not satisfy the frozen sharpness and bootstrap gates; both also produced
excessive conditional-coverage regressions.
The architecture therefore retains the simpler status-conditional v1 artifact;
no Gamma residual-scale model or v2 serving artifact exists. Any
eventual legacy-holdout evaluation must use this frozen choice, and a new
unbiased coverage claim still requires newly authorized later-period data.

Global importance will be computed during evaluation. Per-prediction explanations
will use a method appropriate to the selected model and deployment memory budget.
If lightweight perturbation is used instead of additive SHAP values, the UI will
label it as sensitivity rather than feature attribution.

## Persistence and privacy

Local development uses the Python SQLite driver and a local database file. The
public deployment uses the same repository interface backed by Turso/libSQL,
because Render's free filesystem cannot retain a SQLite file across restarts or
idle spin-downs.

History will contain normalized input features, result values, model version,
explanation data, and timestamp. An anonymous browser-generated identifier will
scope reads. VINs, names, email addresses, IP addresses, and arbitrary free text
are out of scope.

## Operational constraints

The public inference service targets one worker and a 512 MB memory envelope.
Model selection therefore considers artifact size, startup peak memory, warm
latency, and accuracy. Training libraries and raw data will not be shipped in the
serving image unless required for artifact loading.

Configuration comes from environment variables. Production CORS uses an exact
frontend origin. Request validation, bounded payloads, rate limiting, safe logs,
and generic error responses will be verified before public release.
