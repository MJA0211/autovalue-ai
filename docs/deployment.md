# Zero-cost deployment design

**Current state:** local valuation is ready with the authenticated, Git-ignored
RF05 bundle. Public deployment requires provisioning that same bundle privately;
public binary redistribution remains unapproved.

## Approved topology

```text
GitHub monorepo
├── frontend -> any free static-site host
└── backend  -> any free Python web-service tier -> private RF05 bundle
```

The React portfolio and engineering demo require no private data and can deploy
as static files. FastAPI also starts without data or a model, but deliberately
reports `artifact_required` until the pinned manifest, model, runtime,
feature-contract, training-data, and calibration bindings pass. No paid API or
managed database is required.

## Prediction history

Successful estimates are stored in local SQLite and isolated by an anonymous
browser UUID. The browser keeps only that identifier; the API hashes it before
persistence and returns at most five matching rows. SQLite retains at most 25
rows per browser and stores no VIN, dealer, listing, account, source, or free
text. Free-host filesystems are commonly ephemeral, so online durability must be
checked against the selected host rather than assumed.

## Backend deployment budget

- one Uvicorn worker;
- inference only—never train on the host;
- model artifact below the loader's 300 MB hard limit;
- warm memory and startup peak measured on the selected free tier;
- target warm p95 prediction below 500 ms;
- native numerical threads capped at one; and
- no raw data, notebooks, or visualization packages in the runtime image.

These are project acceptance budgets, not provider guarantees. The selected
model must justify accuracy gains against footprint and latency.

## Runtime configuration

Backend variables include the exact frontend origin, trusted model and
calibration roots, bundle and calibration paths, SQLite path, and runtime
environment. Secrets live only on the backend host. `VITE_API_BASE_URL` is
public frontend configuration.

Production startup requires at least one exact HTTPS frontend origin and rejects
localhost, wildcard, non-HTTPS, and path-bearing CORS values. The production
frontend must set `VITE_API_BASE_URL` to the backend's HTTPS origin when the two
are hosted separately; if it is omitted, the built client uses same-origin
relative paths rather than a localhost fallback.

The backend process needs both source roots on its import path:

```powershell
$env:PYTHONPATH = "backend/src;ml/src"
python -m uvicorn autovalue_api.main:app --host 0.0.0.0 --port 8000
```

The host should inject the safe variables documented in
[`backend/.env.example`](../backend/.env.example). Debug mode and reload must
remain off in production.

## Cold-start experience

The static frontend loads independently and preserves form state during a
request. It distinguishes an unreachable API from a reachable API whose model
artifact is unavailable. Artificial keep-alive traffic must not be used to evade
free-tier behavior.

## Release checks

Before deployment, CI verifies linting, strict typing, unit/contract tests,
coverage, and the frontend production build. After the exact estimator is
privately provisioned, deployment smoke tests must authenticate its recorded
hashes and call `/health/live`, `/api/v1/model`, and the golden
`/api/v1/valuations` fixtures before the service is presented as available.

## Private RF05 provisioning

The binary did not exist after model evaluation because that phase intentionally
persisted aggregate evidence only. It was later created through deterministic
reconstruction and packaging of the frozen reference estimator—not post-holdout
tuning. The build used only the 98,552 development rows, reproduced the exact
five-fold development evidence, and bound the model to unchanged calibration
v1. See the aggregate reconstruction report for the artifact hashes and runtime.

The recommended publication strategy is **deployment-private**. Do not commit
the binary, attach it to a GitHub Release, or add it to LFS while downloadable
trained-model rights remain pending. An authorized operator can instead run the
pinned local reconstruction command documented in `models/README.md` or copy
the already-authenticated two-file bundle through a private deployment channel.

Provider subdomains and managed TLS can preserve a strict $0 cost. Free-tier
terms and resource limits change, so verify them immediately before deployment.
A custom domain is optional and may have a registration fee.

## SQLite deployment decision

SQLite remains acceptable for a single-instance portfolio demo whose recent
history may be ephemeral. Mount `AUTOVALUE_PREDICTION_HISTORY_PATH` on a private
persistent volume if the host offers one. Do not share the same file across
multiple workers or instances. If durable multi-instance history becomes a
product requirement, disable the history feature or perform a separately
reviewed migration; it is not a reason to expose the RF05 bundle or add a paid
service now.

The deterministic pre-release and post-deployment checks are recorded in the
[release smoke-test checklist](release/release-smoke-checklist.md).
