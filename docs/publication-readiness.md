# Publication readiness

## Status

The code, aggregate evidence, synthetic fixtures, and portfolio UI are ready for
repository review. Local valuation runs with an authenticated, Git-ignored
RF05 bundle. A public deployment may provision the bundle privately under the
existing hosted-inference approval. Public download or redistribution remains
blocked pending explicit trained-model permission.

## Repository audit

The repository had no earlier history when the first-commit candidate was
reviewed. A scan of that tree found no username, absolute local workspace path,
API token, hard-coded password, bearer credential, or private key in the
publishable source tree.

Ignore-rule checks confirm that the following stay local:

- `.env` files while retaining `.env.example` templates;
- raw, interim, and processed datasets;
- CSV, Parquet, Arrow, Feather, JSONL, and NDJSON row artifacts;
- SQLite databases and journals;
- model binaries and private model manifests;
- caches, virtual environments, frontend build output, and notebooks caches;
- online-learning checkpoints, prediction/outcome logs, and generic log files.

The locally present third-party CSV and Parquet files are ignored. They must not
be force-added. Only project-owned synthetic fixture pages and aggregate JSON
evidence are intended for publication.

## Security and privacy findings

- Public requests do not accept VIN, dealer, seller, listing ID, free-form notes,
  source identity, target price, or filesystem paths.
- Pydantic rejects extra fields, invalid/future model years, negative or extreme
  mileage, unsupported interval levels, blank strings, and control characters.
- The API catches serving failures and returns a fixed public error without the
  underlying exception or local path.
- Joblib is loaded only from the exact bundle beneath a configured trusted root.
  The loader rejects links and extra files, authenticates a pinned manifest,
  checks its runtime/data/specification/calibration bindings, verifies model
  bytes, and only then deserializes those verified bytes. Operators must still
  treat the deployment-private joblib as trusted code and protect its location.
- Local SQLite stores at most 25 successful estimates per anonymous browser and
  the UI returns five. The browser UUID is hashed before storage; entries contain
  only serving-contract features and valuation output, never VIN, dealer,
  listing, account, source, or free-text data.
- River has no public prediction or outcome route and cannot affect RF05.

## Independent data acquisition

Do not publish downloaded datasets. Another developer should review
[`DATA_SOURCES.md`](../DATA_SOURCES.md), the source-specific records in
[`docs/data-reviews`](data-reviews/README.md), and the attribution requirements
before independently obtaining any artifact from its official host. Checksums
in the review records identify the exact artifacts used. Scraping and ML reuse
remain separate permissions.

## Before public deployment

1. Provision the exact authenticated RF05 bundle through a private deployment
   channel; do not publish it as a repository, LFS, or Release asset.
2. Configure an HTTPS host and exact production CORS origin; do not use a wildcard.
3. Run model-readiness, representative valuation, invalid-input, and restart smoke
   tests in the target environment.
4. Confirm the deployment platform never packages local `data/`, `.env`, logs,
   caches, or unapproved model files.
5. Repeat the documented screenshot plan against the hosted URL and replace
   local release-candidate captures only after desktop, tablet, and mobile
   review passes.

A static frontend host and a free compute tier can run the FastAPI application
without a paid service. Free-tier availability and limits change, so provider
terms must be checked again before deployment.

The current release-candidate scan, exact first-commit inventory, screenshots,
and smoke procedure are indexed in [`docs/release`](release/README.md).
