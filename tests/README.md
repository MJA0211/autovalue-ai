# Testing strategy

Tests will be organized by responsibility:

- `tests/backend`: API, validation, configuration, and persistence behavior;
- `tests/ml`: deterministic cleaning, features, splitting, and evaluation;
- `tests/integration`: saved-artifact and API contract tests; and
- `frontend/src/**/*.test.*`: component tests beside the components they cover.

Only tiny, clearly synthetic fixtures belong in Git. Full datasets and trained
artifacts must not be copied into the test suite. Model checks will cover schema
compatibility, unknown categories, serialization round trips, deterministic
splits, finite predictions, and metric calculations.

Acquisition tests use only invented inline pages and `httpx.MockTransport`.
Their autouse socket guard makes accidental live network access fail immediately.
The suite verifies disabled and expired permissions, independent acquisition and
ML-reuse decisions, strict licensed-dataset manifests/checksums, common-schema
parity, normalization, exact/conflicting duplicate handling, quarantine, URL and
robots boundaries, authentication and cookie rejection, bounded retry/backoff and
crawl budgets, HTTP 429 handling, memory-cache integrity/capacity, parser drift,
pagination loops, nested provenance, output checksums, strict scalar
normalization, externally pinned manifests, and River-shaped event creation
without a River dependency or model update. It also verifies the hard
`market_country="US"`/`currency="USD"` contract and requires the final
`.ready.json` marker plus `verify_scrape_artifact_set(...)` checksum validation
before a scrape artifact set is considered complete.

Run the acquisition/ML checks from the repository root:

```powershell
python -m pytest tests/ml
```
