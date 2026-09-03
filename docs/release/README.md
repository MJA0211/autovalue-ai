# Release candidate

AutoValue AI can be demonstrated locally and its public source can be reviewed.
The ML lifecycle is closed. Release work must not retrain, tune, recalibrate,
open the final holdout, or change any frozen experiment decision.

## Verified product flow

The authenticated local flow passed this sequence:

```text
React form
  → FastAPI validation
  → canonical feature pipeline
  → authenticated retail-rf05-v1
  → point valuation
  → authenticated calibration v1
  → calibrated USD interval
  → browser result and bounded recent history
```

The live-server audit exercised six project-owned example requests. Every point
was finite, every interval contained its point, RF05 and calibration versions
matched, warnings were returned where applicable, recent history returned its
five-row browser-specific cap, invalid input returned 422, and the configured
development CORS origin was honored. Automated tests separately prove missing,
corrupt, unexpected, and mismatched bundles remain fail-closed.

## UI release findings

The release pass kept the existing product hierarchy and added five synthetic
presets. It also added keyboard focus for success and error results,
busy/live-region semantics, active-navigation semantics, accessible River
scenario controls, and explicit synthetic/shadow labels. No confidence label
appears in the primary UI. The primary wording remains "90% calibrated valuation
range," with separate data-quality warnings.

Headless Chromium visual checks passed at 1440 × 1200 and a true emulated
390 × 844 viewport. The mobile document reported identical inner and scroll
widths, confirming no horizontal overflow. See the [screenshot plan](screenshot-plan.md).

## Publication boundary

Public source may contain code, tests, aggregate reports, documentation, the
model card, project-owned synthetic fixtures, reconstruction code, and reviewed
screenshots. Private local/deployment state includes every downloaded dataset,
row-level derivative, SQLite file, environment file, log, cache, and the complete
`models/retail-rf05-v1/` directory.

The model and its manifest are intentionally ignored together. Public binary
distribution remains unapproved; private provisioning preserves authenticated
serving without expanding dataset or model rights.

## Release evidence

- [Publication and security audit](../publication-readiness.md)
- [First-commit review](first-commit-review-v1.md)
- [Exact publication manifest](first-commit-manifest-v1.json)
- [Release smoke-test checklist](release-smoke-checklist.md)
- [Screenshot plan](screenshot-plan.md)
- [Deployment plan](../deployment.md)
- [Security policy](../../SECURITY.md)

The local Git commit was created only after these artifacts and every quality
gate were rechecked. No GitHub push or public deployment is part of this release
phase.
