# Release smoke-test checklist

Run this checklist from the repository root. Use only project-owned synthetic
vehicles; never use a source, calibration, or holdout row.

## Before startup

- [ ] `models/retail-rf05-v1/` contains only `manifest.json` and `model.joblib`.
- [ ] Both files are ignored by Git.
- [ ] The calibration artifact and RF05 hashes match the frozen manifest.
- [ ] `backend/.env` contains the intended environment, trusted roots, bundle,
  calibration, history, and exact CORS settings.
- [ ] `frontend/.env` contains only the public API base URL.

## Start locally

```powershell
$env:PYTHONPATH = "backend/src;ml/src"
python -m uvicorn autovalue_api.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
Set-Location frontend
npm ci
npm run dev
```

## Product flow

- [ ] `GET /health/live` returns 200 without disclosing paths or secrets.
- [ ] `GET /api/v1/model` reports `ready`, `can_predict: true`, and
  `retail-rf05-v1`.
- [ ] Each synthetic preset produces a finite USD point valuation.
- [ ] The chosen calibrated interval is returned and contains the point where
  expected.
- [ ] The response identifies `retail-rf05-split-conformal-v1`.
- [ ] Missing mileage produces a data-quality warning.
- [ ] A malformed, extra, future-year, negative-mileage, or unsupported request
  returns a sanitized 422.
- [ ] Recent history returns only the current browser's records and remains
  capped.
- [ ] Temporarily absent, corrupt, or mismatched model components return the
  fail-closed unavailable state; never modify the authentic bundle to test this
  outside an isolated temporary copy.

## Frontend and engineering view

- [ ] Landing, preset, loading, success, warning, error, and unavailable states
  are readable with keyboard-only navigation.
- [ ] Focus moves to the result or error after submission.
- [ ] The primary output says “calibrated valuation range,” not confidence.
- [ ] The engineering view displays final metrics and the three isolated paths.
- [ ] River is labeled synthetic, shadow-only, telemetry-focused, and unable to
  influence user-facing valuations.
- [ ] Desktop, tablet, and mobile layouts have no horizontal overflow.

## Production artifact

- [ ] `npm run build` succeeds and generated JavaScript contains no localhost API
  fallback, source map, credential, or private path.
- [ ] The backend starts without reload/debug behavior.
- [ ] Production CORS accepts only the exact HTTPS frontend origin.
- [ ] The private model is not available through any static or API file route.
- [ ] Raw datasets, SQLite history, environment files, and logs are inaccessible.
- [ ] Repeat health, model readiness, valuation, interval, history, invalid-input,
  restart, and mobile checks against the hosted URLs.
