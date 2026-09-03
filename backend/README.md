# Backend

This directory owns the FastAPI transport layer, runtime configuration, input
validation, inference orchestration, and prediction-history adapters.

The API includes liveness, public model readiness/metrics, strict request
validation, and a calibrated valuation route. The RF05 loader is fail-closed:
it constrains the bundle to an allowed local root, rejects links and unexpected
files, authenticates the pinned manifest, verifies the model bytes before
deserialization, validates the runtime, training-data, feature-contract,
pipeline, parameter, and calibration bindings, and returns a sanitized 503 on
any mismatch.

The backend may depend on the narrow inference surface of `autovalue_ml`. The ML
package must never import from the API package.

Run from the repository root:

```powershell
$env:PYTHONPATH = "backend/src;ml/src"
python -m uvicorn autovalue_api.main:app --reload
```

Both source roots are required because the API imports the narrow inference
surface from `autovalue_ml`. Production must set `AUTOVALUE_ENVIRONMENT` to
`production` and provide one or more exact HTTPS `AUTOVALUE_CORS_ORIGINS`.
Startup rejects localhost, wildcard, non-HTTPS, and path-bearing production
origins.

Routes:

- `GET /health/live`
- `GET /api/v1/model`
- `POST /api/v1/valuations`
- `GET /api/v1/predictions/recent`

Only model year, make, exact model, vehicle status, optional mileage, and the
80/90/95% interval choice are accepted. Extra features fail validation. The
public response contains no local path, raw record, calibration row, or holdout
row.

Successful valuations carrying `X-AutoValue-Client: <uuid>` are saved to local
SQLite. Only a SHA-256 of that anonymous UUID is stored. Retention is capped at
25 rows per browser and the API returns five; SQL is parameterized and the
database is ignored by Git.

`confidence_label` remains in the response for compatibility with the frozen
calibration contract. It describes relative interval width and calibration
bucket support; final evidence showed that it does not rank realized error, so
it is not a probability-of-correctness claim and the primary UI does not show it.

The current local private bundle is authenticated as `retail-rf05-v1`, so the
API is valuation-ready in this workspace. The binary remains Git-ignored. A
public clone without it starts safely and reports `artifact_required` until an
authorized operator provisions the exact authenticated bundle.
