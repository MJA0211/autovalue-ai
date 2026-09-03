# Frontend

The frontend is a React/Vite portfolio product with progressive disclosure. The
main experience contains the exact RF05 vehicle form, calibrated value/range
result, data-quality notes, browser-isolated recent estimates, loading/error states,
and restrained limitations. The separate ML Engineering view shows frozen
metrics, coverage, architecture, governance, experiment decisions, and a replay
of the five aggregate synthetic River scenarios.

From this directory:

```powershell
Copy-Item .env.example .env
npm install
npm run dev
```

The local API now serves the authenticated, privately reconstructed RF05 bundle.
The interface also remains safe in a public clone or deployment where that
private bundle is absent: it disables estimation when the API reports
`artifact_required` and never renders a made-up fallback. localStorage holds
only a random browser UUID. The API hashes it and uses it to isolate bounded
SQLite history; no VIN, dealer, account, listing, source, or location is stored.

`VITE_API_BASE_URL` is public configuration, not a secret. Database credentials
and other secrets must never use the `VITE_` prefix or enter the browser bundle.
Development defaults to `http://localhost:8000`. A production build without an
explicit value uses same-origin relative API paths instead of embedding a
localhost fallback; separate frontend/backend hosts must set the backend's exact
HTTPS origin at build time.

Five form presets cover a common sedan, SUV, truck, older high-mileage vehicle,
and luxury vehicle. They contain example inputs only and always invoke the real
API—no prediction is hardcoded into the interface.
