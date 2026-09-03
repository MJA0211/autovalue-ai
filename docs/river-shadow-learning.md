# River shadow-learning architecture

## Decision

**Classification: architecture validated for shadow simulation.** The River
subsystem is experimental and shadow-only. It is not registered in the public
FastAPI router, cannot alter a user-facing estimate, and is not eligible for
production promotion or real-outcome collection under current permissions.

The machine-readable evidence is
[`river-shadow-simulation-v1.json`](experiments/river-shadow-simulation-v1.json).
Its synthetic metrics validate sequencing and state management, not real-world
vehicle-price accuracy.

The canonical report contains 600 resolved outcomes per scenario (3,000 total),
zero unexpected quarantines, and the following USD error metrics:

| Scenario | River MAE | River RMSE | Static MAE | Static RMSE | Final rolling MAE | ADWIN detections |
|---|---:|---:|---:|---:|---:|---:|
| Stable market | $2,059.35 | $3,277.17 | $1,412.52 | $1,750.13 | $1,795.44 | 0 |
| Gradual price drift | $2,088.86 | $3,304.09 | $3,760.97 | $4,443.34 | $1,880.19 | 0 |
| Abrupt price shift | $2,143.31 | $3,373.00 | $4,696.01 | $5,936.62 | $1,867.99 | 0 |
| Manufacturer-specific drift | $2,286.71 | $3,530.57 | $2,277.51 | $3,620.76 | $1,899.42 | 0 |
| Mileage-related drift | $2,709.41 | $4,008.09 | $2,709.29 | $3,850.23 | $3,063.53 | 1 |

The static reference is intentionally strong in the stable generator, while the
incremental model adapts to the broad gradual and abrupt shifts. It does not
outperform the reference on every localized scenario. ADWIN's single signal is
telemetry, not a requirement that every constructed shift be detected. These
patterns are simulator behavior and carry no production-accuracy meaning.

Checkpoint/restart verification matched the uninterrupted aggregate model,
metric, drift, and counter state after 120 events. The duplicate-delivery probe
accepted the first outcome, quarantined the second as `duplicate_outcome`, and
recorded exactly one update. The report file SHA-256 is
`7c5d8b6a3fd28bcd0f65644a5736e6785645ba1492d3c11015eb33c410528957`.

## Isolation boundary

```text
stable offline model ----------------------> user-facing/reference prediction

approved synthetic event -> River predict_one -> stored shadow prediction
                                                |
approved delayed outcome -> validate -> score --+-> ADWIN telemetry
                                                `-> River learn_one -> checkpoint
```

The current FastAPI router still exposes only `/health/live`. A typed internal
facade provides future integration methods for creating a shadow prediction,
submitting an outcome, reading metrics, reading drift status, and reading model
state/version. The facade is explicitly `experimental`, `shadow`, and
`user_facing=False`; it defines no route.

## Test-then-train lifecycle

1. Validate an event ID, UTC observation time, source permission, target-free
   vehicle features, and a separate static-reference prediction.
2. Call River `predict_one` without updating preprocessing or the estimator.
3. Retain that pre-update prediction and only the predictors needed to resolve a
   delayed outcome.
4. When an outcome arrives, require the event, a matching source, online
   permission, valid predictors, a later UTC timestamp, and a finite positive USD
   target.
5. Update paired cumulative and rolling MAE/RMSE using the stored River and
   static predictions on the same outcome.
6. Feed normalized absolute prequential error to ADWIN. A detection records
   telemetry only.
7. Call `learn_one` once, mark both event and outcome IDs as processed, remove
   resolved features, and atomically checkpoint when configured.

A repeated outcome or a second outcome for the same event is quarantined as
`duplicate_outcome` and cannot produce another update.

## Model and feature contract

The v1 estimator is deliberately simple:

- River `StandardScaler` for numeric features;
- River `OneHotEncoder` for categorical features;
- River `TargetStandardScaler` around `LinearRegression` with SGD;
- version `river-target-scaled-linear-regression-v1`;
- feature contract `shadow-vehicle-features-v1`.

Numeric predictors are year, vehicle age, mileage, mileage per year, accident
count, and owner count. Categorical predictors are make, model, condition,
engine, transmission, drivetrain, vehicle type, and missingness indicators. No
target, KBB value, listing price, deal score, expected price, source identity, or
market-position field can enter the feature map.

Preprocessing is explicitly updated only during `learn_one`; prediction does not
silently train the scaler or encoder. The first target-scaled linear prediction
is `0.0`, retained as honest cold-start behavior rather than replaced with a
manipulated value.

## Fail-closed source policy

The only enabled v1 source is `autovalue.synthetic.shadow.v1`, a project-owned
simulation stream. Cars.com-derived history, wholesale history, Yoad22,
Rebrowser AutoTrader, Carson-Shively, and unknown sources are denied. Offline
batch approval never implies an online grant.

## Simulator

The deterministic simulator creates sequential U.S./USD vehicle events with
outcomes delayed by one to six event-time minutes. Five scenarios change only
the target-generating process:

- stable market;
- gradual price drift;
- abrupt price shift;
- Ford-specific drift; and
- high-mileage drift.

The fixed simulator reference estimates the original target process. It is not
Phase 4 RF05 and is not a production-model proxy. River and the reference are
scored on the same resolved outcomes. Each scenario starts with fresh River
state so results are comparable and isolated.

## Drift, quarantine, and persistence

ADWIN receives normalized absolute error after an outcome. It may emit an event
index and detector state, but it cannot reset the estimator, retrain another
model, change routing, or promote anything.

Quarantine reason codes cover missing events, duplicate outcomes, source
mismatch, blocked sources, invalid predictors, invalid targets, and timestamp
order. Quarantine telemetry contains IDs and reasons, not raw features or target
values.

The checkpoint contains the River estimator and preprocessors, aggregate metric
state, counters, processed IDs, version fields, ADWIN state, and unresolved
synthetic events. Resolved features are deleted. Writes use a temporary file,
flush, filesystem sync, and atomic replacement. The payload has a SHA-256
integrity envelope and version checks; corruption fails closed. Because its model
payload uses Python pickle, loading is restricted to a trusted local path and is
never exposed as an upload endpoint.

## Reproduce

```powershell
$env:PYTHONPATH = "ml/src"
python -m autovalue_ml.online.simulation_cli `
  --output docs/experiments/river-shadow-simulation-v1.json `
  --events-per-scenario 600 `
  --seed 20260902 `
  --rolling-window-size 100 `
  --drift-delta 0.002
```

The report contains only configurations, aggregate metrics, synthetic event
indexes, permission decisions, restart verification, and idempotency evidence.
It contains no third-party rows.

## What remains before real outcomes

Moving to “eligible for future controlled real-outcome collection” requires a
new source-specific online permission, an append-only delayed-label contract,
privacy and retention controls, authentication/authorization, bounded state and
processed-ID retention, operational monitoring, and a separately reviewed
deployment design. It would still begin in shadow mode and require an explicit
promotion decision.
