# 0001 — Initial architecture

- **Status:** accepted
- **Date:** 2026-08-27

## Decision

Use a monorepo with a React/Vite frontend, FastAPI backend, reusable scikit-learn
ML package, SQLite for local history, and a SQLite-compatible Turso/libSQL adapter
for the free public deployment. Train offline and deploy inference only.

Use MAE, RMSE, and R² to compare Linear Regression, Random Forest Regressor,
Gradient Boosting Regressor, and optionally XGBoost. Persist the fitted
preprocessing and model as one versioned, checksummed bundle. Include a calibrated
prediction range when validation supports it.

## Rationale

The design demonstrates original ML work, leakage-safe evaluation, API and UI
engineering, local persistence, deployment-aware model selection, and clear
operational boundaries without a paid API or required paid cloud account.

Separating static frontend hosting from the sleeping free inference service keeps
the interface available during backend cold starts. A persistence adapter avoids
misrepresenting Render's ephemeral local filesystem as durable SQLite storage.

## Consequences

- Two deployable components and exact CORS configuration are required.
- Local and hosted history adapters need the same contract tests.
- The model must fit a conservative single-worker memory budget.
- Dataset-supported inputs take priority over the aspirational product schema.
- Managed free-tier services are replaceable deployment infrastructure; all core
  application and model behavior remains runnable with open-source local tools.

## Deferred decisions

- Acceptance of a legally reusable U.S. price dataset after the Phase 2 audit;
  MUCars-2024 is rejected for geographic mismatch.
- Exact v1 inputs and valuation reference date. Target currency is fixed to USD.
- Final estimator and explanation technique after measured comparison.
- Final interval method and nominal coverage after calibration experiments.
