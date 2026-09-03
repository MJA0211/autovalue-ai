# ADR 0003: United States market only

## Status

Accepted on 2026-08-28.

## Context

Vehicle prices are market-specific. Mixing countries, currencies, tax regimes,
vehicle specifications, and listing conventions would make an apparently broad
model misleading. The previously considered MUCars-2024 data represents Morocco
and cannot support a U.S. valuation claim.

## Decision

AutoValue AI targets vehicles in the United States and reports USD only. The
retail asking-price track retains New, Used, and Certified status explicitly;
the original used-vehicle use case remains a required evaluation slice.
The common listing schema requires `market_country="US"` and `currency="USD"`.
Licensed-dataset manifests and scraping source policies must also declare `US`;
normalization, provenance validation, artifact writing, and training-event
creation fail closed on any other country.

No production model will be trained until each U.S.-specific dataset independently
passes acquisition and ML-reuse permission, target semantics, schema, quality,
and representativeness gates. US Sales Cars v2 and Vehicle Sales Data v1 were
subsequently selected with conditions for separate historical asking-price and
wholesale completed-sale tracks; neither establishes a current national
retail-value claim. Regional U.S.
government fleet data may exercise the ingestion pipeline but must retain its
narrow label and may not be presented as national retail-market evidence.

## Consequences

This reduces immediately available data but makes the intended use honest and
testable. Country is retained as lineage rather than a predictive feature because
it is constant by design. Supporting another country later requires a separate
model, dataset review, currency contract, evaluation, and product surface.
