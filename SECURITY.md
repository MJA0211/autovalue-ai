# Security and privacy

AutoValue AI is an educational portfolio system. Security reports should avoid
including private datasets, model binaries, vehicle records, credentials, or
other sensitive material. Before a public repository exists, report issues
privately to the repository owner; add a dedicated private contact channel to
the GitHub security policy before accepting external reports.

## Trust boundaries

- The API accepts only the documented year, make, model, vehicle-status,
  mileage, and interval fields. VIN, dealer, listing, seller, account, source,
  target price, and free text are rejected.
- The RF05 joblib is trusted code and remains deployment-private. The loader
  constrains its location, rejects links and extra files, authenticates the
  manifest and bytes before deserialization, and validates model, runtime,
  training, feature, and calibration bindings.
- Missing, corrupt, or mismatched model components fail closed; no fallback
  estimate is generated.
- Browser-scoped history stores a hash of a random UUID and bounded valuation
  fields in an ignored SQLite file. It stores no VIN, dealer, account, listing,
  source, or free text.
- River is an isolated synthetic simulator with no public learning endpoint and
  no influence on RF05 valuations.

## Public repository boundary

Source code, aggregate evidence, documentation, the model card, project-owned
synthetic pages, synthetic serving fixtures, and reviewed screenshots may be
public. Downloaded datasets, row-level derivatives, SQLite files, logs, secrets,
and `models/retail-rf05-v1/` must remain private. Do not distribute the model
through normal Git, LFS, or Releases unless trained-model rights are explicitly
approved.
