# Retail RF05 serving reconstruction

## Decision

**Passed for private local serving.** The frozen RF05 reference estimator was
deterministically reconstructed and packaged without changing model selection,
hyperparameters, preprocessing, features, calibration, or evaluation evidence.
This is a deployment artifact, not a new experiment or performance claim.

The model was fitted only on the exact 98,552-row development population. All
10,958 calibration rows remained excluded from point-model fitting. The final
holdout was not requested, loaded, parsed, inspected, or scored.

## Reproduction proof

The reconstruction reran the existing five grouped folds using the frozen RF05
tuple and seed. It reproduced the frozen development evidence:

| Metric | Frozen | Reconstructed |
|---|---:|---:|
| MAE | $10,269.781733328542 | $10,269.781733328542 |
| RMSE | $30,602.265758177135 | $30,602.265758177135 |
| R² | 0.4856628873797728 | 0.4856628873797728 |

The largest absolute difference across overall, fold, and vehicle-status
metrics was `3.637978807091713e-12`, within the preregistered absolute and
relative tolerances. Two independent full-development fits produced exactly
equal float64 predictions for all development predictors and five synthetic
golden vehicles. Their serialized bytes were also identical.

## Authenticated bundle

The private local bundle contains exactly:

```text
models/retail-rf05-v1/
  manifest.json
  model.joblib
```

| Evidence | SHA-256 |
|---|---|
| Reconstruction policy | `becb895893f81cf04744786b722fc2a3c40be9e52257590989e1f9f2b44c831b` |
| Aggregate reconstruction report | `e0dc8686e98a4d6cc51dd17ef61b43ca6a263a498d07aa9f3fb040071f6409f9` |
| Model bytes | `00ceb2680639a555a4705717e21ffe993a04e5731a3143e147d92d43b082e4fd` |
| Manifest | `dd31703302dce38d1a85907d3f818439e70c00f179155609be9bb93f41aaf3a2` |
| Two-file bundle fingerprint | `ef343a2fa275c85c0c666dcbc38e4899eef773d0b6dd3c0c3a159bdc3dc616d4` |
| Development identity | `c131c5b9f2561401e7545f65b491b2f0fd98f5788f9f92ea4faac19abc28b58b` |
| Synthetic golden fixture | `b36878a2c2a7bc9ed1fbc4901e1453120a11aa5a746ee894eb20682d0a43dcef` |

The schema-v2 manifest binds RF05 candidate identity, the exact parameter tuple,
random seed, ordered feature contract, preprocessing implementation, training
data identity, source and split manifests, runtime versions, serialization
policy, and the unchanged calibration v1 checksum.

## Serving security

The backend accepts only `retail-rf05-v1` directly beneath the configured
trusted models root. It rejects symbolic links and unexpected files,
authenticates the pinned manifest, verifies the complete model byte stream, and
only then deserializes those same verified bytes. It also checks the loaded
pipeline structure, fitted state, feature contract, RF05 parameters, runtime,
training boundary, and calibration binding. Any failure produces a sanitized
unavailable response rather than a fallback estimate.

Five project-owned synthetic golden requests pass through the real FastAPI route
and reproduce exact frozen point and interval responses. No source row, VIN,
dealer, listing, or row-level prediction is published in the evidence.

## Rebuild and distribution

An authorized operator with the checksum-pinned local source artifact can run:

```powershell
$env:PYTHONPATH = "ml/src"
python -m autovalue_ml.modeling.rf05_serving_bundle_cli --project-root .
```

The command writes no bundle unless every upstream, OOF, determinism, and
serialization check succeeds. Model binaries and private manifests remain
Git-ignored. The recommended distribution mode is deployment-private; committing
the model, Git LFS, or a downloadable Release remains blocked until trained-model
redistribution rights are explicit.

Complete machine-readable aggregate evidence is in
[`retail-rf05-serving-reconstruction-v1.json`](retail-rf05-serving-reconstruction-v1.json).
