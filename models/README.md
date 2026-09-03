# Model artifacts

The original experiments persisted aggregate evidence and row-free calibration
state, but no estimator binary. A separately governed deployment-only process
has now deterministically reconstructed the already-selected RF05 estimator from
the exact 98,552-row development population. This was packaging, not a new model
experiment, and it did not use calibration or final-holdout rows for fitting.

The serving boundary expects this private local layout:

```text
models/retail-rf05-v1/
  manifest.json
  model.joblib
```

The private local bundle currently has model SHA-256
`00ceb2680639a555a4705717e21ffe993a04e5731a3143e147d92d43b082e4fd`
and manifest SHA-256
`dd31703302dce38d1a85907d3f818439e70c00f179155609be9bb93f41aaf3a2`.
Its complete two-file fingerprint is
`ef343a2fa275c85c0c666dcbc38e4899eef773d0b6dd3c0c3a159bdc3dc616d4`.
The schema-v2 manifest binds the exact candidate, feature contract,
preprocessing, training-data identity, source/split checksums, runtime,
serialization policy, and frozen calibration v1 checksum.

`joblib` and pickle-based artifacts can execute code while loading. The API
therefore accepts only the exact bundle name beneath its configured trusted
models root, rejects links and unexpected files, authenticates the pinned
manifest, verifies the model bytes, and only then deserializes the same verified
bytes. User-uploaded and arbitrary remote artifacts remain forbidden.

Generated binaries and manifests under `models/` remain ignored by Git.
Downloadable publication, Git LFS, and GitHub Release distribution remain
blocked until trained-model redistribution permission is explicit. The current
recommendation is a deployment-private artifact, with controlled local
reconstruction available to an authorized operator:

```powershell
$env:PYTHONPATH = "ml/src"
python -m autovalue_ml.modeling.rf05_serving_bundle_cli --project-root .
```

That command fails closed unless all upstream hashes, the frozen OOF metrics,
and the two-fit determinism check pass. The governed policy and aggregate proof
are in `docs/experiments/retail-rf05-serving-reconstruction-policy-v1.json` and
`retail-rf05-serving-reconstruction-v1.json`.
