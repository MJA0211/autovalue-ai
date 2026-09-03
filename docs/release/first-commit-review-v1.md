# First-commit publication review

## Decision

**The file audit is clean for a first public source commit.** No candidate file
requires removal or additional content review. The private model, downloaded
data, row-level derivatives, local history, environment state, logs, and caches
remain excluded by Git.

The repository is initialized on `main`, has no previous commit and no remote,
so no historical revision could contain removed private material. The exact
candidate path, byte size, SHA-256, and classification inventory is in
[`first-commit-manifest-v1.json`](first-commit-manifest-v1.json).

## Classification

| Classification | Result |
|---|---|
| Safe/public | 280 candidate paths, including this report and its self-describing manifest |
| Ignored/private | Present locally and protected by explicit ignore rules |
| Needs review | 0 files |
| Excluded | All private/runtime artifacts listed below |

Safe/public material consists of source code, tests, CI configuration, safe
environment examples, documentation, aggregate experiment evidence, the model
card, project-owned synthetic pages and fixtures, and four reviewed screenshots.
The screenshots show project-owned example inputs; the valuation screenshot was
produced through the authentic RF05 API rather than embedded in the UI.

## Security and privacy scan

The publishable candidate tree produced no match for a credential assignment,
API token, bearer credential, connection string, private-key header, absolute
local workspace path, machine username, or secret environment file. The only
environment candidates are the two documented `.env.example` templates.

No candidate has a data, database, archive, log, private-key, or model-binary
extension. Textual VIN, dealer, and listing-ID field names remain in schema,
privacy, acquisition, and rejection logic because the code must identify and
remove those fields. They are not source records. The controlled scraper pages
contain six `synthetic-*` listing identifiers owned by this project. Twenty
VIN-format strings are confined to synthetic wholesale unit tests; eight other
pattern hits are dependency-integrity text in the npm lockfile. No real VIN or
dealer record is published.

The four PNGs range from 136,127 to 539,324 bytes. Their metadata contains only
image dimension/resolution properties and no local path. The largest non-image
candidate is a 499,612-byte aggregate uncertainty report. No candidate is near a
Git hosting size limit.

## Ignored/private evidence

The audit observed these protected local categories:

| Category | Ignored paths observed |
|---|---:|
| Raw data | 39 |
| Interim data | 8 |
| Processed data | 10 |
| Private RF05 bundle | 2 |
| Local model reproductions | 2 |
| Local runtime/history/browser state | 1,350 |
| Frontend build output | 11 |
| Virtual-environment dependencies | 20,492 |
| Node dependencies | 4,263 |
| Python/tool caches | 7,862 |

Sensitive ignored extensions include 32 Parquet files, seven CSV files, four
JSONL files, one source archive, one SQLite history file, and the RF05 joblib.
The ignored logs and database-like cache files are runtime/browser/tool state,
not publication candidates.

`git check-ignore` resolves both private bundle files through the `models/*`
rule. Global joblib, pickle, ONNX, and skops rules provide a second model-binary
barrier. Global row-data, database, archive, environment, credential, log,
cache, and build-output rules remain active.

## Git readiness

- Branch: `main`.
- Existing commits: none at audit time.
- Remotes: none.
- Candidate, `git add --dry-run`, manifest, and staged path sets: identical.
- All 280 staged blobs are byte-identical to their reviewed working-tree files.
  The checksum-pinned River report has an explicit no-normalization rule so Git
  cannot silently alter its line endings.
- Candidate file review: clean.
- Public push or deployment: not authorized by this phase.
- Git author name/email: not configured; an owner-supplied identity is required
  before creating the initial commit.

The exact manifest path set is staged. Once an author identity is supplied,
repeat the sensitive-file and ignored-bundle checks and create:

```text
Initial release candidate: AutoValue AI ML engineering system
```

Do not force-add an ignored path and do not push during this step.
