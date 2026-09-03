# Data collection and scraping policy

## Default state

AutoValue AI does not enable website scraping by default. The only default
collection path is a policy-gated synthetic demo adapter, which supports local
UI, API, database, and test development while a real source is under review.

The product and acquisition boundary are U.S.-only. Every source policy must
declare `market_country="US"`; every accepted record and River-shaped event must
retain that value, and every price must use `currency="USD"`. A source from
another country is rejected for this product even if it has an open license.

External crawling is currently disabled even for a reviewed policy. It remains
disabled until transport can pin the validated hostname to its reviewed network
address for the full request. The runnable demo uses a numeric loopback address
and project-owned pages only.

Synthetic records must be visibly labeled as demo data. They must not enter a
real training or evaluation corpus, and their outputs must not be described as
market estimates, accuracy results, or evidence of model performance.

This policy is a project safeguard, not legal advice. When permission or rights
are unclear, do not collect the data; obtain clarification from the source owner
or choose another source.

## Approval gate for a real source

A source remains `pending` and its adapter stays disabled until every applicable
item below is documented in a project issue, pull request, or provenance record:

1. Identify the source owner and canonical dataset, API, or download URL.
2. Capture the dataset version, publication date, license or terms URL, and the
   date those terms were reviewed.
3. Confirm permission for the exact access method: official API, bulk download,
   or automated page retrieval.
4. Record collection/storage approval separately from ML-training and
   public-portfolio approval. One approval never implies the other.
5. Review robots directives and technical policies as operational signals. They
   do not replace a license or affirmative permission.
6. Minimize collection and complete a privacy review for personal, contact,
   location, registration, VIN, or other linkable data.
7. Document rate limits, caching rules, retry behavior, required identification
   or contact details, attribution, retention, refresh, and deletion duties.
8. Record an explicit decision of `approved` or `rejected`. `Pending` is not
   permission to run the adapter.
9. Confirm that the source represents the U.S. market and that all price target
   values are denominated in USD with explicit asking-versus-sale semantics.

## Prohibited behavior

An AutoValue AI adapter must never:

- bypass authentication, authorization, account scope, access controls, or a
  paywall;
- solve, evade, or outsource CAPTCHA or anti-bot challenges;
- extract, replay, or share browser cookies, session tokens, CSRF tokens,
  private API keys, or another person's credentials;
- impersonate a user, use undisclosed private endpoints, or scrape data visible
  only inside a logged-in marketplace account without explicit permission;
- add target-specific bypass logic, rotating proxies, browser-fingerprint
  evasion, or deceptive/stealth headers;
- ignore published rate limits, robots directives, revocation, or a request to
  stop collection; or
- retain unnecessary personal or contact information or commit raw acquired
  data to Git by default.

## Adding a permitted adapter

After the source is approved:

1. Add its approval decision and required attribution to the project record.
2. Implement only the authorized API, bulk-download, or retrieval method. Write
   immutable source artifacts to the Git-ignored raw-data area and create a
   checksum manifest.
3. Keep permitted credentials in environment variables; document only variable
   names in sample configuration and never commit secrets.
4. Enforce bounded timeouts, limited retries with backoff, source rate limits,
   caching, and deterministic pagination or snapshots.
5. Keep acquisition separate from normalization and schema validation so the
   original artifact remains auditable.
6. Test parsing with committed, non-sensitive fixtures. Live network tests must
   be opt-in and disabled by default.
7. Place the adapter behind an explicit source setting or feature flag. Keep the
   synthetic adapter available for local product plumbing, with demo labeling.
8. Document attribution, refresh cadence, retention and deletion requirements,
   and whether raw data, processed data, and trained artifacts may be shared.

`ReviewedScrapingAdapter` is the common adapter boundary. It binds a reviewed
policy, fixed start path, parser, and adapter version. Adapters may vary in page
parsing, but they must emit the same validated vehicle-listing schema and use the
shared transport, retry, pagination, cache, normalization, deduplication,
quarantine, provenance, and metrics behavior.

An acquisition run writes accepted normalized JSONL, rejected-record quarantine
JSONL, a checksum-verifiable manifest, and a `.ready.json` marker written last.
Consumers must call `verify_scrape_artifact_set(...)`; a missing or invalid final
marker means the run is incomplete. The manifest records separate
permission evidence, terms and policy hashes, adapter/parser/normalizer versions,
time and request budgets, HTTP status counts, retries, bytes, cache hits/misses,
duplicates, accepted/rejected counts, and artifact hashes. Raw response HTML is
not written to the generated dataset directory.

Acquired records never flow into training automatically. Downstream ML code must
invoke the separate ML-reuse gate, which revalidates current permission and
source-policy lineage and records exclusions such as an unapproved price kind or
currency. The gate also revalidates `market_country="US"` and USD target
metadata before producing a River-compatible event.

## Required provenance

For every acquired artifact, record:

- source owner, canonical URL, dataset name, and version;
- license or terms URL, the date checked, and the approval decision;
- acquisition timestamp, authorized method, request parameters, and snapshot or
  pagination details;
- filename, byte size, and SHA-256 checksum;
- adapter commit or code version;
- transformation version, row counts, exclusions, and filtering reasons;
- required attribution and regional/temporal limitations; and
- redistribution status plus retention, refresh, and deletion obligations.

This lineage must connect the raw artifact to processed datasets, split
manifests, trained models, and published evaluation results.

## Official U.S. source notes

None of the following is approved as AutoValue AI's national price-labeled
training corpus. Each source has a narrower potential role and still requires
the normal license, permission, provenance, and representativeness review.

- [FuelEconomy.gov Web Services](https://www.fueleconomy.gov/feg/ws/index.shtml)
  and [downloadable data](https://www.fueleconomy.gov/feg/download.shtml) expose
  US EPA/DOE vehicle specifications and fuel-economy fields. Review the
  [EPA data license](https://edg.epa.gov/EPA_Data_License.html) and the exact
  resource before use. Limit a future adapter to official vehicle records;
  community-submitted "My MPG" records require a separate review.
- City of Seattle Sold Fleet Equipment may be reviewed as a public-domain
  licensed-file smoke test only. Its small municipal-auction sample and missing
  mileage prevent production use.
- Any future GSA fleet-sales corpus must be acquired through an official bulk
  release or direct agency request. Do not scrape an auction site to obtain it.

Kaggle Vehicle Sales Data v1 and US Sales Cars v2 are conditionally approved
through version-pinned local-file paths, not scraping adapters. The latter's
upstream repository identifies a historical Cars.com extraction; the project
owner separately attests that collection and portfolio ML reuse were authorized.
That evidence permits the fixed reviewed artifact only. It does not authorize
AutoValue AI to crawl Cars.com, Kaggle, or an upstream auction marketplace.
MUCars-2024 remains rejected because it represents Morocco, and UCI Automobile
lacks the fields and modern coverage required for this U.S. product.
