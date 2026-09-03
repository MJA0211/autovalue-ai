# ADR 0002: Permission-gated acquisition

## Status

Accepted on 2026-08-27.

## Context

AutoValue AI needs a defensible data story as well as a real machine-learning
pipeline. Public visibility, a personal login, or non-commercial intent does not
by itself authorize automated collection or downstream model training.

## Decision

All automated acquisition is disabled unless a reviewed source policy explicitly
authorizes the exact network boundary, fields, retention, collection, and
storage. ML training and public portfolio reuse require an independent, current
approval and a separate fingerprint; changing that later approval does not
rewrite immutable acquisition lineage. Every nested record is revalidated before
artifact writing or event emission. Acquisition never implies downstream use.
`robots.txt` is an additional restriction, not permission. The client is
sequential and bounded, and it does not support authentication, cookies, browser
automation, CAPTCHA bypass, proxy rotation, or arbitrary command-line targets.

Licensed files require a strict manifest whose complete SHA-256 is pinned in an
independent project review record before parsing. Scraped output publishes
accepted JSONL, quarantine JSONL, and its lineage/metrics manifest before writing
a `.ready.json` marker last. Consumers must verify that marker and every artifact
hash; publication readiness does not imply ML-training permission.

All source policies, licensed manifests, normalized records, and future
River-shaped events must satisfy the separate U.S.-market decision:
`market_country="US"` and `currency="USD"` for every price-bearing record.

The first and only enabled adapter uses invented pages owned by this repository
and served over numeric loopback. External crawling remains disabled until an
address-pinned transport is implemented and reviewed. Any future external
adapter also requires a new policy, documented authorization evidence, parser
review, and adversarial tests.

## Consequences

The repository demonstrates scraping architecture without encouraging prohibited
collection. It may take longer to obtain a real price-labeled dataset, but every
training artifact can retain a clear provenance chain. Official open-data sources
may enrich vehicle attributes, while the primary target still must come from an
appropriately licensed dataset.
