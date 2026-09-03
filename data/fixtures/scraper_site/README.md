# Synthetic vehicle marketplace

This is a project-owned HTML fixture used to exercise AutoValue AI's acquisition
pipeline without accessing a third-party marketplace. It represents an invented
U.S. dealership; every record uses `market_country="US"` and USD. Every vehicle,
price, identifier, and attribute is invented. It contains no seller information, VINs,
photos, tracking scripts, or real listings.

The fixture deliberately includes three paginated result pages, an exact
duplicate, optional missing values, and required-field/malformed-price records
that must be quarantined. Its `robots.txt` permits only the AutoValue AI demo bot.
The local server also injects one temporary `503` and one `429` before returning
successful pages. It binds only to `127.0.0.1` and writes normalized output plus
lineage and quarantine artifacts to the ignored `data/interim/` directory. The
writer publishes `.ready.json` last, and consumers verify the complete artifact
set before reading it.

This fixture tests scraper behavior; it is never evidence of model accuracy and
must not be used as the final training dataset.
