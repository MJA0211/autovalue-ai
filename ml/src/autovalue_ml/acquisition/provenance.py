"""Cross-boundary validation for normalized scrape-result lineage."""

from __future__ import annotations

from autovalue_ml.acquisition.contracts import ScrapeResult
from autovalue_ml.acquisition.errors import PolicyViolationError
from autovalue_ml.acquisition.policy import SourcePolicy


def validate_scrape_result_provenance(result: ScrapeResult, policy: SourcePolicy) -> None:
    """Fail closed unless the result and every nested record share one lineage."""
    policy.validate_acquisition_lineage(acquired_on=result.authorization_date)
    if (
        result.source_id != policy.source_id
        or result.policy_id != policy.policy_id
        or result.policy_sha256 != policy.acquisition_fingerprint()
    ):
        raise PolicyViolationError("result provenance does not match its acquisition policy")
    policy.ensure_url_allowed(result.robots_url)
    records_seen = len(result.listings) + result.duplicates_skipped + len(result.rejected_listings)
    if (
        result.pages_fetched > policy.limits.max_pages
        or result.requests_made > policy.limits.max_requests
        or result.response_bytes > policy.limits.max_total_response_bytes
        or records_seen > policy.limits.max_records
        or (result.completed_at - result.started_at).total_seconds()
        > policy.limits.max_runtime_seconds
    ):
        raise PolicyViolationError("result metrics exceed the approved acquisition limits")
    if result.cache_persistent or (
        result.cache_max_bytes is not None and result.cache_max_bytes > policy.max_cache_bytes
    ):
        raise PolicyViolationError("result cache lineage exceeds the approved policy")

    seen_listing_ids: set[str] = set()
    for listing in result.listings:
        if (
            listing.source_id != result.source_id
            or listing.market_country != policy.market_country
            or listing.authorization_policy_id != result.policy_id
            or listing.ingestion_run_id != result.ingestion_run_id
            or listing.observed_at != result.started_at
        ):
            raise PolicyViolationError(
                "listing provenance does not match its containing acquisition result"
            )
        if listing.source_listing_id in seen_listing_ids:
            raise PolicyViolationError("result contains a duplicate normalized listing ID")
        seen_listing_ids.add(listing.source_listing_id)
        policy.ensure_url_allowed(listing.canonical_url)

    for rejection in result.rejected_listings:
        if (
            rejection.source_id != result.source_id
            or rejection.authorization_policy_id != result.policy_id
            or rejection.ingestion_run_id != result.ingestion_run_id
            or rejection.observed_at != result.started_at
        ):
            raise PolicyViolationError(
                "rejection provenance does not match its containing acquisition result"
            )
        policy.ensure_url_allowed(rejection.page_url)


__all__ = ["validate_scrape_result_provenance"]
