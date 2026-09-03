"""Contract tests for the project-owned synthetic dealership fixture."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from autovalue_ml.acquisition.parser import SyntheticMarketplaceParser


def test_fixture_preserves_pagination_duplicates_and_data_quality_cases() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    fixture_root = repository_root / "data" / "fixtures" / "scraper_site"
    parser = SyntheticMarketplaceParser()
    observed_at = datetime(2026, 8, 27, 12, tzinfo=UTC)

    pages = tuple(
        parser.parse_page(
            (fixture_root / filename).read_text(encoding="utf-8"),
            page_url=f"http://127.0.0.1/{filename}",
            source_id="autovalue-synthetic-marketplace",
            observed_at=observed_at,
            ingestion_run_id="fixture-contract-test",
            authorization_policy_id="synthetic-marketplace-v1",
        )
        for filename in ("index.html", "page-2.html", "page-3.html")
    )

    assert pages[0].next_url == "http://127.0.0.1/page-2.html"
    assert pages[1].next_url == "http://127.0.0.1/page-3.html"
    assert pages[2].next_url is None

    assert [len(page.listings) for page in pages] == [2, 2, 1]
    assert pages[0].listings[1] == pages[1].listings[0]

    sparse_listing = pages[2].listings[0]
    assert sparse_listing.source_listing_id == "synthetic-004"
    assert sparse_listing.trim is None
    assert sparse_listing.condition is None
    assert sparse_listing.engine is None
    assert sparse_listing.owner_count is None

    rejected = pages[2].rejected_listings
    assert [record.source_listing_id for record in rejected] == [
        "synthetic-005",
        "synthetic-006",
    ]
    assert all(record.reason_code == "schema_validation_failed" for record in rejected)
    assert "required field is missing: make" in rejected[0].message
    assert "monthly payment text" in rejected[1].message
