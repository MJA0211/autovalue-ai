"""Run the scraper only against the project-owned synthetic marketplace."""

from __future__ import annotations

import hashlib
import hmac
import threading
from collections import Counter
from datetime import date
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from autovalue_ml.acquisition.adapter import ReviewedScrapingAdapter
from autovalue_ml.acquisition.cache import MemoryResponseCache
from autovalue_ml.acquisition.errors import PolicyViolationError
from autovalue_ml.acquisition.parser import SyntheticMarketplaceParser
from autovalue_ml.acquisition.policy import (
    CrawlLimits,
    PermissionGrant,
    Purpose,
    SourcePolicy,
)
from autovalue_ml.acquisition.scraper import SafeVehicleScraper
from autovalue_ml.acquisition.writer import write_scrape_result

DEMO_TERMS_SHA256 = "57eeea6e5817c0e74db86a420131b13cc3400e6c8004f049e81bb3890f328416"


class _QuietFixtureHandler(SimpleHTTPRequestHandler):
    """Serve owned fixtures and inject bounded, deterministic transient faults."""

    _fault_attempts: Counter[str] = Counter()
    _fault_lock = threading.Lock()

    @classmethod
    def reset_faults(cls) -> None:
        with cls._fault_lock:
            cls._fault_attempts.clear()

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        fault = self._next_fault(path)
        if fault is not None:
            status, retry_after = fault
            payload = f"Synthetic temporary HTTP {status}\n".encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            if retry_after is not None:
                self.send_header("Retry-After", retry_after)
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()

    @classmethod
    def _next_fault(cls, path: str) -> tuple[int, str | None] | None:
        fault_by_path = {
            "/page-2.html": (503, None),
            "/page-3.html": (429, "0.05"),
        }
        fault = fault_by_path.get(path)
        if fault is None:
            return None
        with cls._fault_lock:
            cls._fault_attempts[path] += 1
            return fault if cls._fault_attempts[path] == 1 else None

    def log_message(self, format: str, *args: object) -> None:
        return


def build_demo_policy(*, base_url: str, port: int, evidence_sha256: str) -> SourcePolicy:
    """Create the only enabled source policy in the initial scraper slice."""
    if not hmac.compare_digest(evidence_sha256, DEMO_TERMS_SHA256):
        raise PolicyViolationError(
            "synthetic fixture terms changed; review them and intentionally update "
            "the policy digest"
        )
    parser = SyntheticMarketplaceParser()
    return SourcePolicy(
        policy_id="synthetic-marketplace-v1",
        source_id="autovalue-synthetic-marketplace",
        source_owner="AutoValue AI repository maintainers",
        market_country="US",
        base_url=base_url,
        allowed_hosts=frozenset({"127.0.0.1"}),
        allowed_ports=frozenset({port}),
        allowed_path_prefixes=("/",),
        allowed_query_parameters=frozenset(),
        allowed_fields=parser.output_fields,
        terms_url=f"{base_url}terms.html",
        terms_version="project-owned-fixture-v1",
        user_agent="AutoValueAIResearchBot/0.1 (project-owned local fixture)",
        scraping_permission=PermissionGrant(
            approved=True,
            basis="project_owned_synthetic_fixture",
            evidence_reference="data/fixtures/scraper_site/terms.html",
            evidence_sha256=evidence_sha256,
            effective_on=date(2026, 1, 1),
            expires_on=None,
            approved_purposes=frozenset({Purpose.COLLECTION, Purpose.STORAGE}),
            contact="AutoValue AI repository maintainers",
        ),
        ml_training_permission=PermissionGrant(
            approved=True,
            basis="project_owned_synthetic_fixture",
            evidence_reference="data/fixtures/scraper_site/terms.html",
            evidence_sha256=evidence_sha256,
            effective_on=date(2026, 1, 1),
            expires_on=None,
            approved_purposes=frozenset({Purpose.ML_TRAINING, Purpose.PUBLIC_PORTFOLIO}),
            contact="AutoValue AI repository maintainers",
        ),
        limits=CrawlLimits(
            request_delay_seconds=0,
            max_pages=3,
            max_records=10,
            max_requests=10,
            max_response_bytes=200_000,
            max_total_response_bytes=500_000,
            max_runtime_seconds=30,
            max_retries=2,
            max_retry_after_seconds=0.1,
        ),
        raw_html_retention_days=1,
        cache_ttl_seconds=60,
        max_cache_bytes=500_000,
        demo_only=True,
        enabled=True,
    )


def main() -> None:
    """Start a loopback server, scrape three pages, and write normalized artifacts."""
    repository_root = Path(__file__).resolve().parents[4]
    fixture_directory = repository_root / "data" / "fixtures" / "scraper_site"
    terms_path = fixture_directory / "terms.html"
    evidence_sha256 = hashlib.sha256(terms_path.read_bytes()).hexdigest()

    handler = partial(_QuietFixtureHandler, directory=str(fixture_directory))
    _QuietFixtureHandler.reset_faults()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        port = int(server.server_address[1])
        base_url = f"http://127.0.0.1:{port}/"
        policy = build_demo_policy(
            base_url=base_url,
            port=port,
            evidence_sha256=evidence_sha256,
        )
        adapter = ReviewedScrapingAdapter(
            adapter_id="synthetic-dealership-v1",
            adapter_version="1.0.0",
            start_path="/index.html",
            policy=policy,
            parser=SyntheticMarketplaceParser(),
        )
        adapter.validate(today=date.today())
        with SafeVehicleScraper(
            adapter.policy,
            adapter.parser,
            cache=MemoryResponseCache(max_bytes=policy.max_cache_bytes),
        ) as scraper:
            result = scraper.scrape(start_path=adapter.start_path)

        output_path = repository_root / "data" / "interim" / "synthetic_listings.jsonl"
        dataset_path, manifest_path = write_scrape_result(result, policy, output_path)
        print(
            f"Scraped {len(result.listings)} synthetic listings from {result.pages_fetched} pages."
        )
        print(
            f"Handled {result.retries} transient retries, skipped "
            f"{result.duplicates_skipped} duplicate, and quarantined "
            f"{len(result.rejected_listings)} malformed records."
        )
        print(f"Dataset: {dataset_path.relative_to(repository_root)}")
        print(f"Manifest: {manifest_path.relative_to(repository_root)}")
        print(
            "Quarantine: "
            f"{dataset_path.with_suffix('.quarantine.jsonl').relative_to(repository_root)}"
        )
        print(f"Readiness: {dataset_path.with_suffix('.ready.json').relative_to(repository_root)}")
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


if __name__ == "__main__":
    main()
