"""Pure HTML parser for the project-owned synthetic marketplace."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime
from decimal import InvalidOperation
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from autovalue_ml.acquisition.contracts import (
    ParsedPage,
    PriceKind,
    RejectedListing,
    VehicleListingSnapshot,
)
from autovalue_ml.acquisition.errors import ListingParseError
from autovalue_ml.acquisition.scalar_parsing import parse_price_text_cents

PARSER_VERSION = "synthetic-marketplace/1.0.0"
NORMALIZATION_VERSION = "1.0.0"


class SyntheticMarketplaceParser:
    """Parse only the reviewed markup owned by this repository."""

    output_fields = frozenset(VehicleListingSnapshot.__dataclass_fields__)

    def parse_page(
        self,
        html: str,
        *,
        page_url: str,
        source_id: str,
        observed_at: datetime,
        ingestion_run_id: str,
        authorization_policy_id: str,
    ) -> ParsedPage:
        """Parse one page without performing I/O or executing page scripts."""
        soup = BeautifulSoup(html, "html.parser")
        cards = soup.select("article.vehicle-card")
        if not cards:
            raise ListingParseError("no vehicle cards found; parser contract may have drifted")

        listings: list[VehicleListingSnapshot] = []
        rejected_listings: list[RejectedListing] = []
        for card in cards:
            try:
                listings.append(
                    self._parse_card(
                        card,
                        page_url=page_url,
                        source_id=source_id,
                        observed_at=observed_at,
                        ingestion_run_id=ingestion_run_id,
                        authorization_policy_id=authorization_policy_id,
                    )
                )
            except (ListingParseError, ValueError, InvalidOperation) as error:
                listing_id = card.get("data-listing-id")
                normalized_listing_id = listing_id.strip() if isinstance(listing_id, str) else None
                rejected_listings.append(
                    RejectedListing(
                        source_id=source_id,
                        page_url=page_url,
                        source_listing_id=normalized_listing_id or None,
                        observed_at=observed_at,
                        reason_code="schema_validation_failed",
                        message=str(error),
                        raw_content_sha256=hashlib.sha256(str(card).encode("utf-8")).hexdigest(),
                        parser_version=PARSER_VERSION,
                        ingestion_run_id=ingestion_run_id,
                        authorization_policy_id=authorization_policy_id,
                    )
                )

        next_link = soup.select_one('a[rel~="next"][href]')
        next_url = None
        if isinstance(next_link, Tag):
            href = next_link.get("href")
            if isinstance(href, str) and href.strip():
                next_url = urljoin(page_url, href.strip())

        return ParsedPage(
            listings=tuple(listings),
            next_url=next_url,
            rejected_listings=tuple(rejected_listings),
        )

    def _parse_card(
        self,
        card: Tag,
        *,
        page_url: str,
        source_id: str,
        observed_at: datetime,
        ingestion_run_id: str,
        authorization_policy_id: str,
    ) -> VehicleListingSnapshot:
        listing_id = card.get("data-listing-id")
        if not isinstance(listing_id, str) or not listing_id.strip():
            raise ListingParseError("vehicle card is missing its source listing ID")

        try:
            canonical_url = self._listing_url(card, page_url)
            year = _parse_integer(_required_text(card, "year"), field="year")
            make = _required_text(card, "make")
            model = _required_text(card, "model")
            trim = _optional_text(card, "trim")
            mileage = _parse_optional_integer(_optional_text(card, "mileage"), field="mileage")
            condition = _optional_text(card, "condition")
            vehicle_status_text = _optional_text(card, "vehicle-status")
            vehicle_status = (
                vehicle_status_text.casefold() if vehicle_status_text is not None else None
            )
            engine = _optional_text(card, "engine")
            drivetrain = _optional_text(card, "drivetrain")
            accident_status = _optional_text(card, "accident-status")
            accident_count = _parse_optional_integer(
                _optional_text(card, "accident-count"), field="accident-count"
            )
            owner_count = _parse_optional_integer(
                _optional_text(card, "owner-count"), field="owner-count"
            )
            vehicle_type = _optional_text(card, "vehicle-type")
            price_element = _required_element(card, "price")
            currency = _required_attribute(price_element, "data-currency")
            price_kind = PriceKind(_required_attribute(price_element, "data-price-kind"))
            price_cents = parse_price_text_cents(
                price_element.get_text(" ", strip=True),
                expected_currency=currency,
                price_kind=price_kind,
            )
        except (ValueError, InvalidOperation) as error:
            raise ListingParseError(f"invalid listing {listing_id}: {error}") from error

        raw_content_sha256 = hashlib.sha256(str(card).encode("utf-8")).hexdigest()
        return VehicleListingSnapshot(
            source_id=source_id,
            source_listing_id=listing_id.strip(),
            canonical_url=canonical_url,
            observed_at=observed_at,
            market_country="US",
            year=year,
            make=make,
            model=model,
            trim=trim,
            mileage=mileage,
            mileage_unit="miles",
            condition=condition,
            vehicle_status=vehicle_status,
            engine=engine,
            drivetrain=drivetrain,
            accident_status=accident_status,
            accident_count=accident_count,
            owner_count=owner_count,
            vehicle_type=vehicle_type,
            price_cents=price_cents,
            currency=currency,
            price_kind=price_kind,
            sale_status="active",
            raw_content_sha256=raw_content_sha256,
            parser_version=PARSER_VERSION,
            normalization_version=NORMALIZATION_VERSION,
            ingestion_run_id=ingestion_run_id,
            authorization_policy_id=authorization_policy_id,
        )

    @staticmethod
    def _listing_url(card: Tag, page_url: str) -> str:
        link = card.select_one("a.vehicle-card__link[href]")
        if not isinstance(link, Tag):
            raise ListingParseError("vehicle card is missing its canonical link")
        href = link.get("href")
        if not isinstance(href, str) or not href.strip():
            raise ListingParseError("vehicle canonical link is empty")
        return urljoin(page_url, href.strip())


def _required_element(card: Tag, field: str) -> Tag:
    element = card.select_one(f'[data-field="{field}"]')
    if not isinstance(element, Tag):
        raise ListingParseError(f"required field is missing: {field}")
    return element


def _required_text(card: Tag, field: str) -> str:
    value = _normalize_text(_required_element(card, field).get_text(" ", strip=True))
    if not value:
        raise ListingParseError(f"required field is empty: {field}")
    return value


def _optional_text(card: Tag, field: str) -> str | None:
    element = card.select_one(f'[data-field="{field}"]')
    if not isinstance(element, Tag):
        return None
    value = _normalize_text(element.get_text(" ", strip=True))
    return value or None


def _required_attribute(element: Tag, name: str) -> str:
    value = element.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ListingParseError(f"required attribute is missing: {name}")
    return value.strip()


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split())


def _parse_integer(value: str, *, field: str) -> int:
    suffixes = {
        "year": None,
        "mileage": r"mi|miles?",
        "accident-count": r"accidents?",
        "owner-count": r"owners?",
    }
    suffix = suffixes.get(field)
    suffix_pattern = rf"(?:\s*(?:{suffix}))?" if suffix is not None else ""
    normalized = unicodedata.normalize("NFKC", value)
    match = re.fullmatch(
        rf"\s*(?P<number>-?(?:\d{{1,3}}(?:,\d{{3}})+|\d+)){suffix_pattern}\s*",
        normalized,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"{field} is not an integer")
    return int(match.group("number").replace(",", ""))


def _parse_optional_integer(value: str | None, *, field: str) -> int | None:
    if value is None or value.lower() in {"unknown", "not reported", "n/a"}:
        return None
    return _parse_integer(value, field=field)
