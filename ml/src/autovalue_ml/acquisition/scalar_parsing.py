"""Strict scalar parsers shared by reviewed acquisition adapters."""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal

from autovalue_ml.acquisition.contracts import PriceKind

_PRICE_PATTERN = re.compile(
    r"^\s*(?:(?P<prefix>[A-Za-z]{3})\s+)?"
    r"(?P<symbol>[$€£¥])?\s*"
    r"(?P<amount>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)"
    r"\s*(?P<suffix>[A-Za-z]{3})?"
    r"\s*(?P<monthly>/\s*mo(?:nth)?|monthly)?\s*$",
    flags=re.IGNORECASE,
)
_SYMBOL_CURRENCIES = {
    "$": frozenset({"AUD", "CAD", "HKD", "NZD", "SGD", "USD"}),
    "€": frozenset({"EUR"}),
    "£": frozenset({"GBP"}),
    "¥": frozenset({"CNY", "JPY"}),
}


def parse_price_text_cents(
    value: str,
    *,
    expected_currency: str,
    price_kind: PriceKind,
) -> int:
    """Parse one complete price string without discarding unknown characters."""
    if not isinstance(value, str):
        raise ValueError("price must be text")
    if re.fullmatch(r"[A-Z]{3}", expected_currency) is None:
        raise ValueError("expected currency must be a three-letter uppercase code")
    if not isinstance(price_kind, PriceKind):
        raise ValueError("price kind must use the PriceKind enum")

    normalized = unicodedata.normalize("NFKC", value)
    match = _PRICE_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError("price does not match the approved numeric format")

    is_monthly = match.group("monthly") is not None
    if is_monthly and price_kind is not PriceKind.MONTHLY_PAYMENT:
        raise ValueError("monthly payment text cannot be treated as a vehicle price")

    declared_codes = [
        code.upper() for code in (match.group("prefix"), match.group("suffix")) if code
    ]
    if len(declared_codes) > 1 or any(code != expected_currency for code in declared_codes):
        raise ValueError("price currency does not match the reviewed source currency")

    symbol = match.group("symbol")
    if symbol is not None and expected_currency not in _SYMBOL_CURRENCIES[symbol]:
        raise ValueError("price symbol does not match the reviewed source currency")

    amount = Decimal(match.group("amount").replace(",", ""))
    if not amount.is_finite() or amount <= 0:
        raise ValueError("price must be a positive finite amount")
    scaled = amount * 100
    if scaled != scaled.to_integral_value():
        raise ValueError("price has more than two decimal places")
    return int(scaled)


__all__ = ["parse_price_text_cents"]
