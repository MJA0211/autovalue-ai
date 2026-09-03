"""Reusable, reviewed scraping-adapter registry."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from urllib.parse import urljoin

from autovalue_ml.acquisition.errors import PolicyViolationError
from autovalue_ml.acquisition.policy import SourcePolicy
from autovalue_ml.acquisition.scraper import PageParser

_ADAPTER_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,79}$")


@dataclass(frozen=True, slots=True)
class ReviewedScrapingAdapter:
    """Bundle one pure parser with its exact policy and fixed start path."""

    adapter_id: str
    adapter_version: str
    start_path: str
    policy: SourcePolicy
    parser: PageParser

    def validate(self, *, today: date) -> None:
        if not _ADAPTER_ID.fullmatch(self.adapter_id):
            raise PolicyViolationError("adapter_id is invalid")
        if not _VERSION.fullmatch(self.adapter_version):
            raise PolicyViolationError("adapter_version is invalid")
        if not self.start_path.startswith("/") or self.start_path.startswith("//"):
            raise PolicyViolationError("adapter start path must be an absolute path")
        self.policy.validate_for_run(today=today)
        self.policy.ensure_url_allowed(urljoin(self.policy.base_url, self.start_path))
        unexpected = self.parser.output_fields - self.policy.allowed_fields
        if unexpected:
            raise PolicyViolationError("adapter parser emits fields outside its policy")


class AdapterRegistry:
    """In-process registry that rejects ambiguous adapter/source ownership."""

    def __init__(self, *, today: Callable[[], date] = date.today) -> None:
        self._by_id: dict[str, ReviewedScrapingAdapter] = {}
        self._source_ids: set[str] = set()
        self._today = today

    def register(self, adapter: ReviewedScrapingAdapter) -> None:
        if adapter.adapter_id in self._by_id:
            raise ValueError(f"adapter is already registered: {adapter.adapter_id}")
        if adapter.policy.source_id in self._source_ids:
            raise ValueError(f"source already has an adapter: {adapter.policy.source_id}")
        adapter.validate(today=self._today())
        self._by_id[adapter.adapter_id] = adapter
        self._source_ids.add(adapter.policy.source_id)

    def get(self, adapter_id: str) -> ReviewedScrapingAdapter:
        try:
            return self._by_id[adapter_id]
        except KeyError as error:
            raise KeyError(f"unknown acquisition adapter: {adapter_id}") from error

    def list_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_id))
