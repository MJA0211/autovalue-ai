"""Small, bounded in-memory response cache for approved source adapters."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CachedTextResponse:
    """Integrity-protected text retained only in process memory."""

    body: str
    body_sha256: str
    expires_at: float
    size_bytes: int


class ResponseCache(Protocol):
    """Minimal cache boundary accepted by the safe scraper."""

    @property
    def backend_name(self) -> str: ...

    @property
    def persistent(self) -> bool: ...

    @property
    def max_bytes(self) -> int: ...

    def get(self, key: str) -> str | None: ...

    def put(self, key: str, body: str, *, ttl_seconds: float) -> bool: ...


class MemoryResponseCache:
    """TTL/LRU cache with entry and byte limits; it never writes raw pages to disk."""

    def __init__(
        self,
        *,
        max_entries: int = 128,
        max_bytes: int = 5_000_000,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(max_entries) is not int or not 1 <= max_entries <= 10_000:
            raise ValueError("max_entries is outside the safe range")
        if type(max_bytes) is not int or not 1_024 <= max_bytes <= 100_000_000:
            raise ValueError("max_bytes is outside the safe range")
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._monotonic = monotonic
        self._entries: OrderedDict[str, CachedTextResponse] = OrderedDict()
        self._current_bytes = 0
        self._lock = threading.Lock()

    @property
    def backend_name(self) -> str:
        return "memory"

    @property
    def persistent(self) -> bool:
        return False

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def get(self, key: str) -> str | None:
        """Return a verified, unexpired body and promote it in LRU order."""
        if not isinstance(key, str) or not key:
            raise ValueError("cache key must be a non-empty string")
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= self._monotonic():
                self._remove(key)
                return None
            actual_sha256 = hashlib.sha256(entry.body.encode("utf-8")).hexdigest()
            if actual_sha256 != entry.body_sha256:
                self._remove(key)
                return None
            self._entries.move_to_end(key)
            return entry.body

    def put(self, key: str, body: str, *, ttl_seconds: float) -> bool:
        """Cache one text response if it fits the reviewed memory budget."""
        if not isinstance(key, str) or not key:
            raise ValueError("cache key must be a non-empty string")
        if not isinstance(body, str):
            raise ValueError("cache body must be text")
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(float(ttl_seconds))
        ):
            raise ValueError("cache TTL must be a finite number")
        if ttl_seconds <= 0:
            return False
        if ttl_seconds > 86_400:
            raise ValueError("cache TTL exceeds the one-day safety limit")
        payload = body.encode("utf-8")
        size_bytes = len(payload)
        if size_bytes > self._max_bytes:
            return False
        entry = CachedTextResponse(
            body=body,
            body_sha256=hashlib.sha256(payload).hexdigest(),
            expires_at=self._monotonic() + ttl_seconds,
            size_bytes=size_bytes,
        )
        with self._lock:
            if key in self._entries:
                self._remove(key)
            self._entries[key] = entry
            self._current_bytes += size_bytes
            while len(self._entries) > self._max_entries or self._current_bytes > self._max_bytes:
                oldest_key = next(iter(self._entries))
                self._remove(oldest_key)
        return True

    def clear(self) -> None:
        """Drop every ephemeral cached response."""
        with self._lock:
            self._entries.clear()
            self._current_bytes = 0

    def _remove(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._current_bytes -= entry.size_bytes
