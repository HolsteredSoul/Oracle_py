"""Rejection cache — skip markets that recently failed hard gates.

In-memory only (no disk persistence). Resets on agent restart, which is
fine since volume/liquidity can change between sessions.

Only caches "hard" gate failures where conditions are unlikely to change
within the TTL window. Soft failures (no_edge, negative_kelly) are NOT
cached because market prices move.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

# Outcomes that are safe to cache — conditions won't change in 2 hours
CACHEABLE_OUTCOMES = frozenset({
    "skipped_volume",
    "skipped_liquidity",
    "skipped_niche",
    "skipped_inplay",
    "skipped_crossed",
    "skipped_no_model",
})


class RejectionCache:
    """Skip markets that recently failed hard gates."""

    def __init__(self, ttl_minutes: int = 120) -> None:
        self._cache: dict[str, float] = {}  # market_id -> expiry timestamp
        self._ttl = ttl_minutes * 60

    def reject(self, market_id: str, outcome: str) -> None:
        """Cache a rejection if the outcome is a hard gate failure."""
        if outcome in CACHEABLE_OUTCOMES:
            self._cache[market_id] = time.time() + self._ttl

    def is_rejected(self, market_id: str) -> bool:
        """Check if a market was recently rejected. Auto-evicts expired entries."""
        exp = self._cache.get(market_id)
        if exp is None:
            return False
        if time.time() > exp:
            del self._cache[market_id]
            return False
        return True

    def rejected_count(self, market_ids: list[str]) -> int:
        """Count how many of the given market IDs are currently rejected."""
        return sum(1 for mid in market_ids if self.is_rejected(mid))

    def purge_expired(self) -> None:
        """Remove all expired entries."""
        now = time.time()
        self._cache = {k: v for k, v in self._cache.items() if v > now}

    def __len__(self) -> int:
        return len(self._cache)
