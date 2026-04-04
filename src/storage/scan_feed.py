"""Scan feed writer — records per-market outcomes for dashboard display.

Writes to state/scan_feed.json with a rolling window of recent cycles.
Zero coupling to the dashboard — pure JSON writer.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

_SCAN_FEED_PATH = PROJECT_ROOT / "state" / "scan_feed.json"
_MAX_CYCLES = 48  # 24 hours at 30-min intervals


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ScanFeedWriter:
    """Collects per-market outcomes during a scan cycle and flushes to disk."""

    def __init__(self, path: Path = _SCAN_FEED_PATH) -> None:
        self._path = path
        self._current_cycle: dict | None = None
        self._entries: list[dict] = []

    def begin_cycle(self, markets_found: int) -> None:
        """Start tracking a new scan cycle."""
        self._current_cycle = {
            "cycle_id": _utc_now(),
            "started_at": _utc_now(),
            "finished_at": None,
            "markets_found": markets_found,
            "markets_analysed": 0,
            "entries": [],
        }
        self._entries = []

    def log_market(
        self,
        market_id: str,
        question: str,
        outcome: str,
        *,
        reason: str = "",
        delta: float | None = None,
        uncertainty: float | None = None,
        volume: float | None = None,
        back_edge: float | None = None,
        lay_edge: float | None = None,
        direction: str | None = None,
        fill_price: float | None = None,
        f_final: float | None = None,
        p_fair: float | None = None,
    ) -> None:
        """Record the outcome of analysing one market."""
        self._entries.append({
            "market_id": market_id,
            "question": question[:100],
            "outcome": outcome,
            "reason": reason,
            "delta": delta,
            "uncertainty": uncertainty,
            "volume": volume,
            "back_edge": back_edge,
            "lay_edge": lay_edge,
            "direction": direction,
            "fill_price": fill_price,
            "f_final": f_final,
            "p_fair": p_fair,
        })

    def end_cycle(self) -> None:
        """Finalize the current cycle and flush to disk."""
        if self._current_cycle is None:
            return
        self._current_cycle["finished_at"] = _utc_now()
        self._current_cycle["markets_analysed"] = len(self._entries)
        self._current_cycle["entries"] = self._entries
        self._flush()
        self._current_cycle = None
        self._entries = []

    def _load(self) -> dict:
        """Load existing feed from disk."""
        if not self._path.exists():
            return {"cycles": []}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"cycles": []}

    def _flush(self) -> None:
        """Append current cycle, trim to rolling window, atomic write."""
        data = self._load()
        data["cycles"].append(self._current_cycle)
        data["cycles"] = data["cycles"][-_MAX_CYCLES:]

        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")

        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                tmp.replace(self._path)
                return
            except PermissionError:
                if attempt < max_attempts - 1:
                    delay = min(0.5 * 2 ** attempt, 10.0) + random.uniform(0, 0.5)
                    logger.warning(
                        "Scan feed save attempt %d/%d failed (file locked) — retrying in %.1fs",
                        attempt + 1, max_attempts, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Scan feed save failed after %d attempts.", max_attempts,
                    )
