"""Backfill missing resolution/runner_status/exit_price in trade_history.

Trades settled before commit d885f1a had correct P&L but didn't persist
resolution, runner_status, or exit_price to the Trade record. This script
parses the settlement log to recover those values.

Usage:
    python scripts/backfill_trade_metadata.py [--dry-run]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

STATE_PATH = Path("state/oracle_state.json")
LOG_PATH = Path("logs/oracle.log")

# Pattern: Settled | market=X resolution=Y pnl=Z ...
_SETTLED_RE = re.compile(
    r"Settled \| market=(\S+) resolution=(\w+) pnl=([\d.+-]+)"
)

_RESOLUTION_TO_META = {
    "YES":  {"runner_status": "WINNER", "exit_price": 1.0},
    "NO":   {"runner_status": "LOSER",  "exit_price": 0.0},
    "VOID": {"runner_status": "REMOVED", "exit_price": 0.0},
    "MKT":  {"runner_status": None,      "exit_price": 0.5},
}


def parse_log_resolutions(log_path: Path) -> dict[str, str]:
    """Extract market_id -> resolution from settlement log lines.

    Only considers real market IDs (skips mock 'mkt-*' entries).
    """
    resolutions: dict[str, str] = {}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = _SETTLED_RE.search(line)
            if m:
                market_id, resolution, _ = m.groups()
                if not market_id.startswith("mkt-"):
                    resolutions[market_id] = resolution
    return resolutions


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    if not STATE_PATH.exists():
        print(f"ERROR: {STATE_PATH} not found")
        sys.exit(1)
    if not LOG_PATH.exists():
        print(f"ERROR: {LOG_PATH} not found")
        sys.exit(1)

    log_resolutions = parse_log_resolutions(LOG_PATH)
    print(f"Parsed {len(log_resolutions)} real settlement entries from log")

    with open(STATE_PATH, encoding="utf-8") as f:
        state = json.load(f)

    trades = state.get("trade_history", [])
    updated_count = 0
    skipped_count = 0
    already_ok_count = 0

    for i, trade in enumerate(trades):
        if trade.get("status") != "settled":
            continue

        resolution = trade.get("resolution")
        market_id = trade.get("market_id", "")
        needs_resolution = resolution is None
        needs_exit_fix = (
            resolution in ("YES", "NO")
            and trade.get("exit_price") == 0.5
        )

        if not needs_resolution and not needs_exit_fix:
            already_ok_count += 1
            continue

        # Determine the resolution to use
        if needs_resolution:
            log_res = log_resolutions.get(market_id)
            if log_res is None:
                print(f"  SKIP trade {i}: market={market_id} — not found in log")
                skipped_count += 1
                continue
        else:
            log_res = resolution  # already have it, just fix exit_price

        meta = _RESOLUTION_TO_META.get(log_res)
        if meta is None:
            print(f"  SKIP trade {i}: market={market_id} — unknown resolution '{log_res}'")
            skipped_count += 1
            continue

        old_pnl = trade.get("pnl")
        changes = []

        if needs_resolution:
            trade["resolution"] = log_res
            trade["runner_status"] = meta["runner_status"]
            changes.append(f"resolution={log_res}")

        if needs_exit_fix or needs_resolution:
            old_exit = trade.get("exit_price")
            trade["exit_price"] = meta["exit_price"]
            if old_exit != meta["exit_price"]:
                changes.append(f"exit_price={old_exit}->{meta['exit_price']}")

        direction = trade.get("direction", "?")
        label = "FILL" if needs_resolution else "FIX "
        print(
            f"  {label} trade {i}: market={market_id} {direction} "
            f"{', '.join(changes)} pnl={old_pnl} (unchanged)"
        )
        updated_count += 1

    print(f"\nSummary: {updated_count} backfilled/fixed, {already_ok_count} already OK, {skipped_count} skipped")

    if dry_run:
        print("DRY RUN — no changes written")
    else:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        print(f"State saved to {STATE_PATH}")


if __name__ == "__main__":
    main()
