"""Recalculate P&L for all settled trades in the corrupted backup state.

Fetches runner.status from Betfair for each settled trade's market to determine
the actual YES/NO/VOID outcome, then computes corrected P&L.

Output: state/recalculated_trades.json

Usage:
    python scripts/recalculate_trades.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when running from scripts/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BACKUP_PATH = Path("state/oracle_state_pre_fix_backup.json")
OUTPUT_PATH = Path("state/recalculated_trades.json")
COMMISSION_PCT = 0.05  # from config.toml


def _resolve_from_betfair(market_id: str) -> dict:
    """Fetch runner status from Betfair for a settled market."""
    try:
        from src.scanner.betfair_scanner import get_market_detail
        detail = get_market_detail(market_id)
        return {
            "resolution": detail.get("resolution", "UNKNOWN"),
            "runner_status": detail.get("runner_status"),
            "is_resolved": detail.get("isResolved", False),
            "selection_id": detail.get("selection_id"),
        }
    except Exception as exc:
        logger.error("  Fetch error for %s: %s", market_id, exc)
        return {"resolution": "FETCH_FAILED", "error": str(exc)}


def _compute_pnl(
    direction: str,
    resolution: str,
    stake_abs: float,
    liability_abs: float,
    entry_price: float,
) -> tuple[float, float]:
    """Compute correct P&L given binary resolution."""
    if resolution == "VOID":
        return 0.0, 0.0

    if direction == "back":
        if resolution == "YES":
            gross = stake_abs * (1.0 / entry_price - 1.0)
            commission = gross * COMMISSION_PCT
            return round(gross - commission, 4), round(commission, 4)
        else:  # NO
            return round(-stake_abs, 4), 0.0
    else:  # lay
        if resolution == "NO":
            commission = stake_abs * COMMISSION_PCT
            return round(stake_abs - commission, 4), round(commission, 4)
        else:  # YES
            return round(-liability_abs, 4), 0.0


def main() -> None:
    if not BACKUP_PATH.exists():
        logger.error("Backup file not found: %s", BACKUP_PATH)
        return

    state = json.loads(BACKUP_PATH.read_text(encoding="utf-8"))
    settled = [t for t in state.get("trade_history", []) if t.get("status") == "settled"]

    logger.info("Recalculating %d settled trades...\n", len(settled))

    results = []
    total_original_pnl = 0.0
    total_corrected_pnl = 0.0
    fetch_failures = 0

    for i, trade in enumerate(settled):
        market_id = trade["market_id"]
        direction = trade["direction"]
        entry_price = trade["fill_price"]
        stake_abs = trade.get("stake_abs", 0)
        liability_abs = trade.get("liability_abs", 0)
        original_pnl = trade.get("pnl", 0)

        # The true escrowed amount is the bankroll delta at entry.
        # For lay trades, this is the actual liability (not the inflated stake_abs).
        escrowed = trade.get("bankroll_before", 0) - trade.get("bankroll_after", 0)

        # For lay trades: use escrowed as liability, derive stake from it
        if direction == "lay":
            liability_abs = escrowed
            denom = (1.0 / entry_price) - 1.0
            stake_abs = liability_abs / denom if denom > 0 else 0

        logger.info("[%d/%d] %s  dir=%s  entry=%.3f", i + 1, len(settled), market_id, direction, entry_price)

        # Fetch actual resolution from Betfair
        bf_result = _resolve_from_betfair(market_id)
        resolution = bf_result.get("resolution", "UNKNOWN")
        logger.info("        resolution=%s", resolution)

        if resolution in ("YES", "NO", "VOID"):
            corrected_pnl, commission = _compute_pnl(
                direction, resolution, stake_abs, liability_abs, entry_price,
            )
        else:
            corrected_pnl = None
            commission = None
            fetch_failures += 1

        total_original_pnl += original_pnl or 0
        if corrected_pnl is not None:
            total_corrected_pnl += corrected_pnl

        results.append({
            "market_id": market_id,
            "question": trade.get("question", ""),
            "direction": direction,
            "entry_price": entry_price,
            "stake_abs": stake_abs,
            "liability_abs": liability_abs,
            "original_pnl": original_pnl,
            "original_exit_price": trade.get("exit_price"),
            "corrected_resolution": resolution,
            "corrected_pnl": corrected_pnl,
            "corrected_commission": commission,
            "delta": round(corrected_pnl - original_pnl, 4) if corrected_pnl is not None else None,
        })

        # Be polite to the API
        time.sleep(0.5)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Total trades:       %d", len(results))
    logger.info("Fetch failures:     %d", fetch_failures)
    logger.info("Original total P&L: $%.2f", total_original_pnl)
    logger.info("Corrected total P&L: $%.2f", total_corrected_pnl)
    logger.info("Difference:         $%.2f", total_corrected_pnl - total_original_pnl)

    output = {
        "summary": {
            "total_trades": len(results),
            "fetch_failures": fetch_failures,
            "original_total_pnl": round(total_original_pnl, 2),
            "corrected_total_pnl": round(total_corrected_pnl, 2),
            "difference": round(total_corrected_pnl - total_original_pnl, 2),
        },
        "trades": results,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("\nResults written to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
