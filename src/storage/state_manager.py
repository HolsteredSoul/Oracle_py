"""Oracle state persistence.

Single source of truth for bankroll, positions, trade history, and market priors.
All writes are atomic: write to .tmp then Path.replace() — identical pattern to
llm/client.py _save_spend(). A corrupt or missing state file boots a fresh
OracleState rather than crashing the agent.

Public API:
    StateManager(path)     — instantiate with optional custom path
    .load()                — returns OracleState (never raises)
    .save(state)           — atomic write
    .add_trade(state, t)   — append trade + save
    .update_position(...)  — upsert/delete position + save
    .current_exposure(s)   — sum of open filled_sizes
    .drawdown_pct(s)       — (peak - current) / peak; updates peak if exceeded
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

STATE_PATH = PROJECT_ROOT / "state" / "oracle_state.json"
DEFAULT_BANKROLL = 1000.0


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Trade(BaseModel):
    """Immutable record of one paper fill. Fields populated on settlement."""

    trade_id: str
    timestamp: str                                   # ISO-8601 UTC — entry time
    market_id: str
    question: str                                    # truncated to 120 chars
    direction: Literal["back", "lay"]
    requested_size: float                            # fraction of bankroll requested
    filled_size: float                               # fraction actually filled
    fill_price: float                                # p_ask (back) or p_bid (lay)
    edge: float                                      # executable_edge() at entry
    p_fair: float                                    # Bayesian estimate at entry
    conf_score: float                                # _BASE_CONF * (1 - uncertainty_penalty)
    uncertainty_penalty: float                       # raw LLM output — audit trail
    kelly_f_star: float                              # raw commission_aware_kelly()
    kelly_f_final: float                             # after apply_oracle_sizing()
    bankroll_before: float
    bankroll_after: float
    status: Literal["open", "settled"]
    # Populated on settlement:
    exit_price: Optional[float] = None
    pnl: Optional[float] = None                     # realised P&L in paper AUD
    exit_timestamp: Optional[str] = None
    commission_paid: Optional[float] = None
    stake_abs: Optional[float] = None               # stored for dashboard display


class Position(BaseModel):
    """Live open position."""

    market_id: str
    question: str
    direction: Literal["back", "lay"]
    entry_price: float
    filled_size: float                               # fraction of bankroll
    stake_abs: float                                 # absolute AUD staked
    liability_abs: float                             # absolute AUD at risk (lay)
    entry_timestamp: str                             # ISO-8601 UTC
    trade_id: str                                    # FK to Trade record
    p_fair_at_entry: float


class OracleState(BaseModel):
    """Root application state — persisted as JSON after every trade."""

    bankroll: float = DEFAULT_BANKROLL
    peak_bankroll: float = DEFAULT_BANKROLL          # for drawdown calculation
    positions: dict[str, Position] = Field(default_factory=dict)   # key: market_id
    trade_history: list[Trade] = Field(default_factory=list)
    priors: dict[str, float] = Field(default_factory=dict)         # market_id → p_fair
    created_at: str = Field(default_factory=_utc_now)
    last_updated: str = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------

class StateManager:
    """Thread-safe state persistence with atomic writes."""

    def __init__(self, path: Path = STATE_PATH) -> None:
        self._path = path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> OracleState:
        """Load state from disk.

        Returns a fresh OracleState with DEFAULT_BANKROLL if the file is
        missing or corrupt. Never raises.
        """
        if not self._path.exists():
            logger.debug("State file not found at %s — starting fresh.", self._path)
            return OracleState()

        try:
            text = self._path.read_text(encoding="utf-8")
            return OracleState.model_validate_json(text)
        except Exception as exc:
            logger.critical(
                "State file corrupt at %s: %s — starting fresh with DEFAULT_BANKROLL.",
                self._path,
                exc,
            )
            return OracleState()

    def save(self, state: OracleState) -> None:
        """Atomically persist state.

        Writes to <path>.tmp then replaces the main file via Path.replace(),
        which is atomic on POSIX and atomic-within-same-drive on Windows.
        Identical pattern to llm/client.py _save_spend().
        """
        state.last_updated = _utc_now()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def add_trade(self, state: OracleState, trade: Trade) -> OracleState:
        """Append a Trade to history and save. Returns the mutated state."""
        state.trade_history.append(trade)
        self.save(state)
        return state

    def update_position(
        self,
        state: OracleState,
        market_id: str,
        position: Position | None,
    ) -> OracleState:
        """Upsert or delete a position and save.

        Args:
            state:     Current OracleState.
            market_id: Key to update.
            position:  Position to set, or None to remove.

        Returns:
            Mutated state (same object).
        """
        if position is None:
            state.positions.pop(market_id, None)
        else:
            state.positions[market_id] = position
        self.save(state)
        return state

    # ------------------------------------------------------------------
    # Computed metrics (no IO)
    # ------------------------------------------------------------------

    def current_exposure(self, state: OracleState) -> float:
        """Sum of filled_size across all open positions (fraction of bankroll)."""
        return sum(p.filled_size for p in state.positions.values())

    def drawdown_pct(self, state: OracleState) -> float:
        """Current drawdown fraction: (peak - bankroll) / peak.

        Updates state.peak_bankroll if the current bankroll exceeds the peak.
        Returns 0.0 when bankroll is at or above peak.
        """
        if state.bankroll > state.peak_bankroll:
            state.peak_bankroll = state.bankroll
        if state.peak_bankroll <= 0:
            return 0.0
        dd = (state.peak_bankroll - state.bankroll) / state.peak_bankroll
        return max(dd, 0.0)
