"""Paper trading execution engine.

Simulates Manifold prediction-market fills without touching real money.

Public API:
    PaperBroker(state_manager, settings)
        .derive_spread(probability)          -> (p_ask, p_bid)
        .derive_conf_score(uncertainty)      -> float
        .execute(state, ...)                 -> (OracleState, Trade | None)
        .settle_position(state, ...)         -> (OracleState, Trade)
        .check_and_settle_positions(state)   -> OracleState

Fill-or-Kill logic:
    requested_abs = f_final * bankroll
    if requested_abs <= available_liquidity * liquidity_safety_factor:
        fill at 100%
    else:
        fill at random.uniform(60%, 90%) of requested — partial fill

P&L formulae (canonical — dashboard must match these exactly):
    back + YES:  pnl = stake_abs * (1/entry_price - 1) * (1 - commission_pct)
    back + NO:   pnl = -stake_abs
    lay  + NO:   pnl = stake_abs * (1 - commission_pct)
    lay  + YES:  pnl = -liability_abs
    MKT:         use resolution_probability as settlement price in back formula
"""

from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Literal

from src.config import Settings
from src.scanner.manifold import get_market_detail
from src.storage.state_manager import OracleState, Position, StateManager, Trade

logger = logging.getLogger(__name__)

_BASE_CONF = 100.0          # conf_score = _BASE_CONF * (1 - uncertainty_penalty)
_HALF_SPREAD = 0.01         # ±1% approximation of Manifold AMM half-spread
_PARTIAL_FILL_LOW = 0.60
_PARTIAL_FILL_HIGH = 0.90


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PaperBroker:
    """Simulated execution engine for paper trading on Manifold Markets."""

    def __init__(self, state_manager: StateManager, settings: Settings) -> None:
        self._sm = state_manager
        self._cfg = settings

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def derive_spread(probability: float) -> tuple[float, float]:
        """Return (p_ask, p_bid) from a market mid probability.

        Applies a fixed ±0.01 half-spread approximation of Manifold's AMM.

        Args:
            probability: Current market probability from get_market_detail().

        Returns:
            (p_ask, p_bid) — p_ask is the cost to back YES (higher),
            p_bid is the implied price to lay against (lower).
        """
        p_ask = min(probability + _HALF_SPREAD, 0.99)
        p_bid = max(probability - _HALF_SPREAD, 0.01)
        return p_ask, p_bid

    @staticmethod
    def derive_conf_score(uncertainty_penalty: float) -> float:
        """Compute confidence score from LLM uncertainty_penalty.

        Formula: conf_score = _BASE_CONF * (1 - uncertainty_penalty)
        Matches whitepaper §3.3: conf = c_base * (1 - u_pen).

        Args:
            uncertainty_penalty: LLM output in [0.0, 1.0].

        Returns:
            conf_score in [0.0, 100.0].
        """
        return _BASE_CONF * (1.0 - uncertainty_penalty)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        state: OracleState,
        market_id: str,
        question: str,
        direction: Literal["back", "lay"],
        f_final: float,
        fill_price: float,
        edge: float,
        p_fair: float,
        kelly_f_star: float,
        kelly_f_final: float,
        conf_score: float,
        uncertainty_penalty: float,
        available_liquidity: float,
    ) -> tuple[OracleState, Trade | None]:
        """Simulate a Fill-or-Kill paper order.

        Args:
            state:               Current OracleState (mutated in-place and returned).
            market_id:           Manifold market ID.
            question:            Market question text (truncated to 120 chars).
            direction:           "back" or "lay".
            f_final:             Final Kelly fraction (output of apply_oracle_sizing).
            fill_price:          p_ask for back; p_bid for lay.
            edge:                executable_edge() result at entry.
            p_fair:              Bayesian fair probability at entry.
            kelly_f_star:        Raw commission_aware_kelly() output.
            kelly_f_final:       After apply_oracle_sizing(), before fill scaling.
            conf_score:          Derived from derive_conf_score().
            uncertainty_penalty: Raw LLM field (preserved for audit).
            available_liquidity: Pool size from get_market_detail().

        Returns:
            (updated_state, Trade) on success.
            (unchanged_state, None) if fill is zero or bankroll insufficient.
        """
        liquidity_floor = available_liquidity * self._cfg.risk.liquidity_safety_factor
        requested_abs = f_final * state.bankroll

        if requested_abs <= liquidity_floor:
            fill_pct = 1.0
        else:
            fill_pct = random.uniform(_PARTIAL_FILL_LOW, _PARTIAL_FILL_HIGH)

        filled_size = f_final * fill_pct

        # Compute absolute stake and liability
        if direction == "back":
            stake_abs = filled_size * state.bankroll
            # liability = what we stand to lose if YES doesn't happen (the stake itself)
            # Betfair-style: liability is stake * (1/price - 1) but for paper Manifold
            # we simply lose the stake on NO, so liability_abs == stake_abs
            liability_abs = stake_abs
        else:  # lay
            liability_abs = filled_size * state.bankroll
            # stake_abs for a lay = what we win if lay succeeds
            denom = (1.0 / fill_price) - 1.0
            stake_abs = liability_abs / denom if denom > 0 else 0.0

        # Insufficient funds guard
        cost = stake_abs if direction == "back" else liability_abs
        if cost > state.bankroll or cost <= 0:
            logger.warning(
                "Paper execute skipped — cost %.2f exceeds bankroll %.2f or is zero.",
                cost,
                state.bankroll,
            )
            return state, None

        bankroll_before = state.bankroll
        bankroll_after = state.bankroll - cost

        trade = Trade(
            trade_id=str(uuid.uuid4()),
            timestamp=_utc_now(),
            market_id=market_id,
            question=question[:120],
            direction=direction,
            requested_size=f_final,
            filled_size=filled_size,
            fill_price=fill_price,
            edge=edge,
            p_fair=p_fair,
            conf_score=conf_score,
            uncertainty_penalty=uncertainty_penalty,
            kelly_f_star=kelly_f_star,
            kelly_f_final=kelly_f_final,
            bankroll_before=bankroll_before,
            bankroll_after=bankroll_after,
            status="open",
            stake_abs=stake_abs,
        )

        position = Position(
            market_id=market_id,
            question=question[:120],
            direction=direction,
            entry_price=fill_price,
            filled_size=filled_size,
            stake_abs=stake_abs,
            liability_abs=liability_abs,
            entry_timestamp=trade.timestamp,
            trade_id=trade.trade_id,
            p_fair_at_entry=p_fair,
        )

        state.bankroll = bankroll_after
        if state.bankroll > state.peak_bankroll:
            state.peak_bankroll = state.bankroll

        state = self._sm.update_position(state, market_id, position)
        state = self._sm.add_trade(state, trade)

        logger.info(
            "Paper fill | market=%s dir=%s filled=%.4f price=%.3f edge=%.3f "
            "stake=%.2f bankroll=%.2f",
            market_id,
            direction,
            filled_size,
            fill_price,
            edge,
            stake_abs,
            bankroll_after,
        )

        return state, trade

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle_position(
        self,
        state: OracleState,
        market_id: str,
        resolution: str,
        resolution_probability: float,
    ) -> tuple[OracleState, Trade]:
        """Settle an open position on market resolution.

        P&L formulae (canonical):
            back + YES: gross = stake_abs * (1/entry_price - 1)
                        pnl   = gross * (1 - commission_pct)
            back + NO:  pnl = -stake_abs
            lay  + NO:  pnl = stake_abs * (1 - commission_pct)
            lay  + YES: pnl = -liability_abs
            MKT:        use resolution_probability as settlement price

        Args:
            state:                  Current OracleState.
            market_id:              Market being settled.
            resolution:             "YES", "NO", or "MKT".
            resolution_probability: Final probability (used for MKT resolution).

        Returns:
            (updated_state, settled_trade)

        Raises:
            KeyError: If market_id has no open position.
        """
        position = state.positions[market_id]
        commission_pct = self._cfg.risk.commission_pct

        stake_abs = position.stake_abs
        liability_abs = position.liability_abs
        entry_price = position.entry_price
        direction = position.direction

        if resolution == "MKT":
            # Treat as partial YES win at resolution_probability
            eff_price = resolution_probability
            gross = stake_abs * (eff_price / entry_price - 1.0) if direction == "back" else (
                stake_abs * (1.0 - eff_price)
            )
            pnl = gross * (1.0 - commission_pct) if gross > 0 else gross
            commission_paid = gross * commission_pct if gross > 0 else 0.0
        elif direction == "back":
            if resolution == "YES":
                gross = stake_abs * (1.0 / entry_price - 1.0)
                commission_paid = gross * commission_pct
                pnl = gross - commission_paid
            else:  # NO
                pnl = -stake_abs
                commission_paid = 0.0
        else:  # lay
            if resolution == "NO":
                commission_paid = stake_abs * commission_pct
                pnl = stake_abs - commission_paid
            else:  # YES — lay loses
                pnl = -liability_abs
                commission_paid = 0.0

        state.bankroll += pnl

        exit_ts = _utc_now()

        # Find and update the Trade record in history
        settled_trade: Trade | None = None
        for i, t in enumerate(state.trade_history):
            if t.trade_id == position.trade_id:
                # Rebuild as new object (Pydantic v2 models are mutable but let's be explicit)
                updated = t.model_copy(update={
                    "status": "settled",
                    "exit_price": resolution_probability,
                    "pnl": round(pnl, 4),
                    "exit_timestamp": exit_ts,
                    "commission_paid": round(commission_paid, 4),
                })
                state.trade_history[i] = updated
                settled_trade = updated
                break

        if settled_trade is None:
            logger.error(
                "Could not find trade_id=%s in history for market %s",
                position.trade_id,
                market_id,
            )
            settled_trade = state.trade_history[-1]  # fallback — should not happen

        # Remove position
        state = self._sm.update_position(state, market_id, None)

        logger.info(
            "Settled | market=%s resolution=%s pnl=%.2f bankroll=%.2f",
            market_id,
            resolution,
            pnl,
            state.bankroll,
        )
        return state, settled_trade

    # ------------------------------------------------------------------
    # Batch settlement check
    # ------------------------------------------------------------------

    def check_and_settle_positions(self, state: OracleState) -> OracleState:
        """Check all open positions for resolution and settle any that have closed.

        Called at the top of each scan cycle before scanning for new trades.
        Fetches get_market_detail() for each open position market.

        Args:
            state: Current OracleState.

        Returns:
            Updated state (unchanged if no positions resolved this cycle).
        """
        for market_id in list(state.positions.keys()):
            try:
                detail = get_market_detail(market_id)
            except Exception as exc:
                logger.warning(
                    "Failed to fetch detail for open position %s: %s — skipping settlement check.",
                    market_id,
                    exc,
                )
                continue

            if detail.get("isResolved"):
                resolution = detail.get("resolution", "MKT")
                res_prob = detail.get("probability", 0.5)
                try:
                    state, _ = self.settle_position(state, market_id, resolution, res_prob)
                except Exception as exc:
                    logger.error(
                        "Settlement failed for %s: %s", market_id, exc
                    )

        return state
