"""Paper trading execution engine.

Simulates prediction-market fills without touching real money.

Public API:
    PaperBroker(state_manager, settings)
        .derive_spread(probability)          -> (p_ask, p_bid)
        .derive_conf_score(uncertainty)      -> float
        .execute(state, ...)                 -> (OracleState, Trade | None)
        .settle_position(state, ...)         -> (OracleState, Trade)
        .cancel_position(state, ...)         -> (OracleState, Trade)

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
    MKT:         interpolate between YES/NO outcomes at resolution_probability

Escrow accounting:
    Entry:      bankroll -= cost  (stake for back, liability for lay)
    Settlement: bankroll += cost + pnl  (return escrow, apply P&L)
"""

from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Literal

from src.config import Settings
from src.storage.state_manager import OracleState, Position, StateManager, Trade

logger = logging.getLogger(__name__)

_BASE_CONF = 100.0          # conf_score = _BASE_CONF * (1 - uncertainty_penalty)
_HALF_SPREAD = 0.01         # ±1% synthetic half-spread when real prices unavailable
_PARTIAL_FILL_LOW = 0.60
_PARTIAL_FILL_HIGH = 0.90


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PaperBroker:
    """Simulated execution engine for paper trading."""

    def __init__(self, state_manager: StateManager, settings: Settings) -> None:
        self._sm = state_manager
        self._cfg = settings

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def derive_spread(probability: float) -> tuple[float, float]:
        """Return (p_ask, p_bid) from a market mid probability.

        Applies a fixed ±0.01 synthetic half-spread when real back/lay prices
        are not available.

        Args:
            probability: Current market probability.

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
        market_start_time: str | None = None,
        selection_id: int | None = None,
    ) -> tuple[OracleState, Trade | None]:
        """Simulate a Fill-or-Kill paper order.

        Args:
            state:               Current OracleState (mutated in-place and returned).
            market_id:           Market ID.
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
            # For back bets, liability equals stake (we lose the stake on NO)
            liability_abs = stake_abs
        else:  # lay
            liability_abs = filled_size * state.bankroll
            # stake_abs for a lay = what we win if lay succeeds
            denom = (1.0 / fill_price) - 1.0
            if denom <= 0:
                logger.warning(
                    "Lay fill_price=%.6f produces zero/negative denom (%.6f) — skipping trade.",
                    fill_price, denom,
                )
                return state, None
            stake_abs = liability_abs / denom

            # Lay stake cap: prevent stake_abs from exceeding max_pnl_pct of
            # bankroll. Without this, laying at extreme odds (e.g. 0.99) produces
            # stake_abs ~100× liability, so one trade can move bankroll by 500%+.
            lay_max_pnl_pct = self._cfg.risk.lay_max_pnl_pct
            max_stake = state.bankroll * lay_max_pnl_pct
            if stake_abs > max_stake:
                scale_lay = max_stake / stake_abs
                logger.info(
                    "Lay stake cap: capping stake_abs from %.2f to %.2f "
                    "(%.0f%% of bankroll)",
                    stake_abs, max_stake, lay_max_pnl_pct * 100,
                )
                stake_abs *= scale_lay
                liability_abs *= scale_lay
                filled_size *= scale_lay

        # Hard dollar cap per trade — second line of defence against Kelly oversizing
        max_paper_cost = self._cfg.risk.paper_max_cost_aud
        cost = stake_abs if direction == "back" else liability_abs
        if cost > max_paper_cost:
            logger.warning(
                "Paper trade cost %.2f exceeds max %.2f — capping.",
                cost, max_paper_cost,
            )
            scale = max_paper_cost / cost
            filled_size *= scale
            stake_abs *= scale
            liability_abs *= scale
            cost = max_paper_cost

        # Insufficient funds guard
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
            liability_abs=liability_abs,
            selection_id=selection_id,
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
            market_start_time=market_start_time,
            selection_id=selection_id,
        )

        state.bankroll = bankroll_after
        if state.bankroll > state.peak_bankroll:
            state.peak_bankroll = state.bankroll

        # Batch state mutations into a single save to reduce OneDrive lock contention
        state.positions[market_id] = position
        state.trade_history.append(trade)
        self._sm.save(state)

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
        closing_price: float | None = None,    # Phase 5A.1
    ) -> tuple[OracleState, Trade]:
        """Settle an open position on market resolution.

        P&L formulae (canonical):
            back + YES: gross = stake_abs * (1/entry_price - 1)
                        pnl   = gross * (1 - commission_pct)
            back + NO:  pnl = -stake_abs
            lay  + NO:  pnl = stake_abs * (1 - commission_pct)
            lay  + YES: pnl = -liability_abs
            MKT:        interpolate between YES/NO at resolution_probability

        Escrow: bankroll += cost + pnl (returns escrowed capital + applies P&L).

        Args:
            state:                  Current OracleState.
            market_id:              Market being settled.
            resolution:             "YES", "NO", or "MKT".
            resolution_probability: Final probability (used for MKT resolution).
            closing_price:          Last market price before suspension (Phase 5A.1 CLV).

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

        if resolution == "VOID":
            # Runner removed — refund escrowed capital, no P&L
            pnl = 0.0
            commission_paid = 0.0
        elif resolution == "MKT":
            # Fallback: interpolate between YES/NO at resolution_probability.
            # Should rarely be used now that runner.status is read directly.
            eff_price = resolution_probability
            if direction == "back":
                gross = stake_abs * (eff_price / entry_price - 1.0)
                pnl = gross * (1.0 - commission_pct) if gross > 0 else gross
                commission_paid = gross * commission_pct if gross > 0 else 0.0
            else:  # lay
                win_portion = stake_abs * (1.0 - eff_price)
                loss_portion = liability_abs * eff_price
                commission_paid = win_portion * commission_pct if win_portion > 0 else 0.0
                pnl = win_portion - commission_paid - loss_portion
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

        cost = stake_abs if direction == "back" else liability_abs
        state.bankroll += cost + pnl

        exit_ts = _utc_now()

        # Phase 5A.1: Compute CLV
        # back: CLV = closing_price - entry_price  (positive = market moved toward our view)
        # lay:  CLV = entry_price - closing_price  (positive = market moved toward our view)
        clv = None
        if closing_price is not None:
            if direction == "back":
                clv = round(closing_price - entry_price, 6)
            else:  # lay
                clv = round(entry_price - closing_price, 6)

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
                    "closing_price": closing_price,      # Phase 5A.1
                    "clv": clv,                          # Phase 5A.1
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

        # Remove position and save in one write
        state.positions.pop(market_id, None)
        self._sm.save(state)

        logger.info(
            "Settled | market=%s resolution=%s pnl=%.2f clv=%s bankroll=%.2f",
            market_id,
            resolution,
            pnl,
            f"{clv:+.4f}" if clv is not None else "N/A",
            state.bankroll,
        )
        return state, settled_trade

    # ------------------------------------------------------------------
    # Cancellation (escrow refund, no P&L)
    # ------------------------------------------------------------------

    def cancel_position(
        self,
        state: OracleState,
        market_id: str,
        reason: str = "",
    ) -> tuple[OracleState, Trade]:
        """Cancel an open position, returning escrowed capital to bankroll.

        Used when a market becomes unreachable, is filtered out (e.g. aged-out
        outrights), or is otherwise no longer tradeable.

        Args:
            state:     Current OracleState.
            market_id: Market to cancel.
            reason:    Human-readable reason (logged and stored).

        Returns:
            (updated_state, cancelled_trade)

        Raises:
            KeyError: If market_id has no open position.
        """
        position = state.positions[market_id]
        cost = position.stake_abs if position.direction == "back" else position.liability_abs
        state.bankroll += cost  # return escrow, zero P&L

        cancel_ts = _utc_now()

        # Capture CLV from last_seen_price (same formula as settle_position)
        closing_price = position.last_seen_price
        clv = None
        if closing_price is not None:
            if position.direction == "back":
                clv = round(closing_price - position.entry_price, 6)
            else:  # lay
                clv = round(position.entry_price - closing_price, 6)

        cancelled_trade: Trade | None = None
        for i, t in enumerate(state.trade_history):
            if t.trade_id == position.trade_id:
                updated = t.model_copy(update={
                    "status": "cancelled",
                    "exit_timestamp": cancel_ts,
                    "pnl": 0.0,
                    "commission_paid": 0.0,
                    "closing_price": closing_price,
                    "clv": clv,
                })
                state.trade_history[i] = updated
                cancelled_trade = updated
                break

        if cancelled_trade is None:
            logger.error(
                "Could not find trade_id=%s in history for market %s",
                position.trade_id, market_id,
            )
            cancelled_trade = state.trade_history[-1]

        state.positions.pop(market_id, None)
        self._sm.save(state)

        logger.info(
            "Cancelled | market=%s reason=%s refund=%.2f bankroll=%.2f",
            market_id, reason or "unspecified", cost, state.bankroll,
        )
        return state, cancelled_trade

