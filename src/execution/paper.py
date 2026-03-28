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
import math
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


def _depth_fill(
    direction: Literal["back", "lay"],
    cost: float,
    depth_ladder: list[tuple[float, float]],
    queue_model: str = "none",
    queue_factor: float = 0.5,
) -> tuple[float, float]:
    """Walk the order book ladder to determine a realistic fill.

    For backs, the ladder is sorted best-first (highest back price first).
    For lays, the ladder is sorted best-first (lowest lay price first).
    Each entry is (decimal_odds_price, available_size_aud).

    Queue-position models (applied per ladder level):
        "none":          Fill all displayed volume (optimistic baseline).
        "linear":        Discount available volume by your share at each level:
                         effective = available * max(0, 1 - queue_factor * (want / available))
        "probabilistic": Bernoulli draw — fill probability at each level is
                         available / (available + available * queue_factor).
                         Small orders almost always fill; large orders face
                         steep queue friction.

    Returns:
        (filled_fraction, vwap_probability) where filled_fraction is in [0, 1]
        and vwap_probability is the volume-weighted average implied probability
        of the filled portion.
    """
    if not depth_ladder or cost <= 0:
        return 0.0, 0.0

    remaining = cost
    total_filled = 0.0
    weighted_prob_sum = 0.0

    for decimal_price, size_aud in depth_ladder:
        if remaining <= 0:
            break
        if decimal_price <= 1.0:
            continue  # invalid price

        want = min(remaining, size_aud)

        # Apply queue-position discount
        if queue_model == "linear" and queue_factor > 0 and size_aud > 0:
            share = want / size_aud
            effective = want * max(0.0, 1.0 - queue_factor * share)
        elif queue_model == "probabilistic" and queue_factor > 0 and size_aud > 0:
            # Fill probability: your order competes with hidden queue depth
            hidden_queue = size_aud * queue_factor
            fill_prob = size_aud / (size_aud + hidden_queue)
            if random.random() > fill_prob:
                continue  # queue friction — skipped this level entirely
            effective = want  # if you pass the draw, you fill your requested amount
        else:
            effective = want

        if effective <= 0:
            continue

        implied_prob = 1.0 / decimal_price
        total_filled += effective
        weighted_prob_sum += effective * implied_prob
        remaining -= effective

    if total_filled <= 0:
        return 0.0, 0.0

    vwap_prob = weighted_prob_sum / total_filled
    filled_fraction = total_filled / cost
    return min(filled_fraction, 1.0), vwap_prob


def _apply_slippage(
    fill_price: float,
    direction: Literal["back", "lay"],
    cost: float,
    available_liquidity: float,
    model: str,
    factor: float,
) -> float:
    """Apply price slippage based on order size relative to liquidity.

    Back: price increases (worse for buyer).
    Lay: price decreases (worse for seller).

    Returns:
        Slipped fill_price, clamped to [0.01, 0.99].
    """
    if model == "none" or factor <= 0 or available_liquidity <= 0:
        return fill_price

    ratio = cost / available_liquidity
    if model == "sqrt":
        impact = factor * math.sqrt(ratio)
    else:  # "linear" (default)
        impact = factor * ratio

    if direction == "back":
        slipped = fill_price + impact
    else:  # lay
        slipped = fill_price - impact

    return max(0.01, min(0.99, slipped))


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
        depth_ladder: list[tuple[float, float]] | None = None,
        margin_min: float = 0.0,
    ) -> tuple[OracleState, Trade | None]:
        """Simulate a Fill-or-Kill paper order with realistic fills.

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
            depth_ladder:        Order book depth [(decimal_price, size_aud), ...].
            margin_min:          Minimum edge threshold — trade skipped if slippage
                                 degrades edge below this.

        Returns:
            (updated_state, Trade) on success.
            (unchanged_state, None) if fill is zero or bankroll insufficient.
        """
        requested_abs = f_final * state.bankroll

        # --- Depth-aware fill or fallback ---
        if depth_ladder:
            fill_frac, vwap_prob = _depth_fill(
                direction, requested_abs, depth_ladder,
                queue_model=self._cfg.paper.queue_position_model,
                queue_factor=self._cfg.paper.queue_factor,
            )
            if fill_frac <= 0:
                logger.info(
                    "Depth fill exhausted | market=%s dir=%s requested=%.2f — no liquidity on ladder.",
                    market_id, direction, requested_abs,
                )
                return state, None
            filled_size = f_final * fill_frac
            # Use VWAP as the effective fill price (more realistic than top-of-book)
            effective_price = vwap_prob
        else:
            # Legacy fallback: binary full-or-partial
            liquidity_floor = available_liquidity * self._cfg.risk.liquidity_safety_factor
            if requested_abs <= liquidity_floor:
                fill_pct = 1.0
            else:
                fill_pct = random.uniform(_PARTIAL_FILL_LOW, _PARTIAL_FILL_HIGH)
            filled_size = f_final * fill_pct
            effective_price = fill_price

        # --- Fill-rate tracking ---
        if self._cfg.paper.track_fill_rates:
            fill_rate = filled_size / f_final if f_final > 0 else 0.0
            logger.info(
                "Fill rate | market=%s dir=%s fill_rate=%.3f requested=%.2f filled=%.4f",
                market_id, direction, fill_rate, requested_abs, filled_size * state.bankroll,
            )

        # --- Apply slippage model ---
        cost_estimate = filled_size * state.bankroll
        effective_price = _apply_slippage(
            effective_price,
            direction,
            cost_estimate,
            available_liquidity,
            self._cfg.risk.slippage_model,
            self._cfg.risk.slippage_factor,
        )

        # Re-check edge after slippage — skip if edge destroyed
        if direction == "back":
            slipped_edge = p_fair - effective_price
        else:
            slipped_edge = effective_price - p_fair
        if margin_min > 0 and slipped_edge < margin_min:
            logger.info(
                "Slippage killed edge | market=%s dir=%s pre=%.4f post=%.4f min=%.4f",
                market_id, direction, fill_price, effective_price, margin_min,
            )
            return state, None

        # Compute absolute stake and liability using effective (slipped) price
        if direction == "back":
            stake_abs = filled_size * state.bankroll
            liability_abs = stake_abs
        else:  # lay
            liability_abs = filled_size * state.bankroll
            denom = (1.0 / effective_price) - 1.0
            if denom <= 0:
                logger.warning(
                    "Lay fill_price=%.6f produces zero/negative denom (%.6f) — skipping trade.",
                    effective_price, denom,
                )
                return state, None
            stake_abs = liability_abs / denom

            # Lay stake cap
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

        # Hard dollar cap per trade
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
            fill_price=effective_price,
            edge=slipped_edge,
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
            entry_price=effective_price,
            filled_size=filled_size,
            stake_abs=stake_abs,
            liability_abs=liability_abs,
            entry_timestamp=trade.timestamp,
            trade_id=trade.trade_id,
            p_fair_at_entry=p_fair,
            market_start_time=market_start_time,
            selection_id=selection_id,
            last_seen_price=effective_price,
        )

        state.bankroll = bankroll_after
        if state.bankroll > state.peak_bankroll:
            state.peak_bankroll = state.bankroll

        state.positions[market_id] = position
        state.trade_history.append(trade)
        self._sm.save(state)

        logger.info(
            "Paper fill | market=%s dir=%s filled=%.4f price=%.3f edge=%.3f "
            "stake=%.2f bankroll=%.2f",
            market_id,
            direction,
            filled_size,
            effective_price,
            slipped_edge,
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
        runner_status: str | None = None,       # raw Betfair runner status
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

        # For binary resolutions, use the actual outcome as exit_price
        # rather than the raw resolution_probability (which may be a 0.5 fallback
        # from an empty order book on a resolved market).
        if resolution == "YES":
            exit_price = 1.0
        elif resolution == "NO":
            exit_price = 0.0
        else:
            exit_price = resolution_probability  # MKT/VOID: keep the interpolated value

        # Find and update the Trade record in history
        settled_trade: Trade | None = None
        for i, t in enumerate(state.trade_history):
            if t.trade_id == position.trade_id:
                # Rebuild as new object (Pydantic v2 models are mutable but let's be explicit)
                updated = t.model_copy(update={
                    "status": "settled",
                    "exit_price": exit_price,
                    "pnl": round(pnl, 4),
                    "exit_timestamp": exit_ts,
                    "commission_paid": round(commission_paid, 4),
                    "closing_price": closing_price,      # Phase 5A.1
                    "clv": clv,                          # Phase 5A.1
                    "clv_snapshot_stale": position.clv_snapshot_stale,
                    "runner_status": runner_status,       # raw Betfair status
                    "resolution": resolution,             # mapped YES/NO/VOID/MKT
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
                    "clv_snapshot_stale": position.clv_snapshot_stale,
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

