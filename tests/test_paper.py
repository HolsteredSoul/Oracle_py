"""Tests for src/execution/paper.py.

get_market_detail() is mocked everywhere — no network calls.
StateManager uses tmp_path — no side-effects on real state/.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.execution.paper import PaperBroker, _BASE_CONF, _HALF_SPREAD
from src.storage.state_manager import DEFAULT_BANKROLL, OracleState, Position, StateManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sm(tmp_path: Path) -> StateManager:
    return StateManager(path=tmp_path / "oracle_state.json")


@pytest.fixture()
def cfg() -> MagicMock:
    """Minimal settings mock matching src/config.py structure."""
    mock = MagicMock()
    mock.risk.liquidity_safety_factor = 0.70
    mock.risk.commission_pct = 0.05
    mock.risk.lay_max_pnl_pct = 0.15
    mock.risk.paper_max_cost_aud = 100.0
    return mock


@pytest.fixture()
def broker(sm: StateManager, cfg: MagicMock) -> PaperBroker:
    return PaperBroker(sm, cfg)


@pytest.fixture()
def fresh_state() -> OracleState:
    return OracleState(bankroll=DEFAULT_BANKROLL, peak_bankroll=DEFAULT_BANKROLL)


def _execute_back(
    broker: PaperBroker,
    state: OracleState,
    market_id: str = "mkt-1",
    f_final: float = 0.05,
    fill_price: float = 0.50,
    available_liquidity: float = 100_000.0,
) -> tuple[OracleState, object]:
    return broker.execute(
        state=state,
        market_id=market_id,
        question="Will X happen?",
        direction="back",
        f_final=f_final,
        fill_price=fill_price,
        edge=0.05,
        p_fair=0.55,
        kelly_f_star=0.10,
        kelly_f_final=f_final,
        conf_score=70.0,
        uncertainty_penalty=0.30,
        available_liquidity=available_liquidity,
    )


def _execute_lay(
    broker: PaperBroker,
    state: OracleState,
    market_id: str = "mkt-lay",
    f_final: float = 0.05,
    fill_price: float = 0.50,
    available_liquidity: float = 100_000.0,
) -> tuple[OracleState, object]:
    return broker.execute(
        state=state,
        market_id=market_id,
        question="Will X happen?",
        direction="lay",
        f_final=f_final,
        fill_price=fill_price,
        edge=0.05,
        p_fair=0.45,
        kelly_f_star=0.10,
        kelly_f_final=f_final,
        conf_score=70.0,
        uncertainty_penalty=0.30,
        available_liquidity=available_liquidity,
    )


# ---------------------------------------------------------------------------
# derive_spread
# ---------------------------------------------------------------------------

class TestDeriveSpread:
    def test_p_ask_above_probability(self):
        p_ask, _ = PaperBroker.derive_spread(0.50)
        assert p_ask == pytest.approx(0.50 + _HALF_SPREAD)

    def test_p_bid_below_probability(self):
        _, p_bid = PaperBroker.derive_spread(0.50)
        assert p_bid == pytest.approx(0.50 - _HALF_SPREAD)

    def test_p_ask_clamped_at_high_probability(self):
        p_ask, _ = PaperBroker.derive_spread(0.995)
        assert p_ask <= 0.99

    def test_p_bid_clamped_at_low_probability(self):
        _, p_bid = PaperBroker.derive_spread(0.005)
        assert p_bid >= 0.01

    def test_spread_symmetric(self):
        p_ask, p_bid = PaperBroker.derive_spread(0.60)
        assert p_ask - 0.60 == pytest.approx(0.60 - p_bid, abs=1e-9)


# ---------------------------------------------------------------------------
# derive_conf_score
# ---------------------------------------------------------------------------

class TestDeriveConfScore:
    def test_zero_uncertainty_gives_base_conf(self):
        assert PaperBroker.derive_conf_score(0.0) == pytest.approx(_BASE_CONF)

    def test_full_uncertainty_gives_zero(self):
        assert PaperBroker.derive_conf_score(1.0) == pytest.approx(0.0)

    def test_mid_uncertainty(self):
        assert PaperBroker.derive_conf_score(0.30) == pytest.approx(70.0)

    def test_half_uncertainty(self):
        assert PaperBroker.derive_conf_score(0.50) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# execute — full fill
# ---------------------------------------------------------------------------

class TestExecuteFullFill:
    def test_full_fill_within_liquidity(self, broker, fresh_state):
        """Large liquidity → fill_pct = 1.0 → filled_size == f_final."""
        state, trade = _execute_back(broker, fresh_state, f_final=0.05, available_liquidity=100_000.0)
        assert trade is not None
        assert trade.filled_size == pytest.approx(0.05)

    def test_bankroll_decreases_by_stake(self, broker, fresh_state):
        """For a back trade, bankroll decreases by stake_abs = filled_size * bankroll."""
        state, trade = _execute_back(broker, fresh_state, f_final=0.05, fill_price=0.50,
                                     available_liquidity=100_000.0)
        expected_stake = 0.05 * DEFAULT_BANKROLL   # 50.0
        assert trade.bankroll_after == pytest.approx(DEFAULT_BANKROLL - expected_stake)
        assert state.bankroll == pytest.approx(DEFAULT_BANKROLL - expected_stake)

    def test_position_opened_after_execute(self, broker, fresh_state):
        state, trade = _execute_back(broker, fresh_state, market_id="mkt-42")
        assert "mkt-42" in state.positions

    def test_trade_appended_to_history(self, broker, fresh_state):
        state, trade = _execute_back(broker, fresh_state)
        assert len(state.trade_history) == 1
        assert state.trade_history[0].trade_id == trade.trade_id

    def test_trade_status_is_open(self, broker, fresh_state):
        _, trade = _execute_back(broker, fresh_state)
        assert trade.status == "open"

    def test_stake_abs_stored_on_trade(self, broker, fresh_state):
        _, trade = _execute_back(broker, fresh_state, f_final=0.05, fill_price=0.50,
                                  available_liquidity=100_000.0)
        assert trade.stake_abs == pytest.approx(0.05 * DEFAULT_BANKROLL)


# ---------------------------------------------------------------------------
# execute — partial fill
# ---------------------------------------------------------------------------

class TestExecutePartialFill:
    def test_partial_fill_when_exceeds_liquidity(self, broker, fresh_state):
        """Tiny liquidity → partial fill in [60%, 90%] of requested."""
        # requested_abs = 0.10 * 1000 = 100; liquidity_floor = 10 * 0.70 = 7 < 100
        state, trade = _execute_back(
            broker, fresh_state, f_final=0.10, available_liquidity=10.0
        )
        assert trade is not None
        assert trade.filled_size < 0.10
        assert trade.filled_size >= 0.10 * 0.60 - 1e-9
        assert trade.filled_size <= 0.10 * 0.90 + 1e-9

    def test_partial_fill_still_opens_position(self, broker, fresh_state):
        state, trade = _execute_back(
            broker, fresh_state, f_final=0.10, available_liquidity=10.0
        )
        assert trade is not None
        assert "mkt-1" in state.positions


# ---------------------------------------------------------------------------
# execute — edge cases
# ---------------------------------------------------------------------------

class TestExecuteEdgeCases:
    def test_returns_none_trade_when_bankroll_zero(self, broker):
        state = OracleState(bankroll=0.0, peak_bankroll=DEFAULT_BANKROLL)
        result_state, trade = _execute_back(broker, state)
        assert trade is None
        assert result_state.bankroll == pytest.approx(0.0)

    def test_lay_execute_sets_liability(self, broker, fresh_state):
        """For a lay trade, bankroll decreases by liability_abs, not stake_abs."""
        state, trade = broker.execute(
            state=fresh_state,
            market_id="mkt-lay",
            question="Will X happen?",
            direction="lay",
            f_final=0.05,
            fill_price=0.50,
            edge=0.05,
            p_fair=0.45,
            kelly_f_star=0.10,
            kelly_f_final=0.05,
            conf_score=70.0,
            uncertainty_penalty=0.30,
            available_liquidity=100_000.0,
        )
        assert trade is not None
        # For lay: liability_abs = filled_size * bankroll = 0.05 * 1000 = 50
        expected_liability = 0.05 * DEFAULT_BANKROLL
        assert state.bankroll == pytest.approx(DEFAULT_BANKROLL - expected_liability)
        pos = state.positions["mkt-lay"]
        assert pos.liability_abs == pytest.approx(expected_liability)


# ---------------------------------------------------------------------------
# settle_position
# ---------------------------------------------------------------------------

class TestSettlePosition:
    def _setup_back_position(
        self,
        broker: PaperBroker,
        state: OracleState,
        entry_price: float = 0.50,
        f_final: float = 0.05,
    ) -> tuple[OracleState, str]:
        state, trade = _execute_back(
            broker, state, fill_price=entry_price, f_final=f_final,
            available_liquidity=100_000.0
        )
        return state, "mkt-1"

    def test_settle_back_yes_pnl(self, broker, fresh_state):
        """back + YES: gross = stake * (1/price - 1); pnl = gross * (1 - commission).
        Escrow: bankroll += stake (returned) + pnl."""
        state, market_id = self._setup_back_position(broker, fresh_state,
                                                      entry_price=0.50, f_final=0.05)
        stake_abs = 0.05 * DEFAULT_BANKROLL   # 50.0
        bankroll_pre_settle = state.bankroll   # 950.0

        state, trade = broker.settle_position(state, market_id, "YES", 1.0)

        gross = stake_abs * (1.0 / 0.50 - 1.0)          # 50 * 1 = 50
        expected_pnl = gross * (1.0 - 0.05)             # 50 * 0.95 = 47.5
        assert trade.pnl == pytest.approx(expected_pnl, rel=1e-4)
        # Escrow return: bankroll += cost + pnl = 50 + 47.5 = 97.5
        assert state.bankroll == pytest.approx(bankroll_pre_settle + stake_abs + expected_pnl, rel=1e-4)

    def test_settle_back_no_pnl(self, broker, fresh_state):
        """back + NO: pnl = -stake_abs. Escrow: cost + pnl = stake + (-stake) = 0."""
        state, market_id = self._setup_back_position(broker, fresh_state,
                                                      entry_price=0.50, f_final=0.05)
        stake_abs = 0.05 * DEFAULT_BANKROLL
        bankroll_pre = state.bankroll   # 950

        state, trade = broker.settle_position(state, market_id, "NO", 0.0)

        assert trade.pnl == pytest.approx(-stake_abs, rel=1e-4)
        # Escrow consumed: cost + pnl = 50 + (-50) = 0 → bankroll unchanged
        assert state.bankroll == pytest.approx(bankroll_pre, rel=1e-4)

    def test_settle_lay_no_pnl(self, broker, fresh_state):
        """lay + NO: pnl = stake_abs * (1 - commission_pct).
        Escrow: bankroll += liability (returned) + pnl."""
        # Set up a lay position
        state, trade = broker.execute(
            state=fresh_state,
            market_id="mkt-lay",
            question="Will X?",
            direction="lay",
            f_final=0.05,
            fill_price=0.50,
            edge=0.05,
            p_fair=0.45,
            kelly_f_star=0.10,
            kelly_f_final=0.05,
            conf_score=70.0,
            uncertainty_penalty=0.30,
            available_liquidity=100_000.0,
        )
        assert trade is not None
        stake_abs = trade.stake_abs
        liability_abs = state.positions["mkt-lay"].liability_abs
        bankroll_pre = state.bankroll

        state, settled = broker.settle_position(state, "mkt-lay", "NO", 0.0)

        expected_pnl = stake_abs * (1.0 - 0.05)
        assert settled.pnl == pytest.approx(expected_pnl, rel=1e-4)
        # Escrow return: bankroll += liability + pnl
        assert state.bankroll == pytest.approx(bankroll_pre + liability_abs + expected_pnl, rel=1e-4)

    def test_settle_lay_yes_pnl(self, broker, fresh_state):
        """lay + YES: pnl = -liability_abs. Escrow: cost + pnl = liability + (-liability) = 0."""
        state, trade = broker.execute(
            state=fresh_state,
            market_id="mkt-lay",
            question="Will X?",
            direction="lay",
            f_final=0.05,
            fill_price=0.50,
            edge=0.05,
            p_fair=0.45,
            kelly_f_star=0.10,
            kelly_f_final=0.05,
            conf_score=70.0,
            uncertainty_penalty=0.30,
            available_liquidity=100_000.0,
        )
        pos = state.positions["mkt-lay"]
        liability_abs = pos.liability_abs
        bankroll_pre = state.bankroll

        state, settled = broker.settle_position(state, "mkt-lay", "YES", 1.0)

        assert settled.pnl == pytest.approx(-liability_abs, rel=1e-4)
        # Escrow consumed: cost + pnl = 0 → bankroll unchanged
        assert state.bankroll == pytest.approx(bankroll_pre, rel=1e-4)

    def test_settle_removes_position(self, broker, fresh_state):
        state, _ = self._setup_back_position(broker, fresh_state)
        assert "mkt-1" in state.positions
        state, _ = broker.settle_position(state, "mkt-1", "YES", 1.0)
        assert "mkt-1" not in state.positions

    def test_settle_marks_trade_settled(self, broker, fresh_state):
        state, _ = self._setup_back_position(broker, fresh_state)
        state, settled = broker.settle_position(state, "mkt-1", "YES", 1.0)
        assert settled.status == "settled"

    def test_settle_commission_paid_recorded(self, broker, fresh_state):
        state, _ = self._setup_back_position(broker, fresh_state, entry_price=0.50, f_final=0.05)
        state, settled = broker.settle_position(state, "mkt-1", "YES", 1.0)
        assert settled.commission_paid is not None
        assert settled.commission_paid > 0.0

    def test_settle_no_commission_on_loss(self, broker, fresh_state):
        state, _ = self._setup_back_position(broker, fresh_state)
        state, settled = broker.settle_position(state, "mkt-1", "NO", 0.0)
        assert settled.commission_paid == pytest.approx(0.0)

    def test_settle_lay_no_at_low_price(self, broker, fresh_state):
        """lay + NO at p=0.30: stake_abs != liability_abs, pnl must use stake_abs."""
        state, trade = broker.execute(
            state=fresh_state,
            market_id="mkt-lay-low",
            question="Will X?",
            direction="lay",
            f_final=0.05,
            fill_price=0.30,
            edge=0.05,
            p_fair=0.25,
            kelly_f_star=0.10,
            kelly_f_final=0.05,
            conf_score=70.0,
            uncertainty_penalty=0.30,
            available_liquidity=100_000.0,
        )
        assert trade is not None
        pos = state.positions["mkt-lay-low"]
        stake_abs = pos.stake_abs
        liability_abs = pos.liability_abs
        # At p=0.30: stake_abs = liability / (1/0.30 - 1) = liability / 2.333
        # So stake_abs < liability_abs — they are NOT equal
        assert stake_abs < liability_abs * 0.99, "At p=0.30 stake should be much less than liability"

        bankroll_pre = state.bankroll
        state, settled = broker.settle_position(state, "mkt-lay-low", "NO", 0.0)

        expected_pnl = stake_abs * (1.0 - 0.05)
        assert settled.pnl == pytest.approx(expected_pnl, rel=1e-4)
        # Must NOT equal liability_abs * (1 - 0.05) — that would be the old wrong formula
        wrong_pnl = liability_abs * (1.0 - 0.05)
        assert abs(settled.pnl - wrong_pnl) > 1.0, "P&L must use stake_abs, not liability_abs"
        # Escrow return: bankroll += liability + pnl
        assert state.bankroll == pytest.approx(bankroll_pre + liability_abs + expected_pnl, rel=1e-4)


    def test_settle_void_refunds_escrow(self, broker, fresh_state):
        """VOID resolution: pnl=0, escrow returned in full (back)."""
        state, _ = self._setup_back_position(broker, fresh_state, entry_price=0.50, f_final=0.05)
        stake_abs = 0.05 * DEFAULT_BANKROLL
        bankroll_pre = state.bankroll  # 950

        state, settled = broker.settle_position(state, "mkt-1", "VOID", 0.0)

        assert settled.pnl == pytest.approx(0.0)
        assert settled.commission_paid == pytest.approx(0.0)
        # Escrow fully returned: 950 + 50 + 0 = 1000
        assert state.bankroll == pytest.approx(bankroll_pre + stake_abs, rel=1e-4)

    def test_settle_void_lay_refunds_liability(self, broker, fresh_state):
        """VOID resolution: pnl=0, lay escrow (liability) returned in full."""
        state, trade = broker.execute(
            state=fresh_state,
            market_id="mkt-lay-void",
            question="Will X?",
            direction="lay",
            f_final=0.05,
            fill_price=0.50,
            edge=0.05,
            p_fair=0.45,
            kelly_f_star=0.10,
            kelly_f_final=0.05,
            conf_score=70.0,
            uncertainty_penalty=0.30,
            available_liquidity=100_000.0,
        )
        assert trade is not None
        liability_abs = state.positions["mkt-lay-void"].liability_abs
        bankroll_pre = state.bankroll

        state, settled = broker.settle_position(state, "mkt-lay-void", "VOID", 0.0)

        assert settled.pnl == pytest.approx(0.0)
        assert state.bankroll == pytest.approx(bankroll_pre + liability_abs, rel=1e-4)

    def test_settle_lay_mkt_commission_on_win_portion(self, broker, fresh_state):
        """lay + MKT: commission applies to win portion even when net P&L is negative."""
        state, trade = broker.execute(
            state=fresh_state,
            market_id="mkt-lay-mkt",
            question="Will X?",
            direction="lay",
            f_final=0.05,
            fill_price=0.30,
            edge=0.05,
            p_fair=0.25,
            kelly_f_star=0.10,
            kelly_f_final=0.05,
            conf_score=70.0,
            uncertainty_penalty=0.30,
            available_liquidity=100_000.0,
        )
        assert trade is not None
        pos = state.positions["mkt-lay-mkt"]
        stake_abs = pos.stake_abs
        liability_abs = pos.liability_abs
        bankroll_pre = state.bankroll

        # Settle at 0.50 — partial outcome
        state, settled = broker.settle_position(state, "mkt-lay-mkt", "MKT", 0.50)

        # Correct formula: win_portion = stake * (1 - 0.5), loss_portion = liability * 0.5
        win_portion = stake_abs * 0.5
        loss_portion = liability_abs * 0.5
        expected_commission = win_portion * 0.05
        expected_pnl = win_portion - expected_commission - loss_portion
        assert settled.pnl == pytest.approx(expected_pnl, rel=1e-4)
        assert settled.commission_paid == pytest.approx(expected_commission, rel=1e-4)


# ---------------------------------------------------------------------------
# cancel_position
# ---------------------------------------------------------------------------

class TestCancelPosition:
    def test_cancel_back_refunds_stake(self, broker, fresh_state):
        """Cancelling a back position refunds stake_abs to bankroll."""
        state, trade = _execute_back(broker, fresh_state, market_id="mkt-c1")
        stake_abs = state.positions["mkt-c1"].stake_abs
        bankroll_pre = state.bankroll

        state, cancelled = broker.cancel_position(state, "mkt-c1", reason="test")

        assert state.bankroll == pytest.approx(bankroll_pre + stake_abs, rel=1e-4)
        assert cancelled.pnl == 0.0

    def test_cancel_lay_refunds_liability(self, broker, fresh_state):
        """Cancelling a lay position refunds liability_abs to bankroll."""
        state, trade = _execute_lay(broker, fresh_state, market_id="mkt-c2")
        liability_abs = state.positions["mkt-c2"].liability_abs
        bankroll_pre = state.bankroll

        state, cancelled = broker.cancel_position(state, "mkt-c2", reason="test")

        assert state.bankroll == pytest.approx(bankroll_pre + liability_abs, rel=1e-4)
        assert cancelled.pnl == 0.0

    def test_cancel_sets_status_and_timestamp(self, broker, fresh_state):
        """Cancelled trade has status='cancelled' and exit_timestamp set."""
        state, _ = _execute_back(broker, fresh_state, market_id="mkt-c3")
        state, cancelled = broker.cancel_position(state, "mkt-c3")

        assert cancelled.status == "cancelled"
        assert cancelled.exit_timestamp is not None

    def test_cancel_removes_position(self, broker, fresh_state):
        """Position is removed from state after cancellation."""
        state, _ = _execute_back(broker, fresh_state, market_id="mkt-c4")
        assert "mkt-c4" in state.positions

        state, _ = broker.cancel_position(state, "mkt-c4")
        assert "mkt-c4" not in state.positions

    def test_cancel_captures_clv(self, broker, fresh_state):
        """CLV is computed from last_seen_price when available."""
        state, _ = _execute_back(broker, fresh_state, market_id="mkt-c5", fill_price=0.50)
        # Simulate a price update during scan cycle
        state.positions["mkt-c5"].last_seen_price = 0.55

        state, cancelled = broker.cancel_position(state, "mkt-c5")

        assert cancelled.closing_price == 0.55
        # back CLV = closing_price - entry_price = 0.55 - 0.50 = 0.05
        assert cancelled.clv == pytest.approx(0.05, abs=1e-5)

    def test_cancel_clv_none_when_no_last_seen(self, broker, fresh_state):
        """CLV stays None when last_seen_price was never set."""
        state, _ = _execute_back(broker, fresh_state, market_id="mkt-c6")
        assert state.positions["mkt-c6"].last_seen_price is None

        state, cancelled = broker.cancel_position(state, "mkt-c6")

        assert cancelled.closing_price is None
        assert cancelled.clv is None

    def test_cancel_nonexistent_raises(self, broker, fresh_state):
        """Cancelling a non-existent position raises KeyError."""
        with pytest.raises(KeyError):
            broker.cancel_position(fresh_state, "no-such-market")


