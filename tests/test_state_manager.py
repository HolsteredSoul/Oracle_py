"""Tests for src/storage/state_manager.py.

All file IO uses the tmp_path pytest fixture — no side-effects on the real
state/ directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.storage.state_manager import (
    DEFAULT_BANKROLL,
    OracleState,
    Position,
    StateManager,
    Trade,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_position(market_id: str = "mkt-1", filled_size: float = 0.05) -> Position:
    return Position(
        market_id=market_id,
        question="Will X happen?",
        direction="back",
        entry_price=0.50,
        filled_size=filled_size,
        stake_abs=filled_size * DEFAULT_BANKROLL,
        liability_abs=0.0,
        entry_timestamp="2026-03-15T00:00:00+00:00",
        trade_id="trade-1",
        p_fair_at_entry=0.55,
    )


def _make_trade(trade_id: str = "trade-1", filled_size: float = 0.05) -> Trade:
    bankroll = DEFAULT_BANKROLL
    stake = filled_size * bankroll
    return Trade(
        trade_id=trade_id,
        timestamp="2026-03-15T00:00:00+00:00",
        market_id="mkt-1",
        question="Will X happen?",
        direction="back",
        requested_size=filled_size,
        filled_size=filled_size,
        fill_price=0.50,
        edge=0.05,
        p_fair=0.55,
        conf_score=70.0,
        uncertainty_penalty=0.30,
        kelly_f_star=0.10,
        kelly_f_final=filled_size,
        bankroll_before=bankroll,
        bankroll_after=bankroll - stake,
        status="open",
        stake_abs=stake,
    )


def _make_sm(tmp_path: Path) -> StateManager:
    return StateManager(path=tmp_path / "oracle_state.json")


# ---------------------------------------------------------------------------
# load() — missing file
# ---------------------------------------------------------------------------

class TestStateManagerLoad:
    def test_load_returns_fresh_state_when_file_absent(self, tmp_path):
        sm = _make_sm(tmp_path)
        state = sm.load()
        assert state.bankroll == DEFAULT_BANKROLL
        assert state.positions == {}
        assert state.trade_history == []

    def test_load_survives_corrupt_json(self, tmp_path):
        path = tmp_path / "oracle_state.json"
        path.write_text("not-json", encoding="utf-8")
        sm = StateManager(path=path)
        state = sm.load()
        assert state.bankroll == DEFAULT_BANKROLL

    def test_load_survives_truncated_write(self, tmp_path):
        """Simulates a crash mid-write: main file contains truncated bytes."""
        path = tmp_path / "oracle_state.json"
        path.write_bytes(b'{"bankroll": 750')   # truncated — invalid JSON
        sm = StateManager(path=path)
        state = sm.load()
        assert state.bankroll == DEFAULT_BANKROLL   # recovers, does not use partial data

    def test_load_returns_saved_state(self, tmp_path):
        sm = _make_sm(tmp_path)
        original = OracleState(bankroll=750.0, peak_bankroll=750.0)
        sm.save(original)
        loaded = sm.load()
        assert loaded.bankroll == pytest.approx(750.0)
        assert loaded.peak_bankroll == pytest.approx(750.0)


# ---------------------------------------------------------------------------
# save() — atomicity and metadata
# ---------------------------------------------------------------------------

class TestStateManagerSave:
    def test_save_atomic_no_tmp_left(self, tmp_path):
        sm = _make_sm(tmp_path)
        sm.save(OracleState())
        tmp = (tmp_path / "oracle_state.json").with_suffix(".tmp")
        assert not tmp.exists(), ".tmp file should be removed after save()"

    def test_save_updates_last_updated(self, tmp_path):
        sm = _make_sm(tmp_path)
        state = OracleState()
        old_ts = state.last_updated
        sm.save(state)
        # last_updated is set inside save(); reload to confirm persistence
        loaded = sm.load()
        # The timestamp should be a valid ISO string (not the original default)
        assert loaded.last_updated >= old_ts

    def test_save_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "oracle_state.json"
        sm = StateManager(path=nested)
        sm.save(OracleState())
        assert nested.exists()

    def test_save_then_load_round_trip(self, tmp_path):
        sm = _make_sm(tmp_path)
        pos = _make_position()
        state = OracleState(bankroll=800.0)
        state.positions["mkt-1"] = pos
        state.priors["mkt-1"] = 0.55
        sm.save(state)

        loaded = sm.load()
        assert loaded.bankroll == pytest.approx(800.0)
        assert "mkt-1" in loaded.positions
        assert loaded.priors["mkt-1"] == pytest.approx(0.55)

    def test_save_persists_trade_history(self, tmp_path):
        sm = _make_sm(tmp_path)
        trade = _make_trade()
        state = OracleState()
        state.trade_history.append(trade)
        sm.save(state)

        loaded = sm.load()
        assert len(loaded.trade_history) == 1
        assert loaded.trade_history[0].trade_id == "trade-1"


# ---------------------------------------------------------------------------
# add_trade / update_position
# ---------------------------------------------------------------------------

class TestStateManagerMutations:
    def test_add_trade_appends_to_history(self, tmp_path):
        sm = _make_sm(tmp_path)
        state = OracleState()
        trade = _make_trade()
        state = sm.add_trade(state, trade)
        assert len(state.trade_history) == 1

    def test_add_trade_saves_to_disk(self, tmp_path):
        path = tmp_path / "oracle_state.json"
        sm = StateManager(path=path)
        state = OracleState()
        sm.add_trade(state, _make_trade())
        assert path.exists()
        on_disk = json.loads(path.read_text())
        assert len(on_disk["trade_history"]) == 1

    def test_update_position_upserts(self, tmp_path):
        sm = _make_sm(tmp_path)
        state = OracleState()
        pos = _make_position("mkt-42")
        state = sm.update_position(state, "mkt-42", pos)
        assert "mkt-42" in state.positions

    def test_update_position_removes_when_none(self, tmp_path):
        sm = _make_sm(tmp_path)
        state = OracleState()
        state.positions["mkt-42"] = _make_position("mkt-42")
        state = sm.update_position(state, "mkt-42", None)
        assert "mkt-42" not in state.positions

    def test_update_position_noop_remove_missing_key(self, tmp_path):
        """Removing a key that doesn't exist should not raise."""
        sm = _make_sm(tmp_path)
        state = OracleState()
        state = sm.update_position(state, "nonexistent", None)   # must not raise
        assert state.positions == {}


# ---------------------------------------------------------------------------
# current_exposure
# ---------------------------------------------------------------------------

class TestCurrentExposure:
    def test_empty_positions_is_zero(self, tmp_path):
        sm = _make_sm(tmp_path)
        assert sm.current_exposure(OracleState()) == pytest.approx(0.0)

    def test_sums_filled_sizes(self, tmp_path):
        sm = _make_sm(tmp_path)
        state = OracleState()
        state.positions["mkt-1"] = _make_position("mkt-1", filled_size=0.05)
        state.positions["mkt-2"] = _make_position("mkt-2", filled_size=0.10)
        # Simulate bankroll reduction from escrowing stakes (as production does)
        state.bankroll -= 0.05 * DEFAULT_BANKROLL + 0.10 * DEFAULT_BANKROLL
        # exposure = total_risk / equity = 150 / 1000 = 0.15
        assert sm.current_exposure(state) == pytest.approx(0.15)

    def test_single_position(self, tmp_path):
        sm = _make_sm(tmp_path)
        state = OracleState()
        state.positions["mkt-1"] = _make_position(filled_size=0.08)
        state.bankroll -= 0.08 * DEFAULT_BANKROLL
        # exposure = 80 / 1000 = 0.08
        assert sm.current_exposure(state) == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# drawdown_pct
# ---------------------------------------------------------------------------

class TestDrawdownPct:
    def test_zero_when_bankroll_equals_peak(self, tmp_path):
        sm = _make_sm(tmp_path)
        state = OracleState(bankroll=1000.0, peak_bankroll=1000.0)
        assert sm.drawdown_pct(state) == pytest.approx(0.0)

    def test_drawdown_computed_correctly(self, tmp_path):
        sm = _make_sm(tmp_path)
        state = OracleState(bankroll=800.0, peak_bankroll=1000.0)
        assert sm.drawdown_pct(state) == pytest.approx(0.20)

    def test_drawdown_50_percent(self, tmp_path):
        sm = _make_sm(tmp_path)
        state = OracleState(bankroll=500.0, peak_bankroll=1000.0)
        assert sm.drawdown_pct(state) == pytest.approx(0.50)

    def test_peak_updated_when_bankroll_exceeds_peak(self, tmp_path):
        sm = _make_sm(tmp_path)
        state = OracleState(bankroll=1100.0, peak_bankroll=1000.0)
        dd = sm.drawdown_pct(state)
        assert dd == pytest.approx(0.0)
        assert state.peak_bankroll == pytest.approx(1100.0)

    def test_returns_zero_not_negative(self, tmp_path):
        """Drawdown is never negative even if bankroll slightly above peak due to float."""
        sm = _make_sm(tmp_path)
        state = OracleState(bankroll=1000.0000001, peak_bankroll=1000.0)
        assert sm.drawdown_pct(state) >= 0.0
