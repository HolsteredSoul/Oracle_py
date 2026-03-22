"""Tests for the post-cycle settlement validator in main.py."""

from __future__ import annotations

import logging

import pytest

from main import _validate_settlements
from src.storage.state_manager import Trade


def _make_trade(**overrides) -> Trade:
    """Create a minimal settled Trade with sensible defaults."""
    defaults = dict(
        trade_id="t-1",
        timestamp="2026-03-22T12:00:00+00:00",
        market_id="mkt-1",
        question="Test Market",
        direction="back",
        requested_size=0.05,
        filled_size=0.05,
        fill_price=0.50,
        edge=0.10,
        p_fair=0.60,
        conf_score=70.0,
        uncertainty_penalty=0.30,
        kelly_f_star=0.10,
        kelly_f_final=0.05,
        bankroll_before=1000.0,
        bankroll_after=950.0,
        status="settled",
        exit_price=1.0,
        pnl=47.5,
        exit_timestamp="2026-03-22T14:00:00+00:00",
        commission_paid=2.5,
        stake_abs=50.0,
        resolution="YES",
        runner_status="WINNER",
    )
    defaults.update(overrides)
    return Trade(**defaults)


class TestSettlementValidator:
    """Validator should log warnings for suspicious settlements."""

    def test_clean_settlement_no_warnings(self, caplog):
        """A proper YES settlement with WINNER status should not warn."""
        trade = _make_trade(resolution="YES", runner_status="WINNER", pnl=47.5)
        with caplog.at_level(logging.WARNING):
            _validate_settlements([trade])
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 0

    def test_mkt_fallback_warns(self, caplog):
        """MKT resolution should trigger a warning."""
        trade = _make_trade(resolution="MKT", runner_status="", pnl=5.0)
        with caplog.at_level(logging.WARNING):
            _validate_settlements([trade])
        assert any("MKT fallback" in r.message for r in caplog.records)

    def test_none_resolution_warns(self, caplog):
        """None resolution should trigger a warning."""
        trade = _make_trade(resolution=None, runner_status=None, pnl=5.0)
        with caplog.at_level(logging.WARNING):
            _validate_settlements([trade])
        assert any("MKT fallback" in r.message for r in caplog.records)

    def test_exit_price_050_warns(self, caplog):
        """exit_price=0.5 is the old bug symptom — should warn."""
        trade = _make_trade(exit_price=0.5, resolution="YES", runner_status="WINNER", pnl=47.5)
        with caplog.at_level(logging.WARNING):
            _validate_settlements([trade])
        assert any("exit_price=0.500" in r.message for r in caplog.records)

    def test_empty_runner_status_warns(self, caplog):
        """Missing runner_status should warn."""
        trade = _make_trade(runner_status="", resolution="YES", pnl=47.5)
        with caplog.at_level(logging.WARNING):
            _validate_settlements([trade])
        assert any("runner_status is empty" in r.message for r in caplog.records)

    def test_back_yes_negative_pnl_warns(self, caplog):
        """back + YES should not have negative P&L."""
        trade = _make_trade(direction="back", resolution="YES", runner_status="WINNER", pnl=-10.0)
        with caplog.at_level(logging.WARNING):
            _validate_settlements([trade])
        assert any("INCONSISTENT" in r.message for r in caplog.records)

    def test_back_no_positive_pnl_warns(self, caplog):
        """back + NO should not have positive P&L."""
        trade = _make_trade(direction="back", resolution="NO", runner_status="LOSER", pnl=10.0)
        with caplog.at_level(logging.WARNING):
            _validate_settlements([trade])
        assert any("INCONSISTENT" in r.message for r in caplog.records)

    def test_lay_no_negative_pnl_warns(self, caplog):
        """lay + NO (lay wins) should not have negative P&L."""
        trade = _make_trade(direction="lay", resolution="NO", runner_status="LOSER", pnl=-10.0)
        with caplog.at_level(logging.WARNING):
            _validate_settlements([trade])
        assert any("INCONSISTENT" in r.message for r in caplog.records)

    def test_lay_yes_positive_pnl_warns(self, caplog):
        """lay + YES (lay loses) should not have positive P&L."""
        trade = _make_trade(direction="lay", resolution="YES", runner_status="WINNER", pnl=10.0)
        with caplog.at_level(logging.WARNING):
            _validate_settlements([trade])
        assert any("INCONSISTENT" in r.message for r in caplog.records)

    def test_void_no_warnings(self, caplog):
        """VOID with pnl=0 should not warn."""
        trade = _make_trade(resolution="VOID", runner_status="REMOVED", pnl=0.0, exit_price=0.0)
        with caplog.at_level(logging.WARNING):
            _validate_settlements([trade])
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 0
