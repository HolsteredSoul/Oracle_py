"""Unit tests for src/strategy/kelly.py."""

import pytest
from unittest.mock import MagicMock
from src.strategy.kelly import (
    commission_aware_kelly,
    apply_oracle_sizing,
    translate_to_betfair,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(
    kelly_base_fraction=0.50,
    drawdown_throttle_pct=0.20,
    drawdown_throttle_factor=0.50,
    kelly_hard_cap=0.10,
):
    cfg = MagicMock()
    cfg.risk.kelly_base_fraction = kelly_base_fraction
    cfg.risk.drawdown_throttle_pct = drawdown_throttle_pct
    cfg.risk.drawdown_throttle_factor = drawdown_throttle_factor
    cfg.risk.kelly_hard_cap = kelly_hard_cap
    return cfg


# ---------------------------------------------------------------------------
# commission_aware_kelly
# ---------------------------------------------------------------------------

def test_back_zero_edge_returns_zero():
    # p_fair == q_market + gamma → numerator = 0
    gamma = 0.05
    q = 0.50
    p_fair = q + gamma
    f = commission_aware_kelly(p_fair, q, gamma, "back")
    assert f == pytest.approx(0.0)


def test_back_negative_edge_returns_negative():
    f = commission_aware_kelly(0.45, 0.55, 0.05, "back")
    assert f < 0


def test_back_no_commission_standard_kelly():
    # Standard Kelly: f = (p*b - q) / b where b = (1-q)/q, odds-based
    # Simplifies to: f = (p_fair - q_market) / (1 - q_market)
    p, q = 0.60, 0.50
    f = commission_aware_kelly(p, q, gamma=0.0, direction="back")
    expected = (p - q) / (1 - q)
    assert f == pytest.approx(expected)


def test_lay_zero_edge_returns_zero():
    gamma = 0.05
    q = 0.50
    p_fair = q - gamma
    f = commission_aware_kelly(p_fair, q, gamma, "lay")
    assert f == pytest.approx(0.0)


def test_lay_negative_edge_returns_negative():
    f = commission_aware_kelly(0.55, 0.45, 0.05, "lay")
    assert f < 0


# ---------------------------------------------------------------------------
# apply_oracle_sizing
# ---------------------------------------------------------------------------

def test_hard_cap_enforced():
    cfg = make_config(kelly_base_fraction=1.0)
    # f_star=10 would give 10*1.0*1.0*1.0 = 10 without cap → clamped to 0.10
    result = apply_oracle_sizing(10.0, conf_score=100, drawdown_pct=0.0, config=cfg)
    assert result == pytest.approx(0.10)


def test_drawdown_above_threshold_halves_output():
    cfg = make_config()
    # Use small f_star so results stay below the 0.10 hard cap
    no_dd = apply_oracle_sizing(0.05, conf_score=100, drawdown_pct=0.0, config=cfg)
    with_dd = apply_oracle_sizing(0.05, conf_score=100, drawdown_pct=0.25, config=cfg)
    assert with_dd == pytest.approx(no_dd * 0.50, rel=1e-6)


def test_conf_score_50_gives_half_lambda_conf():
    # Use small f_star so neither result hits the 0.10 hard cap
    cfg = make_config(kelly_base_fraction=1.0)
    result_50 = apply_oracle_sizing(0.05, conf_score=50, drawdown_pct=0.0, config=cfg)
    result_100 = apply_oracle_sizing(0.05, conf_score=100, drawdown_pct=0.0, config=cfg)
    # λ_conf at 50 = 0.5, at 100 = 1.0 → ratio should be 0.5
    assert result_50 == pytest.approx(result_100 * 0.5, rel=1e-6)


def test_conf_score_below_50_clamped_to_50():
    cfg = make_config(kelly_base_fraction=1.0)
    result_0 = apply_oracle_sizing(1.0, conf_score=0, drawdown_pct=0.0, config=cfg)
    result_50 = apply_oracle_sizing(1.0, conf_score=50, drawdown_pct=0.0, config=cfg)
    assert result_0 == pytest.approx(result_50)


# ---------------------------------------------------------------------------
# translate_to_betfair
# ---------------------------------------------------------------------------

def test_back_translation():
    result = translate_to_betfair(f=0.10, bankroll=1000.0, decimal_odds=3.0, direction="back")
    assert result["stake"] == pytest.approx(100.0)
    assert result["liability"] == pytest.approx(200.0)


def test_lay_translation_stake_equals_liability_over_odds_minus_one():
    result = translate_to_betfair(f=0.10, bankroll=1000.0, decimal_odds=3.0, direction="lay")
    # liability = 0.10 * 1000 = 100; stake = 100 / (3-1) = 50
    assert result["liability"] == pytest.approx(100.0)
    assert result["stake"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Property-based: f_final never exceeds 0.25
# ---------------------------------------------------------------------------

try:
    from hypothesis import given, settings as h_settings
    from hypothesis import strategies as st

    @given(
        f_star=st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False),
        conf_score=st.floats(0.0, 100.0, allow_nan=False),
        drawdown_pct=st.floats(0.0, 1.0, allow_nan=False),
    )
    @h_settings(max_examples=500)
    def test_f_final_never_exceeds_hard_cap(f_star, conf_score, drawdown_pct):
        cfg = make_config()
        result = apply_oracle_sizing(f_star, conf_score, drawdown_pct, cfg)
        assert result <= 0.25

except ImportError:
    pass  # hypothesis not installed; property test skipped
