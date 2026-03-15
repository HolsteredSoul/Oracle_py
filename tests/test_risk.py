"""Unit tests for src/risk/manager.py."""

from unittest.mock import MagicMock
from src.risk.manager import check_risk_gates


def make_config(confidence_floor=52, max_exposure=0.85):
    cfg = MagicMock()
    cfg.risk.confidence_floor = confidence_floor
    cfg.risk.max_exposure = max_exposure
    return cfg


def _ok_args(cfg):
    """Return kwargs that pass all gates."""
    return dict(
        f=0.05,
        conf_score=60.0,
        current_exposure=0.20,
        drawdown_pct=0.10,
        config=cfg,
    )


def test_approved_path_returns_true_and_ok():
    cfg = make_config()
    approved, reason = check_risk_gates(**_ok_args(cfg))
    assert approved is True
    assert reason == "OK"


def test_gate1_low_confidence_fires():
    cfg = make_config(confidence_floor=52)
    approved, reason = check_risk_gates(
        f=0.05, conf_score=40.0, current_exposure=0.10, drawdown_pct=0.0, config=cfg
    )
    assert approved is False
    assert "confidence" in reason.lower()


def test_gate1_exactly_at_floor_passes():
    cfg = make_config(confidence_floor=52)
    approved, _ = check_risk_gates(
        f=0.05, conf_score=52.0, current_exposure=0.10, drawdown_pct=0.0, config=cfg
    )
    assert approved is True


def test_gate2_exposure_at_cap_fires():
    cfg = make_config(max_exposure=0.85)
    approved, reason = check_risk_gates(
        f=0.05, conf_score=60.0, current_exposure=0.85, drawdown_pct=0.0, config=cfg
    )
    assert approved is False
    assert "cap" in reason.lower()


def test_gate3_trade_would_breach_cap_fires():
    cfg = make_config(max_exposure=0.85)
    # 0.80 + 0.10 = 0.90 > 0.85
    approved, reason = check_risk_gates(
        f=0.10, conf_score=60.0, current_exposure=0.80, drawdown_pct=0.0, config=cfg
    )
    assert approved is False
    assert "breach" in reason.lower()


def test_gates_fire_independently_gate1_only():
    """When only confidence is low, gate 1 fires regardless of exposure state."""
    cfg = make_config(confidence_floor=52, max_exposure=0.85)
    approved, reason = check_risk_gates(
        f=0.01, conf_score=30.0, current_exposure=0.10, drawdown_pct=0.0, config=cfg
    )
    assert approved is False
    assert "confidence" in reason.lower()


def test_gates_fire_independently_gate2_only():
    """When only exposure is at cap, gate 2 fires regardless of conf."""
    cfg = make_config(confidence_floor=52, max_exposure=0.85)
    approved, reason = check_risk_gates(
        f=0.01, conf_score=80.0, current_exposure=0.85, drawdown_pct=0.0, config=cfg
    )
    assert approved is False
    assert "cap" in reason.lower()
