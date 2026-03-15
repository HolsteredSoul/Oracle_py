"""Unit tests for src/strategy/edge.py."""

import pytest
from src.strategy.edge import executable_edge


def test_back_edge_positive_when_fair_above_ask():
    assert executable_edge(0.60, 0.55, 0.50, "back") == pytest.approx(0.05)


def test_back_edge_zero_at_breakeven():
    assert executable_edge(0.55, 0.55, 0.50, "back") == pytest.approx(0.0)


def test_back_edge_negative_when_fair_below_ask():
    assert executable_edge(0.50, 0.55, 0.45, "back") < 0


def test_lay_edge_positive_when_fair_below_bid():
    assert executable_edge(0.40, 0.55, 0.45, "lay") == pytest.approx(0.05)


def test_lay_edge_zero_at_breakeven():
    assert executable_edge(0.45, 0.55, 0.45, "lay") == pytest.approx(0.0)


def test_lay_edge_negative_when_fair_above_bid():
    assert executable_edge(0.50, 0.55, 0.45, "lay") < 0
