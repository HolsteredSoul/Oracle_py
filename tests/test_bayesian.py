"""Unit tests for src/strategy/bayesian.py."""

import pytest
from src.strategy.bayesian import update_probability


def test_positive_delta_increases_probability():
    p_mid = 0.5
    result = update_probability(p_mid, sentiment_delta=0.5, beta=0.15)
    assert result > p_mid


def test_negative_delta_decreases_probability():
    p_mid = 0.5
    result = update_probability(p_mid, sentiment_delta=-0.5, beta=0.15)
    assert result < p_mid


def test_zero_beta_is_identity():
    for p in [0.2, 0.5, 0.75]:
        result = update_probability(p, sentiment_delta=1.0, beta=0.0)
        assert result == pytest.approx(p, abs=1e-9)


def test_boundary_zero_does_not_raise():
    result = update_probability(0.0, sentiment_delta=0.5, beta=0.15)
    assert 0.0 < result < 1.0


def test_boundary_one_does_not_raise():
    result = update_probability(1.0, sentiment_delta=-0.5, beta=0.15)
    assert 0.0 < result < 1.0


def test_output_always_in_open_unit_interval():
    for p_mid in [0.0, 0.001, 0.5, 0.999, 1.0]:
        for delta in [-1.0, 0.0, 1.0]:
            result = update_probability(p_mid, delta, beta=0.15)
            assert 0.0 < result < 1.0, f"Out of bounds for p_mid={p_mid}, delta={delta}"
