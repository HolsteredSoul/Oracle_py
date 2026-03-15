"""Bayesian probability updater using logit-space sentiment adjustment.

Updates a market mid-probability with an LLM-derived sentiment signal.
Uses scipy's numerically stable logit/expit to stay within (0, 1).
"""

from __future__ import annotations

from scipy.special import expit, logit  # type: ignore[import]

_CLAMP_LOW = 0.001
_CLAMP_HIGH = 0.999


def update_probability(p_mid: float, sentiment_delta: float, beta: float) -> float:
    """Update market mid-probability with a sentiment signal.

    Formula:
        p_fair = expit(logit(clamp(p_mid)) + beta * sentiment_delta)

    Args:
        p_mid: Market mid-price as a probability, i.e. (ask + bid) / 2.
        sentiment_delta: LLM sentiment signal in [-1.0, 1.0].
        beta: Sensitivity parameter (default 0.15 from config).
              beta=0 returns p_mid exactly (identity property).

    Returns:
        Updated fair probability in (0, 1).
    """
    p_clamped = max(_CLAMP_LOW, min(_CLAMP_HIGH, p_mid))
    return float(expit(logit(p_clamped) + beta * sentiment_delta))
