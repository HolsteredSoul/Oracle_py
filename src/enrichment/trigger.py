"""Trigger detector — determines whether a deep LLM analysis is warranted.

Pure function, no network calls, fully testable in isolation.
Reads thresholds from config.

Public API:
    should_trigger_deep(sentiment_delta, volatility_z, x_momentum) -> bool
"""

from __future__ import annotations

from src.config import settings


def should_trigger_deep(
    sentiment_delta: float,
    volatility_z: float,
    x_momentum: float,
) -> bool:
    """Return True if any signal exceeds its configured threshold.

    A deep trigger causes the agent to run an expensive deep LLM analysis
    instead of (or in addition to) the cheap light batch scan.

    Args:
        sentiment_delta: Absolute sentiment shift from the light scan [-1, 1].
                         Uses absolute value — strong negative signals also trigger.
        volatility_z:    Z-score of recent price volatility vs. rolling baseline.
        x_momentum:      X/Twitter momentum score for the market's keywords [0, 1].

    Returns:
        True if any signal breaches its threshold.
    """
    cfg = settings.triggers
    return (
        abs(sentiment_delta) > cfg.news_sentiment_delta
        or volatility_z > cfg.volatility_z
        or x_momentum > cfg.x_momentum
    )
