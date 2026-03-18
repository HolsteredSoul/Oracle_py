"""Kelly sizing calculator with commission awareness and Oracle scaling.

Three public functions:
    commission_aware_kelly  — raw f* (uncapped)
    apply_oracle_sizing     — applies k, λ_conf, λ_dd, and the 0.25 hard cap
    translate_to_betfair    — converts fraction → stake/liability dict
"""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

_HARD_CAP = 0.10  # Maximum fraction of bankroll risked on any single trade (paper phase)


def commission_aware_kelly(
    p_fair: float,
    q_market: float,
    gamma: float,
    direction: Literal["back", "lay"],
) -> float:
    """Compute commission-adjusted Kelly fraction f* (uncapped).

    Args:
        p_fair:    Our Bayesian probability estimate.
        q_market:  Market-implied probability (p_ask for back, p_bid for lay).
        gamma:     Commission rate, e.g. 0.05 for 5% Betfair commission.
        direction: "back" or "lay".

    Returns:
        Raw Kelly fraction f*. Negative means no edge — caller should skip.
        May exceed 1.0 for extreme edges; caller applies scaling + cap.
    """
    if direction == "back":
        denominator = 1.0 - q_market
        if denominator == 0.0:
            return 0.0
        f_star = (p_fair - q_market - gamma) / denominator
    else:  # lay
        denominator = q_market
        if denominator == 0.0:
            return 0.0
        f_star = (q_market - p_fair - gamma) / denominator

    if f_star > 1.0:
        logger.warning("f* = %.4f > 1.0 — extreme edge signal; hard cap will apply", f_star)

    return f_star


def apply_oracle_sizing(
    f_star: float,
    conf_score: float,
    drawdown_pct: float,
    config: object,
) -> float:
    """Apply Oracle scaling factors and the 0.25 hard cap to raw f*.

    Scaling:
        f_final = k * f* * λ_conf * λ_dd
        λ_conf  = clamp(conf_score / 100, 0.5, 1.0)
        λ_dd    = drawdown_throttle_factor if drawdown_pct >= drawdown_throttle_pct else 1.0
        Hard cap: f_final = min(f_final, 0.25)

    Args:
        f_star:       Raw Kelly fraction from commission_aware_kelly.
        conf_score:   LLM confidence as integer 0–100.
        drawdown_pct: Current drawdown as a fraction, e.g. 0.22 for 22%.
        config:       Settings object with a `risk` sub-config.

    Returns:
        Final fraction of bankroll to risk (always in [0, 0.25]).
    """
    risk = config.risk
    k = risk.kelly_base_fraction
    lambda_conf = max(0.5, min(1.0, conf_score / 100.0))
    lambda_dd = (
        risk.drawdown_throttle_factor
        if drawdown_pct >= risk.drawdown_throttle_pct
        else 1.0
    )
    f_final = k * f_star * lambda_conf * lambda_dd
    return min(f_final, _HARD_CAP)


def translate_to_betfair(
    f: float,
    bankroll: float,
    decimal_odds: float,
    direction: Literal["back", "lay"],
) -> dict[str, float]:
    """Translate a Kelly fraction to a Betfair stake/liability pair.

    Args:
        f:            Final Kelly fraction (output of apply_oracle_sizing).
        bankroll:     Current bankroll in account currency.
        decimal_odds: Betfair decimal odds (e.g. 3.5).
        direction:    "back" or "lay".

    Returns:
        {"stake": float, "liability": float}
        For backs:  stake = f * bankroll; liability = stake * (odds - 1)
        For lays:   liability = f * bankroll; stake = liability / (odds - 1)
    """
    if direction == "back":
        stake = f * bankroll
        liability = stake * (decimal_odds - 1.0)
    else:
        liability = f * bankroll
        odds_minus_one = decimal_odds - 1.0
        stake = liability / odds_minus_one if odds_minus_one > 0 else 0.0

    return {"stake": round(stake, 2), "liability": round(liability, 2)}
