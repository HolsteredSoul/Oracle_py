"""Risk gate manager.

A single function that checks all pre-trade risk gates in order.
Gates return on first failure — the caller receives (False, reason).
The drawdown throttle is a sizing scaler (applied in kelly.py), not a gate.
"""

from __future__ import annotations


def check_risk_gates(
    f: float,
    conf_score: float,
    current_exposure: float,
    drawdown_pct: float,
    config: object,
) -> tuple[bool, str]:
    """Check all risk gates before allowing a trade.

    Gates (evaluated in order):
        1. Confidence floor — conf_score must meet the minimum.
        2. Exposure cap already reached — current exposure is at or above max.
        3. Trade would breach exposure cap — adding f would exceed max.

    Args:
        f:                Proposed trade size as a fraction of bankroll.
        conf_score:       LLM confidence as a float 0–100.
        current_exposure: Sum of all open position sizes as a fraction of bankroll.
        drawdown_pct:     Current drawdown fraction (informational; throttle is
                          already baked into f via apply_oracle_sizing).
        config:           Settings object with a `risk` sub-config.

    Returns:
        (True, "OK") if all gates pass, else (False, <reason string>).
    """
    risk = config.risk

    if conf_score < risk.confidence_floor:
        return False, f"confidence {conf_score:.1f} below floor {risk.confidence_floor}"

    if current_exposure >= risk.max_exposure:
        return False, f"exposure cap reached ({current_exposure:.2%} >= {risk.max_exposure:.2%})"

    if (current_exposure + f) > risk.max_exposure:
        return (
            False,
            f"trade would breach exposure cap "
            f"({current_exposure:.2%} + {f:.2%} > {risk.max_exposure:.2%})",
        )

    return True, "OK"
