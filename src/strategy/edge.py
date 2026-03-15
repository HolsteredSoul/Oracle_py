"""Edge calculator for back and lay positions.

Commission is NOT applied here — it is applied in the Kelly calculator.
A negative return means no executable edge; caller should skip the trade.
"""

from __future__ import annotations

from typing import Literal


def executable_edge(
    p_fair: float,
    p_ask: float,
    p_bid: float,
    direction: Literal["back", "lay"],
) -> float:
    """Compute signed executable edge for a back or lay position.

    Args:
        p_fair: Our Bayesian estimate of the true probability.
        p_ask: Market-implied probability at the ask (back) price.
        p_bid: Market-implied probability at the bid (lay) price.
        direction: "back" to buy YES, "lay" to sell YES.

    Returns:
        Signed edge. Positive = edge exists; negative = no edge, skip.
    """
    if direction == "back":
        return p_fair - p_ask
    else:
        return p_bid - p_fair
