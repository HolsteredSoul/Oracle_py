"""Manifold Markets scanner — fetches and filters live prediction markets."""

import logging

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.manifold.markets/v0"
_TIMEOUT = 10.0


def get_markets(
    limit: int = 100,
    min_volume: float | None = None,
) -> list[dict]:
    """Fetch and filter live markets from Manifold.

    Args:
        limit: Maximum number of markets to request from the API.
        min_volume: Minimum trading volume. Defaults to config value.

    Returns:
        Filtered list of market dicts with keys: id, question, probability, volume, url.

    Raises:
        httpx.HTTPError: On network or HTTP errors.
    """
    if min_volume is None:
        min_volume = settings.scanner.manifold_min_volume

    prob_low, prob_high = settings.scanner.manifold_min_prob_range

    with httpx.Client(timeout=_TIMEOUT) as client:
        response = client.get(f"{_BASE_URL}/markets", params={"limit": limit})
        response.raise_for_status()
        raw_markets = response.json()

    markets = []
    for m in raw_markets:
        if m.get("isResolved"):
            continue
        prob = m.get("probability")
        if prob is None or not (prob_low <= prob <= prob_high):
            continue
        volume = m.get("volume", 0)
        if volume < min_volume:
            continue
        markets.append(
            {
                "id": m["id"],
                "question": m.get("question", ""),
                "probability": prob,
                "volume": volume,
                "url": m.get("url", ""),
            }
        )

    logger.info("Manifold scan: %d markets after filtering (requested %d)", len(markets), limit)
    return markets


def get_market_detail(market_id: str) -> dict:
    """Fetch full detail for a single Manifold market.

    Args:
        market_id: The Manifold market ID.

    Returns:
        Full market detail dict including bets (order book).

    Raises:
        httpx.HTTPError: On network or HTTP errors.
    """
    with httpx.Client(timeout=_TIMEOUT) as client:
        response = client.get(f"{_BASE_URL}/market/{market_id}")
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    markets = get_markets()
    if not markets:
        print("No markets returned.")
        sys.exit(0)
    for m in markets[:5]:
        print(f"[{m['probability']:.2f}] {m['question'][:80]}  vol={m['volume']:.0f}")
