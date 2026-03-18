"""Betfair Exchange scanner — fetches live Australian markets for paper trading.

Uses read-only API calls only:
    list_market_catalogue()  — market names and metadata
    list_market_book()       — best back/lay prices and volumes

No bets are ever placed by this module.

Public API mirrors src/scanner/manifold.py so main.py can swap scanners
without changing any pipeline logic:

    get_markets()                -> list[dict]
    get_market_detail(market_id) -> dict

Dict shape (same keys as manifold scanner):
    id            : str   — Betfair market ID (e.g. "1.247542121")
    question      : str   — Market name
    probability   : float — Implied probability from best back price (1 / price)
    volume        : float — total_matched for the market
    url           : str   — Betfair AU deep-link
    totalLiquidity: float — Sum of available back + lay volumes at best price
    isResolved    : bool  — True when status is CLOSED or SETTLED
    resolution    : str   — Always "MKT" (Betfair doesn't use YES/NO)
"""

from __future__ import annotations

import logging

import betfairlightweight
from betfairlightweight.filters import market_filter, price_projection

from src.config import settings

logger = logging.getLogger(__name__)

# Module-level client — created once, reused across calls.
_client: betfairlightweight.APIClient | None = None

_RESOLVED_STATUSES = {"CLOSED", "SETTLED"}
_PROB_FLOOR = 0.05
_PROB_CEIL = 0.95
# Betfair's TOO_MUCH_DATA limit is response-size based, not market count.
# AU horse-racing markets have 8-20 runners each; empirical safe limit ~40 per call.
_BOOK_BATCH_SIZE = 40


def _get_client() -> betfairlightweight.APIClient:
    """Return (or lazily create) the authenticated Betfair API client."""
    global _client  # noqa: PLW0603
    if _client is None:
        _client = betfairlightweight.APIClient(
            username=settings.betfair_username,
            password=settings.betfair_password,
            app_key=settings.betfair_app_key,
        )
        _client.login_interactive()
        logger.info("Betfair interactive login successful.")
    return _client


def _implied_prob(best_back_price: float | None) -> float | None:
    """Convert a decimal back price to implied probability.

    Returns None if the price is invalid (≤ 1.0).
    """
    if best_back_price is None or best_back_price <= 1.0:
        return None
    return round(1.0 / best_back_price, 6)


def _liquidity_from_book(book) -> tuple[float | None, float]:
    """Extract (probability, total_liquidity) from a market book.

    probability    = implied prob of the first runner with a valid back price
                     in the tradeable range [PROB_FLOOR, PROB_CEIL].
                     Returns None if no valid back price exists.
    total_liquidity = sum of all available_to_back and available_to_lay sizes
                      across all runners at the best price level.
    """
    prob: float | None = None
    total_liquidity = 0.0

    for runner in book.runners or []:
        ex = runner.ex
        if ex is None:
            continue
        if ex.available_to_back:
            total_liquidity += sum(o.size for o in ex.available_to_back)
            if prob is None:
                p = _implied_prob(ex.available_to_back[0].price)
                if p is not None and _PROB_FLOOR <= p <= _PROB_CEIL:
                    prob = p
        if ex.available_to_lay:
            total_liquidity += sum(o.size for o in ex.available_to_lay)

    return prob, total_liquidity


def _market_url(market_id: str) -> str:
    return f"https://www.betfair.com.au/exchange/plus/en/betting-type-link/{market_id}/"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_markets(
    limit: int = 100,
    country_codes: list[str] | None = None,
) -> list[dict]:
    """Fetch and filter live Australian Betfair exchange markets.

    Applies the same probability range filter as the Manifold scanner
    (settings.scanner.manifold_min_prob_range) so the same config.toml
    controls both scanners.

    Args:
        limit:         Maximum number of markets to return after filtering.
        country_codes: ISO country codes to filter by. Defaults to ["AU"].

    Returns:
        List of market dicts with keys:
            id, question, probability, volume, url, totalLiquidity, isResolved.

    Raises:
        betfairlightweight.exceptions.APIError: On API or auth errors.
    """
    if country_codes is None:
        country_codes = ["AU"]

    client = _get_client()
    prob_low, prob_high = settings.scanner.manifold_min_prob_range

    catalogue = client.betting.list_market_catalogue(
        filter=market_filter(market_countries=country_codes),
        market_projection=["MARKET_START_TIME", "RUNNER_DESCRIPTION", "EVENT", "EVENT_TYPE"],
        max_results=min(limit * 3, 200),  # cap to avoid excessive batches
    )

    if not catalogue:
        logger.info("Betfair scan: no markets returned for countries=%s", country_codes)
        return []

    # list_market_book accepts at most _BOOK_BATCH_SIZE IDs per call — batch if needed.
    market_ids = [m.market_id for m in catalogue]
    all_books: list = []
    for i in range(0, len(market_ids), _BOOK_BATCH_SIZE):
        batch = market_ids[i : i + _BOOK_BATCH_SIZE]
        all_books.extend(
            client.betting.list_market_book(
                market_ids=batch,
                price_projection=price_projection(price_data=["EX_BEST_OFFERS"]),
            )
        )
    book_by_id = {b.market_id: b for b in all_books}

    markets: list[dict] = []
    for cat in catalogue:
        book = book_by_id.get(cat.market_id)
        if book is None:
            continue
        if book.status in _RESOLVED_STATUSES or book.status == "SUSPENDED":
            continue

        prob, total_liquidity = _liquidity_from_book(book)

        # prob is None when no back price falls in [PROB_FLOOR, PROB_CEIL]
        if prob is None or not (prob_low <= prob <= prob_high):
            continue

        total_matched = getattr(book, "total_matched", 0.0) or 0.0

        event_name = (cat.event.name if cat.event else None) or ""
        market_name = cat.market_name or cat.market_id
        question = f"{event_name} {market_name}".strip() if event_name else market_name

        markets.append({
            "id": cat.market_id,
            "question": question,
            "probability": prob,
            "volume": total_matched,
            "url": _market_url(cat.market_id),
            "totalLiquidity": total_liquidity,
            "isResolved": False,
        })

        if len(markets) >= limit:
            break

    logger.info(
        "Betfair scan: %d markets after filtering (catalogue size %d)",
        len(markets),
        len(catalogue),
    )
    return markets


def get_market_detail(market_id: str) -> dict:
    """Fetch full detail for a single Betfair market.

    Returns the same dict shape as manifold.get_market_detail() so
    PaperBroker and the pipeline work unmodified.

    Args:
        market_id: Betfair market ID (e.g. "1.247542121").

    Returns:
        Dict with keys:
            id, question, probability, volume, url,
            totalLiquidity, isResolved, resolution.

    Raises:
        ValueError: If the market book cannot be retrieved.
        betfairlightweight.exceptions.APIError: On API errors.
    """
    client = _get_client()

    catalogue = client.betting.list_market_catalogue(
        filter=market_filter(market_ids=[market_id]),
        market_projection=["MARKET_START_TIME", "RUNNER_DESCRIPTION", "EVENT", "EVENT_TYPE"],
        max_results=1,
    )
    books = client.betting.list_market_book(
        market_ids=[market_id],
        price_projection=price_projection(price_data=["EX_BEST_OFFERS"]),
    )

    if not books:
        raise ValueError(f"No market book returned for market_id={market_id!r}")

    book = books[0]
    cat = catalogue[0] if catalogue else None

    status = book.status or ""
    is_resolved = status in _RESOLVED_STATUSES
    total_matched = getattr(book, "total_matched", 0.0) or 0.0

    prob, total_liquidity = _liquidity_from_book(book)
    # Fall back to 0.5 for detail calls — caller uses this as settlement price
    if prob is None:
        prob = 0.5
    event_name = (cat.event.name if cat and cat.event else None) or ""
    market_name = (cat.market_name if cat else None) or market_id
    name = f"{event_name} {market_name}".strip() if event_name else market_name

    return {
        "id": market_id,
        "question": name,
        "probability": prob,
        "volume": total_matched,
        "url": _market_url(market_id),
        "totalLiquidity": total_liquidity,
        "isResolved": is_resolved,
        # Betfair markets don't resolve YES/NO — always use MKT settlement.
        # PaperBroker.settle_position() handles MKT via resolution_probability.
        "resolution": "MKT",
    }
