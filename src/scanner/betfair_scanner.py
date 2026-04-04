"""Betfair Exchange scanner — fetches live Australian markets for paper trading.

Uses read-only API calls only:
    list_market_catalogue()  — market names and metadata
    list_market_book()       — best back/lay prices and volumes

No bets are ever placed by this module.

Public API:

    get_markets()                -> list[dict]
    get_market_detail(market_id) -> dict

Dict shape:
    id            : str   — Betfair market ID (e.g. "1.247542121")
    question      : str   — Market name (includes runner name where applicable)
    runner_name   : str   — Name of the specific runner/selection
    market_type   : str   — Betfair market type (e.g. MATCH_ODDS, WINNER)
    probability   : float — Implied probability from best back price (1 / price)
    p_back        : float — Implied prob from real best back price (= p_ask)
    p_lay         : float — Implied prob from real best lay price (= p_bid)
    best_back_price: float — Real best back decimal odds
    best_lay_price : float — Real best lay decimal odds
    volume        : float — total_matched for the market
    url           : str   — Betfair AU deep-link
    totalLiquidity: float — Sum of available back + lay volumes at best price
    isResolved    : bool  — True when status is CLOSED or SETTLED
    resolution    : str   — Always "MKT" (Betfair doesn't use YES/NO)
    market_start_time: datetime | None — Event start time (UTC)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import betfairlightweight
from betfairlightweight.filters import market_filter, price_projection

from src.config import settings
from src.enrichment.team_mapping import parse_teams_from_event

logger = logging.getLogger(__name__)

# Module-level client — created once, reused across calls.
_client: betfairlightweight.APIClient | None = None

# Re-auth backoff: avoid rapid-fire login attempts on persistent failures.
_last_auth_failure: float = 0.0
_AUTH_COOLDOWN = 60.0  # seconds

_RESOLVED_STATUSES = {"CLOSED", "SETTLED"}
_PROB_FLOOR = 0.05
_API_TIMEOUT = 30  # seconds — applied to betfairlightweight session
_PROB_CEIL = 0.95
# Betfair's TOO_MUCH_DATA limit is response-size based, not market count.
# AU horse-racing markets have 8-20 runners each; empirical safe limit ~40 per call.
_BOOK_BATCH_SIZE = 40

# Safe market types: runner[0] reliably represents the named selection.
# Excluded: OVER_UNDER_*, ASIAN_HANDICAP, CORRECT_SCORE, BOTH_TEAMS_TO_SCORE,
#           HALF_TIME, DOUBLE_CHANCE, TOTAL_GOALS, ALT_TOTAL_GOALS, FIRST_HALF_GOALS
# — all have ambiguous runner ordering that causes wrong-side trade placement.
_SAFE_MARKET_TYPES = {"MATCH_ODDS", "WINNER", "OUTRIGHT_WINNER", "DRAW_NO_BET", "MONEYLINE"}

# Market type priority: lower number = higher priority.
_MARKET_TYPE_PRIORITY: dict[str, int] = {
    "MATCH_ODDS": 0,
    "MONEYLINE": 0,
    "DRAW_NO_BET": 1,
    "WINNER": 2,
    "OUTRIGHT_WINNER": 2,
}
_DEFAULT_MARKET_PRIORITY = 3  # for unknown types


def _create_and_login() -> betfairlightweight.APIClient:
    """Create a fresh Betfair API client and authenticate."""
    client = betfairlightweight.APIClient(
        username=settings.betfair_username,
        password=settings.betfair_password,
        app_key=settings.betfair_app_key,
    )
    client.session.timeout = _API_TIMEOUT
    client.login_interactive()
    logger.info("Betfair interactive login successful.")
    return client


def _reauth_with_backoff() -> betfairlightweight.APIClient:
    """Re-authenticate with cooldown to avoid rapid-fire login attempts."""
    global _last_auth_failure  # noqa: PLW0603
    now = time.monotonic()
    if now - _last_auth_failure < _AUTH_COOLDOWN:
        wait = _AUTH_COOLDOWN - (now - _last_auth_failure)
        logger.warning("Betfair re-auth cooldown: waiting %.0fs before retry.", wait)
        time.sleep(wait)
    try:
        client = _create_and_login()
    except Exception:
        _last_auth_failure = time.monotonic()
        raise
    return client


def _get_client() -> betfairlightweight.APIClient:
    """Return (or lazily create) the authenticated Betfair API client.

    Performs a lightweight session health check on each call and
    re-authenticates transparently if the session has expired.
    """
    global _client  # noqa: PLW0603
    if _client is None:
        _client = _create_and_login()
    else:
        try:
            # Lightweight health check — triggers re-auth if session expired.
            _client.betting.list_event_types()
        except (betfairlightweight.exceptions.APIError,
                betfairlightweight.exceptions.InvalidResponse) as exc:
            logger.warning("Betfair session expired (%s) — re-authenticating.", exc)
            _client = _reauth_with_backoff()
        except Exception as exc:
            logger.error("Unexpected Betfair error during health check: %s", exc)
            _client = _reauth_with_backoff()
    return _client


def _implied_prob(best_back_price: float | None) -> float | None:
    """Convert a decimal back price to implied probability.

    Returns None if the price is invalid (≤ 1.0).
    """
    if best_back_price is None or best_back_price <= 1.0:
        return None
    return round(1.0 / best_back_price, 6)


def _liquidity_from_book(
    book,
    target_selection_id: int | None = None,
) -> tuple[
    float | None, float, float | None, float | None,
    list[tuple[float, float]], list[tuple[float, float]],
]:
    """Extract pricing, liquidity, and order-book depth from a MarketBook.

    Returns:
        (probability, total_liquidity, best_back_price, best_lay_price,
         depth_back, depth_lay)

    probability      = implied prob of the first runner with a valid back price
                       in the tradeable range [PROB_FLOOR, PROB_CEIL].
                       Returns None if no valid back price exists.
    total_liquidity  = sum of all available_to_back and available_to_lay sizes
                       across all runners.
    best_back_price  = real decimal odds for best back offer (first valid runner).
    best_lay_price   = real decimal odds for best lay offer (first valid runner).
    depth_back       = [(decimal_price, size), ...] for the target runner, sorted
                       best-first (highest price first for backs).
    depth_lay        = [(decimal_price, size), ...] for the target runner, sorted
                       best-first (lowest price first for lays).
    """
    prob: float | None = None
    total_liquidity = 0.0
    best_back_price: float | None = None
    best_lay_price: float | None = None
    depth_back: list[tuple[float, float]] = []
    depth_lay: list[tuple[float, float]] = []

    # Identify the target runner for depth extraction
    target_runner_ex = None

    for runner in book.runners or []:
        ex = runner.ex
        if ex is None:
            continue
        # Track target runner for depth data
        if target_selection_id is not None:
            if getattr(runner, "selection_id", None) == target_selection_id:
                target_runner_ex = ex
        elif target_runner_ex is None:
            target_runner_ex = ex  # fallback: first runner

        if ex.available_to_back:
            total_liquidity += sum(o.size for o in ex.available_to_back)
            if prob is None:
                bp = ex.available_to_back[0].price
                p = _implied_prob(bp)
                if p is not None and _PROB_FLOOR <= p <= _PROB_CEIL:
                    prob = p
                    best_back_price = bp
        if ex.available_to_lay:
            total_liquidity += sum(o.size for o in ex.available_to_lay)
            if best_lay_price is None:
                lp = ex.available_to_lay[0].price
                if lp > 1.0:
                    best_lay_price = lp

    # Extract full depth ladder for the target runner, and override prob/prices.
    # The initial prob/best_* above are taken from whichever book runner happens
    # to appear first — that may differ from the catalogue runner order.  When we
    # have a specific target_selection_id we must use that runner's prices so that
    # mid_price always corresponds to the runner being evaluated (not a different
    # team whose book entry comes first).
    if target_runner_ex is not None:
        if target_runner_ex.available_to_back:
            depth_back = [
                (o.price, o.size)
                for o in target_runner_ex.available_to_back
            ]
            # Override prob / best_back_price with the target runner's price
            bp = target_runner_ex.available_to_back[0].price
            p = _implied_prob(bp)
            if p is not None and _PROB_FLOOR <= p <= _PROB_CEIL:
                prob = p
                best_back_price = bp
        if target_runner_ex.available_to_lay:
            depth_lay = [
                (o.price, o.size)
                for o in target_runner_ex.available_to_lay
            ]
            # Override best_lay_price with the target runner's lay price
            lp = target_runner_ex.available_to_lay[0].price
            if lp > 1.0:
                best_lay_price = lp

    return prob, total_liquidity, best_back_price, best_lay_price, depth_back, depth_lay


def _market_url(market_id: str) -> str:
    return f"https://www.betfair.com.au/exchange/plus/en/betting-type-link/{market_id}/"


def _ensure_utc(dt: datetime | None) -> datetime | None:
    """Ensure a datetime is timezone-aware (UTC)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_markets(
    limit: int = 200,
    country_codes: list[str] | None = None,
    hours_ahead: int = 72,
) -> list[dict]:
    """Fetch and filter live Betfair exchange markets.

    Prioritises near-term MATCH_ODDS and OVER_UNDER markets over long-term
    outright winner futures. Markets are sorted by market type priority then
    by start time (ascending) so the agent focuses on events with active
    price discovery and news flow.

    Args:
        limit:         Maximum number of markets to return after filtering.
        country_codes: ISO country codes to filter by. Defaults to ["AU"].
        hours_ahead:   Only include markets starting within this many hours.
                       Markets with no start time are included as a fallback.

    Returns:
        List of market dicts with keys:
            id, question, runner_name, market_type, probability, p_back, p_lay,
            best_back_price, best_lay_price, volume, url, totalLiquidity,
            isResolved, market_start_time.

    Raises:
        betfairlightweight.exceptions.APIError: On API or auth errors.
    """
    if country_codes is None:
        country_codes = ["AU"]

    client = _get_client()
    prob_low, prob_high = settings.scanner.prob_range

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)

    # --- Per-sport catalogue queries with competition whitelists ---
    # Each sport gets its own query so junk leagues can't crowd out viable ones.
    _SPORT_COMP_MAP: dict[str, list[int]] = {
        "1": settings.scanner.competition_ids_football,       # Football
        "5": settings.scanner.competition_ids_rugby_union,     # Rugby Union
        "61420": settings.scanner.competition_ids_afl,         # AFL
        "1477": settings.scanner.competition_ids_rugby_league, # Rugby League
        "7522": settings.scanner.competition_ids_basketball,   # Basketball
        "7524": settings.scanner.competition_ids_hockey,       # Ice Hockey
        "7511": settings.scanner.competition_ids_baseball,     # Baseball
    }

    allowed_event_types = settings.scanner.betfair_event_types or []
    _projections = [
        "MARKET_START_TIME", "RUNNER_DESCRIPTION", "EVENT",
        "EVENT_TYPE", "MARKET_DESCRIPTION", "COMPETITION",
    ]

    catalogue: list = []
    for eid in allowed_event_types:
        comp_ids = _SPORT_COMP_MAP.get(str(eid), [])
        filt = market_filter(
            market_countries=country_codes,
            event_type_ids=[str(eid)],
            **({"competition_ids": [str(c) for c in comp_ids]} if comp_ids else {}),
        )
        batch = client.betting.list_market_catalogue(
            filter=filt,
            market_projection=_projections,
            max_results=min(limit, 200),
            sort="FIRST_TO_START",
        )
        if batch:
            logger.info(
                "Betfair scan: event_type=%s comp_ids=%s → %d markets",
                eid, comp_ids or "all", len(batch),
            )
            catalogue.extend(batch)

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

        # Start time filtering: skip markets that start more than hours_ahead from now.
        start_dt = _ensure_utc(getattr(cat, "market_start_time", None))
        if start_dt is not None and start_dt > cutoff:
            continue  # Too far in the future — likely a stale futures market

        # Skip markets whose start_time is too far in the PAST — catches
        # season-long outright/futures markets (e.g. "Premiership Winner 2025/26"
        # with a market_start_time of Aug 2025).
        max_age = settings.scanner.betfair_max_market_age_hours
        if max_age > 0 and start_dt is not None:
            age_cutoff = now - timedelta(hours=max_age)
            if start_dt < age_cutoff:
                logger.debug(
                    "Skipping market %s — market_start_time %s is older than %dh",
                    cat.market_id, start_dt.isoformat(), max_age,
                )
                continue

        prob, total_liquidity, best_back_price, best_lay_price, _db, _dl = _liquidity_from_book(book)

        # prob is None when no back price falls in [PROB_FLOOR, PROB_CEIL]
        if prob is None or not (prob_low <= prob <= prob_high):
            continue

        total_matched = getattr(book, "total_matched", 0.0) or 0.0

        _event = getattr(cat, "event", None)
        event_name = (getattr(_event, "name", None) if _event else None) or ""
        market_name = getattr(cat, "market_name", None) or cat.market_id

        # Event type ID for sport detection (1=Soccer, 61420=AFL, 7524=Hockey, etc.)
        _event_type = getattr(cat, "event_type", None)
        event_type_id = (getattr(_event_type, "id", None) if _event_type else None) or ""

        # Competition name for league resolution (e.g. "NBA", "EuroLeague", "ACB")
        _comp = getattr(cat, "competition", None)
        competition_name = (getattr(_comp, "name", None) if _comp else None) or ""

        # Parse home/away teams from event name (e.g. "Sydney FC v Melbourne Victory")
        teams = parse_teams_from_event(event_name)
        home_team = teams[0] if teams else ""
        away_team = teams[1] if teams else ""

        # Market type from MARKET_DESCRIPTION projection
        market_type = ""
        _desc = getattr(cat, "description", None)
        if _desc is not None:
            market_type = getattr(_desc, "market_type", "") or ""

        # Skip markets where runner[0] identity is ambiguous (Over/Under, AH, BTTS, etc.)
        if market_type and market_type not in _SAFE_MARKET_TYPES:
            logger.debug("Skipping market %s — unsafe market type %r", cat.market_id, market_type)
            continue

        # Runner name and selection ID — the specific selection being priced
        runner_name = ""
        selection_id = None
        _runners = getattr(cat, "runners", None)
        if _runners:
            runner_name = getattr(_runners[0], "runner_name", "") or ""
            selection_id = getattr(_runners[0], "selection_id", None)

        # Build a descriptive question string that includes who/what is being priced
        if runner_name:
            question = (
                f"{event_name} {market_name} — {runner_name}".strip()
                if event_name
                else f"{market_name} — {runner_name}"
            )
        else:
            question = f"{event_name} {market_name}".strip() if event_name else market_name

        # Compute real implied probs from back/lay prices
        p_back = round(1.0 / best_back_price, 6) if best_back_price else None
        p_lay = round(1.0 / best_lay_price, 6) if best_lay_price else None

        markets.append({
            "id": cat.market_id,
            "question": question,
            "runner_name": runner_name,
            "market_type": market_type,
            "probability": prob,
            "p_back": p_back,
            "p_lay": p_lay,
            "best_back_price": best_back_price,
            "best_lay_price": best_lay_price,
            "volume": total_matched,
            "url": _market_url(cat.market_id),
            "totalLiquidity": total_liquidity,
            "isResolved": False,
            "market_start_time": start_dt,
            "home_team": home_team,
            "away_team": away_team,
            "event_type_id": str(event_type_id),
            "competition_name": competition_name,
            "selection_id": selection_id,
        })

    # Deduplicate by market ID (catalogue can return duplicates across batches)
    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for m in markets:
        if m["id"] not in seen_ids:
            seen_ids.add(m["id"])
            deduped.append(m)
    markets = deduped

    # Sort: MATCH_ODDS first, then by start time ascending (None last)
    def _sort_key(m: dict) -> tuple[int, datetime]:
        type_priority = _MARKET_TYPE_PRIORITY.get(m["market_type"], _DEFAULT_MARKET_PRIORITY)
        start = m["market_start_time"] or datetime(9999, 12, 31, tzinfo=timezone.utc)
        return (type_priority, start)

    markets.sort(key=_sort_key)
    markets = markets[:limit]

    logger.info(
        "Betfair scan: %d markets after filtering (catalogue %d, cutoff +%dh, %d sports queried)",
        len(markets),
        len(catalogue),
        hours_ahead,
        len(allowed_event_types),
    )
    return markets


def get_market_detail(market_id: str) -> dict:
    """Fetch full detail for a single Betfair market.

    Includes real back/lay prices and implied probabilities from the
    live order book.

    Automatically retries once after re-authentication if the session
    has expired.

    Args:
        market_id: Betfair market ID (e.g. "1.247542121").

    Returns:
        Dict with keys:
            id, question, runner_name, market_type, probability, p_back, p_lay,
            best_back_price, best_lay_price, volume, url,
            totalLiquidity, isResolved, resolution.

    Raises:
        ValueError: If the market book cannot be retrieved.
        betfairlightweight.exceptions.APIError: On API errors.
    """
    return _fetch_market_detail(market_id, retry=True)


def _fetch_market_detail(market_id: str, retry: bool = True) -> dict:
    """Internal implementation with optional retry on session expiry."""
    try:
        client = _get_client()

        catalogue = client.betting.list_market_catalogue(
            filter=market_filter(market_ids=[market_id]),
            market_projection=[
                "MARKET_START_TIME",
                "RUNNER_DESCRIPTION",
                "EVENT",
                "EVENT_TYPE",
                "MARKET_DESCRIPTION",
            ],
            max_results=1,
        )
        books = client.betting.list_market_book(
            market_ids=[market_id],
            price_projection=price_projection(price_data=["EX_ALL_OFFERS"]),
        )
    except Exception as exc:
        if retry:
            logger.warning(
                "Market detail fetch failed for %s (%s) — re-authenticating and retrying.",
                market_id, exc,
            )
            global _client  # noqa: PLW0603
            _client = _create_and_login()
            return _fetch_market_detail(market_id, retry=False)
        raise

    if not books:
        raise ValueError(f"No market book returned for market_id={market_id!r}")

    book = books[0]
    cat = catalogue[0] if catalogue else None

    status = book.status or ""
    is_resolved = status in _RESOLVED_STATUSES
    total_matched = getattr(book, "total_matched", 0.0) or 0.0

    _event = getattr(cat, "event", None) if cat else None
    event_name = (getattr(_event, "name", None) if _event else None) or ""
    market_name = (getattr(cat, "market_name", None) if cat else None) or market_id

    market_type = ""
    _desc = getattr(cat, "description", None) if cat else None
    if _desc is not None:
        market_type = getattr(_desc, "market_type", "") or ""

    runner_name = ""
    selection_id = None
    _runners = getattr(cat, "runners", None) if cat else None
    if _runners:
        runner_name = getattr(_runners[0], "runner_name", "") or ""
        selection_id = getattr(_runners[0], "selection_id", None)

    # Extract liquidity and depth for the target runner
    prob, total_liquidity, best_back_price, best_lay_price, depth_back, depth_lay = (
        _liquidity_from_book(book, target_selection_id=selection_id)
    )
    # raw_probability is None when the order book is empty (e.g. market suspended).
    # Callers that need a real market price (CLV tracking) should use raw_probability
    # and ignore None values rather than treating the 0.5 fallback as real.
    raw_probability = prob
    # Fall back to 0.5 for detail calls — caller uses this as settlement price
    if prob is None:
        prob = 0.5

    if runner_name:
        name = (
            f"{event_name} {market_name} — {runner_name}".strip()
            if event_name
            else f"{market_name} — {runner_name}"
        )
    else:
        name = f"{event_name} {market_name}".strip() if event_name else market_name

    p_back = round(1.0 / best_back_price, 6) if best_back_price else None
    p_lay = round(1.0 / best_lay_price, 6) if best_lay_price else None

    start_dt = _ensure_utc(getattr(cat, "market_start_time", None)) if cat else None

    # Determine resolution from runner status when market is settled/closed.
    resolution = "MKT"  # default fallback
    raw_runner_status: str | None = None
    if is_resolved and book.runners:
        # Match by selection_id if available, otherwise use runners[0]
        target_runner = None
        if selection_id is not None:
            for r in book.runners:
                if getattr(r, "selection_id", None) == selection_id:
                    target_runner = r
                    break
        if target_runner is None:
            target_runner = book.runners[0]

        raw_runner_status = getattr(target_runner, "status", "") or ""
        _STATUS_MAP = {"WINNER": "YES", "LOSER": "NO"}
        if raw_runner_status in _STATUS_MAP:
            resolution = _STATUS_MAP[raw_runner_status]
        elif raw_runner_status in ("REMOVED", "REMOVED_VACANT"):
            resolution = "VOID"
        else:
            logger.warning(
                "Unknown runner status %r for market %s — falling back to MKT",
                raw_runner_status, market_id,
            )

    return {
        "id": market_id,
        "question": name,
        "runner_name": runner_name,
        "market_type": market_type,
        "probability": prob,
        "p_back": p_back,
        "p_lay": p_lay,
        "best_back_price": best_back_price,
        "best_lay_price": best_lay_price,
        "volume": total_matched,
        "url": _market_url(market_id),
        "totalLiquidity": total_liquidity,
        "isResolved": is_resolved,
        "resolution": resolution,
        "runner_status": raw_runner_status,
        "selection_id": selection_id,
        "market_start_time": start_dt,
        "raw_probability": raw_probability,
        "depth_back": depth_back,
        "depth_lay": depth_lay,
        "inplay": getattr(book, "inplay", False) or False,
    }
