"""X (Twitter) API v2 enrichment client.

Public API:
    get_x_summary(keywords, max_tweets=20) -> str

Searches recent tweets for the given keywords and returns a plain-text
summary of the most relevant posts. Rate-limit state is tracked in memory
from response headers — if the remaining request quota hits zero, returns
an empty string (non-fatal).

Returns an empty string when:
  - No X bearer token is configured
  - Rate limit is exhausted
  - The search returns no results
  - Any network/HTTP error occurs (non-fatal)
"""

from __future__ import annotations

import logging
import time

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

_X_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
_TIMEOUT = 10.0

# Rate-limit state (updated from response headers each call)
_remaining_requests: int | None = None
_reset_at_unix: float = 0.0


def get_x_summary(keywords: list[str], max_tweets: int = 20) -> str:
    """Search X for recent tweets matching keywords and return a summary.

    Args:
        keywords: List of search terms to AND-join in the query.
        max_tweets: Maximum number of tweets to include (max 100 per API rules).

    Returns:
        Newline-separated tweet texts, or empty string on failure/rate-limit.
    """
    global _remaining_requests, _reset_at_unix

    if not settings.x_bearer_token:
        logger.debug("No X bearer token configured; skipping X enrichment.")
        return ""

    # Check rate limit state
    if _remaining_requests is not None and _remaining_requests <= 0:
        if time.time() < _reset_at_unix:
            logger.warning("X API rate limit exhausted; skipping until reset at %.0f", _reset_at_unix)
            return ""

    query = " ".join(keywords)
    if not query.strip():
        return ""

    try:
        return _fetch_x(query, min(max_tweets, 100))
    except Exception as exc:  # noqa: BLE001
        logger.warning("X enrichment failed for '%s': %s", query[:60], exc)
        return ""


def _fetch_x(query: str, max_results: int) -> str:
    global _remaining_requests, _reset_at_unix

    headers = {"Authorization": f"Bearer {settings.x_bearer_token}"}
    params = {
        "query": f"{query} -is:retweet lang:en",
        "max_results": max(10, max_results),  # API minimum is 10
        "tweet.fields": "text,created_at,public_metrics",
    }

    with httpx.Client(timeout=_TIMEOUT) as client:
        response = client.get(_X_SEARCH_URL, headers=headers, params=params)

    # Update rate limit state from headers regardless of status
    _remaining_requests = _parse_int_header(response, "x-rate-limit-remaining")
    reset_str = response.headers.get("x-rate-limit-reset")
    if reset_str:
        try:
            _reset_at_unix = float(reset_str)
        except ValueError:
            pass

    if response.status_code == 429:
        logger.warning("X API rate limit hit (429); returning empty summary.")
        return ""

    response.raise_for_status()
    data = response.json()

    tweets: list[dict] = data.get("data") or []
    if not tweets:
        logger.debug("No X results for query: %s", query[:60])
        return ""

    # Sort by engagement (likes + retweets) descending, take top tweets
    def _engagement(t: dict) -> int:
        m = t.get("public_metrics") or {}
        return m.get("like_count", 0) + m.get("retweet_count", 0)

    tweets.sort(key=_engagement, reverse=True)

    lines = [f"- {t['text'][:200]}" for t in tweets[:max_results]]
    summary = "\n".join(lines)
    logger.info("X enrichment: %d tweets for '%s'", len(lines), query[:60])
    return summary


def _parse_int_header(response: httpx.Response, header: str) -> int | None:
    val = response.headers.get(header)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None
