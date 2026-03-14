"""NewsData.io enrichment client.

Public API:
    get_news_summary(query, max_articles=5) -> str

Returns a plain-text summary of recent headlines and descriptions for the
given query. Results are cached in memory for one hour per query to avoid
burning NewsData free-tier credits (200/day).

Returns an empty string when:
  - No API key is configured
  - The API returns no results
  - Any network/HTTP error occurs (non-fatal)
"""

from __future__ import annotations

import logging
import time

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

_NEWSDATA_URL = "https://newsdata.io/api/1/news"
_TIMEOUT = 10.0
_CACHE_TTL = 3600.0  # 1 hour

# In-memory cache: query -> (fetched_at_unix, summary_str)
_cache: dict[str, tuple[float, str]] = {}


def get_news_summary(query: str, max_articles: int = 5) -> str:
    """Fetch recent news for a query and return a plain-text summary.

    Args:
        query: Search terms relevant to the prediction market question.
        max_articles: Maximum number of articles to include in the summary.

    Returns:
        Newline-separated headlines+descriptions, or empty string on failure.
    """
    if not settings.newsdata_api_key:
        logger.debug("No NewsData API key configured; skipping news enrichment.")
        return ""

    # Check cache
    cached = _cache.get(query)
    if cached is not None:
        fetched_at, summary = cached
        if time.monotonic() - fetched_at < _CACHE_TTL:
            logger.debug("News cache hit for query: %s", query[:60])
            return summary

    try:
        summary = _fetch_news(query, max_articles)
    except Exception as exc:  # noqa: BLE001
        logger.warning("News enrichment failed for '%s': %s", query[:60], exc)
        return ""

    _cache[query] = (time.monotonic(), summary)
    return summary


def _fetch_news(query: str, max_articles: int) -> str:
    params = {
        "apikey": settings.newsdata_api_key,
        "q": query,
        "language": "en",
        "size": max_articles,
    }

    with httpx.Client(timeout=_TIMEOUT) as client:
        response = client.get(_NEWSDATA_URL, params=params)
        response.raise_for_status()
        data = response.json()

    results = data.get("results") or []
    if not results:
        logger.debug("No news results for query: %s", query[:60])
        return ""

    lines: list[str] = []
    for article in results[:max_articles]:
        title = article.get("title") or ""
        description = article.get("description") or ""
        if title:
            lines.append(f"- {title}: {description}" if description else f"- {title}")

    summary = "\n".join(lines)
    logger.info("News enrichment: %d articles for '%s'", len(lines), query[:60])
    return summary
