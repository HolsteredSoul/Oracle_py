"""NewsData.io enrichment client with Google News RSS fallback.

Public API:
    rewrite_query(question, runner_name, market_type) -> str
    get_news_summary(query, max_articles=5) -> str

rewrite_query() extracts searchable terms from Betfair-style market names
(e.g. "AFL 2026 Grand Final — Hawthorn" → "AFL Hawthorn Grand Final 2026").

get_news_summary() tries NewsData.io first, then falls back to Google News
RSS when NewsData returns no results. Results are cached in memory for one
hour per query to avoid burning free-tier credits.

Returns an empty string when all sources fail or no results are found.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

_NEWSDATA_URL = "https://newsdata.io/api/1/news"
_GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
_TIMEOUT = 10.0
_CACHE_TTL = 3600.0  # 1 hour

# In-memory cache: query -> (fetched_at_unix, summary_str)
_cache: dict[str, tuple[float, str]] = {}

# Common Betfair boilerplate words to strip when building search queries.
_STRIP_WORDS = {
    "winner", "match odds", "outright", "to win", "tournament",
    "season", "2024", "2025", "2026", "2027", "2028",
}


def rewrite_query(
    question: str,
    runner_name: str = "",
    market_type: str = "",
) -> str:
    """Extract searchable news terms from a Betfair-style market name.

    Args:
        question:    Full Betfair market question (may include runner name suffix).
        runner_name: Specific selection name (e.g. "Melbourne United").
        market_type: Betfair market type string (e.g. "MATCH_ODDS", "WINNER").

    Returns:
        Cleaned search query string suitable for news APIs.

    Examples:
        "AFL 2026/27 Winner — Sydney Swans" → "AFL Sydney Swans"
        "NBL 2025/26 Winner" → "NBL basketball"
        "Chelsea v Arsenal Match Odds" → "Chelsea Arsenal"
    """
    # Start from the question, strip the runner suffix (after " — ")
    base = question.split(" — ")[0].strip()

    # Split on whitespace and slashes
    tokens_orig = re.split(r"[\s/]+", base)
    tokens_lower = [t.lower() for t in tokens_orig]

    # Keep tokens that are not in the strip list and not pure punctuation
    kept = [
        orig for orig, low in zip(tokens_orig, tokens_lower)
        if low not in _STRIP_WORDS and re.search(r"[a-zA-Z0-9]", orig)
    ]

    # Add runner name if it's not already present in the base
    if runner_name:
        runner_lower = runner_name.lower()
        base_lower = base.lower()
        if runner_lower not in base_lower:
            kept.extend(runner_name.split())

    # Limit to 8 tokens to avoid over-specific queries
    query = " ".join(kept[:8]).strip()
    return query if query else question[:80]


def get_news_summary(query: str, max_articles: int = 5) -> str:
    """Fetch recent news for a query and return a plain-text summary.

    Tries NewsData.io first, then falls back to Google News RSS when
    NewsData returns no results. Results are cached per query for 1 hour.

    Args:
        query:        Search terms (ideally from rewrite_query()).
        max_articles: Maximum number of articles to include in the summary.

    Returns:
        Newline-separated headlines+descriptions, or empty string on failure.
    """
    # Check cache
    cached = _cache.get(query)
    if cached is not None:
        fetched_at, summary = cached
        if time.monotonic() - fetched_at < _CACHE_TTL:
            logger.debug("News cache hit for query: %s", query[:60])
            return summary

    summary = ""

    # --- Primary: NewsData.io ---
    if settings.newsdata_api_key:
        try:
            summary = _fetch_newsdata(query, max_articles)
        except Exception as exc:  # noqa: BLE001
            logger.warning("NewsData enrichment failed for '%s': %s", query[:60], exc)
    else:
        logger.debug("No NewsData API key configured; skipping primary news source.")

    # --- Fallback: Google News RSS ---
    if not summary:
        try:
            summary = _fetch_google_news_rss(query, max_articles)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Google News RSS fallback failed for '%s': %s", query[:60], exc)

    _cache[query] = (time.monotonic(), summary)
    return summary


def _fetch_newsdata(query: str, max_articles: int) -> str:
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
        logger.debug("NewsData: no results for query '%s'", query[:60])
        return ""

    lines: list[str] = []
    for article in results[:max_articles]:
        title = article.get("title") or ""
        description = article.get("description") or ""
        if title:
            lines.append(f"- {title}: {description}" if description else f"- {title}")

    summary = "\n".join(lines)
    logger.info("NewsData: %d articles for '%s'", len(lines), query[:60])
    return summary


def _fetch_google_news_rss(query: str, max_articles: int) -> str:
    """Fallback: fetch headlines from Google News RSS."""
    encoded = urllib.parse.quote(query)
    url = f"{_GOOGLE_NEWS_RSS_URL}?q={encoded}&hl=en&gl=AU&ceid=AU:en"

    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        raw_xml = response.text

    root = ET.fromstring(raw_xml)
    items = root.findall(".//item")[:max_articles]

    if not items:
        logger.debug("Google News RSS: no results for '%s'", query[:60])
        return ""

    lines: list[str] = []
    for item in items:
        title = item.findtext("title") or ""
        # Strip HTML tags from description
        raw_desc = item.findtext("description") or ""
        desc = re.sub(r"<[^>]+>", "", raw_desc)[:120].strip()
        if title:
            lines.append(f"- {title}: {desc}" if desc else f"- {title}")

    summary = "\n".join(lines)
    logger.info("Google News RSS fallback: %d articles for '%s'", len(lines), query[:60])
    return summary
