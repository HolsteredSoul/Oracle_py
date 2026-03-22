"""Team name mapping between Betfair runner/event names and stats API names.

Betfair uses names like "Sydney FC", "Man Utd", "Western Sydney".
Football-data.org uses "Sydney FC", "Manchester United FC", "Western Sydney Wanderers FC".
Squiggle uses short AFL names like "Sydney", "Collingwood".

Strategy:
  1. Check runtime cache (permanent per session)
  2. AFL: hardcoded aliases (Squiggle has no dynamic index)
  3. Football: resolve against the football-data.org team index (316+ teams
     with full + short names), built dynamically from the API on first use.
     Falls back to fuzzy matching against the index keys.
  4. Tiny override dict for proven Betfair naming quirks only.
  5. Return None on no confident match — caller falls back to mid_price
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

FUZZY_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# Betfair-specific overrides — ONLY for names that the football-data.org
# index cannot resolve even with fuzzy matching.  Keep this < 10 entries.
# Add entries here only when proven needed from production logs.
# ---------------------------------------------------------------------------

_FOOTBALL_OVERRIDES: dict[str, str] = {
    "nott'm forest": "Nottingham Forest FC",
    "spurs": "Tottenham Hotspur FC",
    "wolves": "Wolverhampton Wanderers FC",
}

# AFL aliases — Squiggle API has no dynamic team index, so these stay hardcoded.
_AFL_ALIASES: dict[str, str] = {
    "sydney swans": "Sydney",
    "sydney": "Sydney",
    "collingwood": "Collingwood",
    "collingwood magpies": "Collingwood",
    "carlton": "Carlton",
    "carlton blues": "Carlton",
    "brisbane": "Brisbane Lions",
    "brisbane lions": "Brisbane Lions",
    "melbourne": "Melbourne",
    "melbourne demons": "Melbourne",
    "geelong": "Geelong",
    "geelong cats": "Geelong",
    "western bulldogs": "Western Bulldogs",
    "bulldogs": "Western Bulldogs",
    "gws": "GWS",
    "gws giants": "GWS",
    "greater western sydney": "GWS",
    "greater western sydney giants": "GWS",
    "hawthorn": "Hawthorn",
    "hawthorn hawks": "Hawthorn",
    "essendon": "Essendon",
    "essendon bombers": "Essendon",
    "richmond": "Richmond",
    "richmond tigers": "Richmond",
    "fremantle": "Fremantle",
    "fremantle dockers": "Fremantle",
    "port adelaide": "Port Adelaide",
    "port adelaide power": "Port Adelaide",
    "west coast": "West Coast",
    "west coast eagles": "West Coast",
    "north melbourne": "North Melbourne",
    "north melbourne kangaroos": "North Melbourne",
    "gold coast": "Gold Coast",
    "gold coast suns": "Gold Coast",
    "adelaide": "Adelaide",
    "adelaide crows": "Adelaide",
    "st kilda": "St Kilda",
    "st kilda saints": "St Kilda",
}

# Runtime cache for successful matches: (betfair_name_lower, sport) -> canonical
_resolved_cache: dict[tuple[str, str], str] = {}


def _normalize(name: str) -> str:
    """Normalize a team name for comparison."""
    n = name.lower().strip()
    for suffix in (" fc", " afc", " sc"):
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    return n


def _get_football_index() -> dict[str, int]:
    """Lazy-load the football-data.org team index.

    Returns an empty dict if the API is unavailable (non-fatal).
    """
    try:
        from src.enrichment.stats import get_fd_team_index
        return get_fd_team_index()
    except Exception as exc:
        logger.debug("Could not load football-data.org team index: %s", exc)
        return {}


def resolve_team(betfair_name: str, sport: str = "football") -> str | None:
    """Map a Betfair runner/event name to the canonical stats API team name.

    Returns None if no confident match is found (caller falls back to mid_price).
    """
    key_lower = betfair_name.lower().strip()
    cache_key = (key_lower, sport)

    # 1. Check runtime cache
    if cache_key in _resolved_cache:
        return _resolved_cache[cache_key]

    # 2. AFL — hardcoded aliases (no dynamic index available)
    if sport == "afl":
        if key_lower in _AFL_ALIASES:
            result = _AFL_ALIASES[key_lower]
            _resolved_cache[cache_key] = result
            return result
        # Fuzzy match against AFL aliases
        return _fuzzy_match_aliases(betfair_name, _AFL_ALIASES, cache_key)

    # 3. Football — check overrides first
    if key_lower in _FOOTBALL_OVERRIDES:
        result = _FOOTBALL_OVERRIDES[key_lower]
        _resolved_cache[cache_key] = result
        return result

    # 4. Football — resolve against football-data.org team index
    index = _get_football_index()
    if not index:
        logger.warning(
            "Team mapping failed for %r — football-data.org index unavailable.",
            betfair_name,
        )
        return None

    # 4a. Exact match (lowercase)
    if key_lower in index:
        # Recover the canonical casing from the index
        canonical = _canonical_from_index(key_lower, index)
        _resolved_cache[cache_key] = canonical
        return canonical

    # 4b. Normalized match (strip FC/AFC/SC suffixes)
    normalized = _normalize(betfair_name)
    for idx_key in index:
        if _normalize(idx_key) == normalized:
            canonical = _canonical_from_index(idx_key, index)
            _resolved_cache[cache_key] = canonical
            logger.debug(
                "Normalized match: %r -> %r", betfair_name, canonical,
            )
            return canonical

    # 4c. Fuzzy match against all index keys
    best_score = 0.0
    best_key: str | None = None

    for idx_key in index:
        score = SequenceMatcher(None, normalized, _normalize(idx_key)).ratio()
        if score > best_score:
            best_score = score
            best_key = idx_key

    if best_score >= FUZZY_THRESHOLD and best_key is not None:
        canonical = _canonical_from_index(best_key, index)
        _resolved_cache[cache_key] = canonical
        logger.debug(
            "Fuzzy-matched %r -> %r (score=%.2f)", betfair_name, canonical, best_score,
        )
        return canonical

    logger.warning(
        "Team mapping failed for %r (sport=%s, best_score=%.2f) "
        "— falling back to mid_price.",
        betfair_name, sport, best_score,
    )
    return None


def _fuzzy_match_aliases(
    betfair_name: str,
    aliases: dict[str, str],
    cache_key: tuple[str, str],
) -> str | None:
    """Fuzzy match against an alias dict. Used for AFL."""
    normalized = _normalize(betfair_name)
    best_score = 0.0
    best_match: str | None = None

    for alias_key, canonical in aliases.items():
        score = SequenceMatcher(None, normalized, _normalize(alias_key)).ratio()
        if score > best_score:
            best_score = score
            best_match = canonical

    if best_score >= FUZZY_THRESHOLD and best_match is not None:
        _resolved_cache[cache_key] = best_match
        logger.debug(
            "Fuzzy-matched %r -> %r (score=%.2f)",
            betfair_name, best_match, best_score,
        )
        return best_match

    logger.warning(
        "Team mapping failed for %r (sport=afl, best_score=%.2f) "
        "— falling back to mid_price.",
        betfair_name, best_score,
    )
    return None


def _canonical_from_index(key_lower: str, index: dict[str, int]) -> str:
    """Recover proper-cased canonical name from a lowercase index key.

    Uses the original casing stored during index build when available,
    falls back to title-case as an approximation.
    """
    try:
        from src.enrichment.stats import get_fd_team_name
        original = get_fd_team_name(key_lower)
        if original:
            return original
    except Exception:
        pass
    return key_lower.title() if key_lower in index else key_lower


def parse_teams_from_event(event_name: str) -> tuple[str, str] | None:
    """Parse 'Home Team v Away Team' from a Betfair event name.

    Handles separators: ' v ', ' vs ', ' vs. ', ' - '.
    Returns (home, away) tuple or None if the format doesn't match.
    """
    parts = re.split(r"\s+(?:vs?\.?|-)\s+", event_name, maxsplit=1)
    if len(parts) == 2:
        home = parts[0].strip()
        away = parts[1].strip()
        if home and away:
            return home, away
    return None
