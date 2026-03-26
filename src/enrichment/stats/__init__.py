"""Statistical data fetcher for sports match analysis.

Fetches team form, goals, standings, and H2H data from:
  - football-data.org (football/soccer) — free tier, 10 req/min
  - Squiggle API (AFL) — free, JSON-based
  - API-Basketball (api-sports.io) — free tier, 100 req/day

Returns MatchStats objects consumed by src/strategy/statistical_model.py.
Caches results in memory with configurable TTL (default 6 hours).
Returns None on any failure — caller falls back to market mid-price.
"""

from __future__ import annotations

import logging

from src.enrichment.team_mapping import resolve_team

from .afl import fetch_afl_stats
from .baseball import fetch_baseball_stats
from .basketball import fetch_basketball_stats, get_bb_team_index, get_bb_team_name
from .football import fetch_football_stats, get_fd_team_index, get_fd_team_name
from .models import MatchStats, cache_get, cache_set, compute_completeness

logger = logging.getLogger(__name__)

# Backwards-compatible alias for tests that import the private name.
_compute_completeness = compute_completeness

__all__ = [
    "MatchStats",
    "get_match_stats",
    "get_fd_team_index",
    "get_fd_team_name",
    "get_bb_team_index",
    "get_bb_team_name",
    "_compute_completeness",
]


def get_match_stats(
    home_team: str,
    away_team: str,
    sport: str = "football",
    competition: str = "",
) -> MatchStats | None:
    """Fetch statistical features for an upcoming match.

    Routes to football-data.org, Squiggle, or API-Basketball based on sport.
    Returns None if teams cannot be resolved or API fails.
    Caches results per (home_team, away_team, sport) tuple.
    """
    # Basketball and baseball use direct name resolution via their own team indexes
    if sport in ("basketball", "baseball"):
        cached = cache_get(home_team, away_team, sport)
        if cached is not None:
            logger.debug("Stats cache hit: %s v %s (%s)", home_team, away_team, sport)
            return cached

        if sport == "basketball":
            stats = fetch_basketball_stats(home_team, away_team, competition)
        else:
            stats = fetch_baseball_stats(home_team, away_team)
        if stats is not None:
            cache_set(home_team, away_team, sport, stats)
            logger.info(
                "Stats fetched: %s v %s (%s) completeness=%.0f%%",
                home_team, away_team, sport, stats.data_completeness * 100,
            )
        return stats

    # Resolve Betfair names to canonical stats API names
    home_canonical = resolve_team(home_team, sport)
    away_canonical = resolve_team(away_team, sport)

    if home_canonical is None and away_canonical is None:
        return None

    # Use original names as fallback for partial resolution
    home_canonical = home_canonical or home_team
    away_canonical = away_canonical or away_team

    # Check cache
    cached = cache_get(home_canonical, away_canonical, sport)
    if cached is not None:
        logger.debug("Stats cache hit: %s v %s (%s)", home_canonical, away_canonical, sport)
        return cached

    # Fetch from API
    if sport == "afl":
        stats = fetch_afl_stats(home_canonical, away_canonical)
    elif sport == "baseball":
        stats = fetch_baseball_stats(home_canonical, away_canonical)
    else:
        stats = fetch_football_stats(home_canonical, away_canonical)

    if stats is not None:
        cache_set(home_canonical, away_canonical, sport, stats)
        logger.info(
            "Stats fetched: %s v %s (%s) completeness=%.0f%%",
            home_canonical, away_canonical, sport, stats.data_completeness * 100,
        )

    return stats
