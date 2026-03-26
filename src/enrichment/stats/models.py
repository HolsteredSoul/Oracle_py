"""Shared data model, cache, and helpers for stats fetchers."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

from src.config import settings

TIMEOUT = 15.0  # seconds for all stats API requests


class MatchStats(BaseModel):
    """Statistical features for a single match."""

    sport: Literal["football", "afl", "basketball", "baseball"]
    home_team: str
    away_team: str
    # Form (last 5 matches)
    home_form_pts_per_game: float | None = None
    away_form_pts_per_game: float | None = None
    home_goals_scored_avg: float | None = None
    home_goals_conceded_avg: float | None = None
    away_goals_scored_avg: float | None = None
    away_goals_conceded_avg: float | None = None
    # Standings
    home_league_position: int | None = None
    away_league_position: int | None = None
    # H2H
    h2h_home_wins: int = 0
    h2h_draws: int = 0
    h2h_away_wins: int = 0
    h2h_total_matches: int = 0
    # Data quality
    data_completeness: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

# (home_canonical, away_canonical, sport) -> (timestamp, MatchStats)
_cache: dict[tuple[str, str, str], tuple[float, MatchStats]] = {}


def cache_get(home: str, away: str, sport: str) -> MatchStats | None:
    key = (home, away, sport)
    if key in _cache:
        ts, stats = _cache[key]
        ttl = settings.stats.cache_ttl_hours * 3600
        if time.time() - ts < ttl:
            return stats
        del _cache[key]
    return None


def cache_set(home: str, away: str, sport: str, stats: MatchStats) -> None:
    _cache[(home, away, sport)] = (time.time(), stats)


def compute_completeness(stats: MatchStats) -> float:
    """Fraction of optional numeric fields that are populated."""
    optional_fields = [
        stats.home_form_pts_per_game,
        stats.away_form_pts_per_game,
        stats.home_goals_scored_avg,
        stats.home_goals_conceded_avg,
        stats.away_goals_scored_avg,
        stats.away_goals_conceded_avg,
        stats.home_league_position,
        stats.away_league_position,
    ]
    filled = sum(1 for f in optional_fields if f is not None)
    return round(filled / len(optional_fields), 2)
