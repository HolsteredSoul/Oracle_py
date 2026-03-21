"""Statistical data fetcher for sports match analysis.

Fetches team form, goals, standings, and H2H data from:
  - football-data.org (football/soccer) — free tier, 10 req/min
  - Squiggle API (AFL) — free, JSON-based

Returns MatchStats objects consumed by src/strategy/statistical_model.py.
Caches results in memory with configurable TTL (default 6 hours).
Returns None on any failure — caller falls back to market mid-price.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from src.config import settings
from src.enrichment.team_mapping import resolve_team

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0  # seconds


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class MatchStats(BaseModel):
    """Statistical features for a single match."""

    sport: Literal["football", "afl"]
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


def _cache_get(home: str, away: str, sport: str) -> MatchStats | None:
    key = (home, away, sport)
    if key in _cache:
        ts, stats = _cache[key]
        ttl = settings.stats.cache_ttl_hours * 3600
        if time.time() - ts < ttl:
            return stats
        del _cache[key]
    return None


def _cache_set(home: str, away: str, sport: str, stats: MatchStats) -> None:
    _cache[(home, away, sport)] = (time.time(), stats)


def _compute_completeness(stats: MatchStats) -> float:
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


# ---------------------------------------------------------------------------
# football-data.org helpers
# ---------------------------------------------------------------------------

# Team ID cache: canonical name -> football-data.org team ID
_team_id_cache: dict[str, int | None] = {}


def _fd_headers() -> dict[str, str]:
    """Auth headers for football-data.org."""
    key = settings.football_data_api_key
    if key:
        return {"X-Auth-Token": key}
    return {}


def _fd_get(path: str, params: dict | None = None) -> dict | list | None:
    """GET request to football-data.org v4 API."""
    url = f"{settings.stats.football_api_base}{path}"
    try:
        resp = httpx.get(url, headers=_fd_headers(), params=params, timeout=_TIMEOUT)
        if resp.status_code == 429:
            logger.warning("football-data.org rate limit hit.")
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("football-data.org request failed: %s %s", path, exc)
        return None


def _fd_resolve_team_id(canonical_name: str) -> int | None:
    """Resolve a canonical team name to a football-data.org team ID."""
    if canonical_name in _team_id_cache:
        return _team_id_cache[canonical_name]

    data = _fd_get("/teams", params={"name": canonical_name})
    if not data or not isinstance(data, dict):
        _team_id_cache[canonical_name] = None
        return None

    teams = data.get("teams", [])
    if not teams:
        # Try shorter search
        short = canonical_name.replace(" FC", "").replace(" AFC", "").strip()
        data = _fd_get("/teams", params={"name": short})
        if not data or not isinstance(data, dict):
            _team_id_cache[canonical_name] = None
            return None
        teams = data.get("teams", [])

    if teams:
        team_id = teams[0].get("id")
        _team_id_cache[canonical_name] = team_id
        return team_id

    _team_id_cache[canonical_name] = None
    return None


def _fd_team_form(team_id: int, limit: int = 5) -> tuple[float | None, float | None, float | None]:
    """Fetch recent form for a team: (pts_per_game, goals_scored_avg, goals_conceded_avg)."""
    data = _fd_get(f"/teams/{team_id}/matches", params={"status": "FINISHED", "limit": limit})
    if not data or not isinstance(data, dict):
        return None, None, None

    matches = data.get("matches", [])
    if not matches:
        return None, None, None

    total_pts = 0
    total_scored = 0
    total_conceded = 0
    count = 0

    for m in matches:
        score = m.get("score", {})
        ft = score.get("fullTime", {})
        home_goals = ft.get("home")
        away_goals = ft.get("away")
        if home_goals is None or away_goals is None:
            continue

        home_id = m.get("homeTeam", {}).get("id")
        if home_id == team_id:
            total_scored += home_goals
            total_conceded += away_goals
            if home_goals > away_goals:
                total_pts += 3
            elif home_goals == away_goals:
                total_pts += 1
        else:
            total_scored += away_goals
            total_conceded += home_goals
            if away_goals > home_goals:
                total_pts += 3
            elif home_goals == away_goals:
                total_pts += 1
        count += 1

    if count == 0:
        return None, None, None

    return (
        round(total_pts / count, 2),
        round(total_scored / count, 2),
        round(total_conceded / count, 2),
    )


def _fd_league_position(team_id: int) -> int | None:
    """Get the team's current league position from standings."""
    # First, find which competition the team is in from their recent matches
    data = _fd_get(f"/teams/{team_id}/matches", params={"status": "FINISHED", "limit": 1})
    if not data or not isinstance(data, dict):
        return None

    matches = data.get("matches", [])
    if not matches:
        return None

    comp_id = matches[0].get("competition", {}).get("id")
    if not comp_id:
        return None

    standings_data = _fd_get(f"/competitions/{comp_id}/standings")
    if not standings_data or not isinstance(standings_data, dict):
        return None

    for standing in standings_data.get("standings", []):
        if standing.get("type") != "TOTAL":
            continue
        for entry in standing.get("table", []):
            if entry.get("team", {}).get("id") == team_id:
                return entry.get("position")

    return None


def _fetch_football_stats(home_canonical: str, away_canonical: str) -> MatchStats | None:
    """Fetch football stats from football-data.org."""
    home_id = _fd_resolve_team_id(home_canonical)
    away_id = _fd_resolve_team_id(away_canonical)

    if home_id is None and away_id is None:
        logger.info("Could not resolve either team ID for %s vs %s", home_canonical, away_canonical)
        return None

    stats = MatchStats(sport="football", home_team=home_canonical, away_team=away_canonical)

    if home_id is not None:
        pts, scored, conceded = _fd_team_form(home_id)
        stats.home_form_pts_per_game = pts
        stats.home_goals_scored_avg = scored
        stats.home_goals_conceded_avg = conceded
        stats.home_league_position = _fd_league_position(home_id)

    if away_id is not None:
        pts, scored, conceded = _fd_team_form(away_id)
        stats.away_form_pts_per_game = pts
        stats.away_goals_scored_avg = scored
        stats.away_goals_conceded_avg = conceded
        stats.away_league_position = _fd_league_position(away_id)

    stats.data_completeness = _compute_completeness(stats)
    return stats


# ---------------------------------------------------------------------------
# Squiggle API helpers (AFL)
# ---------------------------------------------------------------------------


def _squiggle_get(endpoint: str, params: dict | None = None) -> list | None:
    """GET request to Squiggle API."""
    url = f"{settings.stats.afl_api_base}/{endpoint}"
    headers = {"User-Agent": "Oracle_py/1.0 (https://github.com/HolsteredSoul/Oracle_py)"}
    try:
        resp = httpx.get(url, headers=headers, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data  # type: ignore[return-value]
    except Exception as exc:
        logger.warning("Squiggle API request failed: %s %s", endpoint, exc)
        return None


def _fetch_afl_stats(home_canonical: str, away_canonical: str) -> MatchStats | None:
    """Fetch AFL stats from Squiggle API."""
    import datetime

    year = datetime.datetime.now(datetime.timezone.utc).year

    # Get recent games for both teams
    stats = MatchStats(sport="afl", home_team=home_canonical, away_team=away_canonical)

    for team_name, is_home in [(home_canonical, True), (away_canonical, False)]:
        data = _squiggle_get("", params={"q": "games", "team": team_name, "year": year, "complete": 100})
        if not data or not isinstance(data, dict):
            continue

        games = data.get("games", [])
        if not games:
            continue

        # Take last 5 completed games
        recent = sorted(games, key=lambda g: g.get("date", ""), reverse=True)[:5]
        if not recent:
            continue

        total_pts = 0
        total_scored = 0
        total_conceded = 0
        count = 0

        for g in recent:
            hscore = g.get("hscore", 0) or 0
            ascore = g.get("ascore", 0) or 0
            hteam = g.get("hteam", "")

            if hteam == team_name:
                total_scored += hscore
                total_conceded += ascore
                if hscore > ascore:
                    total_pts += 4  # AFL uses 4 pts for a win
                elif hscore == ascore:
                    total_pts += 2
            else:
                total_scored += ascore
                total_conceded += hscore
                if ascore > hscore:
                    total_pts += 4
                elif hscore == ascore:
                    total_pts += 2
            count += 1

        if count > 0:
            if is_home:
                stats.home_form_pts_per_game = round(total_pts / count, 2)
                stats.home_goals_scored_avg = round(total_scored / count, 2)
                stats.home_goals_conceded_avg = round(total_conceded / count, 2)
            else:
                stats.away_form_pts_per_game = round(total_pts / count, 2)
                stats.away_goals_scored_avg = round(total_scored / count, 2)
                stats.away_goals_conceded_avg = round(total_conceded / count, 2)

    # Squiggle standings
    standings_data = _squiggle_get("", params={"q": "standings", "year": year})
    if standings_data and isinstance(standings_data, dict):
        for entry in standings_data.get("standings", []):
            team = entry.get("name", "")
            rank = entry.get("rank")
            if team == home_canonical and rank is not None:
                stats.home_league_position = rank
            elif team == away_canonical and rank is not None:
                stats.away_league_position = rank

    stats.data_completeness = _compute_completeness(stats)
    return stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_match_stats(
    home_team: str,
    away_team: str,
    sport: str = "football",
) -> MatchStats | None:
    """Fetch statistical features for an upcoming match.

    Routes to football-data.org or Squiggle API based on sport.
    Returns None if teams cannot be resolved or API fails.
    Caches results per (home_team, away_team, sport) tuple.
    """
    # Resolve Betfair names to canonical stats API names
    home_canonical = resolve_team(home_team, sport)
    away_canonical = resolve_team(away_team, sport)

    if home_canonical is None and away_canonical is None:
        return None

    # Use original names as fallback for partial resolution
    home_canonical = home_canonical or home_team
    away_canonical = away_canonical or away_team

    # Check cache
    cached = _cache_get(home_canonical, away_canonical, sport)
    if cached is not None:
        logger.debug("Stats cache hit: %s v %s (%s)", home_canonical, away_canonical, sport)
        return cached

    # Fetch from API
    if sport == "afl":
        stats = _fetch_afl_stats(home_canonical, away_canonical)
    else:
        stats = _fetch_football_stats(home_canonical, away_canonical)

    if stats is not None:
        _cache_set(home_canonical, away_canonical, sport, stats)
        logger.info(
            "Stats fetched: %s v %s (%s) completeness=%.0f%%",
            home_canonical, away_canonical, sport, stats.data_completeness * 100,
        )

    return stats
