"""Ice Hockey (NHL) stats fetcher — NHL Stats API (free, no auth).

Uses the public NHL API at api-web.nhle.com.
Only covers NHL. Other hockey leagues fall back to mid-price.
"""

from __future__ import annotations

import datetime as _dt
import logging
from difflib import SequenceMatcher

import httpx

from src.enrichment.team_mapping import FUZZY_THRESHOLD, normalize_team_name

from .models import TIMEOUT, MatchStats, compute_completeness

logger = logging.getLogger(__name__)

_NHL_API_BASE = "https://api-web.nhle.com/v1"
_NHL_STATS_BASE = "https://api.nhle.com/stats/rest/en"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _nhl_get(url: str, params: dict | None = None) -> dict | list | None:
    """GET request to NHL API (no auth needed)."""
    try:
        resp = httpx.get(url, params=params, timeout=TIMEOUT, follow_redirects=True)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("NHL API request failed: %s %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Team index
# ---------------------------------------------------------------------------

# {name_lower: {"id": int, "name": str, "abbrev": str}}
_nhl_team_index: dict[str, dict] = {}


def _build_team_index() -> None:
    """Build NHL team index from standings (always current roster of teams)."""
    if _nhl_team_index:
        return

    data = _nhl_get(f"{_NHL_API_BASE}/standings/now")
    if not data or not isinstance(data, dict):
        logger.warning("Failed to build NHL team index from standings")
        return

    for entry in data.get("standings", []):
        team_name = entry.get("teamName", {})
        # teamName has {"default": "Utah Hockey Club", "fr": "..."}
        name = team_name.get("default", "") if isinstance(team_name, dict) else str(team_name)
        abbrev = entry.get("teamAbbrev", {})
        abbrev_str = abbrev.get("default", "") if isinstance(abbrev, dict) else str(abbrev)
        # Build a full name like "Utah Hockey Club"
        place = entry.get("placeName", {})
        place_str = place.get("default", "") if isinstance(place, dict) else str(place)
        full_name = f"{place_str} {name}".strip() if place_str else name

        team_info = {
            "id": abbrev_str,  # NHL API uses abbreviation as primary ID
            "name": name,
            "full_name": full_name,
            "abbrev": abbrev_str,
        }

        _nhl_team_index[full_name.lower()] = team_info
        _nhl_team_index[name.lower()] = team_info
        if abbrev_str:
            _nhl_team_index[abbrev_str.lower()] = team_info

    logger.debug("NHL team index built: %d entries", len(_nhl_team_index))


def _resolve_team(name: str) -> dict | None:
    """Resolve a Betfair runner name to an NHL team dict."""
    _build_team_index()
    key = name.lower().strip()

    if key in _nhl_team_index:
        return _nhl_team_index[key]

    normalized = normalize_team_name(name)
    for idx_key, team in _nhl_team_index.items():
        if normalize_team_name(idx_key) == normalized:
            _nhl_team_index[key] = team
            return team

    best_score = 0.0
    best_team: dict | None = None
    seen: set[str] = set()
    for idx_key, team in _nhl_team_index.items():
        tid = team["abbrev"]
        if tid in seen:
            continue
        seen.add(tid)
        score = SequenceMatcher(None, normalized, normalize_team_name(team["full_name"])).ratio()
        if score > best_score:
            best_score = score
            best_team = team

    if best_score >= FUZZY_THRESHOLD and best_team is not None:
        _nhl_team_index[key] = best_team
        logger.debug("Fuzzy-matched NHL team %r -> %s (score=%.2f)", name, best_team["full_name"], best_score)
        return best_team

    logger.debug("No NHL match for %r (best=%.2f)", name, best_score)
    return None


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

_standings_cache: dict[str, list[dict]] = {}


def _fetch_standings_raw() -> list[dict]:
    """Fetch current NHL standings. Returns list of team standing entries."""
    cache_key = "current"
    if cache_key in _standings_cache:
        return _standings_cache[cache_key]

    data = _nhl_get(f"{_NHL_API_BASE}/standings/now")
    if not data or not isinstance(data, dict):
        _standings_cache[cache_key] = []
        return []

    entries = data.get("standings", [])
    _standings_cache[cache_key] = entries
    return entries


def _get_team_standing(team_abbrev: str) -> tuple[int | None, float | None, float | None]:
    """Get (league_rank, goals_for_avg, goals_against_avg) from standings."""
    entries = _fetch_standings_raw()
    for i, entry in enumerate(entries):
        abbrev = entry.get("teamAbbrev", {})
        abbrev_str = abbrev.get("default", "") if isinstance(abbrev, dict) else str(abbrev)
        if abbrev_str == team_abbrev:
            gp = entry.get("gamesPlayed", 0) or 1
            gf = entry.get("goalFor", 0) or 0
            ga = entry.get("goalAgainst", 0) or 0
            # Conference rank — standings are sorted by points pct
            rank = entry.get("conferenceSequence") or entry.get("leagueSequence") or (i + 1)
            return int(rank), round(gf / gp, 2), round(ga / gp, 2)

    return None, None, None


def _get_team_form(
    team_abbrev: str, limit: int = 5,
) -> tuple[float | None, float | None, float | None]:
    """Fetch recent form from club schedule: (win_rate_pts, goals_scored_avg, goals_allowed_avg)."""
    today = _dt.date.today()
    # Get schedule for current month and previous month
    data = _nhl_get(
        f"{_NHL_API_BASE}/club-schedule-season/{team_abbrev}/now",
    )
    if not data or not isinstance(data, dict):
        return None, None, None

    games = data.get("games", [])
    # Filter to completed games (gameState = "OFF" means final)
    finished = [g for g in games if g.get("gameState") in ("OFF", "FINAL", "7")]
    # Sort by date descending
    finished.sort(key=lambda g: g.get("gameDate", ""), reverse=True)
    recent = finished[:limit]

    if not recent:
        return None, None, None

    wins = 0
    total_scored = 0
    total_conceded = 0

    for g in recent:
        home_abbrev = g.get("homeTeam", {}).get("abbrev", "")
        home_score = g.get("homeTeam", {}).get("score", 0) or 0
        away_score = g.get("awayTeam", {}).get("score", 0) or 0

        if home_abbrev == team_abbrev:
            total_scored += home_score
            total_conceded += away_score
            if home_score > away_score:
                wins += 1
        else:
            total_scored += away_score
            total_conceded += home_score
            if away_score > home_score:
                wins += 1

    count = len(recent)
    win_rate_pts = round(2.0 * wins / count, 2)
    return (
        win_rate_pts,
        round(total_scored / count, 2),
        round(total_conceded / count, 2),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_hockey_stats(
    home_canonical: str,
    away_canonical: str,
    competition: str = "",
) -> MatchStats | None:
    """Fetch ice hockey stats from NHL API.

    Only covers NHL. Non-NHL competitions return None.
    """
    # Only NHL is supported
    comp_lower = competition.lower() if competition else ""
    is_nhl = (
        not comp_lower
        or "nhl" in comp_lower
        or "national hockey" in comp_lower
        or "ice hockey" in comp_lower
    )
    if not is_nhl:
        logger.debug("Non-NHL hockey competition %r — skipping.", competition)
        return None

    home_team = _resolve_team(home_canonical)
    away_team = _resolve_team(away_canonical)

    if home_team is None and away_team is None:
        logger.info("Could not resolve either NHL team: %s vs %s", home_canonical, away_canonical)
        return None

    stats = MatchStats(sport="hockey", home_team=home_canonical, away_team=away_canonical)

    # Form (recent games) — prefer this over standings averages for form fields
    if home_team is not None:
        pts, scored, conceded = _get_team_form(home_team["abbrev"])
        stats.home_form_pts_per_game = pts
        stats.home_goals_scored_avg = scored
        stats.home_goals_conceded_avg = conceded

    if away_team is not None:
        pts, scored, conceded = _get_team_form(away_team["abbrev"])
        stats.away_form_pts_per_game = pts
        stats.away_goals_scored_avg = scored
        stats.away_goals_conceded_avg = conceded

    # Standings — use for league position (and fill goals avgs if form is empty)
    if home_team is not None:
        rank, gf_avg, ga_avg = _get_team_standing(home_team["abbrev"])
        stats.home_league_position = rank
        if stats.home_goals_scored_avg is None:
            stats.home_goals_scored_avg = gf_avg
        if stats.home_goals_conceded_avg is None:
            stats.home_goals_conceded_avg = ga_avg

    if away_team is not None:
        rank, gf_avg, ga_avg = _get_team_standing(away_team["abbrev"])
        stats.away_league_position = rank
        if stats.away_goals_scored_avg is None:
            stats.away_goals_scored_avg = gf_avg
        if stats.away_goals_conceded_avg is None:
            stats.away_goals_conceded_avg = ga_avg

    stats.data_completeness = compute_completeness(stats)
    return stats
