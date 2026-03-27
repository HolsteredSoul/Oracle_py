"""Rugby League (NRL) stats fetcher — TheSportsDB (free, no key required).

Covers NRL and UK Super League via TheSportsDB V1 API.

Limitations on free tier:
  - Team search works (searchteams.php)
  - Last events returns only 1 recent event per team (eventslast.php)
  - League standings are NOT available (limited to featured soccer on free tier)
  - lookup_all_teams returns wrong data for NRL league ID on free key

Despite these limits, we can still build a partial MatchStats with
recent form from last event data. Completeness will be lower than
other providers but still provides some signal vs mid-price fallback.
"""

from __future__ import annotations

import datetime as _dt
import logging
from difflib import SequenceMatcher

import httpx

from src.enrichment.team_mapping import FUZZY_THRESHOLD, normalize_team_name

from .models import TIMEOUT, MatchStats, compute_completeness

logger = logging.getLogger(__name__)

# TheSportsDB V1 free API — test key "3" is the documented free key
_TSDB_BASE = "https://www.thesportsdb.com/api/v1/json/3"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _tsdb_get(endpoint: str, params: dict | None = None) -> dict | None:
    """GET request to TheSportsDB V1 API."""
    url = f"{_TSDB_BASE}/{endpoint}"
    try:
        resp = httpx.get(url, params=params, timeout=TIMEOUT, follow_redirects=True)
        if resp.status_code == 429:
            logger.warning("TheSportsDB rate limit hit.")
            return None
        resp.raise_for_status()
        text = resp.text.strip()
        if not text:
            return None
        return resp.json()
    except Exception as exc:
        logger.warning("TheSportsDB request failed: %s %s", endpoint, exc)
        return None


# ---------------------------------------------------------------------------
# Team resolution via search
# ---------------------------------------------------------------------------

# Cache: betfair_name_lower -> {"id": str, "name": str}
_rl_team_cache: dict[str, dict | None] = {}

# Known NRL team aliases for better matching
_NRL_ALIASES: dict[str, str] = {
    "panthers": "Penrith Panthers",
    "storm": "Melbourne Storm",
    "roosters": "Sydney Roosters",
    "rabbitohs": "South Sydney Rabbitohs",
    "broncos": "Brisbane Broncos",
    "cowboys": "North Queensland Cowboys",
    "sea eagles": "Manly Sea Eagles",
    "manly warringah": "Manly Sea Eagles",
    "eels": "Parramatta Eels",
    "sharks": "Cronulla Sharks",
    "knights": "Newcastle Knights",
    "raiders": "Canberra Raiders",
    "dragons": "St George Illawarra Dragons",
    "bulldogs": "Canterbury Bulldogs",
    "warriors": "New Zealand Warriors",
    "titans": "Gold Coast Titans",
    "tigers": "Wests Tigers",
    "dolphins": "The Dolphins",
}


def _search_team(name: str) -> dict | None:
    """Search for a team in TheSportsDB."""
    key = name.lower().strip()
    if key in _rl_team_cache:
        return _rl_team_cache[key]

    # Try alias first
    search_name = _NRL_ALIASES.get(key, name)

    data = _tsdb_get("searchteams.php", params={"t": search_name})
    if not data:
        _rl_team_cache[key] = None
        return None

    teams = data.get("teams") or []

    # Filter to rugby league teams
    rl_teams = [t for t in teams if t.get("strSport", "").lower() in ("rugby", "rugby league")]

    if not rl_teams:
        # Try without sport filter if nothing found
        rl_teams = teams

    if not rl_teams:
        _rl_team_cache[key] = None
        return None

    # Find best match
    normalized = normalize_team_name(name)
    best_score = 0.0
    best_team: dict | None = None

    for t in rl_teams:
        team_name = t.get("strTeam", "")
        score = SequenceMatcher(None, normalized, normalize_team_name(team_name)).ratio()
        if score > best_score:
            best_score = score
            best_team = t

    if best_team is not None and best_score >= FUZZY_THRESHOLD:
        result = {
            "id": str(best_team.get("idTeam", "")),
            "name": best_team.get("strTeam", ""),
        }
        _rl_team_cache[key] = result
        logger.debug("TheSportsDB matched RL team %r -> %s (score=%.2f)", name, result["name"], best_score)
        return result

    # Direct match — take the first rugby result
    if rl_teams:
        t = rl_teams[0]
        result = {
            "id": str(t.get("idTeam", "")),
            "name": t.get("strTeam", ""),
        }
        _rl_team_cache[key] = result
        return result

    _rl_team_cache[key] = None
    return None


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def _fetch_last_events(team_id: str) -> list[dict]:
    """Fetch last completed events for a team.

    NOTE: Free tier returns only 1 last event (home games only).
    """
    data = _tsdb_get("eventslast.php", params={"id": team_id})
    if not data:
        return []
    return data.get("results") or []


def _compute_form(
    events: list[dict], team_id: str,
) -> tuple[float | None, float | None, float | None]:
    """Compute form from recent events: (win_rate_pts, pts_scored_avg, pts_conceded_avg).

    With free tier limit of 1 event, this gives a single-game snapshot.
    Still better than no data.
    """
    if not events:
        return None, None, None

    wins = 0
    draws = 0
    total_scored = 0
    total_conceded = 0
    counted = 0

    for e in events:
        home_id = str(e.get("idHomeTeam", ""))
        away_id = str(e.get("idAwayTeam", ""))
        home_score = e.get("intHomeScore")
        away_score = e.get("intAwayScore")

        if home_score is None or away_score is None:
            continue

        try:
            hs = int(home_score)
            as_ = int(away_score)
        except (ValueError, TypeError):
            continue

        is_home = home_id == team_id

        if is_home:
            total_scored += hs
            total_conceded += as_
            if hs > as_:
                wins += 1
            elif hs == as_:
                draws += 1
        else:
            total_scored += as_
            total_conceded += hs
            if as_ > hs:
                wins += 1
            elif hs == as_:
                draws += 1

        counted += 1

    if counted == 0:
        return None, None, None

    pts = round((wins * 2 + draws) / counted, 2)
    return (
        pts,
        round(total_scored / counted, 2),
        round(total_conceded / counted, 2),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_rugby_league_stats(
    home_canonical: str,
    away_canonical: str,
    competition: str = "",
) -> MatchStats | None:
    """Fetch rugby league stats from TheSportsDB.

    Covers NRL and UK Super League.
    Free tier provides limited data (1 last event, no standings).
    """
    home_team = _search_team(home_canonical)
    away_team = _search_team(away_canonical)

    if home_team is None and away_team is None:
        logger.info("Could not resolve either RL team: %s vs %s", home_canonical, away_canonical)
        return None

    stats = MatchStats(sport="rugby_league", home_team=home_canonical, away_team=away_canonical)

    # Form from last events
    if home_team is not None:
        events = _fetch_last_events(home_team["id"])
        pts, scored, conceded = _compute_form(events, home_team["id"])
        stats.home_form_pts_per_game = pts
        stats.home_goals_scored_avg = scored
        stats.home_goals_conceded_avg = conceded

    if away_team is not None:
        events = _fetch_last_events(away_team["id"])
        pts, scored, conceded = _compute_form(events, away_team["id"])
        stats.away_form_pts_per_game = pts
        stats.away_goals_scored_avg = scored
        stats.away_goals_conceded_avg = conceded

    # Standings NOT available on TheSportsDB free tier for NRL
    # league_position fields remain None

    stats.data_completeness = compute_completeness(stats)
    return stats
