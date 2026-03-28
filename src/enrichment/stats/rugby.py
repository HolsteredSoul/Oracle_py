"""Rugby Union stats fetcher — TheSportsDB (free, no key required).

Covers Super Rugby Pacific, URC, Premiership Rugby, Top 14, and more.
Uses the same TheSportsDB V1 API as the rugby_league provider.

Limitations on free tier:
  - Team search works (searchteams.php)
  - Last events returns only 1 recent event per team (eventslast.php)
  - League standings are NOT available on free tier
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

import httpx

from src.enrichment.team_mapping import FUZZY_THRESHOLD, normalize_team_name

from .models import TIMEOUT, MatchStats, compute_completeness

logger = logging.getLogger(__name__)

_TSDB_BASE = "https://www.thesportsdb.com/api/v1/json/3"


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


# Cache: betfair_name_lower -> {"id": str, "name": str} | None
_ru_team_cache: dict[str, dict | None] = {}


def _search_team(name: str) -> dict | None:
    """Search for a rugby union team in TheSportsDB."""
    key = name.lower().strip()
    if key in _ru_team_cache:
        return _ru_team_cache[key]

    data = _tsdb_get("searchteams.php", params={"t": name})
    if not data:
        _ru_team_cache[key] = None
        return None

    teams = data.get("teams") or []

    # Filter to rugby union teams
    rugby_teams = [t for t in teams if t.get("strSport", "").lower() == "rugby"]

    if not rugby_teams:
        _ru_team_cache[key] = None
        return None

    # Find best match
    normalized = normalize_team_name(name)
    best_score = 0.0
    best_team: dict | None = None

    for t in rugby_teams:
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
        _ru_team_cache[key] = result
        return result

    # Fallback: take the first rugby result
    if rugby_teams:
        t = rugby_teams[0]
        result = {
            "id": str(t.get("idTeam", "")),
            "name": t.get("strTeam", ""),
        }
        _ru_team_cache[key] = result
        return result

    _ru_team_cache[key] = None
    return None


def _fetch_last_events(team_id: str) -> list[dict]:
    """Fetch last completed events for a team."""
    data = _tsdb_get("eventslast.php", params={"id": team_id})
    if not data:
        return []
    return data.get("results") or []


def _compute_form(
    events: list[dict], team_id: str,
) -> tuple[float | None, float | None, float | None]:
    """Compute form from recent events: (win_rate_pts, pts_scored_avg, pts_conceded_avg)."""
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


def fetch_rugby_stats(
    home_canonical: str,
    away_canonical: str,
    competition: str = "",
) -> MatchStats | None:
    """Fetch rugby union stats from TheSportsDB."""
    home_team = _search_team(home_canonical)
    away_team = _search_team(away_canonical)

    if home_team is None and away_team is None:
        logger.info("Could not resolve either rugby team: %s vs %s", home_canonical, away_canonical)
        return None

    stats = MatchStats(sport="rugby", home_team=home_canonical, away_team=away_canonical)

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

    stats.data_completeness = compute_completeness(stats)
    return stats
