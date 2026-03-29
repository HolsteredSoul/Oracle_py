"""Rugby League stats fetcher — ESPN (NRL) + TheSportsDB fallback (Super League).

NRL: Full stats via ESPN hidden API — standings, form from last 5 games.
Super League: ESPN has no data; falls back to TheSportsDB (1 last event, no standings).
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

import httpx

from src.enrichment.team_mapping import FUZZY_THRESHOLD, normalize_team_name

from .espn import (
    compute_team_form,
    fetch_recent_matches,
    map_competition,
    resolve_team_position,
)
from .models import TIMEOUT, MatchStats, compute_completeness

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TheSportsDB fallback (Super League only)
# ---------------------------------------------------------------------------

_TSDB_BASE = "https://www.thesportsdb.com/api/v1/json/3"


def _tsdb_get(endpoint: str, params: dict | None = None) -> dict | None:
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


_rl_team_cache: dict[str, dict | None] = {}


def _search_team_tsdb(name: str) -> dict | None:
    key = name.lower().strip()
    if key in _rl_team_cache:
        return _rl_team_cache[key]

    data = _tsdb_get("searchteams.php", params={"t": name})
    if not data:
        _rl_team_cache[key] = None
        return None

    teams = data.get("teams") or []
    rl_teams = [t for t in teams if t.get("strSport", "").lower() in ("rugby", "rugby league")]
    if not rl_teams:
        rl_teams = teams
    if not rl_teams:
        _rl_team_cache[key] = None
        return None

    normalized = normalize_team_name(name)
    best_score = 0.0
    best_team: dict | None = None
    for t in rl_teams:
        score = SequenceMatcher(None, normalized, normalize_team_name(t.get("strTeam", ""))).ratio()
        if score > best_score:
            best_score = score
            best_team = t

    if best_team is not None and best_score >= FUZZY_THRESHOLD:
        result = {"id": str(best_team.get("idTeam", "")), "name": best_team.get("strTeam", "")}
        _rl_team_cache[key] = result
        return result

    if rl_teams:
        t = rl_teams[0]
        result = {"id": str(t.get("idTeam", "")), "name": t.get("strTeam", "")}
        _rl_team_cache[key] = result
        return result

    _rl_team_cache[key] = None
    return None


def _fetch_last_events(team_id: str) -> list[dict]:
    data = _tsdb_get("eventslast.php", params={"id": team_id})
    if not data:
        return []
    return data.get("results") or []


def _compute_form_tsdb(
    events: list[dict], team_id: str,
) -> tuple[float | None, float | None, float | None]:
    if not events:
        return None, None, None

    wins = draws = total_scored = total_conceded = counted = 0
    for e in events:
        home_id = str(e.get("idHomeTeam", ""))
        home_score = e.get("intHomeScore")
        away_score = e.get("intAwayScore")
        if home_score is None or away_score is None:
            continue
        try:
            hs, as_ = int(home_score), int(away_score)
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
    return pts, round(total_scored / counted, 2), round(total_conceded / counted, 2)


def _fetch_via_thesportsdb(home: str, away: str) -> MatchStats | None:
    """Super League fallback — TheSportsDB (1 last event, no standings)."""
    home_team = _search_team_tsdb(home)
    away_team = _search_team_tsdb(away)

    if home_team is None and away_team is None:
        logger.info("Could not resolve either RL team (TSDB): %s vs %s", home, away)
        return None

    stats = MatchStats(sport="rugby_league", home_team=home, away_team=away)

    if home_team is not None:
        events = _fetch_last_events(home_team["id"])
        pts, scored, conceded = _compute_form_tsdb(events, home_team["id"])
        stats.home_form_pts_per_game = pts
        stats.home_goals_scored_avg = scored
        stats.home_goals_conceded_avg = conceded

    if away_team is not None:
        events = _fetch_last_events(away_team["id"])
        pts, scored, conceded = _compute_form_tsdb(events, away_team["id"])
        stats.away_form_pts_per_game = pts
        stats.away_goals_scored_avg = scored
        stats.away_goals_conceded_avg = conceded

    stats.data_completeness = compute_completeness(stats)
    return stats


# ---------------------------------------------------------------------------
# ESPN path (NRL)
# ---------------------------------------------------------------------------

def _fetch_via_espn(home: str, away: str, league_key: str) -> MatchStats | None:
    """NRL via ESPN — standings + form from last 5 games."""
    stats = MatchStats(sport="rugby_league", home_team=home, away_team=away)

    # Standings
    home_pos = resolve_team_position(home, league_key)
    away_pos = resolve_team_position(away, league_key)
    stats.home_league_position = home_pos
    stats.away_league_position = away_pos

    # Form from recent matches
    matches = fetch_recent_matches(league_key, days_back=60)
    if matches:
        h_pts, h_scored, h_conceded = compute_team_form(home, matches)
        stats.home_form_pts_per_game = h_pts
        stats.home_goals_scored_avg = h_scored
        stats.home_goals_conceded_avg = h_conceded

        a_pts, a_scored, a_conceded = compute_team_form(away, matches)
        stats.away_form_pts_per_game = a_pts
        stats.away_goals_scored_avg = a_scored
        stats.away_goals_conceded_avg = a_conceded

    stats.data_completeness = compute_completeness(stats)

    if stats.data_completeness == 0.0:
        logger.info("ESPN returned no usable data for %s v %s (%s)", home, away, league_key)
        return None

    logger.info(
        "ESPN RL stats: %s v %s — completeness=%.0f%% (pos=%s/%s)",
        home, away, stats.data_completeness * 100, home_pos, away_pos,
    )
    return stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_rugby_league_stats(
    home_canonical: str,
    away_canonical: str,
    competition: str = "",
) -> MatchStats | None:
    """Fetch rugby league stats — ESPN for NRL, TheSportsDB for Super League."""
    league_key = map_competition(competition)

    if league_key == "nrl":
        result = _fetch_via_espn(home_canonical, away_canonical, league_key)
        if result is not None:
            return result
        logger.info("ESPN failed for NRL %s v %s, falling back to TheSportsDB",
                     home_canonical, away_canonical)

    return _fetch_via_thesportsdb(home_canonical, away_canonical)
