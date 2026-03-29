"""ESPN hidden API client for rugby league and rugby union stats.

Provides standings, recent match scores, and team lookup for:
  - NRL (rugby-league, league 3)
  - Super Rugby Pacific (rugby, league 242041)
  - Premiership Rugby (rugby, league 267979)
  - Top 14 (rugby, league 270559)
  - URC (rugby, league 270557)

No API key required.  No documented rate limit.
Unofficial — may break without notice; callers should handle failures gracefully.
"""

from __future__ import annotations

import logging
import time
from difflib import SequenceMatcher

import httpx

from src.enrichment.team_mapping import normalize_team_name

from .models import TIMEOUT

logger = logging.getLogger(__name__)

_ESPN_BASE = "https://site.api.espn.com/apis"

# (espn_sport, espn_league_id)
LEAGUE_MAP: dict[str, tuple[str, str]] = {
    "nrl": ("rugby-league", "3"),
    "super_rugby": ("rugby", "242041"),
    "premiership": ("rugby", "267979"),
    "top14": ("rugby", "270559"),
    "urc": ("rugby", "270557"),
}

# Betfair competition ID → ESPN league key
BETFAIR_COMP_TO_ESPN: dict[int, str] = {
    10564377: "nrl",
    10536616: "super_rugby",
    11960510: "premiership",
    31201: "top14",
    # URC not in Betfair whitelist currently, but map it anyway
}

# Competition name substrings → ESPN league key
_COMP_NAME_HINTS: dict[str, str] = {
    "nrl": "nrl",
    "national rugby league": "nrl",
    "telstra": "nrl",
    "super rugby": "super_rugby",
    "premiership": "premiership",
    "gallagher": "premiership",
    "top 14": "top14",
    "urc": "urc",
    "united rugby": "urc",
}


def map_competition(competition: str) -> str | None:
    """Map a Betfair competition name/ID to an ESPN league key.

    Returns None if the competition isn't covered by ESPN.
    """
    # Try numeric competition ID first
    try:
        comp_id = int(competition)
        if comp_id in BETFAIR_COMP_TO_ESPN:
            return BETFAIR_COMP_TO_ESPN[comp_id]
    except (ValueError, TypeError):
        pass

    # Try name-based hints
    comp_lower = competition.lower()
    for hint, key in _COMP_NAME_HINTS.items():
        if hint in comp_lower:
            return key

    return None


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _espn_get(url: str, params: dict | None = None) -> dict | None:
    try:
        resp = httpx.get(url, params=params, timeout=TIMEOUT, follow_redirects=True)
        if resp.status_code != 200:
            logger.warning("ESPN %s returned %d", url, resp.status_code)
            return None
        return resp.json()
    except Exception as exc:
        logger.warning("ESPN request failed: %s — %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Standings cache: league_key -> (timestamp, {team_display_name_lower: rank})
# ---------------------------------------------------------------------------

_standings_cache: dict[str, tuple[float, dict[str, int]]] = {}
_STANDINGS_TTL = 24 * 3600  # 24 hours


def fetch_standings(league_key: str) -> dict[str, int]:
    """Fetch league standings: {team_display_name_lower: position}.

    Cached for 24 hours.
    """
    if league_key in _standings_cache:
        ts, cached = _standings_cache[league_key]
        if time.time() - ts < _STANDINGS_TTL:
            return cached

    sport, league_id = LEAGUE_MAP[league_key]
    url = f"{_ESPN_BASE}/v2/sports/{sport}/{league_id}/standings"
    data = _espn_get(url)
    if not data:
        return {}

    standings: dict[str, int] = {}
    for child in data.get("children", []):
        for entry in child.get("standings", {}).get("entries", []):
            team_name = entry.get("team", {}).get("displayName", "")
            stats = {s["name"]: s.get("value") for s in entry.get("stats", [])}
            rank = stats.get("rank")
            if team_name and rank is not None:
                standings[team_name.lower()] = int(rank)

    if standings:
        _standings_cache[league_key] = (time.time(), standings)
        logger.info("ESPN standings fetched: %s — %d teams", league_key, len(standings))

    return standings


# ---------------------------------------------------------------------------
# Team index cache: (league_key) -> (timestamp, {team_display_name_lower: team_id})
# ---------------------------------------------------------------------------

_teams_cache: dict[str, tuple[float, dict[str, dict]]] = {}
_TEAMS_TTL = 7 * 24 * 3600  # 7 days (team lists don't change mid-season)


def _fetch_teams(league_key: str) -> dict[str, dict]:
    """Fetch all teams: {name_lower: {"id": str, "displayName": str}}.

    Cached for 7 days.
    """
    if league_key in _teams_cache:
        ts, cached = _teams_cache[league_key]
        if time.time() - ts < _TEAMS_TTL:
            return cached

    sport, league_id = LEAGUE_MAP[league_key]
    url = f"{_ESPN_BASE}/site/v2/sports/{sport}/{league_id}/teams"
    data = _espn_get(url)
    if not data:
        return {}

    teams: dict[str, dict] = {}
    for sport_block in data.get("sports", []):
        for league_block in sport_block.get("leagues", []):
            for team_wrap in league_block.get("teams", []):
                t = team_wrap.get("team", {})
                tid = str(t.get("id", ""))
                display = t.get("displayName", "")
                short = t.get("shortDisplayName", "")
                if tid and display:
                    entry = {"id": tid, "displayName": display}
                    teams[display.lower()] = entry
                    if short:
                        teams[short.lower()] = entry

    if teams:
        _teams_cache[league_key] = (time.time(), teams)

    return teams


def resolve_team_id(name: str, league_key: str) -> str | None:
    """Resolve a Betfair runner name to an ESPN team ID."""
    teams = _fetch_teams(league_key)
    if not teams:
        return None

    name_lower = name.lower().strip()

    # Exact match on display name or short name
    if name_lower in teams:
        return teams[name_lower]["id"]

    # Score-based match
    best_score = 0.0
    best_id: str | None = None
    for key, entry in teams.items():
        score = _name_match_score(name, entry["displayName"])
        if score > best_score:
            best_score = score
            best_id = entry["id"]

    if best_score >= 0.60:
        return best_id

    logger.info("ESPN team not resolved: %s (best=%.2f) in %s", name, best_score, league_key)
    return None


def _name_match_score(query: str, candidate: str) -> float:
    """Score how well *query* (Betfair name) matches *candidate* (ESPN name).

    Combines fuzzy ratio with substring containment bonus.
    ESPN often uses short names ("Titans") while Betfair uses full
    names ("Gold Coast Titans"), so substring match is critical.
    """
    q_norm = normalize_team_name(query)
    c_norm = normalize_team_name(candidate)

    # Exact
    if q_norm == c_norm:
        return 1.0

    # Substring: ESPN short name inside Betfair full name (or vice versa)
    if c_norm in q_norm or q_norm in c_norm:
        return 0.90

    # Check individual words — "Titans" in "Gold Coast Titans"
    q_words = set(q_norm.split())
    c_words = set(c_norm.split())
    if c_words and c_words.issubset(q_words):
        return 0.85
    if q_words and q_words.issubset(c_words):
        return 0.85

    return SequenceMatcher(None, q_norm, c_norm).ratio()


def resolve_team_position(name: str, league_key: str) -> int | None:
    """Get a team's league position by matching against standings."""
    standings = fetch_standings(league_key)
    if not standings:
        return None

    name_lower = name.lower().strip()

    # Exact match
    if name_lower in standings:
        return standings[name_lower]

    # Score-based match
    best_score = 0.0
    best_rank: int | None = None
    for team_key, rank in standings.items():
        score = _name_match_score(name, team_key)
        if score > best_score:
            best_score = score
            best_rank = rank

    if best_score >= 0.60:
        return best_rank

    logger.info("ESPN standings: no match for %s (best=%.2f) in %s", name, best_score, league_key)
    return None


# ---------------------------------------------------------------------------
# Recent matches (scoreboard with date range)
# ---------------------------------------------------------------------------

def fetch_recent_matches(
    league_key: str,
    days_back: int = 30,
) -> list[dict]:
    """Fetch recent completed matches for a league.

    Returns list of dicts: {"home": str, "away": str, "home_score": int, "away_score": int, "date": str}
    """
    import datetime

    sport, league_id = LEAGUE_MAP[league_key]
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days_back)
    date_range = f"{start:%Y%m%d}-{end:%Y%m%d}"

    url = f"{_ESPN_BASE}/site/v2/sports/{sport}/{league_id}/scoreboard"
    data = _espn_get(url, params={"dates": date_range})
    if not data:
        return []

    matches: list[dict] = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            continue

        home_team = away_team = ""
        home_score = away_score = 0
        for c in comp.get("competitors", []):
            team_name = c.get("team", {}).get("displayName", "")
            try:
                score = int(c.get("score", 0))
            except (ValueError, TypeError):
                score = 0

            if c.get("homeAway") == "home":
                home_team = team_name
                home_score = score
            else:
                away_team = team_name
                away_score = score

        if home_team and away_team:
            matches.append({
                "home": home_team,
                "away": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "date": event.get("date", ""),
            })

    return matches


def compute_team_form(
    team_name: str,
    matches: list[dict],
    last_n: int = 5,
) -> tuple[float | None, float | None, float | None]:
    """Compute form for a team from recent matches.

    Returns (win_rate_pts, avg_scored, avg_conceded) or (None, None, None).
    win_rate_pts: 2 for win, 1 for draw, 0 for loss — averaged.
    """
    team_matches: list[dict] = []

    for m in matches:
        home_score = _name_match_score(team_name, m["home"])
        away_score = _name_match_score(team_name, m["away"])

        if home_score >= 0.60:
            team_matches.append({"scored": m["home_score"], "conceded": m["away_score"],
                                 "is_home": True})
        elif away_score >= 0.60:
            team_matches.append({"scored": m["away_score"], "conceded": m["home_score"],
                                 "is_home": False})

    if not team_matches:
        return None, None, None

    # Take most recent N
    team_matches = team_matches[-last_n:]

    wins = 0
    draws = 0
    total_scored = 0
    total_conceded = 0
    for m in team_matches:
        total_scored += m["scored"]
        total_conceded += m["conceded"]
        if m["scored"] > m["conceded"]:
            wins += 1
        elif m["scored"] == m["conceded"]:
            draws += 1

    n = len(team_matches)
    pts = round((wins * 2 + draws) / n, 2)
    return pts, round(total_scored / n, 2), round(total_conceded / n, 2)
