"""Cricket stats fetcher — CricketData.org API (free, 100 req/day).

Covers international cricket (T20I, ODI, Test), IPL, BBL, PSL, and more.
Uses cricketdata.org (free API key required — set CRICKET_API_KEY in .env).
"""

from __future__ import annotations

import datetime as _dt
import logging
from difflib import SequenceMatcher

import httpx

from src.config import settings
from src.enrichment.team_mapping import FUZZY_THRESHOLD, normalize_team_name

from .models import TIMEOUT, MatchStats, compute_completeness

logger = logging.getLogger(__name__)

_CRICKET_API_BASE = "https://api.cricapi.com/v1"

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

_cr_daily_count = 0
_cr_daily_date: str = ""
_CR_DAILY_LIMIT = 90  # Leave 10 credits buffer from the 100 free


def _cr_rate_limit() -> bool:
    global _cr_daily_count, _cr_daily_date  # noqa: PLW0603
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    if today != _cr_daily_date:
        _cr_daily_count = 0
        _cr_daily_date = today

    if _cr_daily_count >= _CR_DAILY_LIMIT:
        logger.warning("CricketData daily limit reached (%d/%d)", _cr_daily_count, _CR_DAILY_LIMIT)
        return False

    _cr_daily_count += 1
    return True


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _cr_get(endpoint: str, params: dict | None = None) -> dict | None:
    """GET request to CricketData.org API."""
    api_key = getattr(settings, "cricket_api_key", "")
    if not api_key:
        logger.debug("No CRICKET_API_KEY configured — skipping cricket stats.")
        return None

    if not _cr_rate_limit():
        return None

    url = f"{_CRICKET_API_BASE}/{endpoint}"
    full_params = {"apikey": api_key}
    if params:
        full_params.update(params)

    try:
        resp = httpx.get(url, params=full_params, timeout=TIMEOUT)
        if resp.status_code == 429:
            logger.warning("CricketData rate limit hit.")
            return None
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            logger.debug("CricketData API error: %s", data.get("reason", "unknown"))
            return None
        return data
    except Exception as exc:
        logger.warning("CricketData request failed: %s %s", endpoint, exc)
        return None


# ---------------------------------------------------------------------------
# Team resolution
# ---------------------------------------------------------------------------

# Cricket team names are relatively standardized — Betfair uses country names
# for internationals and franchise names for T20 leagues
_CRICKET_TEAM_ALIASES: dict[str, str] = {
    "csk": "Chennai Super Kings",
    "mi": "Mumbai Indians",
    "rcb": "Royal Challengers Bangalore",
    "kkr": "Kolkata Knight Riders",
    "dc": "Delhi Capitals",
    "pbks": "Punjab Kings",
    "rr": "Rajasthan Royals",
    "srh": "Sunrisers Hyderabad",
    "gt": "Gujarat Titans",
    "lsg": "Lucknow Super Giants",
    # BBL
    "sydney sixers": "Sydney Sixers",
    "sydney thunder": "Sydney Thunder",
    "melbourne stars": "Melbourne Stars",
    "melbourne renegades": "Melbourne Renegades",
    "brisbane heat": "Brisbane Heat",
    "perth scorchers": "Perth Scorchers",
    "adelaide strikers": "Adelaide Strikers",
    "hobart hurricanes": "Hobart Hurricanes",
    # International short forms
    "eng": "England",
    "aus": "Australia",
    "ind": "India",
    "pak": "Pakistan",
    "sa": "South Africa",
    "nz": "New Zealand",
    "wi": "West Indies",
    "sl": "Sri Lanka",
    "ban": "Bangladesh",
    "afg": "Afghanistan",
    "zim": "Zimbabwe",
    "ire": "Ireland",
}


def _resolve_cricket_team(name: str) -> str:
    """Resolve a Betfair cricket team name to a standardized name."""
    lower = name.lower().strip()

    # Check aliases
    if lower in _CRICKET_TEAM_ALIASES:
        return _CRICKET_TEAM_ALIASES[lower]

    # Already a reasonable name — return as-is
    return name.strip()


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def _fetch_recent_matches(team_name: str) -> list[dict]:
    """Fetch recent completed matches for a team.

    Uses the /matches endpoint to find recent results.
    """
    # CricketData.org currentMatches endpoint returns ongoing + recent matches
    data = _cr_get("currentMatches")
    if not data:
        return []

    matches = data.get("data", [])
    team_lower = team_name.lower()

    team_matches = []
    for m in matches:
        if not m.get("matchEnded", False):
            continue
        teams = [
            (m.get("teamInfo") or [{}])[0].get("name", ""),
            (m.get("teamInfo") or [{}, {}])[-1].get("name", "") if len(m.get("teamInfo", [])) > 1 else "",
        ]
        # Also check team names in t1 and t2 fields
        t1 = m.get("t1", "")
        t2 = m.get("t2", "")

        if any(team_lower in t.lower() for t in [t1, t2] + teams if t):
            team_matches.append(m)

    return team_matches


def _compute_cricket_form(
    matches: list[dict], team_name: str, limit: int = 5,
) -> tuple[float | None, float | None, float | None]:
    """Compute form from recent matches: (win_rate_pts, runs_scored_avg, runs_conceded_avg).

    Cricket scoring is complex — we use total team score for runs_scored/conceded
    and win/loss for form points.
    """
    recent = matches[:limit]
    if not recent:
        return None, None, None

    team_lower = team_name.lower()
    wins = 0
    draws = 0
    total_scored = 0
    total_conceded = 0
    counted = 0

    for m in recent:
        status = (m.get("status") or "").lower()
        t1 = m.get("t1", "")
        t2 = m.get("t2", "")

        # Determine if our team is t1 or t2
        is_t1 = team_lower in t1.lower() if t1 else False
        is_t2 = team_lower in t2.lower() if t2 else False

        if not is_t1 and not is_t2:
            continue

        # Parse scores from t1s and t2s (e.g., "185/4 (20 ov)")
        t1s = m.get("t1s", "") or ""
        t2s = m.get("t2s", "") or ""

        def parse_score(s: str) -> int | None:
            s = s.strip()
            if not s or s == "-":
                return None
            # Handle "185/4 (20 ov)" or "185" or "185/4"
            parts = s.split("/")
            try:
                return int(parts[0].strip())
            except (ValueError, IndexError):
                return None

        s1 = parse_score(t1s)
        s2 = parse_score(t2s)

        if s1 is not None and s2 is not None:
            if is_t1:
                total_scored += s1
                total_conceded += s2
            else:
                total_scored += s2
                total_conceded += s1
            counted += 1

        # Determine win/loss from status
        if "won" in status:
            # Check if winning team matches our team
            if team_lower in status:
                wins += 1
            # else: loss
        elif "draw" in status or "no result" in status or "tied" in status:
            draws += 1

    if counted == 0:
        return None, None, None

    pts = round((wins * 2 + draws) / max(counted, 1), 2)
    return (
        pts,
        round(total_scored / counted, 2),
        round(total_conceded / counted, 2),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_cricket_stats(
    home_canonical: str,
    away_canonical: str,
    competition: str = "",
) -> MatchStats | None:
    """Fetch cricket stats from CricketData.org.

    Covers IPL, BBL, international cricket, and other major tournaments.
    """
    api_key = getattr(settings, "cricket_api_key", "")
    if not api_key:
        logger.debug("No CRICKET_API_KEY — skipping cricket stats.")
        return None

    home_name = _resolve_cricket_team(home_canonical)
    away_name = _resolve_cricket_team(away_canonical)

    stats = MatchStats(sport="cricket", home_team=home_canonical, away_team=away_canonical)

    # Fetch recent matches for both teams
    home_matches = _fetch_recent_matches(home_name)
    away_matches = _fetch_recent_matches(away_name)

    if not home_matches and not away_matches:
        logger.info("No recent cricket matches found for %s vs %s", home_canonical, away_canonical)
        return None

    # Compute form
    if home_matches:
        pts, scored, conceded = _compute_cricket_form(home_matches, home_name)
        stats.home_form_pts_per_game = pts
        stats.home_goals_scored_avg = scored
        stats.home_goals_conceded_avg = conceded

    if away_matches:
        pts, scored, conceded = _compute_cricket_form(away_matches, away_name)
        stats.away_form_pts_per_game = pts
        stats.away_goals_scored_avg = scored
        stats.away_goals_conceded_avg = conceded

    # Cricket doesn't have universal league standings in the same way —
    # standings are competition-specific and not easily queried from this API.
    # Leave league_position as None; form data alone should provide value.

    stats.data_completeness = compute_completeness(stats)
    return stats
