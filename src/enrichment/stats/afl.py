"""AFL stats fetcher — official AFL API (aflapi.afl.com.au, free, no auth)."""

from __future__ import annotations

import logging

import httpx

from .models import MatchStats, TIMEOUT, compute_completeness

logger = logging.getLogger(__name__)

_AFL_API_BASE = "https://aflapi.afl.com.au/afl/v2"
_COMP_SEASON_ID = 85  # 2026 Toyota AFL Premiership
_HEADERS = {"User-Agent": "Oracle_py/1.0"}


def _afl_get(path: str, params: dict | None = None) -> dict | None:
    """GET request to AFL API. Returns parsed JSON or None on error."""
    url = f"{_AFL_API_BASE}/{path}"
    try:
        resp = httpx.get(url, headers=_HEADERS, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("AFL API request failed: %s %s", path, exc)
        return None


def _get_completed_matches() -> list[dict]:
    """Fetch all completed matches for the current season."""
    all_matches: list[dict] = []
    for rnd in range(1, 30):  # AFL has up to ~25 rounds
        data = _afl_get("matches", params={
            "roundNumber": str(rnd),
            "compSeasonId": str(_COMP_SEASON_ID),
        })
        if not data:
            break
        matches = data.get("matches", [])
        if not matches:
            break
        concluded = [m for m in matches if m.get("status") == "CONCLUDED"]
        all_matches.extend(concluded)
        # Stop when we hit a round with no concluded matches (future rounds)
        if not concluded:
            break
    return all_matches


def _team_form(matches: list[dict], team_name: str) -> tuple[float, float, float, int] | None:
    """Calculate form stats for a team from match list.

    Returns (pts_per_game, scored_avg, conceded_avg, count) or None.
    """
    # Filter to matches involving this team
    team_matches = []
    for m in matches:
        home = m.get("home", {}).get("team", {}).get("name", "")
        away = m.get("away", {}).get("team", {}).get("name", "")
        if team_name in (home, away):
            team_matches.append(m)

    if not team_matches:
        return None

    # Take last 5
    recent = team_matches[-5:]

    total_pts = 0
    total_scored = 0
    total_conceded = 0

    for m in recent:
        home_name = m.get("home", {}).get("team", {}).get("name", "")
        home_score = m.get("home", {}).get("score", {}).get("totalScore", 0) or 0
        away_score = m.get("away", {}).get("score", {}).get("totalScore", 0) or 0

        if home_name == team_name:
            total_scored += home_score
            total_conceded += away_score
            if home_score > away_score:
                total_pts += 4
            elif home_score == away_score:
                total_pts += 2
        else:
            total_scored += away_score
            total_conceded += home_score
            if away_score > home_score:
                total_pts += 4
            elif home_score == away_score:
                total_pts += 2

    count = len(recent)
    return (
        round(total_pts / count, 2),
        round(total_scored / count, 2),
        round(total_conceded / count, 2),
        count,
    )


def fetch_afl_stats(home_canonical: str, away_canonical: str) -> MatchStats | None:
    """Fetch AFL stats from official AFL API."""
    matches = _get_completed_matches()
    if not matches:
        logger.info("AFL API: no completed matches found for season %s", _COMP_SEASON_ID)
        return None

    stats = MatchStats(sport="afl", home_team=home_canonical, away_team=away_canonical)

    home_form = _team_form(matches, home_canonical)
    if home_form:
        stats.home_form_pts_per_game = home_form[0]
        stats.home_goals_scored_avg = home_form[1]
        stats.home_goals_conceded_avg = home_form[2]

    away_form = _team_form(matches, away_canonical)
    if away_form:
        stats.away_form_pts_per_game = away_form[0]
        stats.away_goals_scored_avg = away_form[1]
        stats.away_goals_conceded_avg = away_form[2]

    # Build ladder from results for league position
    wins: dict[str, int] = {}
    for m in matches:
        home_name = m.get("home", {}).get("team", {}).get("name", "")
        away_name = m.get("away", {}).get("team", {}).get("name", "")
        home_score = m.get("home", {}).get("score", {}).get("totalScore", 0) or 0
        away_score = m.get("away", {}).get("score", {}).get("totalScore", 0) or 0
        wins.setdefault(home_name, 0)
        wins.setdefault(away_name, 0)
        if home_score > away_score:
            wins[home_name] += 1
        elif away_score > home_score:
            wins[away_name] += 1

    ladder = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    for rank, (team, _) in enumerate(ladder, 1):
        if team == home_canonical:
            stats.home_league_position = rank
        elif team == away_canonical:
            stats.away_league_position = rank

    stats.data_completeness = compute_completeness(stats)
    return stats
