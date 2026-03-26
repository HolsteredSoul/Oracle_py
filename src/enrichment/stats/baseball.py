"""MLB stats fetcher — MLB Stats API (free, no auth required)."""

from __future__ import annotations

import datetime
import logging

import httpx

from src.config import settings

from .models import TIMEOUT, MatchStats, compute_completeness

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Team index cache
# ---------------------------------------------------------------------------
_team_index: dict[str, int] = {}  # lowercase name/abbrev -> team ID
_team_index_built = False


def _mlb_get(path: str, params: dict | None = None) -> dict | None:
    """GET request to MLB Stats API. Returns parsed JSON or None on error."""
    url = f"{settings.stats.mlb_api_base}{path}"
    headers = {"User-Agent": "Oracle_py/1.0 (https://github.com/HolsteredSoul/Oracle_py)"}
    try:
        resp = httpx.get(url, headers=headers, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("MLB Stats API request failed: %s %s", path, exc)
        return None


def _build_team_index() -> None:
    """Build name -> team_id index from MLB teams endpoint."""
    global _team_index_built
    if _team_index_built:
        return

    _team_index_built = True
    year = datetime.datetime.now(datetime.timezone.utc).year
    data = _mlb_get("/teams", params={"sportId": 1, "season": year})
    if not data:
        return

    for team in data.get("teams", []):
        tid = team.get("id")
        name = team.get("name", "")
        abbrev = team.get("abbreviation", "")
        short = team.get("shortName", "")
        club = team.get("clubName", "")
        if tid and name:
            _team_index[name.lower()] = tid
            if abbrev:
                _team_index[abbrev.lower()] = tid
            if short:
                _team_index[short.lower()] = tid
            if club:
                _team_index[club.lower()] = tid

    logger.debug("MLB team index built: %d entries", len(_team_index))


def _resolve_team_id(name: str) -> int | None:
    """Resolve a Betfair team name to an MLB team ID."""
    _build_team_index()

    key = name.lower()
    if key in _team_index:
        return _team_index[key]

    # Strip common suffixes Betfair adds like "(J Smith)" for pitcher names
    stripped = key.split("(")[0].strip()
    if stripped in _team_index:
        return _team_index[stripped]

    # Fuzzy: check if any index key is contained in the Betfair name or vice versa
    for idx_key, tid in _team_index.items():
        if idx_key in key or key in idx_key:
            _team_index[key] = tid
            return tid

    logger.debug("No MLB team match for %r", name)
    return None


def _fetch_form(team_id: int, limit: int = 5) -> tuple[float | None, float | None, float | None]:
    """Fetch recent form: (pts_per_game, runs_scored_avg, runs_allowed_avg)."""
    today = datetime.date.today()
    start = today - datetime.timedelta(days=30)
    # Try regular season first, fall back to spring training early in season
    data = _mlb_get(
        "/schedule",
        params={
            "sportId": 1,
            "teamId": team_id,
            "startDate": start.strftime("%Y-%m-%d"),
            "endDate": today.strftime("%Y-%m-%d"),
            "gameType": "R,S",
            "hydrate": "linescore",
        },
    )
    if not data:
        return None, None, None

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            if status == "Final":
                games.append(game)

    if not games:
        return None, None, None

    # Take last N completed games
    recent = games[-limit:]
    total_wins = 0
    total_scored = 0
    total_allowed = 0
    count = 0

    for game in recent:
        teams = game.get("teams", {})
        home = teams.get("home", {})
        away = teams.get("away", {})

        home_id = home.get("team", {}).get("id")
        home_score = home.get("score", 0) or 0
        away_score = away.get("score", 0) or 0

        if home_id == team_id:
            total_scored += home_score
            total_allowed += away_score
            if home.get("isWinner"):
                total_wins += 1
        else:
            total_scored += away_score
            total_allowed += home_score
            if away.get("isWinner"):
                total_wins += 1
        count += 1

    if count == 0:
        return None, None, None

    # pts_per_game on 0-2 scale (2 = all wins)
    pts_per_game = round((total_wins / count) * 2.0, 2)
    scored_avg = round(total_scored / count, 2)
    allowed_avg = round(total_allowed / count, 2)
    return pts_per_game, scored_avg, allowed_avg


def _fetch_standings(team_id: int) -> int | None:
    """Fetch league rank for a team from standings."""
    year = datetime.datetime.now(datetime.timezone.utc).year
    data = _mlb_get("/standings", params={"leagueId": "103,104", "season": year})
    if not data:
        return None

    for record in data.get("records", []):
        for entry in record.get("teamRecords", []):
            if entry.get("team", {}).get("id") == team_id:
                # Use leagueRank (1-15 per league)
                rank_str = entry.get("leagueRank", "")
                try:
                    return int(rank_str)
                except (ValueError, TypeError):
                    return None
    return None


def fetch_baseball_stats(
    home_canonical: str, away_canonical: str,
) -> MatchStats | None:
    """Fetch baseball stats from MLB Stats API."""
    home_id = _resolve_team_id(home_canonical)
    away_id = _resolve_team_id(away_canonical)

    if home_id is None and away_id is None:
        logger.info("Could not resolve either MLB team: %s vs %s", home_canonical, away_canonical)
        return None

    stats = MatchStats(sport="baseball", home_team=home_canonical, away_team=away_canonical)

    if home_id is not None:
        pts, scored, allowed = _fetch_form(home_id)
        stats.home_form_pts_per_game = pts
        stats.home_goals_scored_avg = scored
        stats.home_goals_conceded_avg = allowed
        stats.home_league_position = _fetch_standings(home_id)

    if away_id is not None:
        pts, scored, allowed = _fetch_form(away_id)
        stats.away_form_pts_per_game = pts
        stats.away_goals_scored_avg = scored
        stats.away_goals_conceded_avg = allowed
        stats.away_league_position = _fetch_standings(away_id)

    stats.data_completeness = compute_completeness(stats)
    return stats
