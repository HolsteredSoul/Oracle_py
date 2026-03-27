"""Basketball stats fetcher — nba_api (NBA.com, free, no key required).

Covers NBA only. NBL (Australian) and other leagues fall back to mid-price.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

from src.enrichment.team_mapping import FUZZY_THRESHOLD, normalize_team_name

from .models import TIMEOUT, MatchStats, compute_completeness

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — nba_api is optional; return None gracefully if missing
# ---------------------------------------------------------------------------

_nba_available: bool | None = None


def _check_nba_api() -> bool:
    global _nba_available  # noqa: PLW0603
    if _nba_available is None:
        try:
            import nba_api  # noqa: F401

            _nba_available = True
        except ImportError:
            logger.warning("nba_api not installed — basketball stats disabled. pip install nba_api")
            _nba_available = False
    return _nba_available


# ---------------------------------------------------------------------------
# Team index (static — NBA teams don't change mid-season)
# ---------------------------------------------------------------------------

# {name_lower: {"id": int, "full_name": str, "abbreviation": str}}
_nba_team_index: dict[str, dict] = {}


def _build_team_index() -> None:
    if _nba_team_index:
        return
    from nba_api.stats.static import teams as nba_teams

    for t in nba_teams.get_teams():
        key = t["full_name"].lower()
        _nba_team_index[key] = t
        # Also index by abbreviation and nickname
        _nba_team_index[t["abbreviation"].lower()] = t
        _nba_team_index[t["nickname"].lower()] = t


def _resolve_team(name: str) -> dict | None:
    """Resolve a Betfair runner name to an NBA team dict."""
    _build_team_index()
    key = name.lower().strip()

    # Exact match
    if key in _nba_team_index:
        return _nba_team_index[key]

    # Normalized match
    normalized = normalize_team_name(name)
    for idx_key, team in _nba_team_index.items():
        if normalize_team_name(idx_key) == normalized:
            _nba_team_index[key] = team
            return team

    # Fuzzy match
    best_score = 0.0
    best_team: dict | None = None
    seen_ids: set[int] = set()
    for idx_key, team in _nba_team_index.items():
        if team["id"] in seen_ids:
            continue
        seen_ids.add(team["id"])
        score = SequenceMatcher(None, normalized, normalize_team_name(team["full_name"])).ratio()
        if score > best_score:
            best_score = score
            best_team = team

    if best_score >= FUZZY_THRESHOLD and best_team is not None:
        _nba_team_index[key] = best_team
        logger.debug("Fuzzy-matched NBA team %r -> %s (score=%.2f)", name, best_team["full_name"], best_score)
        return best_team

    logger.debug("No NBA match for %r (best=%.2f)", name, best_score)
    return None


# Public accessors for team_mapping compatibility
def get_bb_team_index(league_id: int = 0) -> dict[str, int]:
    """Return the NBA team index {name_lower: team_id}."""
    _build_team_index()
    return {k: v["id"] for k, v in _nba_team_index.items()}


def get_bb_team_name(league_id: int = 0, key_lower: str = "") -> str | None:
    """Return original-cased team name for a lowercase key."""
    team = _nba_team_index.get(key_lower)
    return team["full_name"] if team else None


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

_standings_cache: dict[str, dict[int, int]] = {}  # season -> {team_id: rank}


def _fetch_standings(season: str) -> dict[int, int]:
    """Fetch NBA standings for a season. Returns {team_id: conference_rank}."""
    if season in _standings_cache:
        return _standings_cache[season]

    from nba_api.stats.endpoints import LeagueStandings

    try:
        resp = LeagueStandings(season=season, timeout=int(TIMEOUT))
        rows = resp.get_normalized_dict().get("Standings", [])
    except Exception as exc:
        logger.warning("NBA standings fetch failed: %s", exc)
        _standings_cache[season] = {}
        return {}

    result: dict[int, int] = {}
    for row in rows:
        tid = row.get("TeamID")
        # PlayoffRank is overall conference seed (1-15)
        rank = row.get("PlayoffRank") or row.get("ConferenceRecord")
        if tid and rank is not None:
            try:
                result[tid] = int(rank)
            except (ValueError, TypeError):
                pass

    _standings_cache[season] = result
    logger.debug("NBA standings loaded: %d teams for season %s", len(result), season)
    return result


def _fetch_team_form(
    team_id: int, season: str, limit: int = 5,
) -> tuple[float | None, float | None, float | None]:
    """Fetch recent form: (win_rate_pts, points_scored_avg, points_allowed_avg).

    Uses LeagueGameLog to get team game logs for the season.
    """
    from nba_api.stats.endpoints import LeagueGameLog

    try:
        resp = LeagueGameLog(
            season=season,
            player_or_team_abbreviation="T",
            timeout=int(TIMEOUT),
        )
        rows = resp.get_normalized_dict().get("LeagueGameLog", [])
    except Exception as exc:
        logger.warning("NBA game log fetch failed for team %d: %s", team_id, exc)
        return None, None, None

    # Filter to our team's games, sorted by date descending (default order)
    team_games = [r for r in rows if r.get("TEAM_ID") == team_id][:limit]

    if not team_games:
        return None, None, None

    wins = 0
    total_scored = 0
    total_conceded = 0

    for g in team_games:
        pts = g.get("PTS", 0) or 0
        # WL is "W" or "L"
        if g.get("WL") == "W":
            wins += 1

        total_scored += pts
        # Calculate opponent points from PLUS_MINUS: opp_pts = pts - plus_minus
        plus_minus = g.get("PLUS_MINUS", 0) or 0
        opp_pts = pts - plus_minus
        total_conceded += opp_pts

    count = len(team_games)
    win_rate_pts = round(2.0 * wins / count, 2)  # 0-2 scale
    return (
        win_rate_pts,
        round(total_scored / count, 2),
        round(total_conceded / count, 2),
    )


def _current_season() -> str:
    """Return the current NBA season string (e.g., '2025-26')."""
    import datetime as _dt

    today = _dt.date.today()
    # NBA season starts in October, so if we're before October,
    # the "current" season started the previous year
    if today.month >= 10:
        start_year = today.year
    else:
        start_year = today.year - 1

    end_year_short = str(start_year + 1)[-2:]
    return f"{start_year}-{end_year_short}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_basketball_stats(
    home_canonical: str,
    away_canonical: str,
    competition: str = "",
) -> MatchStats | None:
    """Fetch basketball stats from NBA.com via nba_api.

    Only covers NBA. Non-NBA competitions (NBL, EuroLeague, etc.)
    return None and fall back to mid-price.
    """
    if not _check_nba_api():
        return None

    # Only NBA is supported — skip non-NBA competitions
    comp_lower = competition.lower() if competition else ""
    is_nba = (
        not comp_lower
        or "nba" in comp_lower
        or "national basketball" in comp_lower
    )
    if not is_nba:
        logger.debug("Non-NBA basketball competition %r — skipping (no provider).", competition)
        return None

    home_team = _resolve_team(home_canonical)
    away_team = _resolve_team(away_canonical)

    if home_team is None and away_team is None:
        logger.info("Could not resolve either NBA team: %s vs %s", home_canonical, away_canonical)
        return None

    season = _current_season()
    stats = MatchStats(sport="basketball", home_team=home_canonical, away_team=away_canonical)

    # Form
    if home_team is not None:
        pts, scored, conceded = _fetch_team_form(home_team["id"], season)
        stats.home_form_pts_per_game = pts
        stats.home_goals_scored_avg = scored
        stats.home_goals_conceded_avg = conceded

    if away_team is not None:
        pts, scored, conceded = _fetch_team_form(away_team["id"], season)
        stats.away_form_pts_per_game = pts
        stats.away_goals_scored_avg = scored
        stats.away_goals_conceded_avg = conceded

    # Standings
    standings = _fetch_standings(season)
    if home_team and home_team["id"] in standings:
        stats.home_league_position = standings[home_team["id"]]
    if away_team and away_team["id"] in standings:
        stats.away_league_position = standings[away_team["id"]]

    stats.data_completeness = compute_completeness(stats)
    return stats
