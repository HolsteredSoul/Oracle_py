"""Rugby Union stats fetcher — API-Sports (api-sports.io, 100 req/day free).

Covers Super Rugby and other Union competitions.
NRL (Rugby League) is NOT covered by this API.
"""

from __future__ import annotations

import datetime as _dt
import logging

import httpx

from src.config import settings

from .models import TIMEOUT, MatchStats, compute_completeness

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter (daily budget — separate from basketball)
# ---------------------------------------------------------------------------

_rg_daily_count = 0
_rg_daily_date: str = ""
_RG_DAILY_LIMIT = 100


def _rg_rate_limit() -> bool:
    """Check and increment the daily request counter."""
    global _rg_daily_count, _rg_daily_date  # noqa: PLW0603
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    if today != _rg_daily_date:
        _rg_daily_count = 0
        _rg_daily_date = today

    if _rg_daily_count >= _RG_DAILY_LIMIT:
        logger.warning("API-Rugby daily limit reached (%d/%d)", _rg_daily_count, _RG_DAILY_LIMIT)
        return False

    _rg_daily_count += 1
    return True


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _rg_get(path: str, params: dict | None = None) -> dict | None:
    """GET request to API-Sports rugby endpoint."""
    if not _rg_rate_limit():
        return None
    url = f"{settings.stats.rugby_api_base}{path}"
    headers = {"x-apisports-key": settings.basketball_api_key}
    try:
        resp = httpx.get(url, headers=headers, params=params, timeout=TIMEOUT)
        if resp.status_code == 429:
            logger.warning("API-Rugby rate limit hit — backing off.")
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("API-Rugby request failed: %s %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# League resolution
# ---------------------------------------------------------------------------

_RG_LEAGUE_OVERRIDES: dict[str, int] = {
    "super rugby": 71,
    "super rugby pacific": 71,
    "premiership rugby": 13,
    "english premiership": 13,
    "top 14": 17,
    "united rugby championship": 69,
    "urc": 69,
    "pro14": 69,
    "six nations": 52,
    "rugby championship": 51,
}

_rg_league_cache: dict[str, int | None] = {}


def _rg_resolve_league(competition_name: str) -> int | None:
    """Map a Betfair competition name to an API-Rugby league ID."""
    if not competition_name:
        return None

    norm = competition_name.lower().strip()
    if norm in _rg_league_cache:
        return _rg_league_cache[norm]

    for key, lid in _RG_LEAGUE_OVERRIDES.items():
        if key in norm or norm in key:
            _rg_league_cache[norm] = lid
            return lid

    # Search API by keywords
    words = [w for w in competition_name.split() if len(w) > 2]
    for word in words:
        data = _rg_get("/leagues", params={"search": word})
        if not data or not isinstance(data, dict):
            continue
        for league in data.get("response", []):
            league_name = league.get("name", "").lower()
            league_id = league.get("id")
            league_type = league.get("type", "")
            if league_id and league_type == "League" and (norm in league_name or league_name in norm):
                _rg_league_cache[norm] = league_id
                logger.info("Resolved rugby league %r -> ID %d", competition_name, league_id)
                return league_id

    _rg_league_cache[norm] = None
    logger.info("Could not resolve rugby league for competition=%r", competition_name)
    return None


# ---------------------------------------------------------------------------
# Season resolution
# ---------------------------------------------------------------------------

_rg_season_cache: dict[int, str | None] = {}


def _rg_get_season(league_id: int) -> str | None:
    """Get current season for a league, cached. Picks the season covering today."""
    if league_id in _rg_season_cache:
        return _rg_season_cache[league_id]

    data = _rg_get("/leagues", params={"id": league_id})
    if not data or not isinstance(data, dict):
        return None

    results = data.get("response", [])
    if not results:
        return None

    seasons = results[0].get("seasons", [])
    if not seasons:
        return None

    today = _dt.date.today().isoformat()

    # Prefer season whose date range covers today
    for s in reversed(seasons):
        start = s.get("start", "")
        end = s.get("end", "")
        if start <= today <= end:
            season = str(s.get("season", ""))
            _rg_season_cache[league_id] = season
            return season

    # Fallback: latest season with start in the past
    for s in reversed(seasons):
        if s.get("start", "") <= today:
            season = str(s.get("season", ""))
            _rg_season_cache[league_id] = season
            return season

    season = str(seasons[-1].get("season", ""))
    _rg_season_cache[league_id] = season
    return season


# ---------------------------------------------------------------------------
# Team index
# ---------------------------------------------------------------------------

_rg_team_indexes: dict[int, dict[str, int]] = {}
_rg_team_name_casing: dict[int, dict[str, str]] = {}


def _rg_build_team_index(league_id: int, season: str) -> None:
    """Build team name -> ID index. Falls back to previous season if empty."""
    if league_id in _rg_team_indexes:
        return

    data = _rg_get("/teams", params={"league": league_id, "season": season})
    teams = data.get("response", []) if data and isinstance(data, dict) else []

    # Fallback to previous seasons if API hasn't populated current one
    if not teams:
        try:
            year = int(season)
        except ValueError:
            year = None
        if year:
            for offset in range(1, 4):  # Try up to 3 years back
                prev = str(year - offset)
                logger.info("No rugby teams for league %d season %s, trying %s", league_id, season, prev)
                data = _rg_get("/teams", params={"league": league_id, "season": prev})
                teams = data.get("response", []) if data and isinstance(data, dict) else []
                if teams:
                    _rg_season_cache[league_id] = prev
                    break

    index: dict[str, int] = {}
    casing: dict[str, str] = {}
    for entry in teams:
        tid = entry.get("id")
        name = entry.get("name", "")
        if tid and name:
            index[name.lower()] = tid
            casing[name.lower()] = name

    _rg_team_indexes[league_id] = index
    _rg_team_name_casing[league_id] = casing
    logger.info("API-Rugby team index built for league %d: %d entries", league_id, len(index))


def _rg_resolve_team_id(name: str, league_id: int) -> int | None:
    """Resolve a team name to an API-Rugby team ID."""
    from difflib import SequenceMatcher

    from src.enrichment.team_mapping import FUZZY_THRESHOLD, normalize_team_name

    index = _rg_team_indexes.get(league_id, {})
    if not index:
        return None

    key = name.lower()
    if key in index:
        return index[key]

    normalized = normalize_team_name(name)
    for idx_key, tid in index.items():
        if normalize_team_name(idx_key) == normalized:
            index[key] = tid
            return tid

    best_score = 0.0
    best_id: int | None = None
    for idx_key, tid in index.items():
        score = SequenceMatcher(None, normalized, normalize_team_name(idx_key)).ratio()
        if score > best_score:
            best_score = score
            best_id = tid

    if best_score >= FUZZY_THRESHOLD and best_id is not None:
        index[key] = best_id
        return best_id

    logger.debug("No API-Rugby match for %r in league %d (best=%.2f)", name, league_id, best_score)
    return None


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def _rg_team_form(
    team_id: int, league_id: int, season: str, limit: int = 5,
) -> tuple[float | None, float | None, float | None]:
    """Fetch recent form: (win_rate_pts, points_scored_avg, points_conceded_avg)."""
    data = _rg_get("/games", params={
        "team": team_id, "league": league_id, "season": season,
    })
    if not data or not isinstance(data, dict):
        return None, None, None

    games = data.get("response", [])
    finished = [g for g in games if g.get("status", {}).get("short") == "FT"]
    finished.sort(key=lambda g: g.get("date", ""), reverse=True)
    recent = finished[:limit]

    if not recent:
        return None, None, None

    wins = 0
    draws = 0
    total_scored = 0
    total_conceded = 0

    for g in recent:
        home_score = g.get("scores", {}).get("home", 0) or 0
        away_score = g.get("scores", {}).get("away", 0) or 0
        home_id = g.get("teams", {}).get("home", {}).get("id")

        if home_id == team_id:
            total_scored += home_score
            total_conceded += away_score
            if home_score > away_score:
                wins += 1
            elif home_score == away_score:
                draws += 1
        else:
            total_scored += away_score
            total_conceded += home_score
            if away_score > home_score:
                wins += 1
            elif home_score == away_score:
                draws += 1

    count = len(recent)
    # pts_per_game on 0-2 scale: win=2, draw=1, loss=0 (normalised)
    pts = round((wins * 2 + draws) / count, 2)
    return pts, round(total_scored / count, 2), round(total_conceded / count, 2)


_rg_standings_cache: dict[tuple[int, str], dict[int, int]] = {}


def _rg_fetch_standings(league_id: int, season: str) -> dict[int, int]:
    """Fetch standings: {team_id: position}."""
    cache_key = (league_id, season)
    if cache_key in _rg_standings_cache:
        return _rg_standings_cache[cache_key]

    data = _rg_get("/standings", params={"league": league_id, "season": season})
    result: dict[int, int] = {}
    if data and isinstance(data, dict):
        for group in data.get("response", []):
            entries = group if isinstance(group, list) else [group]
            for entry in entries:
                tid = entry.get("team", {}).get("id")
                pos = entry.get("position")
                if tid and pos is not None:
                    result[tid] = pos

    _rg_standings_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Main fetch
# ---------------------------------------------------------------------------

def fetch_rugby_stats(
    home_canonical: str,
    away_canonical: str,
    competition: str = "",
) -> MatchStats | None:
    """Fetch rugby union stats from API-Sports."""
    if not settings.basketball_api_key:
        logger.debug("No API-Sports key configured — skipping rugby stats.")
        return None

    league_id = _rg_resolve_league(competition)
    if league_id is None:
        return None

    season = _rg_get_season(league_id)
    if not season:
        logger.info("Could not determine season for rugby league %d", league_id)
        return None

    _rg_build_team_index(league_id, season)
    # Re-read season — build may have fallen back
    season = _rg_season_cache.get(league_id, season)

    home_id = _rg_resolve_team_id(home_canonical, league_id)
    away_id = _rg_resolve_team_id(away_canonical, league_id)

    if home_id is None and away_id is None:
        logger.info("Could not resolve either rugby team: %s vs %s (league=%d)",
                     home_canonical, away_canonical, league_id)
        return None

    stats = MatchStats(sport="rugby", home_team=home_canonical, away_team=away_canonical)

    if home_id is not None:
        pts, scored, conceded = _rg_team_form(home_id, league_id, season)
        stats.home_form_pts_per_game = pts
        stats.home_goals_scored_avg = scored
        stats.home_goals_conceded_avg = conceded

    if away_id is not None:
        pts, scored, conceded = _rg_team_form(away_id, league_id, season)
        stats.away_form_pts_per_game = pts
        stats.away_goals_scored_avg = scored
        stats.away_goals_conceded_avg = conceded

    standings = _rg_fetch_standings(league_id, season)
    if home_id and home_id in standings:
        stats.home_league_position = standings[home_id]
    if away_id and away_id in standings:
        stats.away_league_position = standings[away_id]

    stats.data_completeness = compute_completeness(stats)
    return stats
