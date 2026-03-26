"""Basketball stats fetcher — API-Basketball (api-sports.io, 100 req/day free)."""

from __future__ import annotations

import logging
import time

import httpx

from src.config import settings
from src.enrichment.team_mapping import FUZZY_THRESHOLD, normalize_team_name

from .models import MatchStats, TIMEOUT, compute_completeness

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiter (daily budget)
# ---------------------------------------------------------------------------

_bb_daily_count = 0
_bb_daily_date: str = ""
_BB_DAILY_LIMIT = 100


def _bb_rate_limit() -> bool:
    """Check and increment the daily request counter.

    Returns True if we're within budget, False if exhausted.
    """
    import datetime as _dt

    global _bb_daily_count, _bb_daily_date  # noqa: PLW0603
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    if today != _bb_daily_date:
        _bb_daily_count = 0
        _bb_daily_date = today

    if _bb_daily_count >= _BB_DAILY_LIMIT:
        logger.warning("API-Basketball daily limit reached (%d/%d)", _bb_daily_count, _BB_DAILY_LIMIT)
        return False

    if _bb_daily_count >= 90:
        logger.info("API-Basketball daily budget low: %d/%d", _bb_daily_count, _BB_DAILY_LIMIT)

    _bb_daily_count += 1
    return True


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _bb_headers() -> dict[str, str]:
    """Auth headers for API-Basketball."""
    key = settings.basketball_api_key
    if key:
        return {"x-apisports-key": key}
    return {}


def _bb_get(path: str, params: dict | None = None) -> dict | list | None:
    """GET request to API-Basketball."""
    if not _bb_rate_limit():
        return None
    url = f"{settings.stats.basketball_api_base}{path}"
    try:
        resp = httpx.get(url, headers=_bb_headers(), params=params, timeout=TIMEOUT)
        if resp.status_code == 429:
            logger.warning("API-Basketball rate limit hit — backing off.")
            return None
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as exc:
        logger.warning("API-Basketball request failed: %s %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# League resolution
# ---------------------------------------------------------------------------

_BB_LEAGUE_OVERRIDES: dict[str, int] = {
    # Major leagues
    "nba": 12,
    "ncaa": 116,
    "euroleague": 120,
    "eurocup": 121,
    # European national leagues
    "acb": 117,
    "liga acb": 117,
    "liga endesa": 117,
    "lega a": 136,
    "lega basket serie a": 136,
    "serie a": 136,
    "lnb": 130,
    "pro a": 130,
    "betclic elite": 130,
    "lkl": 149,
    "bbl": 132,
    "czech nbl": 135,
    "aba league": 142,
    "vtb united league": 150,
    "turkish bsl": 145,
    "basket league": 45,
    "greek basket league": 45,
    "korisliiga": 37,
    "finnish korisliiga": 37,
    "plk": 72,
    "polish plk": 72,
    "tauron basket liga": 72,
    "cba": 31,
    "chinese cba": 31,
    "kbl": 91,
    "korean kbl": 91,
    # Oceania
    "nbl": 20,
    "australian nbl": 20,
    "nbb": 177,
    # Scandinavian
    "sbl": 157,
    "basketligan": 157,
}

_bb_league_cache: dict[str, int | None] = {}


def _bb_resolve_league(competition_name: str) -> int | None:
    """Map a Betfair competition name to an API-Basketball league ID."""
    if not competition_name:
        return None

    norm = competition_name.lower().strip()
    if norm in _bb_league_cache:
        return _bb_league_cache[norm]

    for key, lid in _BB_LEAGUE_OVERRIDES.items():
        if key in norm or norm in key:
            _bb_league_cache[norm] = lid
            return lid

    # Try searching by each word in the competition name (most specific first)
    words = [w for w in competition_name.split() if len(w) > 2]
    for word in words:
        data = _bb_get("/leagues", params={"search": word})
        if not data or not isinstance(data, dict):
            continue
        for league in data.get("response", []):
            league_name = league.get("name", "").lower()
            league_id = league.get("id")
            league_type = league.get("type", "")
            # Prefer actual leagues over cups
            if league_id and league_type == "League" and (norm in league_name or league_name in norm):
                _bb_league_cache[norm] = league_id
                logger.info("Resolved basketball league %r -> ID %d", competition_name, league_id)
                return league_id

    _bb_league_cache[norm] = None
    logger.info("Could not resolve basketball league for competition=%r", competition_name)
    return None


# ---------------------------------------------------------------------------
# Season resolution
# ---------------------------------------------------------------------------

_bb_season_cache: dict[int, str | None] = {}


def _bb_current_season(league_id: int) -> str | None:
    """Get the current season string for a league.

    The /leagues endpoint returns all available seasons. We pick the latest
    season whose date range covers today (or the most recent past season
    if none is current). This avoids picking a future season with no data.
    """
    import datetime as _dt

    data = _bb_get("/leagues", params={"id": league_id})
    if not data or not isinstance(data, dict):
        return None

    results = data.get("response", [])
    if not results:
        return None

    seasons = results[0].get("seasons", [])
    if not seasons:
        return None

    today = _dt.date.today().isoformat()

    # Prefer the season whose start <= today <= end
    for s in reversed(seasons):
        start = s.get("start", "")
        end = s.get("end", "")
        if start <= today <= end:
            return str(s.get("season", ""))

    # Fallback: latest season whose start date is in the past
    for s in reversed(seasons):
        start = s.get("start", "")
        if start and start <= today:
            return str(s.get("season", ""))

    # Last resort: last season in the list
    return str(seasons[-1].get("season", ""))


def _bb_get_season(league_id: int) -> str | None:
    """Get current season for a league, cached."""
    if league_id in _bb_season_cache:
        return _bb_season_cache[league_id]
    season = _bb_current_season(league_id)
    _bb_season_cache[league_id] = season
    logger.debug("Basketball season for league %d: %s", league_id, season)
    return season


# ---------------------------------------------------------------------------
# Team index
# ---------------------------------------------------------------------------

# Team index per league: league_id -> {name_lower: team_id}
_bb_team_indexes: dict[int, dict[str, int]] = {}
# Original casing per league: league_id -> {name_lower: original_name}
_bb_team_name_casing: dict[int, dict[str, str]] = {}


def _bb_build_team_index(league_id: int, season: str) -> None:
    """Build a team name -> ID index for a league+season.

    If the requested season has no teams (API data not yet populated),
    falls back to the previous season.
    """
    if league_id in _bb_team_indexes:
        return

    data = _bb_get("/teams", params={"league": league_id, "season": season})
    teams = data.get("response", []) if data and isinstance(data, dict) else []

    # Fallback: try previous seasons if API hasn't populated current one
    if not teams:
        try:
            year = int(season)
        except ValueError:
            year = None
        if year:
            for offset in range(1, 4):  # Try up to 3 years back
                prev = str(year - offset)
                logger.info("No teams for league %d season %s, trying %s", league_id, season, prev)
                data = _bb_get("/teams", params={"league": league_id, "season": prev})
                teams = data.get("response", []) if data and isinstance(data, dict) else []
                if teams:
                    _bb_season_cache[league_id] = prev
                    break

    index: dict[str, int] = {}
    casing: dict[str, str] = {}
    for entry in teams:
        tid = entry.get("id")
        name = entry.get("name", "")
        if tid and name:
            index[name.lower()] = tid
            casing[name.lower()] = name

    _bb_team_indexes[league_id] = index
    _bb_team_name_casing[league_id] = casing
    logger.info("API-Basketball team index built for league %d: %d entries", league_id, len(index))


def get_bb_team_index(league_id: int) -> dict[str, int]:
    """Return the basketball team index for a league. Public for team_mapping."""
    return _bb_team_indexes.get(league_id, {})


def get_bb_team_name(league_id: int, key_lower: str) -> str | None:
    """Return original-cased team name for a lowercase key."""
    return _bb_team_name_casing.get(league_id, {}).get(key_lower)


# ---------------------------------------------------------------------------
# Team resolution
# ---------------------------------------------------------------------------

def _bb_resolve_team_id(canonical_name: str, league_id: int) -> int | None:
    """Resolve a canonical team name to an API-Basketball team ID."""
    index = _bb_team_indexes.get(league_id, {})
    if not index:
        return None

    key = canonical_name.lower()
    if key in index:
        return index[key]

    # Normalized match
    normalized = normalize_team_name(canonical_name)
    for idx_key, tid in index.items():
        if normalize_team_name(idx_key) == normalized:
            index[key] = tid
            return tid

    # Fuzzy match
    from difflib import SequenceMatcher

    best_score = 0.0
    best_id: int | None = None
    for idx_key, tid in index.items():
        score = SequenceMatcher(None, normalized, normalize_team_name(idx_key)).ratio()
        if score > best_score:
            best_score = score
            best_id = tid

    if best_score >= FUZZY_THRESHOLD and best_id is not None:
        index[key] = best_id
        logger.debug("Fuzzy-matched basketball team %r to ID %d (score=%.2f)", canonical_name, best_id, best_score)
        return best_id

    logger.debug("No API-Basketball match for %r in league %d (best=%.2f)", canonical_name, league_id, best_score)
    return None


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def _bb_team_form(
    team_id: int, league_id: int, season: str, limit: int = 5,
) -> tuple[float | None, float | None, float | None]:
    """Fetch recent form: (win_rate_as_pts, points_scored_avg, points_allowed_avg)."""
    data = _bb_get("/games", params={
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
    total_scored = 0
    total_conceded = 0

    for g in recent:
        scores = g.get("scores", {})
        home_score = scores.get("home", {}).get("total")
        away_score = scores.get("away", {}).get("total")
        if home_score is None or away_score is None:
            continue

        home_id = g.get("teams", {}).get("home", {}).get("id")
        if home_id == team_id:
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
    if count == 0:
        return None, None, None

    win_rate_pts = round(2.0 * wins / count, 2)
    return (
        win_rate_pts,
        round(total_scored / count, 2),
        round(total_conceded / count, 2),
    )


# Standings cache: (league_id, season) -> {team_id: position}
_bb_standings_cache: dict[tuple[int, str], dict[int, int]] = {}


def _bb_fetch_standings(league_id: int, season: str) -> dict[int, int]:
    """Fetch standings for a league, returning {team_id: position}."""
    cache_key = (league_id, season)
    if cache_key in _bb_standings_cache:
        return _bb_standings_cache[cache_key]

    data = _bb_get("/standings", params={"league": league_id, "season": season})
    result: dict[int, int] = {}
    if data and isinstance(data, dict):
        for group in data.get("response", []):
            entries = group if isinstance(group, list) else [group]
            for entry in entries:
                tid = entry.get("team", {}).get("id")
                pos = entry.get("position")
                if tid and pos is not None:
                    result[tid] = pos

    _bb_standings_cache[cache_key] = result
    return result


def _bb_head_to_head(team_id_1: int, team_id_2: int) -> tuple[int, int, int]:
    """Fetch H2H record between two basketball teams.

    Returns (team1_wins, team2_wins, total_matches). No draws in basketball.
    """
    data = _bb_get("/games", params={"h2h": f"{team_id_1}-{team_id_2}"})
    if not data or not isinstance(data, dict):
        return 0, 0, 0

    t1_wins = t2_wins = 0
    for g in data.get("response", []):
        if g.get("status", {}).get("short") != "FT":
            continue
        scores = g.get("scores", {})
        hs = scores.get("home", {}).get("total")
        as_ = scores.get("away", {}).get("total")
        if hs is None or as_ is None:
            continue

        home_id = g.get("teams", {}).get("home", {}).get("id")
        if hs > as_:
            winner_id = home_id
        else:
            winner_id = g.get("teams", {}).get("away", {}).get("id")

        if winner_id == team_id_1:
            t1_wins += 1
        elif winner_id == team_id_2:
            t2_wins += 1

    total = t1_wins + t2_wins
    if total:
        logger.debug("Basketball H2H: %d meetings (t1=%d: %dW, t2=%d: %dW)",
                      total, team_id_1, t1_wins, team_id_2, t2_wins)
    return t1_wins, t2_wins, total


def fetch_basketball_stats(
    home_canonical: str,
    away_canonical: str,
    competition: str = "",
) -> MatchStats | None:
    """Fetch basketball stats from API-Basketball."""
    if not settings.basketball_api_key:
        logger.debug("No BASKETBALL_API_KEY configured — skipping basketball stats.")
        return None

    league_id = _bb_resolve_league(competition)
    if league_id is None:
        logger.info("Could not resolve basketball league for competition=%r", competition)
        return None

    season = _bb_get_season(league_id)
    if not season:
        logger.info("Could not determine season for basketball league %d", league_id)
        return None

    _bb_build_team_index(league_id, season)
    # Re-read season — build_team_index may have fallen back to a previous season
    season = _bb_season_cache.get(league_id, season)

    home_id = _bb_resolve_team_id(home_canonical, league_id)
    away_id = _bb_resolve_team_id(away_canonical, league_id)

    if home_id is None and away_id is None:
        logger.info("Could not resolve either basketball team: %s vs %s (league=%d)",
                     home_canonical, away_canonical, league_id)
        return None

    stats = MatchStats(sport="basketball", home_team=home_canonical, away_team=away_canonical)

    if home_id is not None:
        pts, scored, conceded = _bb_team_form(home_id, league_id, season)
        stats.home_form_pts_per_game = pts
        stats.home_goals_scored_avg = scored
        stats.home_goals_conceded_avg = conceded

    if away_id is not None:
        pts, scored, conceded = _bb_team_form(away_id, league_id, season)
        stats.away_form_pts_per_game = pts
        stats.away_goals_scored_avg = scored
        stats.away_goals_conceded_avg = conceded

    # Standings
    standings = _bb_fetch_standings(league_id, season)
    if home_id and home_id in standings:
        stats.home_league_position = standings[home_id]
    if away_id and away_id in standings:
        stats.away_league_position = standings[away_id]

    # H2H — best-effort
    if home_id is not None and away_id is not None:
        try:
            hw, aw, total = _bb_head_to_head(home_id, away_id)
            stats.h2h_home_wins = hw
            stats.h2h_draws = 0  # No draws in basketball
            stats.h2h_away_wins = aw
            stats.h2h_total_matches = total
        except Exception as exc:  # noqa: BLE001
            logger.debug("Basketball H2H fetch failed: %s vs %s: %s",
                         home_canonical, away_canonical, exc)

    stats.data_completeness = compute_completeness(stats)
    return stats
