"""Statistical data fetcher for sports match analysis.

Fetches team form, goals, standings, and H2H data from:
  - football-data.org (football/soccer) — free tier, 10 req/min
  - Squiggle API (AFL) — free, JSON-based

Returns MatchStats objects consumed by src/strategy/statistical_model.py.
Caches results in memory with configurable TTL (default 6 hours).
Returns None on any failure — caller falls back to market mid-price.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from src.config import settings
from src.enrichment.team_mapping import FUZZY_THRESHOLD, resolve_team

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0  # seconds


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class MatchStats(BaseModel):
    """Statistical features for a single match."""

    sport: Literal["football", "afl", "basketball"]
    home_team: str
    away_team: str
    # Form (last 5 matches)
    home_form_pts_per_game: float | None = None
    away_form_pts_per_game: float | None = None
    home_goals_scored_avg: float | None = None
    home_goals_conceded_avg: float | None = None
    away_goals_scored_avg: float | None = None
    away_goals_conceded_avg: float | None = None
    # Standings
    home_league_position: int | None = None
    away_league_position: int | None = None
    # H2H
    h2h_home_wins: int = 0
    h2h_draws: int = 0
    h2h_away_wins: int = 0
    h2h_total_matches: int = 0
    # Data quality
    data_completeness: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

# (home_canonical, away_canonical, sport) -> (timestamp, MatchStats)
_cache: dict[tuple[str, str, str], tuple[float, MatchStats]] = {}


def _cache_get(home: str, away: str, sport: str) -> MatchStats | None:
    key = (home, away, sport)
    if key in _cache:
        ts, stats = _cache[key]
        ttl = settings.stats.cache_ttl_hours * 3600
        if time.time() - ts < ttl:
            return stats
        del _cache[key]
    return None


def _cache_set(home: str, away: str, sport: str, stats: MatchStats) -> None:
    _cache[(home, away, sport)] = (time.time(), stats)


def _compute_completeness(stats: MatchStats) -> float:
    """Fraction of optional numeric fields that are populated."""
    optional_fields = [
        stats.home_form_pts_per_game,
        stats.away_form_pts_per_game,
        stats.home_goals_scored_avg,
        stats.home_goals_conceded_avg,
        stats.away_goals_scored_avg,
        stats.away_goals_conceded_avg,
        stats.home_league_position,
        stats.away_league_position,
    ]
    filled = sum(1 for f in optional_fields if f is not None)
    return round(filled / len(optional_fields), 2)


# ---------------------------------------------------------------------------
# football-data.org helpers
# ---------------------------------------------------------------------------

# Team ID cache: canonical name -> football-data.org team ID
_team_id_cache: dict[str, int | None] = {}
# Original casing cache: lowercase key -> original API name
_team_name_cache: dict[str, str] = {}

# Rate limiter: football-data.org free tier allows 10 requests per minute.
# Track timestamps of recent requests and pause when approaching the limit.
_fd_request_times: list[float] = []
_FD_MAX_REQUESTS_PER_MINUTE = 10
_FD_RATE_WINDOW = 60.0  # seconds


def _fd_rate_limit() -> None:
    """Block until we're under the rate limit for football-data.org."""
    now = time.time()
    # Purge requests older than the window
    _fd_request_times[:] = [t for t in _fd_request_times if now - t < _FD_RATE_WINDOW]

    if len(_fd_request_times) >= _FD_MAX_REQUESTS_PER_MINUTE:
        # Wait until the oldest request falls outside the window
        wait = _FD_RATE_WINDOW - (now - _fd_request_times[0]) + 0.5
        if wait > 0:
            logger.debug("football-data.org rate limiter: sleeping %.1fs", wait)
            time.sleep(wait)
            _fd_request_times[:] = [t for t in _fd_request_times if time.time() - t < _FD_RATE_WINDOW]

    _fd_request_times.append(time.time())


def _fd_headers() -> dict[str, str]:
    """Auth headers for football-data.org."""
    key = settings.football_data_api_key
    if key:
        return {"X-Auth-Token": key}
    return {}


def _fd_get(path: str, params: dict | None = None) -> dict | list | None:
    """GET request to football-data.org v4 API."""
    _fd_rate_limit()
    url = f"{settings.stats.football_api_base}{path}"
    try:
        resp = httpx.get(url, headers=_fd_headers(), params=params, timeout=_TIMEOUT)
        if resp.status_code == 429:
            logger.warning("football-data.org rate limit hit — backing off 60s.")
            time.sleep(60)
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("football-data.org request failed: %s %s", path, exc)
        return None


# Available free-tier competition codes on football-data.org
_FD_COMPETITIONS = ["PL", "BL1", "SA", "PD", "FL1", "DED", "PPL", "ELC", "CL"]
_fd_team_index_built = False


def _fd_build_team_index() -> None:
    """Build a name→ID index from all available competitions (called once)."""
    global _fd_team_index_built  # noqa: PLW0603
    if _fd_team_index_built:
        return
    _fd_team_index_built = True

    for comp_code in _FD_COMPETITIONS:
        data = _fd_get(f"/competitions/{comp_code}/teams")
        if not data or not isinstance(data, dict):
            continue
        for team in data.get("teams", []):
            tid = team.get("id")
            name = team.get("name", "")
            short = team.get("shortName", "")
            if tid and name:
                _team_id_cache[name.lower()] = tid
                _team_name_cache[name.lower()] = name
                if short:
                    _team_id_cache[short.lower()] = tid
                    _team_name_cache[short.lower()] = short

    logger.info("football-data.org team index built: %d entries", len(_team_id_cache))


def get_fd_team_index() -> dict[str, int]:
    """Return the football-data.org team name → ID index.

    Triggers a one-time build from the API if not yet populated.
    Keys are lowercase team names (both full and short variants).
    """
    _fd_build_team_index()
    return _team_id_cache


def get_fd_team_name(key_lower: str) -> str | None:
    """Return the original-cased team name for a lowercase key, or None."""
    _fd_build_team_index()
    return _team_name_cache.get(key_lower)


def _normalize_for_lookup(name: str) -> str:
    """Normalize a team name for index lookup."""
    n = name.lower().strip()
    for suffix in (" fc", " afc", " sc"):
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    return n


def _fd_resolve_team_id(canonical_name: str) -> int | None:
    """Resolve a canonical team name to a football-data.org team ID."""
    _fd_build_team_index()

    key = canonical_name.lower()
    if key in _team_id_cache:
        return _team_id_cache[key]

    # Try normalized (without FC/AFC suffix)
    normalized = _normalize_for_lookup(canonical_name)
    if normalized in _team_id_cache:
        _team_id_cache[key] = _team_id_cache[normalized]
        return _team_id_cache[normalized]

    # Fuzzy search against index keys
    from difflib import SequenceMatcher
    best_score = 0.0
    best_id: int | None = None
    for idx_key, tid in _team_id_cache.items():
        score = SequenceMatcher(None, normalized, idx_key).ratio()
        if score > best_score:
            best_score = score
            best_id = tid

    if best_score >= FUZZY_THRESHOLD and best_id is not None:
        _team_id_cache[key] = best_id
        logger.debug("Fuzzy-matched team %r to ID %d (score=%.2f)", canonical_name, best_id, best_score)
        return best_id

    _team_id_cache[key] = None  # type: ignore[assignment]
    logger.debug("No football-data.org match for %r (best=%.2f)", canonical_name, best_score)
    return None


def _fd_team_form(team_id: int, limit: int = 5) -> tuple[float | None, float | None, float | None]:
    """Fetch recent form for a team: (pts_per_game, goals_scored_avg, goals_conceded_avg)."""
    data = _fd_get(f"/teams/{team_id}/matches", params={"status": "FINISHED", "limit": limit})
    if not data or not isinstance(data, dict):
        return None, None, None

    matches = data.get("matches", [])
    if not matches:
        return None, None, None

    total_pts = 0
    total_scored = 0
    total_conceded = 0
    count = 0

    for m in matches:
        score = m.get("score", {})
        ft = score.get("fullTime", {})
        home_goals = ft.get("home")
        away_goals = ft.get("away")
        if home_goals is None or away_goals is None:
            continue

        home_id = m.get("homeTeam", {}).get("id")
        if home_id == team_id:
            total_scored += home_goals
            total_conceded += away_goals
            if home_goals > away_goals:
                total_pts += 3
            elif home_goals == away_goals:
                total_pts += 1
        else:
            total_scored += away_goals
            total_conceded += home_goals
            if away_goals > home_goals:
                total_pts += 3
            elif home_goals == away_goals:
                total_pts += 1
        count += 1

    if count == 0:
        return None, None, None

    return (
        round(total_pts / count, 2),
        round(total_scored / count, 2),
        round(total_conceded / count, 2),
    )


def _fd_league_position(team_id: int) -> int | None:
    """Get the team's current league position from standings."""
    # First, find which competition the team is in from their recent matches
    data = _fd_get(f"/teams/{team_id}/matches", params={"status": "FINISHED", "limit": 1})
    if not data or not isinstance(data, dict):
        return None

    matches = data.get("matches", [])
    if not matches:
        return None

    comp_id = matches[0].get("competition", {}).get("id")
    if not comp_id:
        return None

    standings_data = _fd_get(f"/competitions/{comp_id}/standings")
    if not standings_data or not isinstance(standings_data, dict):
        return None

    for standing in standings_data.get("standings", []):
        if standing.get("type") != "TOTAL":
            continue
        for entry in standing.get("table", []):
            if entry.get("team", {}).get("id") == team_id:
                return entry.get("position")

    return None


def _fd_head_to_head(
    home_id: int, away_id: int, limit: int = 50,
) -> tuple[int, int, int, int]:
    """Fetch head-to-head record between two teams from football-data.org.

    Returns (home_wins, draws, away_wins, total_matches).
    Uses the home team's match history and filters for meetings with away_id.
    """
    data = _fd_get(f"/teams/{home_id}/matches", params={"status": "FINISHED", "limit": limit})
    if not data or not isinstance(data, dict):
        return 0, 0, 0, 0

    home_wins = draws = away_wins = 0
    for m in data.get("matches", []):
        h_id = m.get("homeTeam", {}).get("id")
        a_id = m.get("awayTeam", {}).get("id")
        if not ({h_id, a_id} == {home_id, away_id}):
            continue

        score = m.get("score", {})
        ft = score.get("fullTime", {})
        hg, ag = ft.get("home"), ft.get("away")
        if hg is None or ag is None:
            continue

        # Determine result relative to home_id (not necessarily the home side in this match)
        if h_id == home_id:
            if hg > ag:
                home_wins += 1
            elif hg == ag:
                draws += 1
            else:
                away_wins += 1
        else:  # home_id was away in this fixture
            if ag > hg:
                home_wins += 1
            elif hg == ag:
                draws += 1
            else:
                away_wins += 1

    total = home_wins + draws + away_wins
    if total:
        logger.debug("H2H: %d meetings (home_id=%d: %dW %dD %dL)", total, home_id, home_wins, draws, away_wins)
    return home_wins, draws, away_wins, total


def _fetch_football_stats(home_canonical: str, away_canonical: str) -> MatchStats | None:
    """Fetch football stats from football-data.org."""
    home_id = _fd_resolve_team_id(home_canonical)
    away_id = _fd_resolve_team_id(away_canonical)

    if home_id is None and away_id is None:
        logger.info("Could not resolve either team ID for %s vs %s", home_canonical, away_canonical)
        return None

    stats = MatchStats(sport="football", home_team=home_canonical, away_team=away_canonical)

    if home_id is not None:
        pts, scored, conceded = _fd_team_form(home_id)
        stats.home_form_pts_per_game = pts
        stats.home_goals_scored_avg = scored
        stats.home_goals_conceded_avg = conceded
        stats.home_league_position = _fd_league_position(home_id)

    if away_id is not None:
        pts, scored, conceded = _fd_team_form(away_id)
        stats.away_form_pts_per_game = pts
        stats.away_goals_scored_avg = scored
        stats.away_goals_conceded_avg = conceded
        stats.away_league_position = _fd_league_position(away_id)

    # H2H — best-effort, don't fail the whole stats fetch
    if home_id is not None and away_id is not None:
        try:
            hw, dr, aw, total = _fd_head_to_head(home_id, away_id)
            stats.h2h_home_wins = hw
            stats.h2h_draws = dr
            stats.h2h_away_wins = aw
            stats.h2h_total_matches = total
        except Exception as exc:  # noqa: BLE001
            logger.debug("H2H fetch failed for %s vs %s: %s", home_canonical, away_canonical, exc)

    stats.data_completeness = _compute_completeness(stats)
    return stats


# ---------------------------------------------------------------------------
# Squiggle API helpers (AFL)
# ---------------------------------------------------------------------------


def _squiggle_get(endpoint: str, params: dict | None = None) -> dict | list | None:
    """GET request to Squiggle API. Returns parsed JSON (dict or list) or None on error."""
    url = f"{settings.stats.afl_api_base}/{endpoint}"
    headers = {"User-Agent": "Oracle_py/1.0 (https://github.com/HolsteredSoul/Oracle_py)"}
    try:
        resp = httpx.get(url, headers=headers, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data  # type: ignore[return-value]
    except Exception as exc:
        logger.warning("Squiggle API request failed: %s %s", endpoint, exc)
        return None


def _fetch_afl_stats(home_canonical: str, away_canonical: str) -> MatchStats | None:
    """Fetch AFL stats from Squiggle API."""
    import datetime

    year = datetime.datetime.now(datetime.timezone.utc).year

    # Get recent games for both teams
    stats = MatchStats(sport="afl", home_team=home_canonical, away_team=away_canonical)

    for team_name, is_home in [(home_canonical, True), (away_canonical, False)]:
        data = _squiggle_get("", params={"q": "games", "team": team_name, "year": year, "complete": 100})
        if not data or not isinstance(data, dict):
            continue

        games = data.get("games", [])
        if not games:
            continue

        # Take last 5 completed games
        recent = sorted(games, key=lambda g: g.get("date", ""), reverse=True)[:5]
        if not recent:
            continue

        total_pts = 0
        total_scored = 0
        total_conceded = 0
        count = 0

        for g in recent:
            hscore = g.get("hscore", 0) or 0
            ascore = g.get("ascore", 0) or 0
            hteam = g.get("hteam", "")

            if hteam == team_name:
                total_scored += hscore
                total_conceded += ascore
                if hscore > ascore:
                    total_pts += 4  # AFL uses 4 pts for a win
                elif hscore == ascore:
                    total_pts += 2
            else:
                total_scored += ascore
                total_conceded += hscore
                if ascore > hscore:
                    total_pts += 4
                elif hscore == ascore:
                    total_pts += 2
            count += 1

        if count > 0:
            if is_home:
                stats.home_form_pts_per_game = round(total_pts / count, 2)
                stats.home_goals_scored_avg = round(total_scored / count, 2)
                stats.home_goals_conceded_avg = round(total_conceded / count, 2)
            else:
                stats.away_form_pts_per_game = round(total_pts / count, 2)
                stats.away_goals_scored_avg = round(total_scored / count, 2)
                stats.away_goals_conceded_avg = round(total_conceded / count, 2)

    # Squiggle standings
    standings_data = _squiggle_get("", params={"q": "standings", "year": year})
    if standings_data and isinstance(standings_data, dict):
        for entry in standings_data.get("standings", []):
            team = entry.get("name", "")
            rank = entry.get("rank")
            if team == home_canonical and rank is not None:
                stats.home_league_position = rank
            elif team == away_canonical and rank is not None:
                stats.away_league_position = rank

    stats.data_completeness = _compute_completeness(stats)
    return stats


# ---------------------------------------------------------------------------
# API-Basketball helpers (basketball)
# ---------------------------------------------------------------------------

# Daily request counter for API-Basketball free tier (100 req/day).
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
        resp = httpx.get(url, headers=_bb_headers(), params=params, timeout=_TIMEOUT)
        if resp.status_code == 429:
            logger.warning("API-Basketball rate limit hit — backing off.")
            return None
        resp.raise_for_status()
        data = resp.json()
        # API-Basketball wraps results in {"response": [...], "results": N, ...}
        return data
    except Exception as exc:
        logger.warning("API-Basketball request failed: %s %s", path, exc)
        return None


# Hardcoded Betfair competition name -> API-Basketball league ID.
# These are looked up first before querying the /leagues endpoint.
_BB_LEAGUE_OVERRIDES: dict[str, int] = {
    "nba": 12,
    "ncaa": 116,
    "euroleague": 120,
    "acb": 117,
    "liga acb": 117,
    "lega a": 136,
    "lega basket serie a": 136,
    "serie a": 136,
    "lnb": 130,
    "pro a": 130,
    "betclic elite": 130,
    "lkl": 149,
    "bbl": 132,
    "nbl": 20,
    "nbb": 177,
    "sbl": 157,
    "basketligan": 157,
    "czech nbl": 135,
    "aba league": 142,
    "vtb united league": 150,
    "turkish bsl": 145,
}

# League search cache: normalized competition name -> league ID
_bb_league_cache: dict[str, int | None] = {}

# Team index per league: league_id -> {name_lower: team_id}
_bb_team_indexes: dict[int, dict[str, int]] = {}
# Original casing per league: league_id -> {name_lower: original_name}
_bb_team_name_casing: dict[int, dict[str, str]] = {}

# Standings cache: (league_id, season) -> {team_id: position}
_bb_standings_cache: dict[tuple[int, str], dict[int, int]] = {}


def _bb_resolve_league(competition_name: str) -> int | None:
    """Map a Betfair competition name to an API-Basketball league ID."""
    if not competition_name:
        return None

    norm = competition_name.lower().strip()
    if norm in _bb_league_cache:
        return _bb_league_cache[norm]

    # Check hardcoded overrides
    for key, lid in _BB_LEAGUE_OVERRIDES.items():
        if key in norm or norm in key:
            _bb_league_cache[norm] = lid
            return lid

    # Search the API
    data = _bb_get("/leagues", params={"search": competition_name.split()[0]})
    if data and isinstance(data, dict):
        results = data.get("response", [])
        for league in results:
            league_name = league.get("name", "").lower()
            league_id = league.get("id")
            if league_id and (norm in league_name or league_name in norm):
                _bb_league_cache[norm] = league_id
                logger.info("Resolved basketball league %r -> ID %d", competition_name, league_id)
                return league_id

    _bb_league_cache[norm] = None
    logger.debug("Could not resolve basketball league: %r", competition_name)
    return None


def _bb_current_season(league_id: int) -> str | None:
    """Get the current season string for a league (e.g. '2025-2026' or '2025')."""
    data = _bb_get("/leagues", params={"id": league_id})
    if data and isinstance(data, dict):
        results = data.get("response", [])
        if results:
            seasons = results[0].get("seasons", [])
            if seasons:
                # Seasons are ordered; last one is current
                return str(seasons[-1].get("season", ""))
    return None


# Season cache: league_id -> season string
_bb_season_cache: dict[int, str | None] = {}


def _bb_get_season(league_id: int) -> str | None:
    """Get current season for a league, cached."""
    if league_id in _bb_season_cache:
        return _bb_season_cache[league_id]
    season = _bb_current_season(league_id)
    _bb_season_cache[league_id] = season
    return season


def _bb_build_team_index(league_id: int, season: str) -> None:
    """Build a team name -> ID index for a league+season."""
    if league_id in _bb_team_indexes:
        return

    data = _bb_get("/teams", params={"league": league_id, "season": season})
    if not data or not isinstance(data, dict):
        _bb_team_indexes[league_id] = {}
        _bb_team_name_casing[league_id] = {}
        return

    index: dict[str, int] = {}
    casing: dict[str, str] = {}
    for entry in data.get("response", []):
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


def _bb_resolve_team_id(canonical_name: str, league_id: int) -> int | None:
    """Resolve a canonical team name to an API-Basketball team ID."""
    index = _bb_team_indexes.get(league_id, {})
    if not index:
        return None

    key = canonical_name.lower()
    if key in index:
        return index[key]

    # Normalized match
    normalized = _normalize_for_lookup(canonical_name)
    for idx_key, tid in index.items():
        if _normalize_for_lookup(idx_key) == normalized:
            index[key] = tid
            return tid

    # Fuzzy match
    from difflib import SequenceMatcher

    best_score = 0.0
    best_id: int | None = None
    for idx_key, tid in index.items():
        score = SequenceMatcher(None, normalized, _normalize_for_lookup(idx_key)).ratio()
        if score > best_score:
            best_score = score
            best_id = tid

    if best_score >= FUZZY_THRESHOLD and best_id is not None:
        index[key] = best_id
        logger.debug("Fuzzy-matched basketball team %r to ID %d (score=%.2f)", canonical_name, best_id, best_score)
        return best_id

    logger.debug("No API-Basketball match for %r in league %d (best=%.2f)", canonical_name, league_id, best_score)
    return None


def _bb_team_form(
    team_id: int, league_id: int, season: str, limit: int = 5,
) -> tuple[float | None, float | None, float | None]:
    """Fetch recent form: (win_rate_as_pts, points_scored_avg, points_allowed_avg).

    win_rate_as_pts: 2 * win_rate (to scale similarly to football pts/game).
    """
    data = _bb_get("/games", params={
        "team": team_id, "league": league_id, "season": season,
    })
    if not data or not isinstance(data, dict):
        return None, None, None

    games = data.get("response", [])
    # Filter to finished games and take last N
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

    # Scale win_rate to 0-2 range to match football pts/game scale
    win_rate_pts = round(2.0 * wins / count, 2)
    return (
        win_rate_pts,
        round(total_scored / count, 2),
        round(total_conceded / count, 2),
    )


def _bb_fetch_standings(league_id: int, season: str) -> dict[int, int]:
    """Fetch standings for a league, returning {team_id: position}."""
    cache_key = (league_id, season)
    if cache_key in _bb_standings_cache:
        return _bb_standings_cache[cache_key]

    data = _bb_get("/standings", params={"league": league_id, "season": season})
    result: dict[int, int] = {}
    if data and isinstance(data, dict):
        for group in data.get("response", []):
            # response is a list of lists (groups/conferences)
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


def _fetch_basketball_stats(
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

    stats.data_completeness = _compute_completeness(stats)
    return stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_match_stats(
    home_team: str,
    away_team: str,
    sport: str = "football",
    competition: str = "",
) -> MatchStats | None:
    """Fetch statistical features for an upcoming match.

    Routes to football-data.org, Squiggle, or API-Basketball based on sport.
    Returns None if teams cannot be resolved or API fails.
    Caches results per (home_team, away_team, sport) tuple.
    """
    # Basketball uses direct name resolution via its own team index
    if sport == "basketball":
        # Check cache with original names (basketball doesn't use team_mapping)
        cached = _cache_get(home_team, away_team, sport)
        if cached is not None:
            logger.debug("Stats cache hit: %s v %s (%s)", home_team, away_team, sport)
            return cached

        stats = _fetch_basketball_stats(home_team, away_team, competition)
        if stats is not None:
            _cache_set(home_team, away_team, sport, stats)
            logger.info(
                "Stats fetched: %s v %s (%s) completeness=%.0f%%",
                home_team, away_team, sport, stats.data_completeness * 100,
            )
        return stats

    # Resolve Betfair names to canonical stats API names
    home_canonical = resolve_team(home_team, sport)
    away_canonical = resolve_team(away_team, sport)

    if home_canonical is None and away_canonical is None:
        return None

    # Use original names as fallback for partial resolution
    home_canonical = home_canonical or home_team
    away_canonical = away_canonical or away_team

    # Check cache
    cached = _cache_get(home_canonical, away_canonical, sport)
    if cached is not None:
        logger.debug("Stats cache hit: %s v %s (%s)", home_canonical, away_canonical, sport)
        return cached

    # Fetch from API
    if sport == "afl":
        stats = _fetch_afl_stats(home_canonical, away_canonical)
    else:
        stats = _fetch_football_stats(home_canonical, away_canonical)

    if stats is not None:
        _cache_set(home_canonical, away_canonical, sport, stats)
        logger.info(
            "Stats fetched: %s v %s (%s) completeness=%.0f%%",
            home_canonical, away_canonical, sport, stats.data_completeness * 100,
        )

    return stats
