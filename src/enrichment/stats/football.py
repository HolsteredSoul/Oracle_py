"""Football stats fetcher — football-data.org (free tier, 10 req/min)."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone

import httpx

from src.config import PROJECT_ROOT, settings
from src.enrichment.team_mapping import FUZZY_THRESHOLD, normalize_team_name

from .models import MatchStats, TIMEOUT, compute_completeness

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

_fd_request_times: list[float] = []
_FD_MAX_REQUESTS_PER_MINUTE = 10
_FD_RATE_WINDOW = 60.0  # seconds


def _fd_rate_limit() -> None:
    """Block until we're under the rate limit for football-data.org."""
    now = time.time()
    _fd_request_times[:] = [t for t in _fd_request_times if now - t < _FD_RATE_WINDOW]

    if len(_fd_request_times) >= _FD_MAX_REQUESTS_PER_MINUTE:
        wait = _FD_RATE_WINDOW - (now - _fd_request_times[0]) + 0.5
        if wait > 0:
            logger.debug("football-data.org rate limiter: sleeping %.1fs", wait)
            time.sleep(wait)
            _fd_request_times[:] = [t for t in _fd_request_times if time.time() - t < _FD_RATE_WINDOW]

    _fd_request_times.append(time.time())


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

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
        resp = httpx.get(url, headers=_fd_headers(), params=params, timeout=TIMEOUT)
        if resp.status_code == 429:
            logger.warning("football-data.org rate limit hit — backing off 60s.")
            time.sleep(60)
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("football-data.org request failed: %s %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Team index
# ---------------------------------------------------------------------------

# Team ID cache: canonical name -> football-data.org team ID
_team_id_cache: dict[str, int | None] = {}
# Original casing cache: lowercase key -> original API name
_team_name_cache: dict[str, str] = {}

_FD_COMPETITIONS = ["PL", "BL1", "SA", "PD", "FL1", "DED", "PPL", "ELC", "CL"]
_fd_team_index_built = False
_FD_TEAM_INDEX_PATH = PROJECT_ROOT / "state" / "fd_team_index.json"


def _fd_load_cached_index() -> bool:
    """Try to load team index from disk cache. Returns True if cache is fresh."""
    global _fd_team_index_built  # noqa: PLW0603
    if not _FD_TEAM_INDEX_PATH.exists():
        return False
    try:
        data = json.loads(_FD_TEAM_INDEX_PATH.read_text(encoding="utf-8"))
        built_at = datetime.fromisoformat(data["built_at"])
        ttl = timedelta(hours=settings.stats.cache_ttl_hours)
        if datetime.now(timezone.utc) - built_at > ttl:
            logger.info(
                "fd_team_index cache stale (built_at=%s, ttl=%dh)",
                data["built_at"], settings.stats.cache_ttl_hours,
            )
            return False
        # Restore caches — JSON keys are already lowercase strings
        for k, v in data.get("team_id_cache", {}).items():
            _team_id_cache[k] = v
        for k, v in data.get("team_name_cache", {}).items():
            _team_name_cache[k] = v
        _fd_team_index_built = True
        logger.info(
            "football-data.org team index loaded from cache: %d entries (built_at=%s)",
            len(_team_id_cache), data["built_at"],
        )
        return True
    except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
        logger.warning("fd_team_index cache unreadable: %s", exc)
        return False


def _fd_save_index_cache() -> None:
    """Persist the team index to disk (atomic write)."""
    data = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "team_id_cache": {k: v for k, v in _team_id_cache.items() if v is not None},
        "team_name_cache": dict(_team_name_cache),
    }
    _FD_TEAM_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _FD_TEAM_INDEX_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(_FD_TEAM_INDEX_PATH)
    logger.info("fd_team_index saved to cache: %d entries", len(_team_id_cache))


def _fd_build_team_index() -> None:
    """Build a name->ID index. Loads from disk cache if fresh, else hits API."""
    global _fd_team_index_built  # noqa: PLW0603
    if _fd_team_index_built:
        return

    # Try disk cache first
    if _fd_load_cached_index():
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

    logger.info("football-data.org team index built from API: %d entries", len(_team_id_cache))
    _fd_save_index_cache()


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


# ---------------------------------------------------------------------------
# Team resolution
# ---------------------------------------------------------------------------

def _fd_resolve_team_id(canonical_name: str) -> int | None:
    """Resolve a canonical team name to a football-data.org team ID."""
    _fd_build_team_index()

    key = canonical_name.lower()
    if key in _team_id_cache:
        return _team_id_cache[key]

    # Try normalized (without FC/AFC suffix)
    normalized = normalize_team_name(canonical_name)
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


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

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

        if h_id == home_id:
            if hg > ag:
                home_wins += 1
            elif hg == ag:
                draws += 1
            else:
                away_wins += 1
        else:
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


def fetch_football_stats(home_canonical: str, away_canonical: str) -> MatchStats | None:
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

    stats.data_completeness = compute_completeness(stats)
    return stats
