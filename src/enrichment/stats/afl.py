"""AFL stats fetcher — Squiggle API (free, no auth required)."""

from __future__ import annotations

import datetime
import logging

import httpx

from src.config import settings

from .models import MatchStats, TIMEOUT, compute_completeness

logger = logging.getLogger(__name__)


def _squiggle_get(endpoint: str, params: dict | None = None) -> dict | list | None:
    """GET request to Squiggle API. Returns parsed JSON or None on error."""
    url = f"{settings.stats.afl_api_base}/{endpoint}"
    headers = {"User-Agent": "Oracle_py/1.0 (https://github.com/HolsteredSoul/Oracle_py)"}
    try:
        resp = httpx.get(url, headers=headers, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data  # type: ignore[return-value]
    except Exception as exc:
        logger.warning("Squiggle API request failed: %s %s", endpoint, exc)
        return None


def fetch_afl_stats(home_canonical: str, away_canonical: str) -> MatchStats | None:
    """Fetch AFL stats from Squiggle API."""
    year = datetime.datetime.now(datetime.timezone.utc).year

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

    stats.data_completeness = compute_completeness(stats)
    return stats
