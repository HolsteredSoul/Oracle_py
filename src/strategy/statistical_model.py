"""Statistical prediction model for match outcome probabilities.

Phase 1: Independent Poisson model for football, logistic for AFL.
No ML training required — uses recent form data directly.

Produces calibrated probability estimates that replace market mid-price
as the Bayesian prior in the main pipeline.
"""

from __future__ import annotations

import logging

from scipy.stats import poisson

from src.config import settings
from src.enrichment.stats import MatchStats

logger = logging.getLogger(__name__)

# Default league average goals per game (used when team data is incomplete)
_DEFAULT_LAMBDA = 1.30

# AFL home advantage (historical ~57%)
_AFL_HOME_ADVANTAGE = 0.57

# Basketball home-court advantage (historical ~57%, varies by league)
_BASKETBALL_HOME_ADVANTAGE = 0.57


def _poisson_match_probs(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 7,
) -> dict[str, float]:
    """Compute home/draw/away probabilities from independent Poisson model.

    Enumerates all goal combinations up to max_goals and sums joint
    probabilities for each outcome.
    """
    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    for h in range(max_goals + 1):
        p_h = poisson.pmf(h, lambda_home)
        for a in range(max_goals + 1):
            p_a = poisson.pmf(a, lambda_away)
            joint = p_h * p_a
            if h > a:
                home_win += joint
            elif h == a:
                draw += joint
            else:
                away_win += joint

    # Normalize to sum to 1.0 (truncation at max_goals loses a tiny amount)
    total = home_win + draw + away_win
    if total <= 0:
        return {"home": 0.33, "draw": 0.34, "away": 0.33}

    return {
        "home": round(home_win / total, 6),
        "draw": round(draw / total, 6),
        "away": round(away_win / total, 6),
    }


def _predict_football(stats: MatchStats) -> dict[str, float]:
    """Football prediction using independent Poisson model.

    lambda_home = (home_goals_scored_avg + away_goals_conceded_avg) / 2
    lambda_away = (away_goals_scored_avg + home_goals_conceded_avg) / 2
    """
    home_scored = stats.home_goals_scored_avg
    home_conceded = stats.home_goals_conceded_avg
    away_scored = stats.away_goals_scored_avg
    away_conceded = stats.away_goals_conceded_avg

    # Build lambdas from available data, falling back to league average
    if home_scored is not None and away_conceded is not None:
        lambda_home = (home_scored + away_conceded) / 2
    elif home_scored is not None:
        lambda_home = home_scored
    elif away_conceded is not None:
        lambda_home = away_conceded
    else:
        lambda_home = _DEFAULT_LAMBDA

    if away_scored is not None and home_conceded is not None:
        lambda_away = (away_scored + home_conceded) / 2
    elif away_scored is not None:
        lambda_away = away_scored
    elif home_conceded is not None:
        lambda_away = home_conceded
    else:
        lambda_away = _DEFAULT_LAMBDA

    # Clamp to reasonable range
    lambda_home = max(0.2, min(4.0, lambda_home))
    lambda_away = max(0.2, min(4.0, lambda_away))

    return _poisson_match_probs(lambda_home, lambda_away)


def _predict_afl(stats: MatchStats) -> dict[str, float]:
    """AFL prediction using form differential + home advantage.

    No draws in AFL. Uses a simple logistic-style approach based on
    form points differential with a home advantage intercept.
    """
    from scipy.special import expit

    # Form differential: higher means home team is in better form
    home_form = stats.home_form_pts_per_game
    away_form = stats.away_form_pts_per_game

    if home_form is not None and away_form is not None:
        # Normalize: AFL wins = 4 pts, max ~4 pts/game
        form_diff = (home_form - away_form) / 4.0  # range roughly [-1, 1]
        # Logistic: home advantage intercept + form differential
        # logit(0.57) ≈ 0.28, scale form_diff by 1.5 for reasonable sensitivity
        p_home = float(expit(0.28 + 1.5 * form_diff))
    else:
        p_home = _AFL_HOME_ADVANTAGE

    p_home = max(0.05, min(0.95, p_home))
    return {"home": round(p_home, 6), "away": round(1 - p_home, 6)}


def _predict_basketball(stats: MatchStats) -> dict[str, float]:
    """Basketball prediction using net rating differential + home-court advantage.

    No draws in basketball. Uses a logistic model based on:
      - Net rating differential (points scored - points allowed per game)
      - Standings position differential
      - Home-court advantage intercept
    """
    from scipy.special import expit

    home_scored = stats.home_goals_scored_avg  # points scored per game
    home_conceded = stats.home_goals_conceded_avg  # points allowed per game
    away_scored = stats.away_goals_scored_avg
    away_conceded = stats.away_goals_conceded_avg

    # logit(0.57) ≈ 0.28 — home-court advantage intercept
    logit_p = 0.28

    # Primary signal: net rating differential, normalized by 20 (typical spread range)
    if (home_scored is not None and home_conceded is not None
            and away_scored is not None and away_conceded is not None):
        home_net = home_scored - home_conceded
        away_net = away_scored - away_conceded
        net_diff = (home_net - away_net) / 20.0  # roughly [-1, 1]
        logit_p += 1.5 * net_diff

    # Secondary signal: standings position (lower = better)
    if stats.home_league_position is not None and stats.away_league_position is not None:
        # Positive when home is ranked higher (lower number)
        standings_diff = (stats.away_league_position - stats.home_league_position)
        # Normalize by a typical league size (~16-30 teams)
        standings_diff_norm = standings_diff / 20.0
        logit_p += 0.5 * standings_diff_norm

    p_home = float(expit(logit_p))
    p_home = max(0.05, min(0.95, p_home))
    return {"home": round(p_home, 6), "away": round(1 - p_home, 6)}


def predict_match_odds(stats: MatchStats) -> dict[str, float] | None:
    """Predict match outcome probabilities from statistical features.

    Returns {"home": p, "draw": p, "away": p} for football (sum to 1.0),
    or {"home": p, "away": p} for AFL (no draws).

    Returns None if data_completeness is below the configured threshold.
    """
    if stats.data_completeness < settings.stats.min_data_completeness:
        logger.info(
            "Insufficient data for %s v %s (completeness=%.0f%% < %.0f%% threshold)",
            stats.home_team, stats.away_team,
            stats.data_completeness * 100,
            settings.stats.min_data_completeness * 100,
        )
        return None

    if stats.sport == "afl":
        probs = _predict_afl(stats)
    elif stats.sport == "basketball":
        probs = _predict_basketball(stats)
    else:
        probs = _predict_football(stats)

    logger.info(
        "Statistical model: %s v %s -> %s",
        stats.home_team, stats.away_team,
        {k: f"{v:.3f}" for k, v in probs.items()},
    )
    return probs


def select_runner_prob(
    model_probs: dict[str, float],
    runner_name: str,
    market_type: str,
    home_team: str,
    away_team: str,
) -> float | None:
    """Map a Betfair runner name + market type to the correct model probability.

    For MATCH_ODDS: runner is home, away, or "The Draw".
    For DRAW_NO_BET: renormalize home/away (exclude draw).
    For WINNER/OUTRIGHT_WINNER: match runner to team.
    """
    runner_lower = runner_name.lower().strip()
    home_lower = home_team.lower().strip()
    away_lower = away_team.lower().strip()

    # Determine if runner is home, away, or draw
    if runner_lower == "the draw" or runner_lower == "draw":
        if "draw" in model_probs:
            return model_probs["draw"]
        return None

    is_home = (
        runner_lower == home_lower
        or runner_lower in home_lower
        or home_lower in runner_lower
    )
    is_away = (
        runner_lower == away_lower
        or runner_lower in away_lower
        or away_lower in runner_lower
    )

    if not is_home and not is_away:
        logger.debug(
            "Cannot map runner %r to home=%r or away=%r",
            runner_name, home_team, away_team,
        )
        return None

    if market_type == "DRAW_NO_BET":
        # Renormalize: exclude draw probability
        p_home = model_probs.get("home", 0.5)
        p_away = model_probs.get("away", 0.5)
        total = p_home + p_away
        if total <= 0:
            return None
        if is_home:
            return round(p_home / total, 6)
        return round(p_away / total, 6)

    # MATCH_ODDS, WINNER, OUTRIGHT_WINNER, MONEYLINE
    if is_home:
        return model_probs.get("home")
    return model_probs.get("away")
