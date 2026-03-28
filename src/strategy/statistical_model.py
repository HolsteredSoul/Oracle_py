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

# Hockey home-ice advantage (historical ~55%)
_HOCKEY_HOME_ADVANTAGE = 0.55

# Rugby League home advantage (historical ~58%, NRL is heavily home-biased)
_RUGBY_LEAGUE_HOME_ADVANTAGE = 0.58

# Cricket home advantage (historical ~55% in limited-overs, higher in Tests)
_CRICKET_HOME_ADVANTAGE = 0.55


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


def _predict_rugby(stats: MatchStats) -> dict[str, float]:
    """Rugby Union prediction using scoring differential + home advantage.

    Uses a logistic model based on:
      - Net points differential (points scored - conceded per game)
      - Standings position differential
      - Home advantage intercept (~56% historical rugby home win rate)
    Returns home/away only — draws are rare in modern Super Rugby.
    """
    from scipy.special import expit

    home_scored = stats.home_goals_scored_avg
    home_conceded = stats.home_goals_conceded_avg
    away_scored = stats.away_goals_scored_avg
    away_conceded = stats.away_goals_conceded_avg

    # logit(0.56) ≈ 0.24 — rugby home advantage intercept
    logit_p = 0.24

    # Primary: net points differential, normalized by 15 (typical rugby spread)
    if (home_scored is not None and home_conceded is not None
            and away_scored is not None and away_conceded is not None):
        home_net = home_scored - home_conceded
        away_net = away_scored - away_conceded
        net_diff = (home_net - away_net) / 15.0
        logit_p += 1.5 * net_diff

    # Secondary: standings position
    if stats.home_league_position is not None and stats.away_league_position is not None:
        standings_diff = (stats.away_league_position - stats.home_league_position)
        standings_diff_norm = standings_diff / 12.0  # ~12 teams in Super Rugby
        logit_p += 0.5 * standings_diff_norm

    p_home = float(expit(logit_p))
    p_home = max(0.05, min(0.95, p_home))
    return {"home": round(p_home, 6), "away": round(1 - p_home, 6)}


def _predict_baseball(stats: MatchStats) -> dict[str, float]:
    """Baseball prediction using net run differential + home advantage.

    No draws in baseball. Uses a logistic model based on:
      - Net run differential (runs scored - runs allowed per game)
      - Standings position differential
      - Home-field advantage intercept (~54% historical MLB home win rate)
    """
    from scipy.special import expit

    home_scored = stats.home_goals_scored_avg   # runs scored per game
    home_conceded = stats.home_goals_conceded_avg  # runs allowed per game
    away_scored = stats.away_goals_scored_avg
    away_conceded = stats.away_goals_conceded_avg

    # logit(0.54) ≈ 0.16 — MLB home-field advantage intercept
    logit_p = 0.16

    # Primary signal: net run differential, normalized by 5 (typical MLB range)
    if (home_scored is not None and home_conceded is not None
            and away_scored is not None and away_conceded is not None):
        home_net = home_scored - home_conceded
        away_net = away_scored - away_conceded
        net_diff = (home_net - away_net) / 5.0
        logit_p += 1.5 * net_diff

    # Secondary signal: standings position (lower = better)
    if stats.home_league_position is not None and stats.away_league_position is not None:
        standings_diff = (stats.away_league_position - stats.home_league_position)
        standings_diff_norm = standings_diff / 15.0  # 15 teams per league
        logit_p += 0.5 * standings_diff_norm

    p_home = float(expit(logit_p))
    p_home = max(0.05, min(0.95, p_home))
    return {"home": round(p_home, 6), "away": round(1 - p_home, 6)}


def _predict_hockey(stats: MatchStats) -> dict[str, float]:
    """Ice hockey prediction using goal differential + home-ice advantage.

    Uses a Poisson model for goals (like football but with hockey-specific lambdas).
    Hockey has OT/shootout so draws in regulation are possible but rare on Betfair
    (most markets are for regulation + OT result). We model home/away only.
    """
    from scipy.special import expit

    home_scored = stats.home_goals_scored_avg
    home_conceded = stats.home_goals_conceded_avg
    away_scored = stats.away_goals_scored_avg
    away_conceded = stats.away_goals_conceded_avg

    # logit(0.55) ≈ 0.20 — home-ice advantage intercept
    logit_p = 0.20

    # Primary: net goal differential, normalized by 3 (typical NHL spread range)
    if (home_scored is not None and home_conceded is not None
            and away_scored is not None and away_conceded is not None):
        home_net = home_scored - home_conceded
        away_net = away_scored - away_conceded
        net_diff = (home_net - away_net) / 3.0
        logit_p += 1.5 * net_diff

    # Secondary: standings position
    if stats.home_league_position is not None and stats.away_league_position is not None:
        standings_diff = (stats.away_league_position - stats.home_league_position)
        standings_diff_norm = standings_diff / 16.0  # ~16 teams per conference
        logit_p += 0.5 * standings_diff_norm

    p_home = float(expit(logit_p))
    p_home = max(0.05, min(0.95, p_home))
    return {"home": round(p_home, 6), "away": round(1 - p_home, 6)}


def _predict_cricket(stats: MatchStats) -> dict[str, float]:
    """Cricket prediction using run differential + home advantage.

    No draws in limited-overs cricket. Uses a logistic model based on:
      - Net run rate differential (runs scored - runs conceded per match)
      - Form differential
      - Home advantage intercept (~55% historical)
    """
    from scipy.special import expit

    home_scored = stats.home_goals_scored_avg   # runs scored per match
    home_conceded = stats.home_goals_conceded_avg
    away_scored = stats.away_goals_scored_avg
    away_conceded = stats.away_goals_conceded_avg

    # logit(0.55) ≈ 0.20
    logit_p = 0.20

    # Primary: net run differential, normalized by 50 (typical T20 margin range)
    if (home_scored is not None and home_conceded is not None
            and away_scored is not None and away_conceded is not None):
        home_net = home_scored - home_conceded
        away_net = away_scored - away_conceded
        net_diff = (home_net - away_net) / 50.0
        logit_p += 1.5 * net_diff

    # Form differential as secondary signal
    if stats.home_form_pts_per_game is not None and stats.away_form_pts_per_game is not None:
        form_diff = (stats.home_form_pts_per_game - stats.away_form_pts_per_game) / 2.0
        logit_p += 0.5 * form_diff

    p_home = float(expit(logit_p))
    p_home = max(0.05, min(0.95, p_home))
    return {"home": round(p_home, 6), "away": round(1 - p_home, 6)}


def _predict_rugby_league(stats: MatchStats) -> dict[str, float]:
    """Rugby League (NRL) prediction using scoring differential + home advantage.

    Uses a logistic model based on:
      - Net points differential (points scored - conceded per game)
      - Standings position differential
      - Home advantage intercept (~58% historical NRL home win rate)
    """
    from scipy.special import expit

    home_scored = stats.home_goals_scored_avg
    home_conceded = stats.home_goals_conceded_avg
    away_scored = stats.away_goals_scored_avg
    away_conceded = stats.away_goals_conceded_avg

    # logit(0.58) ≈ 0.32 — NRL has strong home advantage
    logit_p = 0.32

    # Primary: net points differential, normalized by 15 (typical NRL spread)
    if (home_scored is not None and home_conceded is not None
            and away_scored is not None and away_conceded is not None):
        home_net = home_scored - home_conceded
        away_net = away_scored - away_conceded
        net_diff = (home_net - away_net) / 15.0
        logit_p += 1.5 * net_diff

    # Secondary: standings position
    if stats.home_league_position is not None and stats.away_league_position is not None:
        standings_diff = (stats.away_league_position - stats.home_league_position)
        standings_diff_norm = standings_diff / 8.0  # ~16 teams, midpoint ~8
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
    elif stats.sport == "baseball":
        probs = _predict_baseball(stats)
    elif stats.sport == "rugby":
        probs = _predict_rugby(stats)
    elif stats.sport == "hockey":
        probs = _predict_hockey(stats)
    elif stats.sport == "cricket":
        probs = _predict_cricket(stats)
    elif stats.sport == "rugby_league":
        probs = _predict_rugby_league(stats)
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

    # Common Betfair abbreviations for runner names
    _ABBREVS: dict[str, str] = {
        "qld": "queensland", "nsw": "new south wales", "vic": "victoria",
        "sa": "south australia", "wa": "western australia", "tas": "tasmania",
        "melb": "melbourne", "syd": "sydney", "bris": "brisbane",
        "adel": "adelaide", "pth": "perth", "canb": "canberra",
        "man": "manchester", "utd": "united", "fc": "football club",
        "wst": "western", "nth": "north", "sth": "south",
    }

    def _expand(text: str) -> str:
        """Expand known abbreviations in a name."""
        words = text.split()
        return " ".join(_ABBREVS.get(w, w) for w in words)

    def _matches(runner: str, team: str) -> bool:
        """Check if runner name matches team via substring or fuzzy."""
        if runner == team or runner in team or team in runner:
            return True
        # Expand abbreviations and retry
        runner_exp = _expand(runner)
        team_exp = _expand(team)
        if runner_exp in team_exp or team_exp in runner_exp:
            return True
        # Check if all runner words appear as prefixes of team words
        runner_words = runner_exp.split()
        team_words = team_exp.split()
        if runner_words and team_words:
            matched = 0
            for rw in runner_words:
                for tw in team_words:
                    if tw.startswith(rw) or rw.startswith(tw):
                        matched += 1
                        break
            if matched == len(runner_words):
                return True
        return False

    is_home = _matches(runner_lower, home_lower)
    is_away = _matches(runner_lower, away_lower)

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
