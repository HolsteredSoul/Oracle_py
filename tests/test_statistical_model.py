"""Tests for src/strategy/statistical_model.py."""

import pytest

from src.enrichment.stats import MatchStats
from src.strategy.statistical_model import (
    _poisson_match_probs,
    predict_match_odds,
    select_runner_prob,
)


class TestPoissonMatchProbs:
    """Tests for the independent Poisson model."""

    def test_symmetric_lambdas_symmetric_probs(self):
        probs = _poisson_match_probs(1.3, 1.3)
        # Symmetric input should give equal home/away
        assert abs(probs["home"] - probs["away"]) < 0.001
        assert probs["draw"] > 0

    def test_probabilities_sum_to_one(self):
        probs = _poisson_match_probs(1.5, 1.0)
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.001

    def test_strong_home_favours_home(self):
        probs = _poisson_match_probs(3.0, 0.5)
        assert probs["home"] > probs["away"]
        assert probs["home"] > probs["draw"]

    def test_strong_away_favours_away(self):
        probs = _poisson_match_probs(0.5, 3.0)
        assert probs["away"] > probs["home"]

    def test_all_probs_positive(self):
        probs = _poisson_match_probs(1.0, 1.0)
        for v in probs.values():
            assert v > 0

    @pytest.mark.parametrize("lh,la", [(0.5, 0.5), (1.0, 1.5), (2.0, 0.8), (3.5, 3.5)])
    def test_probs_sum_to_one_parametrized(self, lh, la):
        probs = _poisson_match_probs(lh, la)
        assert abs(sum(probs.values()) - 1.0) < 0.001

    def test_zero_lambda_handled(self):
        # Edge case: very low lambda
        probs = _poisson_match_probs(0.2, 0.2)
        assert abs(sum(probs.values()) - 1.0) < 0.001


class TestPredictMatchOdds:
    """Tests for predict_match_odds()."""

    def test_football_returns_three_outcomes(self):
        stats = MatchStats(
            sport="football",
            home_team="Team A",
            away_team="Team B",
            home_goals_scored_avg=1.5,
            home_goals_conceded_avg=1.0,
            away_goals_scored_avg=1.2,
            away_goals_conceded_avg=1.3,
            home_form_pts_per_game=2.0,
            away_form_pts_per_game=1.5,
            data_completeness=0.75,
        )
        probs = predict_match_odds(stats)
        assert probs is not None
        assert set(probs.keys()) == {"home", "draw", "away"}
        assert abs(sum(probs.values()) - 1.0) < 0.001

    def test_afl_returns_two_outcomes(self):
        stats = MatchStats(
            sport="afl",
            home_team="Sydney",
            away_team="Collingwood",
            home_form_pts_per_game=3.0,
            away_form_pts_per_game=2.0,
            home_goals_scored_avg=90,
            away_goals_scored_avg=80,
            data_completeness=0.5,
        )
        probs = predict_match_odds(stats)
        assert probs is not None
        assert "draw" not in probs
        assert set(probs.keys()) == {"home", "away"}
        assert abs(sum(probs.values()) - 1.0) < 0.001

    def test_insufficient_data_returns_none(self):
        stats = MatchStats(
            sport="football",
            home_team="Team A",
            away_team="Team B",
            data_completeness=0.2,  # Below threshold
        )
        assert predict_match_odds(stats) is None

    def test_afl_home_advantage(self):
        # Equal form should still show home advantage
        stats = MatchStats(
            sport="afl",
            home_team="Sydney",
            away_team="Collingwood",
            home_form_pts_per_game=2.5,
            away_form_pts_per_game=2.5,
            home_goals_scored_avg=85,
            away_goals_scored_avg=85,
            data_completeness=0.5,
        )
        probs = predict_match_odds(stats)
        assert probs is not None
        assert probs["home"] > probs["away"]


class TestSelectRunnerProb:
    """Tests for select_runner_prob()."""

    def test_home_runner(self):
        probs = {"home": 0.45, "draw": 0.25, "away": 0.30}
        p = select_runner_prob(probs, "Sydney FC", "MATCH_ODDS", "Sydney FC", "Melbourne Victory")
        assert p == 0.45

    def test_away_runner(self):
        probs = {"home": 0.45, "draw": 0.25, "away": 0.30}
        p = select_runner_prob(probs, "Melbourne Victory", "MATCH_ODDS", "Sydney FC", "Melbourne Victory")
        assert p == 0.30

    def test_draw_runner(self):
        probs = {"home": 0.45, "draw": 0.25, "away": 0.30}
        p = select_runner_prob(probs, "The Draw", "MATCH_ODDS", "Sydney FC", "Melbourne Victory")
        assert p == 0.25

    def test_draw_no_bet_renormalization(self):
        probs = {"home": 0.4, "draw": 0.2, "away": 0.4}
        p = select_runner_prob(probs, "Sydney FC", "DRAW_NO_BET", "Sydney FC", "Melbourne Victory")
        # DNB: p_home / (p_home + p_away) = 0.4 / 0.8 = 0.5
        assert p is not None
        assert abs(p - 0.5) < 0.001

    def test_unknown_runner_returns_none(self):
        probs = {"home": 0.45, "draw": 0.25, "away": 0.30}
        p = select_runner_prob(probs, "Random Team", "MATCH_ODDS", "Sydney FC", "Melbourne Victory")
        assert p is None

    def test_partial_name_match(self):
        probs = {"home": 0.45, "draw": 0.25, "away": 0.30}
        # "Sydney" is contained in "Sydney FC"
        p = select_runner_prob(probs, "Sydney", "MATCH_ODDS", "Sydney FC", "Melbourne Victory")
        assert p == 0.45
