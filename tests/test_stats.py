"""Tests for src/enrichment/stats.py."""

from src.enrichment.stats import MatchStats, _compute_completeness


class TestComputeCompleteness:
    """Tests for data completeness calculation."""

    def test_all_fields_filled(self):
        stats = MatchStats(
            sport="football",
            home_team="A",
            away_team="B",
            home_form_pts_per_game=2.0,
            away_form_pts_per_game=1.5,
            home_goals_scored_avg=1.5,
            home_goals_conceded_avg=1.0,
            away_goals_scored_avg=1.2,
            away_goals_conceded_avg=1.3,
            home_league_position=3,
            away_league_position=7,
        )
        assert _compute_completeness(stats) == 1.0

    def test_no_fields_filled(self):
        stats = MatchStats(sport="football", home_team="A", away_team="B")
        assert _compute_completeness(stats) == 0.0

    def test_half_filled(self):
        stats = MatchStats(
            sport="football",
            home_team="A",
            away_team="B",
            home_form_pts_per_game=2.0,
            away_form_pts_per_game=1.5,
            home_goals_scored_avg=1.5,
            home_goals_conceded_avg=1.0,
        )
        assert _compute_completeness(stats) == 0.5

    def test_single_field(self):
        stats = MatchStats(
            sport="football",
            home_team="A",
            away_team="B",
            home_form_pts_per_game=2.0,
        )
        assert _compute_completeness(stats) == 0.12  # 1/8 rounded


class TestMatchStatsModel:
    """Tests for MatchStats Pydantic model."""

    def test_defaults(self):
        stats = MatchStats(sport="football", home_team="A", away_team="B")
        assert stats.h2h_home_wins == 0
        assert stats.h2h_draws == 0
        assert stats.h2h_away_wins == 0
        assert stats.data_completeness == 0.0

    def test_afl_sport(self):
        stats = MatchStats(sport="afl", home_team="Sydney", away_team="Collingwood")
        assert stats.sport == "afl"

    def test_baseball_sport(self):
        stats = MatchStats(sport="baseball", home_team="New York Yankees", away_team="Boston Red Sox")
        assert stats.sport == "baseball"

    def test_rugby_sport(self):
        stats = MatchStats(sport="rugby", home_team="Hurricanes", away_team="Blues")
        assert stats.sport == "rugby"
