"""Tests for src/enrichment/team_mapping.py."""

from src.enrichment.team_mapping import parse_teams_from_event, resolve_team


class TestResolveTeam:
    """Tests for resolve_team()."""

    def test_exact_alias_football(self):
        assert resolve_team("Man Utd", "football") == "Manchester United FC"

    def test_exact_alias_case_insensitive(self):
        assert resolve_team("MAN UTD", "football") == "Manchester United FC"

    def test_exact_alias_afl(self):
        assert resolve_team("Sydney Swans", "afl") == "Sydney"

    def test_exact_alias_afl_short(self):
        assert resolve_team("Collingwood", "afl") == "Collingwood"

    def test_fuzzy_match_close_name(self):
        # "Melbourne Victory FC" should fuzzy-match to alias "melbourne victory"
        result = resolve_team("Melbourne Victory FC", "football")
        assert result == "Melbourne Victory FC"

    def test_no_match_returns_none(self):
        result = resolve_team("Completely Unknown Team XYZ 12345", "football")
        assert result is None

    def test_no_match_garbage_afl(self):
        result = resolve_team("Nonexistent Team", "afl")
        assert result is None

    def test_caches_result(self):
        # First call resolves, second should hit cache
        r1 = resolve_team("Spurs", "football")
        r2 = resolve_team("Spurs", "football")
        assert r1 == r2 == "Tottenham Hotspur FC"

    def test_a_league_teams(self):
        assert resolve_team("Western Sydney", "football") == "Western Sydney Wanderers FC"
        assert resolve_team("Central Coast", "football") == "Central Coast Mariners FC"
        assert resolve_team("Brisbane Roar", "football") == "Brisbane Roar FC"

    def test_european_aliases(self):
        assert resolve_team("Bayern Munich", "football") == "FC Bayern München"
        assert resolve_team("PSG", "football") == "Paris Saint-Germain FC"
        assert resolve_team("Inter Milan", "football") == "FC Internazionale Milano"
        assert resolve_team("Barca", "football") == "FC Barcelona"


class TestParseTeamsFromEvent:
    """Tests for parse_teams_from_event()."""

    def test_standard_format(self):
        result = parse_teams_from_event("Sydney FC v Melbourne Victory")
        assert result == ("Sydney FC", "Melbourne Victory")

    def test_with_whitespace(self):
        result = parse_teams_from_event("  Team A  v  Team B  ")
        assert result == ("Team A", "Team B")

    def test_no_v_separator(self):
        result = parse_teams_from_event("AFL 2026 Grand Final")
        assert result is None

    def test_empty_string(self):
        result = parse_teams_from_event("")
        assert result is None

    def test_multiple_v(self):
        # Edge case: "v" appears in team name — split only on first " v "
        # Actually split() splits on all occurrences, so this returns None (3 parts)
        result = parse_teams_from_event("Team v Name v Other")
        assert result is None

    def test_epl_fixture(self):
        result = parse_teams_from_event("Arsenal v Chelsea")
        assert result == ("Arsenal", "Chelsea")
