"""Tests for src/enrichment/team_mapping.py."""

from unittest.mock import patch

from src.enrichment.team_mapping import (
    _resolved_cache,
    parse_teams_from_event,
    resolve_team,
)

# Simulated football-data.org team index (lowercase key -> team ID)
_MOCK_FD_INDEX = {
    "manchester united fc": 66,
    "man united": 66,
    "tottenham hotspur fc": 73,
    "wolverhampton wanderers fc": 76,
    "wolves": 76,
    "nottingham forest fc": 65,
    "forest": 65,
    "fc bayern münchen": 5,
    "bayern": 5,
    "paris saint-germain fc": 524,
    "psg": 524,
    "fc internazionale milano": 108,
    "inter": 108,
    "fc barcelona": 81,
    "barça": 81,
    "arsenal fc": 57,
    "arsenal": 57,
    "chelsea fc": 61,
    "chelsea": 61,
    "barnsley fc": 357,
    "barnsley": 357,
    "reading fc": 355,
    "reading": 355,
    "bromley fc": 1044,
    "bromley": 1044,
    "brighton & hove albion fc": 397,
    "brighton": 397,
    "western sydney wanderers fc": 7011,
    "sydney fc": 7010,
    "melbourne victory fc": 7013,
    "central coast mariners fc": 7014,
    "brisbane roar fc": 7015,
}


def _mock_get_football_index():
    return _MOCK_FD_INDEX


@patch(
    "src.enrichment.team_mapping._get_football_index",
    side_effect=_mock_get_football_index,
)
class TestResolveTeam:
    """Tests for resolve_team() against the football-data.org team index."""

    def setup_method(self):
        _resolved_cache.clear()

    def test_exact_match_from_index(self, _mock):
        result = resolve_team("Arsenal FC", "football")
        # Result should contain "Arsenal" with correct casing from name cache or .title() fallback
        assert result is not None
        assert result.lower() == "arsenal fc"

    def test_exact_match_short_name(self, _mock):
        assert resolve_team("Arsenal", "football") == "Arsenal"

    def test_normalized_match_strips_fc(self, _mock):
        # "Barnsley" should match "barnsley fc" after normalization
        result = resolve_team("Barnsley", "football")
        assert result is not None
        assert "barnsley" in result.lower()

    def test_fuzzy_match(self, _mock):
        # "Melbourne Victory" should fuzzy-match "melbourne victory fc"
        result = resolve_team("Melbourne Victory", "football")
        assert result is not None
        assert "melbourne" in result.lower()

    def test_override_takes_precedence(self, _mock):
        # "Nott'm Forest" is in overrides -> "Nottingham Forest FC"
        assert resolve_team("Nott'm Forest", "football") == "Nottingham Forest FC"

    def test_override_spurs(self, _mock):
        assert resolve_team("Spurs", "football") == "Tottenham Hotspur FC"

    def test_no_match_returns_none(self, _mock):
        result = resolve_team("Completely Unknown Team XYZ 12345", "football")
        assert result is None

    def test_caches_result(self, _mock):
        r1 = resolve_team("Chelsea", "football")
        r2 = resolve_team("Chelsea", "football")
        assert r1 == r2

    def test_case_insensitive(self, _mock):
        r = resolve_team("ARSENAL", "football")
        assert r is not None

    def test_western_sydney(self, _mock):
        r = resolve_team("Western Sydney Wanderers FC", "football")
        assert r is not None
        assert "western sydney" in r.lower()


class TestResolveTeamAFL:
    """Tests for AFL alias resolution (no mock needed — uses hardcoded aliases)."""

    def setup_method(self):
        _resolved_cache.clear()

    def test_exact_alias_afl(self):
        assert resolve_team("Sydney Swans", "afl") == "Sydney"

    def test_exact_alias_afl_short(self):
        assert resolve_team("Collingwood", "afl") == "Collingwood"

    def test_no_match_garbage_afl(self):
        result = resolve_team("Nonexistent Team", "afl")
        assert result is None

    def test_afl_fuzzy(self):
        result = resolve_team("Brisbane Lion", "afl")
        assert result is not None


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
        # maxsplit=1: first separator splits, rest stays with away team
        result = parse_teams_from_event("Team v Name v Other")
        assert result == ("Team", "Name v Other")

    def test_vs_separator(self):
        result = parse_teams_from_event("Sydney vs Melbourne")
        assert result == ("Sydney", "Melbourne")

    def test_vs_dot_separator(self):
        result = parse_teams_from_event("Sydney vs. Melbourne")
        assert result == ("Sydney", "Melbourne")

    def test_dash_separator(self):
        result = parse_teams_from_event("Sydney - Melbourne")
        assert result == ("Sydney", "Melbourne")

    def test_epl_fixture(self):
        result = parse_teams_from_event("Arsenal v Chelsea")
        assert result == ("Arsenal", "Chelsea")
