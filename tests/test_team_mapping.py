"""Tests for src/enrichment/team_mapping.py."""

from unittest.mock import patch

from src.enrichment.team_mapping import (
    _build_canonical_team_list,
    _parse_perplexity_team_response,
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


# ---------------------------------------------------------------------------
# Perplexity fallback tests
# ---------------------------------------------------------------------------

class TestParsePerplexityTeamResponse:
    """Tests for _parse_perplexity_team_response()."""

    def test_exact_match(self):
        result = _parse_perplexity_team_response("Arsenal FC", _MOCK_FD_INDEX)
        assert result is not None
        assert "arsenal" in result.lower()

    def test_none_sentinel(self):
        assert _parse_perplexity_team_response("NONE", _MOCK_FD_INDEX) is None

    def test_none_sentinel_lowercase(self):
        assert _parse_perplexity_team_response("none", _MOCK_FD_INDEX) is None

    def test_multiline_takes_first(self):
        resp = "Arsenal FC\nThis is the team from North London."
        result = _parse_perplexity_team_response(resp, _MOCK_FD_INDEX)
        assert result is not None
        assert "arsenal" in result.lower()

    def test_normalized_match(self):
        # "Arsenal" (no FC) should match "arsenal fc" via normalization
        result = _parse_perplexity_team_response("Arsenal", _MOCK_FD_INDEX)
        assert result is not None

    def test_empty_response(self):
        assert _parse_perplexity_team_response("", _MOCK_FD_INDEX) is None

    def test_garbage_response(self):
        assert _parse_perplexity_team_response("asdfqwerty123", _MOCK_FD_INDEX) is None


class TestBuildCanonicalTeamList:
    """Tests for _build_canonical_team_list()."""

    def test_deduplicates_by_team_id(self):
        names = _build_canonical_team_list(_MOCK_FD_INDEX)
        # Each team ID should appear only once
        assert len(names) == len(set(names))
        # Arsenal has two keys (arsenal fc: 57, arsenal: 57) but should appear once
        arsenal_names = [n for n in names if "arsenal" in n.lower()]
        assert len(arsenal_names) == 1

    def test_returns_nonempty_for_nonempty_index(self):
        assert len(_build_canonical_team_list(_MOCK_FD_INDEX)) > 0

    def test_empty_index(self):
        assert _build_canonical_team_list({}) == []


@patch(
    "src.enrichment.team_mapping._get_football_index",
    return_value=_MOCK_FD_INDEX,
)
class TestPerplexityFallback:
    """Integration tests for the Perplexity fallback path in resolve_team()."""

    def setup_method(self):
        _resolved_cache.clear()

    @patch("src.enrichment.team_mapping._load_perplexity_cache", return_value={})
    @patch("src.enrichment.team_mapping._save_perplexity_cache")
    @patch("src.llm.client.call_perplexity", return_value="Arsenal FC")
    def test_perplexity_resolves_unknown_team(
        self, mock_pplx, mock_save, mock_load, _mock_idx
    ):
        # "Gunners" won't fuzzy-match but Perplexity returns "Arsenal FC"
        result = resolve_team("Gunners", "football")
        assert result is not None
        assert "arsenal" in result.lower()
        mock_pplx.assert_called_once()
        mock_save.assert_called_once()

    @patch(
        "src.enrichment.team_mapping._load_perplexity_cache",
        return_value={"gunners|football": "Arsenal Fc"},
    )
    @patch("src.llm.client.call_perplexity")
    def test_disk_cache_hit_skips_api(self, mock_pplx, mock_load, _mock_idx):
        result = resolve_team("Gunners", "football")
        # Should return cached value without calling Perplexity
        assert result == "Arsenal Fc"
        mock_pplx.assert_not_called()

    @patch(
        "src.enrichment.team_mapping._load_perplexity_cache",
        return_value={"some random club|football": None},
    )
    @patch("src.llm.client.call_perplexity")
    def test_cached_none_skips_api(self, mock_pplx, mock_load, _mock_idx):
        result = resolve_team("Some Random Club", "football")
        # Cached as None (confirmed miss) — should not call Perplexity
        assert result is None
        mock_pplx.assert_not_called()

    @patch("src.enrichment.team_mapping._load_perplexity_cache", return_value={})
    @patch("src.enrichment.team_mapping._save_perplexity_cache")
    @patch("src.llm.client.call_perplexity", return_value="NONE")
    def test_perplexity_none_cached_as_miss(
        self, mock_pplx, mock_save, mock_load, _mock_idx
    ):
        result = resolve_team("Bagmati Province", "football")
        assert result is None
        # Should save None to cache
        saved = mock_save.call_args[0][0]
        assert saved["bagmati province|football"] is None

    @patch("src.enrichment.team_mapping._load_perplexity_cache", return_value={})
    @patch("src.enrichment.team_mapping._save_perplexity_cache")
    @patch("src.llm.client.call_perplexity", return_value=None)
    def test_api_error_not_cached(
        self, mock_pplx, mock_save, mock_load, _mock_idx
    ):
        result = resolve_team("Some Obscure FC", "football")
        assert result is None
        # API error → should NOT cache (allow retry)
        mock_save.assert_not_called()

    @patch("src.enrichment.team_mapping._load_perplexity_cache", return_value={})
    @patch("src.enrichment.team_mapping._save_perplexity_cache")
    @patch(
        "src.llm.client.call_perplexity",
        return_value="I think this might be Arsenal FC based on my analysis.",
    )
    def test_malformed_response_handles_gracefully(
        self, mock_pplx, mock_save, mock_load, _mock_idx
    ):
        # First line doesn't match any team — should cache as None
        result = resolve_team("Mystery Team", "football")
        assert result is None
        saved = mock_save.call_args[0][0]
        assert saved["mystery team|football"] is None
