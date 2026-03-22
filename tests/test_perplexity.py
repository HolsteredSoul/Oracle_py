"""Tests for call_perplexity() in src/llm/client.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_response(content: str = "enriched news", cost: float = 0.005) -> MagicMock:
    """Build a mock httpx.Response with OpenRouter-style JSON."""
    resp = MagicMock(spec=httpx.Response)
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"cost": cost},
    }
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCallPerplexity:
    @patch("src.llm.client.settings")
    def test_returns_none_when_no_api_key(self, mock_settings):
        mock_settings.openrouter_api_key = ""
        from src.llm.client import call_perplexity
        assert call_perplexity("test prompt") is None

    @patch("src.llm.client._get_today_spend", return_value=10.0)
    @patch("src.llm.client.settings")
    def test_returns_none_when_daily_cap_hit(self, mock_settings, mock_spend):
        mock_settings.openrouter_api_key = "sk-test"
        mock_settings.llm.daily_cap_usd = 5.0
        from src.llm.client import call_perplexity
        assert call_perplexity("test prompt") is None

    @patch("src.llm.client._add_spend")
    @patch("src.llm.client._call_perplexity_http")
    @patch("src.llm.client._get_today_spend", return_value=0.0)
    @patch("src.llm.client.settings")
    def test_successful_call_returns_content(
        self, mock_settings, mock_spend, mock_http, mock_add_spend,
    ):
        mock_settings.openrouter_api_key = "sk-test"
        mock_settings.llm.daily_cap_usd = 5.0
        mock_settings.llm.perplexity_model = "perplexity/sonar"
        mock_http.return_value = _mock_response("latest injury news")
        from src.llm.client import call_perplexity
        result = call_perplexity("Who is injured?")
        assert result == "latest injury news"

    @patch("src.llm.client._add_spend")
    @patch("src.llm.client._call_perplexity_http")
    @patch("src.llm.client._get_today_spend", return_value=0.0)
    @patch("src.llm.client.settings")
    def test_tracks_cost(
        self, mock_settings, mock_spend, mock_http, mock_add_spend,
    ):
        mock_settings.openrouter_api_key = "sk-test"
        mock_settings.llm.daily_cap_usd = 5.0
        mock_settings.llm.perplexity_model = "perplexity/sonar"
        mock_http.return_value = _mock_response(cost=0.0042)
        from src.llm.client import call_perplexity
        call_perplexity("test")
        mock_add_spend.assert_called_once_with(0.0042)

    @patch("src.llm.client._call_perplexity_http")
    @patch("src.llm.client._get_today_spend", return_value=0.0)
    @patch("src.llm.client.settings")
    def test_returns_none_on_http_error(
        self, mock_settings, mock_spend, mock_http,
    ):
        mock_settings.openrouter_api_key = "sk-test"
        mock_settings.llm.daily_cap_usd = 5.0
        mock_settings.llm.perplexity_model = "perplexity/sonar"
        mock_http.side_effect = httpx.HTTPStatusError(
            "429", request=MagicMock(), response=MagicMock(),
        )
        from src.llm.client import call_perplexity
        assert call_perplexity("test") is None

    @patch("src.llm.client._call_perplexity_http")
    @patch("src.llm.client._get_today_spend", return_value=0.0)
    @patch("src.llm.client.settings")
    def test_returns_none_on_empty_choices(
        self, mock_settings, mock_spend, mock_http,
    ):
        mock_settings.openrouter_api_key = "sk-test"
        mock_settings.llm.daily_cap_usd = 5.0
        mock_settings.llm.perplexity_model = "perplexity/sonar"
        resp = MagicMock(spec=httpx.Response)
        resp.json.return_value = {"choices": [], "usage": {}}
        mock_http.return_value = resp
        from src.llm.client import call_perplexity
        assert call_perplexity("test") is None
