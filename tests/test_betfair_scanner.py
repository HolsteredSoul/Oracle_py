"""Unit tests for src/scanner/betfair_scanner.py.

All Betfair API calls are mocked — no network required.

Tests verify:
1. get_markets() returns dicts with the same keys as manifold.get_markets()
2. get_market_detail() returns a dict with the same keys as manifold.get_market_detail()
3. probability = 1 / best_back_price
4. totalLiquidity = sum of back + lay sizes
5. isResolved is True for CLOSED/SETTLED status, False otherwise
6. Markets outside the probability range are filtered out
7. CLOSED/SUSPENDED markets are excluded from get_markets()
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — build fake betfairlightweight objects
# ---------------------------------------------------------------------------

def _make_price_size(price: float, size: float) -> SimpleNamespace:
    return SimpleNamespace(price=price, size=size)


def _make_runner(back_prices=None, lay_prices=None) -> SimpleNamespace:
    ex = SimpleNamespace(
        available_to_back=back_prices or [],
        available_to_lay=lay_prices or [],
    )
    return SimpleNamespace(ex=ex)


def _make_book(
    market_id: str,
    status: str = "OPEN",
    runners=None,
    total_matched: float = 500.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        market_id=market_id,
        status=status,
        runners=runners or [],
        total_matched=total_matched,
    )


def _make_catalogue(market_id: str, market_name: str = "Test Market") -> SimpleNamespace:
    return SimpleNamespace(market_id=market_id, market_name=market_name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_client():
    """Ensure the module-level _client is reset between tests."""
    import src.scanner.betfair_scanner as bs
    original = bs._client
    bs._client = None
    yield
    bs._client = original


@pytest.fixture
def mock_client():
    """Return a MagicMock Betfair API client and patch _get_client."""
    client = MagicMock()
    with patch("src.scanner.betfair_scanner._get_client", return_value=client):
        yield client


# ---------------------------------------------------------------------------
# get_markets() tests
# ---------------------------------------------------------------------------

class TestGetMarkets:
    def test_returns_correct_keys(self, mock_client):
        """get_markets() dicts must contain the same keys as manifold.get_markets()."""
        runner = _make_runner(
            back_prices=[_make_price_size(2.0, 100.0)],
            lay_prices=[_make_price_size(2.1, 50.0)],
        )
        book = _make_book("1.111", runners=[runner])
        cat = _make_catalogue("1.111", "AFL Grand Final")

        mock_client.betting.list_market_catalogue.return_value = [cat]
        mock_client.betting.list_market_book.return_value = [book]

        from src.scanner.betfair_scanner import get_markets
        markets = get_markets()

        assert len(markets) == 1
        m = markets[0]
        # These are the keys manifold.get_markets() produces + extras used by paper.py
        for key in ("id", "question", "probability", "volume", "url",
                    "totalLiquidity", "isResolved"):
            assert key in m, f"Missing key: {key!r}"

    def test_probability_is_inverse_of_back_price(self, mock_client):
        back_price = 2.5
        runner = _make_runner(back_prices=[_make_price_size(back_price, 200.0)])
        book = _make_book("1.222", runners=[runner])
        mock_client.betting.list_market_catalogue.return_value = [_make_catalogue("1.222")]
        mock_client.betting.list_market_book.return_value = [book]

        from src.scanner.betfair_scanner import get_markets
        markets = get_markets()

        assert len(markets) == 1
        assert abs(markets[0]["probability"] - (1.0 / back_price)) < 1e-6

    def test_total_liquidity_sums_back_and_lay(self, mock_client):
        runner = _make_runner(
            back_prices=[_make_price_size(2.0, 100.0), _make_price_size(2.2, 50.0)],
            lay_prices=[_make_price_size(2.1, 75.0)],
        )
        book = _make_book("1.333", runners=[runner])
        mock_client.betting.list_market_catalogue.return_value = [_make_catalogue("1.333")]
        mock_client.betting.list_market_book.return_value = [book]

        from src.scanner.betfair_scanner import get_markets
        markets = get_markets()

        assert len(markets) == 1
        assert markets[0]["totalLiquidity"] == pytest.approx(100.0 + 50.0 + 75.0)

    def test_closed_markets_excluded(self, mock_client):
        runner = _make_runner(back_prices=[_make_price_size(2.0, 100.0)])
        book = _make_book("1.444", status="CLOSED", runners=[runner])
        mock_client.betting.list_market_catalogue.return_value = [_make_catalogue("1.444")]
        mock_client.betting.list_market_book.return_value = [book]

        from src.scanner.betfair_scanner import get_markets
        markets = get_markets()

        assert markets == []

    def test_suspended_markets_excluded(self, mock_client):
        runner = _make_runner(back_prices=[_make_price_size(2.0, 100.0)])
        book = _make_book("1.555", status="SUSPENDED", runners=[runner])
        mock_client.betting.list_market_catalogue.return_value = [_make_catalogue("1.555")]
        mock_client.betting.list_market_book.return_value = [book]

        from src.scanner.betfair_scanner import get_markets
        markets = get_markets()

        assert markets == []

    def test_probability_outside_range_filtered(self, mock_client):
        """A back price of 1.05 → prob ≈ 0.952 — above default max of 0.95 — should be dropped."""
        runner = _make_runner(back_prices=[_make_price_size(1.05, 1000.0)])
        book = _make_book("1.666", runners=[runner])
        mock_client.betting.list_market_catalogue.return_value = [_make_catalogue("1.666")]
        mock_client.betting.list_market_book.return_value = [book]

        from src.scanner.betfair_scanner import get_markets
        markets = get_markets()

        assert markets == []

    def test_is_resolved_false_for_open_market(self, mock_client):
        runner = _make_runner(back_prices=[_make_price_size(3.0, 50.0)])
        book = _make_book("1.777", status="OPEN", runners=[runner])
        mock_client.betting.list_market_catalogue.return_value = [_make_catalogue("1.777")]
        mock_client.betting.list_market_book.return_value = [book]

        from src.scanner.betfair_scanner import get_markets
        markets = get_markets()

        assert len(markets) == 1
        assert markets[0]["isResolved"] is False

    def test_empty_catalogue_returns_empty_list(self, mock_client):
        mock_client.betting.list_market_catalogue.return_value = []

        from src.scanner.betfair_scanner import get_markets
        assert get_markets() == []


# ---------------------------------------------------------------------------
# get_market_detail() tests
# ---------------------------------------------------------------------------

class TestGetMarketDetail:
    def test_returns_correct_keys(self, mock_client):
        """get_market_detail() must return the same keys manifold.get_market_detail() does."""
        runner = _make_runner(
            back_prices=[_make_price_size(4.0, 200.0)],
            lay_prices=[_make_price_size(4.2, 100.0)],
        )
        book = _make_book("1.888", runners=[runner])
        cat = _make_catalogue("1.888", "Test Race")

        mock_client.betting.list_market_catalogue.return_value = [cat]
        mock_client.betting.list_market_book.return_value = [book]

        from src.scanner.betfair_scanner import get_market_detail
        detail = get_market_detail("1.888")

        for key in ("id", "question", "probability", "volume", "url",
                    "totalLiquidity", "isResolved", "resolution"):
            assert key in detail, f"Missing key: {key!r}"

    def test_is_resolved_true_for_closed(self, mock_client):
        book = _make_book("1.999", status="CLOSED")
        mock_client.betting.list_market_catalogue.return_value = []
        mock_client.betting.list_market_book.return_value = [book]

        from src.scanner.betfair_scanner import get_market_detail
        detail = get_market_detail("1.999")

        assert detail["isResolved"] is True

    def test_is_resolved_true_for_settled(self, mock_client):
        book = _make_book("1.100", status="SETTLED")
        mock_client.betting.list_market_catalogue.return_value = []
        mock_client.betting.list_market_book.return_value = [book]

        from src.scanner.betfair_scanner import get_market_detail
        detail = get_market_detail("1.100")

        assert detail["isResolved"] is True

    def test_is_resolved_false_for_open(self, mock_client):
        runner = _make_runner(back_prices=[_make_price_size(2.0, 50.0)])
        book = _make_book("1.200", status="OPEN", runners=[runner])
        mock_client.betting.list_market_catalogue.return_value = []
        mock_client.betting.list_market_book.return_value = [book]

        from src.scanner.betfair_scanner import get_market_detail
        detail = get_market_detail("1.200")

        assert detail["isResolved"] is False

    def test_resolution_always_mkt(self, mock_client):
        """Betfair doesn't have YES/NO resolution — must always be 'MKT'."""
        book = _make_book("1.300", status="CLOSED")
        mock_client.betting.list_market_catalogue.return_value = []
        mock_client.betting.list_market_book.return_value = [book]

        from src.scanner.betfair_scanner import get_market_detail
        detail = get_market_detail("1.300")

        assert detail["resolution"] == "MKT"

    def test_raises_on_empty_book(self, mock_client):
        mock_client.betting.list_market_catalogue.return_value = []
        mock_client.betting.list_market_book.return_value = []

        from src.scanner.betfair_scanner import get_market_detail
        with pytest.raises(ValueError, match="No market book returned"):
            get_market_detail("1.400")

    def test_probability_matches_back_price(self, mock_client):
        back_price = 5.0
        runner = _make_runner(back_prices=[_make_price_size(back_price, 300.0)])
        book = _make_book("1.500", runners=[runner])
        mock_client.betting.list_market_catalogue.return_value = []
        mock_client.betting.list_market_book.return_value = [book]

        from src.scanner.betfair_scanner import get_market_detail
        detail = get_market_detail("1.500")

        assert abs(detail["probability"] - (1.0 / back_price)) < 1e-6

    def test_market_id_preserved(self, mock_client):
        book = _make_book("1.600", runners=[])
        mock_client.betting.list_market_catalogue.return_value = []
        mock_client.betting.list_market_book.return_value = [book]

        from src.scanner.betfair_scanner import get_market_detail
        detail = get_market_detail("1.600")

        assert detail["id"] == "1.600"


# ---------------------------------------------------------------------------
# Shape compatibility: manifold vs betfair
# ---------------------------------------------------------------------------

class TestShapeCompatibility:
    """Verify that betfair_scanner output is a superset of manifold output keys."""

    MANIFOLD_GET_MARKETS_KEYS = {"id", "question", "probability", "volume", "url"}
    MANIFOLD_DETAIL_KEYS = {
        "id", "question", "probability", "volume", "url",
        "totalLiquidity", "isResolved",
    }

    def test_get_markets_is_superset_of_manifold(self, mock_client):
        runner = _make_runner(back_prices=[_make_price_size(3.0, 100.0)])
        book = _make_book("1.700", runners=[runner])
        mock_client.betting.list_market_catalogue.return_value = [_make_catalogue("1.700")]
        mock_client.betting.list_market_book.return_value = [book]

        from src.scanner.betfair_scanner import get_markets
        markets = get_markets()

        assert len(markets) == 1
        assert self.MANIFOLD_GET_MARKETS_KEYS.issubset(markets[0].keys())

    def test_get_market_detail_is_superset_of_manifold(self, mock_client):
        runner = _make_runner(back_prices=[_make_price_size(3.0, 100.0)])
        book = _make_book("1.800", runners=[runner])
        mock_client.betting.list_market_catalogue.return_value = [_make_catalogue("1.800")]
        mock_client.betting.list_market_book.return_value = [book]

        from src.scanner.betfair_scanner import get_market_detail
        detail = get_market_detail("1.800")

        assert self.MANIFOLD_DETAIL_KEYS.issubset(detail.keys())
