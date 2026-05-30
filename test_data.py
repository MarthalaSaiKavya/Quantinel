"""
Tests for LAYER 1 · DATA  (data.py)

Verifies that MockDataSource and TinyMockDataSource fulfil the DataSource
protocol and that the MarketData helpers (close_prices, returns, slice_until)
work exactly as the downstream pipeline expects.

Run:  python -m pytest test_data.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contracts import DataSource, MarketData
from data import MockDataSource, TinyMockDataSource


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_source() -> MockDataSource:
    """Default MockDataSource (504 days, seed=7, NVDA+GOOG)."""
    return MockDataSource()


@pytest.fixture
def mock_data(mock_source) -> MarketData:
    return mock_source.load()


@pytest.fixture
def tiny_source() -> TinyMockDataSource:
    return TinyMockDataSource()


@pytest.fixture
def tiny_data(tiny_source) -> MarketData:
    return tiny_source.load()


# ============================================================================
# 1. PROTOCOL CONFORMANCE
# ============================================================================

class TestProtocol:
    """Both data sources must satisfy the DataSource protocol."""

    def test_mock_source_is_datasource(self, mock_source):
        assert isinstance(mock_source, DataSource)

    def test_tiny_source_is_datasource(self, tiny_source):
        assert isinstance(tiny_source, DataSource)

    def test_load_returns_market_data(self, mock_source):
        result = mock_source.load()
        assert isinstance(result, MarketData)


# ============================================================================
# 2. MarketData STRUCTURE (shape, dtypes, index)
# ============================================================================

class TestMarketDataStructure:
    """Ensure bars have the correct columns, index type, and lengths."""

    def test_tickers_match_bars_keys(self, mock_data):
        assert set(mock_data.tickers) == set(mock_data.bars.keys())

    def test_default_tickers_are_nvda_goog(self, mock_data):
        assert mock_data.tickers == ["NVDA", "GOOG"]

    def test_bar_columns(self, mock_data):
        expected_cols = {"open", "high", "low", "close", "volume"}
        for t in mock_data.tickers:
            assert set(mock_data.bars[t].columns) == expected_cols

    def test_bar_index_is_datetime(self, mock_data):
        for t in mock_data.tickers:
            assert isinstance(mock_data.bars[t].index, pd.DatetimeIndex)

    def test_bar_length_matches_n_days(self):
        src = MockDataSource(n_days=100, seed=42)
        data = src.load()
        for t in data.tickers:
            assert len(data.bars[t]) == 100

    def test_all_tickers_same_length(self, mock_data):
        lengths = {len(mock_data.bars[t]) for t in mock_data.tickers}
        assert len(lengths) == 1, "All tickers should have the same number of bars"

    def test_all_tickers_same_dates(self, mock_data):
        idx0 = mock_data.bars[mock_data.tickers[0]].index
        for t in mock_data.tickers[1:]:
            pd.testing.assert_index_equal(mock_data.bars[t].index, idx0)

    def test_no_nans_in_bars(self, mock_data):
        for t in mock_data.tickers:
            assert not mock_data.bars[t].isna().any().any(), f"NaNs in {t} bars"

    def test_prices_are_positive(self, mock_data):
        for t in mock_data.tickers:
            df = mock_data.bars[t]
            for col in ["open", "high", "low", "close"]:
                assert (df[col] > 0).all(), f"Non-positive {col} in {t}"

    def test_volume_is_positive(self, mock_data):
        for t in mock_data.tickers:
            assert (mock_data.bars[t]["volume"] > 0).all()

    def test_high_gte_low(self, mock_data):
        for t in mock_data.tickers:
            df = mock_data.bars[t]
            assert (df["high"] >= df["low"]).all(), f"high < low in {t}"

    def test_high_gte_open_and_close(self, mock_data):
        for t in mock_data.tickers:
            df = mock_data.bars[t]
            assert (df["high"] >= df["open"]).all()
            assert (df["high"] >= df["close"]).all()

    def test_low_lte_open_and_close(self, mock_data):
        for t in mock_data.tickers:
            df = mock_data.bars[t]
            assert (df["low"] <= df["open"]).all()
            assert (df["low"] <= df["close"]).all()


# ============================================================================
# 3. close_prices() HELPER
# ============================================================================

class TestClosePrices:
    """Verify close_prices() returns the right DataFrame."""

    def test_returns_dataframe(self, mock_data):
        cp = mock_data.close_prices()
        assert isinstance(cp, pd.DataFrame)

    def test_columns_are_tickers(self, mock_data):
        cp = mock_data.close_prices()
        assert list(cp.columns) == mock_data.tickers

    def test_values_match_bars_close(self, mock_data):
        cp = mock_data.close_prices()
        for t in mock_data.tickers:
            pd.testing.assert_series_equal(
                cp[t], mock_data.bars[t]["close"], check_names=False
            )

    def test_tiny_close_prices_exact(self, tiny_data):
        cp = tiny_data.close_prices()
        np.testing.assert_array_equal(cp["AAA"].values, [102, 101, 105, 103, 107])
        np.testing.assert_array_equal(cp["BBB"].values, [51, 52, 50, 53, 54])

    def test_index_preserved(self, mock_data):
        cp = mock_data.close_prices()
        pd.testing.assert_index_equal(
            cp.index, mock_data.bars[mock_data.tickers[0]].index
        )


# ============================================================================
# 4. returns() HELPER
# ============================================================================

class TestReturns:
    """Verify returns() computes percent-change correctly."""

    def test_returns_dataframe(self, mock_data):
        ret = mock_data.returns()
        assert isinstance(ret, pd.DataFrame)

    def test_returns_drops_first_row(self, mock_data):
        cp = mock_data.close_prices()
        ret = mock_data.returns()
        assert len(ret) == len(cp) - 1

    def test_returns_no_nans(self, mock_data):
        ret = mock_data.returns()
        assert not ret.isna().any().any()

    def test_tiny_returns_exact(self, tiny_data):
        ret = tiny_data.returns()
        # AAA closes: 102, 101, 105, 103, 107
        # returns:     -1/102, 4/101, -2/105, 4/103
        expected_aaa = np.array([-1 / 102, 4 / 101, -2 / 105, 4 / 103])
        np.testing.assert_allclose(ret["AAA"].values, expected_aaa, rtol=1e-10)

        # BBB closes: 51, 52, 50, 53, 54
        # returns:     1/51, -2/52, 3/50, 1/53
        expected_bbb = np.array([1 / 51, -2 / 52, 3 / 50, 1 / 53])
        np.testing.assert_allclose(ret["BBB"].values, expected_bbb, rtol=1e-10)

    def test_returns_columns_match_tickers(self, mock_data):
        ret = mock_data.returns()
        assert list(ret.columns) == mock_data.tickers


# ============================================================================
# 5. slice_until() HELPER
# ============================================================================

class TestSliceUntil:
    """Verify slice_until() provides point-in-time views (no look-ahead)."""

    def test_returns_market_data(self, mock_data):
        as_of = mock_data.bars[mock_data.tickers[0]].index[50]
        sliced = mock_data.slice_until(as_of)
        assert isinstance(sliced, MarketData)

    def test_tickers_unchanged(self, mock_data):
        as_of = mock_data.bars[mock_data.tickers[0]].index[50]
        sliced = mock_data.slice_until(as_of)
        assert sliced.tickers == mock_data.tickers

    def test_no_future_data(self, mock_data):
        dates = mock_data.bars[mock_data.tickers[0]].index
        as_of = dates[100]
        sliced = mock_data.slice_until(as_of)
        for t in sliced.tickers:
            assert sliced.bars[t].index.max() <= as_of

    def test_sliced_length_correct(self, mock_data):
        dates = mock_data.bars[mock_data.tickers[0]].index
        as_of = dates[99]  # 0-indexed → first 100 rows
        sliced = mock_data.slice_until(as_of)
        for t in sliced.tickers:
            assert len(sliced.bars[t]) == 100

    def test_tiny_slice_until_midpoint(self, tiny_data):
        dates = tiny_data.bars["AAA"].index
        as_of = dates[2]  # 3rd date
        sliced = tiny_data.slice_until(as_of)

        assert len(sliced.bars["AAA"]) == 3
        assert len(sliced.bars["BBB"]) == 3
        np.testing.assert_array_equal(
            sliced.close_prices()["AAA"].values, [102, 101, 105]
        )

    def test_slice_then_close_prices(self, mock_data):
        """Sliced data's close_prices() must be consistent with sliced bars."""
        dates = mock_data.bars[mock_data.tickers[0]].index
        as_of = dates[200]
        sliced = mock_data.slice_until(as_of)
        cp = sliced.close_prices()
        for t in sliced.tickers:
            pd.testing.assert_series_equal(
                cp[t], sliced.bars[t]["close"], check_names=False
            )

    def test_slice_then_returns(self, mock_data):
        """Sliced data's returns() must not include future data."""
        dates = mock_data.bars[mock_data.tickers[0]].index
        as_of = dates[200]
        sliced = mock_data.slice_until(as_of)
        ret = sliced.returns()
        assert ret.index.max() <= as_of

    def test_slice_at_first_date(self, tiny_data):
        """Edge case: slice at the very first date yields single row."""
        dates = tiny_data.bars["AAA"].index
        sliced = tiny_data.slice_until(dates[0])
        for t in sliced.tickers:
            assert len(sliced.bars[t]) == 1

    def test_slice_at_last_date(self, tiny_data):
        """Edge case: slice at last date yields all data."""
        dates = tiny_data.bars["AAA"].index
        sliced = tiny_data.slice_until(dates[-1])
        for t in sliced.tickers:
            assert len(sliced.bars[t]) == 5


# ============================================================================
# 6. REPRODUCIBILITY — same seed, same data
# ============================================================================

class TestReproducibility:
    """The same seed must produce identical results."""

    def test_same_seed_same_data(self):
        d1 = MockDataSource(seed=42).load()
        d2 = MockDataSource(seed=42).load()
        for t in d1.tickers:
            pd.testing.assert_frame_equal(d1.bars[t], d2.bars[t])

    def test_different_seed_different_data(self):
        d1 = MockDataSource(seed=1).load()
        d2 = MockDataSource(seed=2).load()
        # At least one value should differ
        for t in d1.tickers:
            assert not d1.bars[t]["close"].equals(d2.bars[t]["close"])


# ============================================================================
# 7. CUSTOM PARAMETERS
# ============================================================================

class TestCustomParams:
    """MockDataSource must respect user-provided params."""

    def test_custom_tickers(self):
        params = {
            "AAPL": (0.20, 0.30, 190.0),
            "MSFT": (0.18, 0.25, 420.0),
            "AMZN": (0.22, 0.35, 180.0),
        }
        data = MockDataSource(params=params).load()
        assert data.tickers == ["AAPL", "MSFT", "AMZN"]
        assert set(data.bars.keys()) == {"AAPL", "MSFT", "AMZN"}

    def test_custom_n_days(self):
        data = MockDataSource(n_days=50).load()
        for t in data.tickers:
            assert len(data.bars[t]) == 50

    def test_single_ticker(self):
        """Edge case: works with a single ticker (1×1 correlation matrix)."""
        params = {"SOLO": (0.10, 0.20, 100.0)}
        data = MockDataSource(params=params, corr=0.0).load()
        assert data.tickers == ["SOLO"]
        assert len(data.bars["SOLO"]) == 504


# ============================================================================
# 8. FROZEN DATACLASS — MarketData immutability
# ============================================================================

class TestFrozenContract:
    """MarketData is a frozen dataclass — attributes cannot be reassigned."""

    def test_cannot_reassign_tickers(self, mock_data):
        with pytest.raises(AttributeError):
            mock_data.tickers = ["X"]

    def test_cannot_reassign_bars(self, mock_data):
        with pytest.raises(AttributeError):
            mock_data.bars = {}


# ============================================================================
# 9. DOWNSTREAM INTEGRATION SMOKE TEST
# ============================================================================

class TestDownstreamSmoke:
    """Simulate exactly what backtest.py does with MarketData.

    This ensures data.py's output plugs into the pipeline without errors.
    """

    def test_backtest_pattern(self, mock_data):
        """Walk forward through dates the way Backtest.run() does."""
        closes = mock_data.close_prices()
        dates = closes.index
        lookback = 60
        horizon = 5
        rebalance = 5

        step_count = 0
        for i in range(lookback, len(dates) - horizon, rebalance):
            t = dates[i]
            t_fwd = dates[i + horizon]

            # slice_until — point-in-time view
            window = mock_data.slice_until(t)

            # returns for the window
            rets = window.returns()
            assert not rets.empty
            assert rets.index.max() <= t

            # forward return computation (what backtest.py does)
            fwd = {
                tk: float(closes[tk].loc[t_fwd] / closes[tk].loc[t] - 1)
                for tk in mock_data.tickers
            }
            for v in fwd.values():
                assert np.isfinite(v)

            step_count += 1

        assert step_count > 0, "Backtest loop should have executed at least once"

    def test_forecaster_input_pattern(self, mock_data):
        """Simulate what MomentumForecaster.predict() reads from data."""
        dates = mock_data.close_prices().index
        as_of = dates[100]
        window = mock_data.slice_until(as_of)

        rets = window.returns().loc[:as_of].tail(20)
        assert len(rets) == 20
        exp_daily = rets.mean()
        for t in mock_data.tickers:
            assert np.isfinite(exp_daily[t])

    def test_risk_input_pattern(self, mock_data):
        """Simulate what SampleCovRisk.estimate() reads from data."""
        dates = mock_data.close_prices().index
        as_of = dates[100]
        window = mock_data.slice_until(as_of)

        rets = window.returns().loc[:as_of].tail(60)
        cov = rets.cov() * 252
        assert cov.shape == (len(mock_data.tickers), len(mock_data.tickers))
        assert not cov.isna().any().any()
