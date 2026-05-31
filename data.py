"""
LAYER 1 · DATA   (owner: Rahul)

MockDataSource generates synthetic-but-realistic, *correlated* daily OHLCV for
NVDA and GOOG, so the rest of the pipeline can be built and tested before any
real API (Alpaca/Polygon) is wired in.

How it works
------------
1. Build a Cholesky-correlated pair of daily log-return streams
   (annual drift & vol per ticker, configurable cross-correlation).
2. Integrate returns into a close-price path via ``cumprod``.
3. Derive open / high / low / volume from the close with small
   random perturbations so every bar looks plausible on a chart.

To go live later: write an ``AlpacaDataSource`` with the same ``load()``
method that returns a ``MarketData``. Nothing downstream changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from contracts import MarketData


# ============================================================================
# PRIMARY DATA SOURCE — synthetic correlated OHLCV
# ============================================================================

class MockDataSource:
    """Implements DataSource:  load() -> MarketData.

    Parameters
    ----------
    n_days : int
        Number of business days of history to generate (default ~2 years).
    seed : int
        Random seed for full reproducibility.
    corr : float
        Pairwise return correlation between the two tickers (0–1).
    params : dict | None
        Per-ticker generation parameters::

            {ticker: (annual_drift, annual_vol, start_price), ...}

        Defaults to NVDA (high drift/vol) and GOOG (moderate drift/vol).
    """

    def __init__(
        self,
        n_days: int = 504,
        seed: int = 7,
        corr: float = 0.6,
        params: dict | None = None,
    ):
        self.n_days = n_days
        self.seed = seed
        self.corr = corr
        # (annual drift, annual vol, start price)
        self.params = params or {
            "NVDA": (0.35, 0.45, 480.0),
            "GOOG": (0.15, 0.28, 140.0),
        }

    def load(self) -> MarketData:
        """Generate synthetic OHLCV bars and return a ``MarketData`` contract.

        Returns
        -------
        MarketData
            Frozen dataclass with ``tickers``, ``bars``, and helpers
            ``close_prices()``, ``returns()``, ``slice_until(as_of)``.
        """
        rng = np.random.default_rng(self.seed)
        tickers = list(self.params)

        # --- date axis --------------------------------------------------
        dates = pd.bdate_range(
            end=pd.Timestamp.today().normalize(), periods=self.n_days
        )
        n = len(dates)  # source of truth for array length

        # --- correlated daily returns ------------------------------------
        n_assets = len(tickers)
        corr_matrix = np.full((n_assets, n_assets), self.corr)
        np.fill_diagonal(corr_matrix, 1.0)
        L = np.linalg.cholesky(corr_matrix)  # lower-triangular factor

        mu = np.array([self.params[t][0] for t in tickers]) / 252
        sig = np.array([self.params[t][1] for t in tickers]) / np.sqrt(252)

        z = rng.standard_normal((n, n_assets))
        daily_returns = mu + sig * (z @ L.T)  # correlated daily returns

        # --- build OHLCV per ticker -------------------------------------
        bars: dict[str, pd.DataFrame] = {}
        for i, t in enumerate(tickers):
            close = self.params[t][2] * np.cumprod(1 + daily_returns[:, i])

            # open ≈ previous close with tiny overnight gap noise
            open_ = np.concatenate([[close[0]], close[:-1]]) * (
                1 + rng.normal(0, 0.001, n)
            )
            # high >= max(open, close), low <= min(open, close)
            high = np.maximum(open_, close) * (
                1 + np.abs(rng.normal(0, 0.004, n))
            )
            low = np.minimum(open_, close) * (
                1 - np.abs(rng.normal(0, 0.004, n))
            )
            volume = rng.integers(2_000_000, 8_000_000, n).astype(float)

            bars[t] = pd.DataFrame(
                {
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                },
                index=dates,
            )

        return MarketData(tickers=tickers, bars=bars)


# ============================================================================
# LIVE DATA SOURCE — real OHLCV via yfinance
# ============================================================================

class YFinanceDataSource:
    """Implements DataSource: load() -> MarketData.

    Fetches real daily OHLCV from Yahoo Finance for the given tickers and
    date range. Drop-in replacement for MockDataSource — same contract, no
    downstream changes needed.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols, e.g. ["NVDA", "GOOG"].
    start : str
        Start date in "YYYY-MM-DD" format.
    end : str | None
        End date (exclusive). Defaults to today.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        start: str = "2023-01-01",
        end: str | None = None,
    ):
        self.tickers = tickers or ["NVDA", "GOOG"]
        self.start = start
        self.end = end or pd.Timestamp.today().strftime("%Y-%m-%d")

    def load(self) -> MarketData:
        import yfinance as yf

        raw = yf.download(
            self.tickers,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False,
        )

        # yfinance returns a MultiIndex frame when multiple tickers are given;
        # a flat frame when only one ticker is given. Normalise to flat per ticker.
        if isinstance(raw.columns, pd.MultiIndex):
            frames = {t: raw.xs(t, axis=1, level=1).dropna() for t in self.tickers}
        else:
            frames = {self.tickers[0]: raw.dropna()}

        bars: dict[str, pd.DataFrame] = {}
        for ticker, df in frames.items():
            df = df.rename(columns=str.lower)
            bars[ticker] = df[["open", "high", "low", "close", "volume"]]

        return MarketData(tickers=self.tickers, bars=bars)


# ============================================================================
# TINY STATIC DATA SOURCE — deterministic 5-row dataset for unit tests
# ============================================================================

class TinyMockDataSource:
    """A hard-coded, deterministic 5-day dataset for fast unit tests.

    No randomness, no Cholesky — just two tickers with hand-picked OHLCV
    so that expected values (close_prices, returns, slice_until) can be
    asserted by hand.

    Also provides ``load_news()`` for mock news/world-event data that
    downstream layers (e.g. sentiment-aware forecasters) can consume.
    """

    def load(self) -> MarketData:
        dates = pd.bdate_range("2025-01-06", periods=5, freq="B")
        bars: dict[str, pd.DataFrame] = {
            "AAA": pd.DataFrame(
                {
                    "open":   [100.0, 102.0, 101.0, 105.0, 103.0],
                    "high":   [103.0, 104.0, 106.0, 107.0, 108.0],
                    "low":    [ 99.0, 100.0,  99.0, 103.0, 101.0],
                    "close":  [102.0, 101.0, 105.0, 103.0, 107.0],
                    "volume": [1e6,   1.2e6, 0.8e6, 1.5e6, 1.1e6],
                },
                index=dates,
            ),
            "BBB": pd.DataFrame(
                {
                    "open":   [50.0, 51.0, 52.0, 50.0, 53.0],
                    "high":   [52.0, 53.0, 54.0, 53.0, 55.0],
                    "low":    [49.0, 50.0, 50.0, 49.0, 51.0],
                    "close":  [51.0, 52.0, 50.0, 53.0, 54.0],
                    "volume": [5e5,  6e5,  4e5,  7e5,  5.5e5],
                },
                index=dates,
            ),
        }
        return MarketData(tickers=["AAA", "BBB"], bars=bars)

    @staticmethod
    def load_news() -> list[dict]:
        """Return deterministic mock news / world-event data.

        Each item is a JSON-serializable dict with:
            - ``date``       : ISO-8601 date string (aligned with OHLCV dates)
            - ``ticker``     : affected ticker, or ``"MACRO"`` for broad events
            - ``headline``   : human-readable headline
            - ``source``     : news outlet name
            - ``sentiment``  : float in [-1.0, +1.0]  (neg = bearish, pos = bullish)
            - ``category``   : ``"earnings"`` | ``"macro"`` | ``"product"`` | ``"analyst"`` | ``"geopolitical"``
            - ``relevance``  : float in [0.0, 1.0]  (how relevant to the ticker)

        Headlines are chosen to logically match the hand-picked price
        movements so tests can assert sentiment ↔ return correlation.

        Returns
        -------
        list[dict]
            A list of 10 news items spanning the 5 trading days.
        """
        return [
            # ── Day 1  (2025-01-06) ── AAA +2%, BBB +2%  → bullish news
            {
                "date": "2025-01-06",
                "ticker": "AAA",
                "headline": "AAA Corp beats Q4 earnings estimates by 12%",
                "source": "Reuters",
                "sentiment": 0.85,
                "category": "earnings",
                "relevance": 0.95,
            },
            {
                "date": "2025-01-06",
                "ticker": "MACRO",
                "headline": "Fed signals potential rate cuts in H1 2025",
                "source": "Bloomberg",
                "sentiment": 0.60,
                "category": "macro",
                "relevance": 0.70,
            },
            # ── Day 2  (2025-01-07) ── AAA −1%, BBB +2%  → mixed
            {
                "date": "2025-01-07",
                "ticker": "AAA",
                "headline": "Analysts downgrade AAA citing valuation concerns",
                "source": "CNBC",
                "sentiment": -0.45,
                "category": "analyst",
                "relevance": 0.80,
            },
            {
                "date": "2025-01-07",
                "ticker": "BBB",
                "headline": "BBB Inc announces new product line, shares rise",
                "source": "MarketWatch",
                "sentiment": 0.70,
                "category": "product",
                "relevance": 0.90,
            },
            # ── Day 3  (2025-01-08) ── AAA +4%, BBB −3.8%  → divergence
            {
                "date": "2025-01-08",
                "ticker": "AAA",
                "headline": "AAA secures $2B government contract for AI infrastructure",
                "source": "WSJ",
                "sentiment": 0.90,
                "category": "product",
                "relevance": 0.95,
            },
            {
                "date": "2025-01-08",
                "ticker": "BBB",
                "headline": "BBB faces supply chain disruptions in Asia-Pacific",
                "source": "Financial Times",
                "sentiment": -0.65,
                "category": "geopolitical",
                "relevance": 0.85,
            },
            # ── Day 4  (2025-01-09) ── AAA −1.9%, BBB +6%  → reversal
            {
                "date": "2025-01-09",
                "ticker": "AAA",
                "headline": "AAA CFO sells $15M in insider shares",
                "source": "SEC Filing",
                "sentiment": -0.40,
                "category": "earnings",
                "relevance": 0.75,
            },
            {
                "date": "2025-01-09",
                "ticker": "BBB",
                "headline": "BBB resolves supply issues; analyst upgrades to Buy",
                "source": "Goldman Sachs",
                "sentiment": 0.80,
                "category": "analyst",
                "relevance": 0.90,
            },
            # ── Day 5  (2025-01-10) ── AAA +3.9%, BBB +1.9%  → broad rally
            {
                "date": "2025-01-10",
                "ticker": "MACRO",
                "headline": "US jobs report beats expectations; markets rally broadly",
                "source": "Bloomberg",
                "sentiment": 0.75,
                "category": "macro",
                "relevance": 0.80,
            },
            {
                "date": "2025-01-10",
                "ticker": "AAA",
                "headline": "AAA launches next-gen chip platform at CES 2025",
                "source": "The Verge",
                "sentiment": 0.65,
                "category": "product",
                "relevance": 0.85,
            },
        ]
