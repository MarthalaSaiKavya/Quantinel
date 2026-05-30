"""
LAYER 1 · DATA   (owner: ____)

MockDataSource generates synthetic-but-realistic, *correlated* daily OHLCV for
NVDA and GOOG, so the rest of the pipeline can be built and tested before any
real API (Alpaca/Polygon) is wired in.

To go live later: write an `AlpacaDataSource` with the same `load()` method that
returns a MarketData. Nothing downstream changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from contracts import MarketData


class MockDataSource:
    """Implements DataSource:  load() -> MarketData."""

    def __init__(self, n_days: int = 504, seed: int = 7, corr: float = 0.6, params: dict | None = None):
        self.n_days = n_days
        self.seed = seed
        self.corr = corr
        # (annual drift, annual vol, start price)
        self.params = params or {
            "NVDA": (0.35, 0.45, 480.0),
            "GOOG": (0.15, 0.28, 140.0),
        }

    def load(self) -> MarketData:
        rng = np.random.default_rng(self.seed)
        tickers = list(self.params)

        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=self.n_days)
        n = len(dates)                                      # source of truth for length

        corr = np.array([[1.0, self.corr], [self.corr, 1.0]])
        L = np.linalg.cholesky(corr)                       # correlate the two return streams
        mu = np.array([self.params[t][0] for t in tickers]) / 252
        sig = np.array([self.params[t][1] for t in tickers]) / np.sqrt(252)

        z = rng.standard_normal((n, len(tickers)))
        daily = mu + sig * (z @ L.T)                        # correlated daily returns

        bars: dict[str, pd.DataFrame] = {}
        for i, t in enumerate(tickers):
            close = self.params[t][2] * np.cumprod(1 + daily[:, i])
            open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.001, n))
            high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
            low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
            volume = rng.integers(2_000_000, 8_000_000, n).astype(float)
            bars[t] = pd.DataFrame(
                {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                index=dates,
            )
        return MarketData(tickers=tickers, bars=bars)
