"""
LAYER 3 · RISK   (owner: ____)

SampleCovRisk: annualized sample covariance + per-asset vol from a rolling window.
This is what tells the optimizer that NVDA and GOOG move together, so a long/short
pair cancels most of the shared "tech/AI" swing.

Swap ideas (same interface): EWMA covariance, Ledoit-Wolf shrinkage, GARCH vol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from contracts import MarketData, RiskModel


class SampleCovRisk:
    """Implements RiskEstimator: estimate(data, as_of) -> RiskModel."""

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def estimate(self, data: MarketData, as_of) -> RiskModel:
        rets = data.returns().loc[:as_of].tail(self.lookback)
        cov = rets.cov() * 252                              # annualize
        vol = {t: float(np.sqrt(cov.loc[t, t])) for t in data.tickers}
        return RiskModel(as_of=pd.Timestamp(as_of), cov=cov, vol=vol)
