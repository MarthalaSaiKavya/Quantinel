"""
contracts.py  —  THE SHARED LANGUAGE BETWEEN LAYERS.

Nobody touches this file alone. It defines:
  1. The DATA CONTRACTS  — the typed objects that travel between layers.
  2. The LAYER INTERFACES — the Protocol each teammate implements.

As long as your layer consumes the right input contract and returns the right
output contract, the rest of the team does not care what is inside it
(momentum, LSTM, Chronos, QSVM, Markowitz, QAOA — all interchangeable).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd


# ============================================================================
# DATA CONTRACTS  (the messages passed between layers)
# ============================================================================

@dataclass(frozen=True)
class MarketData:
    """OUTPUT of Layer 1 (Data).  INPUT to Forecast & Risk."""
    tickers: list[str]
    bars: dict[str, pd.DataFrame]   # ticker -> DataFrame[open, high, low, close, volume], DatetimeIndex

    def close_prices(self) -> pd.DataFrame:
        return pd.DataFrame({t: self.bars[t]["close"] for t in self.tickers})

    def returns(self) -> pd.DataFrame:
        return self.close_prices().pct_change().dropna()

    def slice_until(self, as_of) -> "MarketData":
        """Point-in-time view: only data up to `as_of` (no look-ahead)."""
        return MarketData(self.tickers, {t: df.loc[:as_of] for t, df in self.bars.items()})


@dataclass(frozen=True)
class Forecast:
    """OUTPUT of Layer 2 (Forecast).  INPUT to Pick & size."""
    as_of: pd.Timestamp
    horizon_days: int
    expected_returns: dict[str, float]                       # ticker -> expected return over the horizon
    direction: dict[str, int] = field(default_factory=dict)   # ticker -> +1 (up) / -1 (down)
    confidence: dict[str, float] = field(default_factory=dict)  # ticker -> 0..1


@dataclass(frozen=True)
class RiskModel:
    """OUTPUT of Layer 3 (Risk).  INPUT to Pick & size."""
    as_of: pd.Timestamp
    cov: pd.DataFrame              # annualized covariance matrix, tickers x tickers
    vol: dict[str, float]          # ticker -> annualized volatility

    def portfolio_vol(self, weights: dict[str, float]) -> float:
        w = pd.Series(weights).reindex(self.cov.index).fillna(0.0)
        return float((w @ self.cov @ w) ** 0.5)


@dataclass(frozen=True)
class TargetPortfolio:
    """OUTPUT of Layer 4 (Pick & size).  INPUT to Execute."""
    as_of: pd.Timestamp
    weights: dict[str, float]      # signed target weights; dollar-neutral => sum ~ 0

    @property
    def gross(self) -> float:
        return sum(abs(w) for w in self.weights.values())

    @property
    def net(self) -> float:
        return sum(self.weights.values())


@dataclass(frozen=True)
class Fill:
    ticker: str
    side: str          # "buy" / "sell"
    qty: float
    price: float


@dataclass(frozen=True)
class ExecutionResult:
    """OUTPUT of Layer 5 (Execute).  INPUT to Score."""
    as_of: pd.Timestamp
    fills: list[Fill]
    realized_weights: dict[str, float]   # what we ACTUALLY hold after fills/rounding/slippage


@dataclass(frozen=True)
class Scorecard:
    """OUTPUT of Layer 6 (Score).  The deliverable you show the judges."""
    sharpe: float
    total_return: float
    directional_accuracy: float
    information_coefficient: float
    vs_baseline_sharpe: float
    equity_curve: pd.Series


# ============================================================================
# LAYER INTERFACES  (each teammate owns exactly one of these)
# ============================================================================

@runtime_checkable
class DataSource(Protocol):
    def load(self) -> MarketData: ...


@runtime_checkable
class Forecaster(Protocol):
    def predict(self, data: MarketData, as_of: pd.Timestamp, horizon_days: int) -> Forecast: ...


@runtime_checkable
class RiskEstimator(Protocol):
    def estimate(self, data: MarketData, as_of: pd.Timestamp) -> RiskModel: ...


@runtime_checkable
class Optimizer(Protocol):
    def solve(self, forecast: Forecast, risk: RiskModel) -> TargetPortfolio: ...


@runtime_checkable
class Executor(Protocol):
    def execute(self, target: TargetPortfolio, data: MarketData, as_of: pd.Timestamp) -> ExecutionResult: ...
