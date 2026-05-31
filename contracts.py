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
    bars: dict[
        str, pd.DataFrame
    ]  # ticker -> DataFrame[open, high, low, close, volume], DatetimeIndex

    def close_prices(self) -> pd.DataFrame:
        return pd.DataFrame({t: self.bars[t]["close"] for t in self.tickers})

    def returns(self) -> pd.DataFrame:
        return self.close_prices().pct_change().dropna()

    def slice_until(self, as_of) -> "MarketData":
        """Point-in-time view: only data up to `as_of` (no look-ahead)."""
        return MarketData(
            self.tickers, {t: df.loc[:as_of] for t, df in self.bars.items()}
        )


@dataclass(frozen=True)
class NewsArticle:
    """A single news article associated with a ticker, including NLP sentiment."""

    ticker: str
    title: str
    snippet: str
    url: str
    published_date: pd.Timestamp
    sentiment_score: float  # [-1, +1]  +1 = bullish, -1 = bearish


@dataclass(frozen=True)
class NewsFeed:
    """OUTPUT of Layer 1 (Data, news path).  INPUT to Risk."""

    as_of: pd.Timestamp
    articles: list[NewsArticle]

    def sentiment_scores(self, lookback_days: int = 5) -> dict[str, float]:
        """Per-ticker average sentiment over recent lookback window."""
        cutoff = self.as_of - pd.Timedelta(days=lookback_days)
        groups: dict[str, list[float]] = {}
        for a in self.articles:
            if a.published_date >= cutoff:
                groups.setdefault(a.ticker, []).append(a.sentiment_score)
        return {t: sum(s) / len(s) for t, s in groups.items() if s}

    def article_count(self, ticker: str) -> int:
        """Number of articles for a ticker in this feed."""
        return sum(1 for a in self.articles if a.ticker == ticker)


@dataclass(frozen=True)
class Forecast:
    """OUTPUT of Layer 2 (Forecast).  INPUT to Pick & size."""

    as_of: pd.Timestamp
    horizon_days: int
    expected_returns: dict[str, float]  # ticker -> expected return over the horizon
    direction: dict[str, int] = field(
        default_factory=dict
    )  # ticker -> +1 (up) / -1 (down)
    confidence: dict[str, float] = field(default_factory=dict)  # ticker -> 0..1


@dataclass(frozen=True)
class ChaosSignal:
    """OUTPUT of ChaosEngine (Layer 2.5).  Optionally consumed by Forecast and Optimizer."""

    as_of: pd.Timestamp
    crash_probability: float          # [0, 1] — blended tail-event estimate
    event_label: str                  # "normal" / "elevated_risk" / "market_crash"
    confidence: float                 # [0, 1] — how far probability is from 0.5 × 2
    ticker_adjustments: dict[str, float]  # per-ticker weight multipliers
    reasoning: str                    # plain-English explanation


@dataclass(frozen=True)
class CrystalBallPrediction:
    """OUTPUT of CrystalBall.  1-year scenario forecast fusing Forecast and ChaosSignal."""

    as_of: pd.Timestamp
    horizon_days: int                           # 252 = ~1 trading year
    base_returns: dict[str, float]              # compounded central estimate
    bull_returns: dict[str, float]              # base + 1.5 × annual_vol
    bear_returns: dict[str, float]              # base − 1.5 × annual_vol
    crash_adjusted_returns: dict[str, float]    # base scaled by chaos ticker_adjustments
    annual_volatility: dict[str, float]         # per-ticker annualised vol (from factor model)
    crash_probability: float                    # from ChaosEngine
    dominant_factor_var: float                  # leading eigenvalue × 252 (market factor strength)
    confidence: dict[str, float]                # per-ticker, inherited from short-horizon Forecast
    reasoning: str                              # plain-English scenario summary


@dataclass(frozen=True)
class PerSubAgentRisk:
    """VaR/CVaR from a single risk sub-agent (GBM, Markov, bootstrap)."""

    agent_label: str  # "gbm", "markov", "bootstrap"
    var_95: dict[str, float]  # ticker -> 95% VaR (5-day horizon)
    cvar_95: dict[str, float]  # ticker -> 95% CVaR


@dataclass(frozen=True)
class RiskModel:
    """OUTPUT of Layer 3 (Risk).  INPUT to Pick & size."""

    as_of: pd.Timestamp
    cov: pd.DataFrame  # annualized covariance matrix, tickers x tickers
    vol: dict[str, float]  # ticker -> annualized volatility
    var_95: dict[str, float] = field(
        default_factory=dict
    )  # aggregated: median VaR across sub-agents
    cvar_95: dict[str, float] = field(
        default_factory=dict
    )  # aggregated: worst CVaR across sub-agents
    sub_agent_results: list[PerSubAgentRisk] = field(
        default_factory=list
    )  # per-agent breakdown
    disagreement: float = 0.0  # [0..1] model divergence

    def portfolio_vol(self, weights: dict[str, float]) -> float:
        w = pd.Series(weights).reindex(self.cov.index).fillna(0.0)
        return float((w @ self.cov @ w) ** 0.5)


@dataclass(frozen=True)
class TargetPortfolio:
    """OUTPUT of Layer 4 (Pick & size).  INPUT to Execute."""

    as_of: pd.Timestamp
    weights: dict[str, float]  # signed target weights; dollar-neutral => sum ~ 0

    @property
    def gross(self) -> float:
        return sum(abs(w) for w in self.weights.values())

    @property
    def net(self) -> float:
        return sum(self.weights.values())


@dataclass(frozen=True)
class Fill:
    ticker: str
    side: str  # "buy" / "sell"
    qty: float
    price: float


@dataclass(frozen=True)
class ExecutionResult:
    """OUTPUT of Layer 5 (Execute).  INPUT to Score."""

    as_of: pd.Timestamp
    fills: list[Fill]
    realized_weights: dict[
        str, float
    ]  # what we ACTUALLY hold after fills/rounding/slippage


@dataclass(frozen=True)
class SubAgentReport:
    """Per-sub-agent calibration report."""

    agent_label: str
    avg_var_95: float
    var_breach_rate: float  # fraction of periods where actual return < -VaR


@dataclass(frozen=True)
class RiskReport:
    """Risk model honesty diagnostics. Produced by Layer 6 alongside Scorecard."""

    var_breaches: int  # total periods where actual return < -VaR
    var_breach_rate: float  # breach_rate / expectation (0.05 for 95% VaR)
    avg_disagreement: float  # mean disagreement across rebalances
    max_disagreement: float  # peak disagreement
    sub_agent_reports: list[SubAgentReport]  # per-agent calibration


@dataclass(frozen=True)
class Scorecard:
    """OUTPUT of Layer 6 (Score).  The deliverable you show the judges."""

    sharpe: float
    total_return: float
    directional_accuracy: float
    information_coefficient: float
    vs_baseline_sharpe: float
    equity_curve: pd.Series


@dataclass(frozen=True)
class MarketIntelligence:
    """OUTPUT of MarketIntelligenceAgent.  INPUT to MasterAgent."""

    as_of: pd.Timestamp
    headlines: dict[str, list[str]]   # ticker -> recent headlines
    sentiment: dict[str, float]        # ticker -> score in [-1.0, 1.0]
    key_themes: list[str]              # top market-wide themes across all tickers
    urls: dict[str, list[str]]         # ticker -> source URLs


# ============================================================================
# LAYER INTERFACES  (each teammate owns exactly one of these)
# ============================================================================


@runtime_checkable
class DataSource(Protocol):
    def load(self) -> MarketData: ...


@runtime_checkable
class NewsSource(Protocol):
    def fetch(self, tickers: list[str], as_of: pd.Timestamp) -> NewsFeed: ...


@runtime_checkable
class Forecaster(Protocol):
    def predict(
        self, data: MarketData, as_of: pd.Timestamp, horizon_days: int
    ) -> Forecast: ...


@runtime_checkable
class RiskEstimator(Protocol):
    def estimate(
        self, data: MarketData, news: NewsFeed, forecast: Forecast, as_of: pd.Timestamp
    ) -> RiskModel: ...


@runtime_checkable
class Optimizer(Protocol):
    def solve(self, forecast: Forecast, risk: RiskModel) -> TargetPortfolio: ...


@runtime_checkable
class Executor(Protocol):
    def execute(
        self, target: TargetPortfolio, data: MarketData, as_of: pd.Timestamp
    ) -> ExecutionResult: ...