"""
backtest.py  —  THE WIRING (owner: ____ , usually whoever owns Score).

This is the only place that knows the order of the layers. It walks forward
through time and, at each rebalance, calls each layer through its interface.
NO layer logic lives here — swap any component and this file is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .contracts import (
    DataSource,
    Executor,
    Forecast,
    Forecaster,
    NewsFeed,
    NewsSource,
    Optimizer,
    RiskEstimator,
    RiskModel,
)


@dataclass
class StepRecord:
    as_of: pd.Timestamp
    weights: dict[str, float]
    forward_returns: dict[str, float]
    forecast: Forecast | None = None
    risk_model: RiskModel | None = None


class Backtest:
    def __init__(
        self,
        source: DataSource,
        forecaster: Forecaster,
        risk: RiskEstimator,
        optimizer: Optimizer,
        executor: Executor,
        news_source: NewsSource | None = None,
        rebalance_every: int = 5,
        lookback_days: int = 60,
        horizon_days: int = 5,
    ):
        self.source = source
        self.forecaster = forecaster
        self.risk = risk
        self.optimizer = optimizer
        self.executor = executor
        self.news_source = news_source
        self.rebalance_every = rebalance_every
        self.lookback_days = lookback_days
        self.horizon_days = horizon_days

    def run(self):
        data = self.source.load()
        closes = data.close_prices()
        dates = closes.index

        records: list[StepRecord] = []
        baseline: list[StepRecord] = []

        for i in range(
            self.lookback_days, len(dates) - self.horizon_days, self.rebalance_every
        ):
            t = dates[i]
            t_fwd = dates[i + self.horizon_days]
            window = data.slice_until(t)

            forecast = self.forecaster.predict(window, t, self.horizon_days)  # Layer 2
            news = (
                self.news_source.fetch(data.tickers, t)
                if self.news_source
                else NewsFeed(as_of=t, articles=[])
            )
            risk = self.risk.estimate(window, news, forecast, t)  # Layer 3
            target = self.optimizer.solve(forecast, risk)  # Layer 4
            execed = self.executor.execute(target, data, t)  # Layer 5

            fwd = {
                tk: float(closes[tk].loc[t_fwd] / closes[tk].loc[t] - 1)
                for tk in data.tickers
            }
            records.append(StepRecord(t, execed.realized_weights, fwd, forecast, risk))
            baseline.append(
                StepRecord(t, {tk: 0.5 for tk in data.tickers}, fwd, None, None)
            )  # 50/50 hold

        return data, records, baseline
