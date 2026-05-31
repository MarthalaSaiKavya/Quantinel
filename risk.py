"""
LAYER 3 · RISK   (owner: ____)

SampleCovRisk: annualized sample covariance + multi-agent VaR/CVaR ensemble.

Sub-agents:
  - GBM: Geometric Brownian Motion (drift from forecast, vol from history)
  - Markov: 2-regime bull/bear switching (sentiment-adjusted transitions)
  - Bootstrap: block resampling of historical returns

Each sub-agent simulates 5-day return paths. Ensemble aggregates to median VaR
and worst CVaR, plus a disagreement score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from contracts import Forecast, MarketData, NewsFeed, PerSubAgentRisk, RiskModel


class SampleCovRisk:
    """Implements RiskEstimator: estimate(data, news, forecast, as_of) -> RiskModel."""

    def __init__(
        self, lookback: int = 60, n_paths: int = 10_000, horizon_days: int = 5
    ):
        self.lookback = lookback
        self.n_paths = n_paths
        self.horizon_days = horizon_days

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def estimate(
        self, data: MarketData, news: NewsFeed, forecast: Forecast, as_of
    ) -> RiskModel:
        rets = data.returns().loc[:as_of].tail(self.lookback)
        cov = rets.cov() * 252
        vol = {t: float(np.sqrt(cov.loc[t, t])) for t in data.tickers}

        # Run sub-agents
        results: list[PerSubAgentRisk] = [
            self._run_gbm(data.tickers, forecast, vol),
            self._run_markov(data.tickers, forecast, rets, news),
            self._run_bootstrap(data.tickers, rets),
        ]

        # Aggregate
        var_95, cvar_95 = self._aggregate(results)
        disagreement = self._disagreement(results)

        return RiskModel(
            as_of=pd.Timestamp(as_of),
            cov=cov,
            vol=vol,
            var_95=var_95,
            cvar_95=cvar_95,
            sub_agent_results=results,
            disagreement=disagreement,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(results: list[PerSubAgentRisk]):
        """Median VaR, worst CVaR across sub-agents."""
        tickers = list(results[0].var_95)
        var_95 = {}
        cvar_95 = {}
        for t in tickers:
            vars_t = [r.var_95[t] for r in results]
            cvars_t = [r.cvar_95[t] for r in results]
            var_95[t] = float(np.median(vars_t))
            cvar_95[t] = float(min(cvars_t))  # worst = most negative
        return var_95, cvar_95

    @staticmethod
    def _disagreement(results: list[PerSubAgentRisk]) -> float:
        """Mean relative deviation from median across agents and tickers."""
        tickers = list(results[0].var_95)
        rel_diffs = []
        for t in tickers:
            vals = [r.var_95[t] for r in results]
            med = np.median(vals)
            if abs(med) > 1e-12:
                rel_diffs.extend(abs(v - med) / abs(med) for v in vals)
        return float(np.mean(rel_diffs)) if rel_diffs else 0.0

    # ------------------------------------------------------------------
    # Sub-agent: Geometric Brownian Motion
    # ------------------------------------------------------------------

    def _run_gbm(
        self, tickers: list[str], forecast: Forecast, vol: dict[str, float]
    ) -> PerSubAgentRisk:
        """Simulate horizon-day return paths via GBM.

        Drift = forecast.expected_returns (scaled to daily).
        Diffusion = historical annualized vol (scaled to daily).
        """
        rng = np.random.default_rng(42)  # deterministic seed
        dt = 1 / 252
        horizon = self.horizon_days

        var_95: dict[str, float] = {}
        cvar_95: dict[str, float] = {}

        for t in tickers:
            mu_daily = forecast.expected_returns[t] / horizon
            sigma_daily = vol[t] / np.sqrt(252)

            # Simulate: N paths x H days
            Z = rng.standard_normal((self.n_paths, horizon))
            daily_ret = mu_daily * dt + sigma_daily * np.sqrt(dt) * Z
            cum_ret = np.cumprod(1 + daily_ret, axis=1)[:, -1] - 1

            var_95[t] = float(np.percentile(cum_ret, 5))
            tail = cum_ret[cum_ret <= var_95[t]]
            cvar_95[t] = float(tail.mean()) if len(tail) > 0 else var_95[t]

        return PerSubAgentRisk(
            agent_label="gbm",
            var_95=var_95,
            cvar_95=cvar_95,
        )

    # ------------------------------------------------------------------
    # Sub-agent: Markov regime-switching (bull/bear)
    # ------------------------------------------------------------------

    def _run_markov(
        self, tickers: list[str], forecast: Forecast, rets: pd.DataFrame, news: NewsFeed
    ) -> PerSubAgentRisk:
        """Simulate via 2-regime Markov switching.

        Base transition matrix from K-means clustering of daily returns.
        News sentiment adjusts P(bear|bull) and P(bull|bear).
        """
        rng = np.random.default_rng(99)
        horizon = self.horizon_days
        sentiment = news.sentiment_scores()

        var_95: dict[str, float] = {}
        cvar_95: dict[str, float] = {}

        for t in tickers:
            r = rets[t].dropna().values
            if len(r) < 10:
                var_95[t] = 0.0
                cvar_95[t] = 0.0
                continue

            # K-means: 2 clusters → bull (higher mean) / bear (lower mean)
            from sklearn.cluster import KMeans  # noqa: PLC0415

            X = r.reshape(-1, 1)
            km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
            means = km.cluster_centers_.flatten()
            bull_idx = int(np.argmax(means))
            bear_idx = 1 - bull_idx

            # Base transition: count empirical transitions
            labels = km.labels_
            trans = np.zeros((2, 2))
            for i in range(len(labels) - 1):
                trans[labels[i], labels[i + 1]] += 1
            trans = trans / trans.sum(axis=1, keepdims=True)

            # Sentiment adjustment: negative sentiment → stickier bear, less sticky bull
            s = sentiment.get(t, 0.0)
            bias = np.clip(s, -0.5, 0.5) * 0.15
            P = trans.copy()
            P[bear_idx, bear_idx] -= bias  # negative s → -bias positive → stickier bear
            P[bull_idx, bull_idx] += (
                bias  # negative s → +bias negative → less sticky bull
            )
            P = np.clip(P, 0.01, 0.99)
            P = P / P.sum(axis=1, keepdims=True)

            # Stationary distribution
            eigvals, eigvecs = np.linalg.eig(P.T)
            stationary = np.real(eigvecs[:, np.argmax(np.real(eigvals))])
            stationary = stationary / stationary.sum()

            # Simulate paths (daily returns, summed to horizon)
            mu_regime = np.array([means[bear_idx], means[bull_idx]])
            sigma_daily = float(r.std())

            paths = np.zeros((self.n_paths, horizon))
            for p in range(self.n_paths):
                state = int(rng.choice([0, 1], p=stationary))
                for h in range(horizon):
                    paths[p, h] = rng.normal(mu_regime[state], sigma_daily)
                    state = int(rng.choice([0, 1], p=P[state]))

            cum_ret = paths.sum(axis=1)  # sum of daily returns = horizon return
            var_95[t] = float(np.percentile(cum_ret, 5))
            tail = cum_ret[cum_ret <= var_95[t]]
            cvar_95[t] = float(tail.mean()) if len(tail) > 0 else var_95[t]

        return PerSubAgentRisk(
            agent_label="markov",
            var_95=var_95,
            cvar_95=cvar_95,
        )

    # ------------------------------------------------------------------
    # Sub-agent: Block bootstrap
    # ------------------------------------------------------------------

    def _run_bootstrap(self, tickers: list[str], rets: pd.DataFrame) -> PerSubAgentRisk:
        """Non-parametric block bootstrap of historical returns."""
        rng = np.random.default_rng(777)
        horizon = self.horizon_days
        block_len = max(3, horizon // 2)

        var_95: dict[str, float] = {}
        cvar_95: dict[str, float] = {}

        for t in tickers:
            r = rets[t].dropna().values
            daily_ret = np.array(r)
            n = len(daily_ret)
            if n < block_len:
                var_95[t] = 0.0
                cvar_95[t] = 0.0
                continue

            n_blocks = n - block_len + 1
            paths = np.zeros(self.n_paths)
            for p in range(self.n_paths):
                cum = 0.0
                steps = 0
                while steps < horizon:
                    start = rng.integers(0, n_blocks)
                    blk = daily_ret[start : start + block_len]
                    cum += blk.sum()
                    steps += len(blk)
                paths[p] = cum

            var_95[t] = float(np.percentile(paths, 5))
            tail = paths[paths <= var_95[t]]
            cvar_95[t] = float(tail.mean()) if len(tail) > 0 else var_95[t]

        return PerSubAgentRisk(
            agent_label="bootstrap",
            var_95=var_95,
            cvar_95=cvar_95,
        )
