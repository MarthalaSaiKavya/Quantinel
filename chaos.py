"""
LAYER 2.5 — CHAOS ENGINE  (owner: GT (10780))

Wildcard / tail-event detector that fuses quantum-encoded market signals
with world-news sentiment to estimate the probability of an adverse
"black-swan" event (crash, liquidity crisis, sector collapse, etc.).

Pipeline integration example (run_baseline.py)::

    from chaos import ChaosEngine
    from news import MockNewsSource, ExaNewsSource

    engine = ChaosEngine()                                        # XPYQ_KEY read from env
    news   = MockNewsSource().fetch(tickers, as_of=as_of)         # swap for ExaNewsSource
    signal = engine.evaluate(data, news, as_of=as_of)
    print(signal.reasoning)                          # plain-English recommendation

    forecast  = engine.adjust_forecast(forecast, signal)   # dampen / flip forecasts
    portfolio = engine.adjust_portfolio(portfolio, signal)  # scale / short positions

XpyQ backend
------------
Set XPYQ_KEY in the environment (or pass api_key= to the constructor).
The engine submits crash-cluster covariance matrices to xpyq's linalg.eig
endpoint; eigenvectors define the principal crash directions used to score
the current market feature vector.
Falls back to a classical centroid-distance estimate when no key is set or
the API is unreachable.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from contracts import ChaosSignal, Forecast, MarketData, NewsFeed, TargetPortfolio

_XPYQ_BASE = "https://xpyq-lib-production.up.railway.app"


# ============================================================================
# CHAOS ENGINE
# ============================================================================


class ChaosEngine:
    """
    Wildcard event predictor — quantum-powered tail-risk detector.

    Workflow
    --------
    1. Build market features from ``MarketData`` (vol regime, momentum,
       drawdown, volatility spike ratio).
    2. Label historical windows:
           label = 1  if worst return over next ``crash_horizon`` days
                      falls below ``crash_threshold``  (i.e. a crash occurred)
           label = 0  otherwise
    3. Submit the crash-cluster covariance matrix to xpyq's linalg.eig
       endpoint. Eigenvectors define the principal crash directions.
    4. Project the current feature vector and both cluster centroids into
       eigen-space; convert the distance ratio to P(crash).
    5. Adjust P(crash) up/down based on live news sentiment (Bayesian boost).
    6. Return a ``ChaosSignal`` with:
           crash_probability    — final blended estimate
           ticker_adjustments   — weight multipliers for adjust_portfolio()
           reasoning            — plain-English recommendation string

    XpyQ backend
    ------------
    Reads XPYQ_KEY from the environment (or pass api_key= to the constructor).
    Falls back to a classical centroid-distance estimate when no key is set or
    the API is unreachable.
    """

    # Severity thresholds for action labels and position adjustments
    _HIGH: float = 0.65
    _MODERATE: float = 0.40

    def __init__(
        self,
        lookback: int = 120,
        crash_threshold: float = -0.04,   # -4 % in crash_horizon days = "crash"
        crash_horizon: int = 5,
        api_key: str | None = None,
        poll_secs: float = 0.4,
        timeout: float = 20.0,
    ) -> None:
        self.lookback = lookback
        self.crash_threshold = crash_threshold
        self.crash_horizon = crash_horizon
        self.api_key = api_key or os.getenv("XPYQ_KEY", "")
        self.poll_secs = poll_secs
        self.timeout = timeout
        self._disabled = not bool(self.api_key)

    # ------------------------------------------------------------------
    # xpyq helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _run_code(self, code: str, name: str = "chaos") -> dict:
        """Submit code to xpyq and block until a terminal status is reached."""
        import requests

        if self._disabled:
            return {"status": "disabled", "stdout": ""}

        h = self._headers()
        run = requests.post(
            f"{_XPYQ_BASE}/api/v1/compute/runs",
            headers=h,
            json={"code": code, "name": name},
            timeout=10,
        ).json()
        run_id = run.get("run_id") or run.get("id")
        if not run_id:
            self._disabled = True
            return {"status": "failed", "stdout": ""}

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            r = requests.get(
                f"{_XPYQ_BASE}/api/v1/compute/runs/{run_id}",
                headers=h,
                timeout=10,
            ).json()
            if r["status"] in ("completed", "failed", "timed_out", "cancelled"):
                return r
            time.sleep(self.poll_secs)
        return {"status": "timed_out", "stdout": ""}

    @staticmethod
    def _parse_json_stdout(stdout: str) -> dict:
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        raise ValueError("xpyq stdout did not contain a JSON object")

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _market_features(self, returns: pd.DataFrame, as_of) -> pd.DataFrame:
        """
        Build a daily feature DataFrame up to ``as_of`` from portfolio returns.

        Columns
        -------
        vol_regime   : 10d rolling std / 30d rolling std  (> 1 = elevated vol)
        momentum_5   : 5-day cumulative portfolio return
        drawdown     : current close / rolling 60d max close − 1
        vol_ratio    : 5d std / 20d std  (sudden spike detector)
        """
        port = returns.loc[:as_of].mean(axis=1).tail(self.lookback + 65)

        vol_10  = port.rolling(10).std()
        vol_30  = port.rolling(30).std()
        vol_5   = port.rolling(5).std()
        vol_20  = port.rolling(20).std()
        mom_5   = port.rolling(5).sum()

        prices  = (1 + port).cumprod()
        peak_60 = prices.rolling(60, min_periods=1).max()
        dd      = prices / peak_60 - 1.0

        feat = pd.DataFrame(
            {
                "vol_regime": (vol_10 / vol_30.replace(0, np.nan)).fillna(1.0),
                "momentum_5": mom_5,
                "drawdown":   dd,
                "vol_ratio":  (vol_5 / vol_20.replace(0, np.nan)).fillna(1.0),
            }
        ).dropna()

        return feat.loc[:as_of]

    def _make_dataset(
        self, feat: pd.DataFrame, returns: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align feature rows with forward crash labels.

        A label of 1 means the portfolio's minimum 1-day return over the
        next ``crash_horizon`` days fell below ``crash_threshold``.
        """
        port = returns.mean(axis=1)
        X, y = [], []
        dates = feat.index[: -self.crash_horizon]   # leave room for forward look

        for date in dates:
            iloc = port.index.get_loc(date)
            if iloc + self.crash_horizon >= len(port):
                continue
            fwd = port.iloc[iloc + 1 : iloc + 1 + self.crash_horizon]
            cum_ret = float((1 + fwd).prod() - 1)
            X.append(feat.loc[date].to_numpy(dtype=float))
            y.append(1 if cum_ret < self.crash_threshold else 0)

        return np.array(X, dtype=float), np.array(y, dtype=int)

    # ------------------------------------------------------------------
    # XpyQ eigen-classification
    # ------------------------------------------------------------------

    def _xpyq_classify(self, X: np.ndarray, y: np.ndarray, feat_now: np.ndarray) -> float:
        """
        Estimate P(crash) via xpyq eigendecomposition of the crash cluster.

        Steps
        -----
        1. Normalise all features to [0, 1].
        2. Compute the covariance matrix of crash-labelled samples.
        3. Send it to xpyq linalg.eig to get principal crash directions.
        4. Project the current feature vector and both cluster centroids into
           eigen-space; return  dist_normal / (dist_crash + dist_normal).
           A point close to the crash centroid yields a high probability.
        Falls back to ``_classical_classify`` if xpyq is unavailable.
        """
        X_crash  = X[y == 1]
        X_normal = X[y == 0]

        if len(X_crash) < 2:
            return self._classical_classify(X, y, feat_now)

        # Normalise to [0, 1]
        X_min   = X.min(axis=0)
        X_range = X.max(axis=0) - X_min
        X_range[X_range == 0] = 1.0
        X_crash_s  = (X_crash  - X_min) / X_range
        X_normal_s = (X_normal - X_min) / X_range
        feat_s     = (feat_now - X_min) / X_range

        crash_centroid  = X_crash_s.mean(axis=0)
        normal_centroid = X_normal_s.mean(axis=0)
        cov_crash = np.cov(X_crash_s.T).tolist()

        code = f"""
import numpy as _np, json

cov  = from_numpy(_np.array({cov_crash}, dtype=_np.float32))
feat = _np.array({feat_s.tolist()}, dtype=_np.float32)
cc   = _np.array({crash_centroid.tolist()}, dtype=_np.float32)
nc   = _np.array({normal_centroid.tolist()}, dtype=_np.float32)

eigvals_mat, eigvecs_mat = linalg.eig(cov)
eigvals_arr, eigvecs_arr = eigvals_mat.numpy()

delta_crash  = feat - cc
delta_normal = feat - nc
proj_crash   = eigvecs_arr.T @ delta_crash
proj_normal  = eigvecs_arr.T @ delta_normal
dist_crash   = float(_np.linalg.norm(proj_crash))
dist_normal  = float(_np.linalg.norm(proj_normal))

print(json.dumps({{
    "dist_crash":  dist_crash,
    "dist_normal": dist_normal,
}}))
"""
        try:
            result = self._run_code(code, name="chaos_eig")
            if result["status"] != "completed" or not result.get("stdout", "").strip():
                if result["status"] in ("failed", "timed_out", "cancelled"):
                    self._disabled = True
                return self._classical_classify(X, y, feat_now)
            out = self._parse_json_stdout(result["stdout"])
        except Exception:
            self._disabled = True
            return self._classical_classify(X, y, feat_now)

        dist_crash  = float(out["dist_crash"])
        dist_normal = float(out["dist_normal"])
        total = dist_crash + dist_normal + 1e-8
        return float(dist_normal / total)   # closer to crash cluster → higher P

    @staticmethod
    def _classical_classify(X: np.ndarray, y: np.ndarray, feat_now: np.ndarray) -> float:
        """Classical fallback: centroid distance ratio in normalised feature space."""
        X_min   = X.min(axis=0)
        X_range = X.max(axis=0) - X_min
        X_range[X_range == 0] = 1.0
        X_s    = (X - X_min) / X_range
        feat_s = (feat_now - X_min) / X_range
        crash_centroid  = X_s[y == 1].mean(axis=0) if (y == 1).any() else feat_s
        normal_centroid = X_s[y == 0].mean(axis=0) if (y == 0).any() else feat_s
        d_crash  = float(np.linalg.norm(feat_s - crash_centroid))
        d_normal = float(np.linalg.norm(feat_s - normal_centroid))
        total = d_crash + d_normal + 1e-8
        return float(d_normal / total)

    # ------------------------------------------------------------------
    # News sentiment aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_sentiment(news: NewsFeed) -> float:
        """Return mean sentiment score across all articles; 0.0 if none."""
        if not news.articles:
            return 0.0
        return float(np.mean([a.sentiment_score for a in news.articles]))

    @staticmethod
    def _news_multiplier(avg_sentiment: float) -> float:
        """
        Bayesian-style multiplier on P(crash) based on news sentiment.

        Strongly negative news boosts the crash probability estimate;
        positive news dampens it.
        """
        if avg_sentiment < -0.5:
            return 1.60
        if avg_sentiment < -0.2:
            return 1.25
        if avg_sentiment > 0.4:
            return 0.75
        if avg_sentiment > 0.2:
            return 0.90
        return 1.00   # neutral news → no adjustment

    # ------------------------------------------------------------------
    # Reasoning string
    # ------------------------------------------------------------------

    def _build_reasoning(
        self,
        as_of,
        quantum_p: float,
        avg_sentiment: float,
        multiplier: float,
        crash_prob: float,
        feat_now: np.ndarray,
        ticker_adjustments: dict[str, float],
    ) -> str:
        level = (
            "HIGH — CRASH ALERT"       if crash_prob >= self._HIGH
            else "MODERATE — CAUTION"  if crash_prob >= self._MODERATE
            else "LOW — NORMAL"
        )
        feat_labels = ["vol_regime", "momentum_5", "drawdown", "vol_ratio"]
        feat_str = "  ".join(f"{k}: {v:+.3f}" for k, v in zip(feat_labels, feat_now))

        actions = []
        for t, adj in ticker_adjustments.items():
            if adj < 0:
                actions.append(f"    {t}: SHORT  (weight multiplier {adj:+.2f})")
            elif adj < 1.0:
                actions.append(f"    {t}: REDUCE  (weight multiplier {adj:+.2f})")
            else:
                actions.append(f"    {t}: HOLD  (no change)")

        return (
            f"ChaosEngine [{pd.Timestamp(as_of).date()}] — {level}\n"
            f"  Quantum P(crash)          : {quantum_p:.3f}\n"
            f"  News sentiment            : {avg_sentiment:+.3f}  (multiplier ×{multiplier:.2f})\n"
            f"  Adjusted crash probability: {crash_prob:.3f}\n"
            f"  Market signals            : {feat_str}\n"
            f"  Recommended actions:\n" + "\n".join(actions)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        data: MarketData,
        news: NewsFeed,
        as_of,
    ) -> ChaosSignal:
        """
        Run the full ChaosEngine pipeline and return a ``ChaosSignal``.

        Parameters
        ----------
        data   : point-in-time ``MarketData`` (no look-ahead)
        news   : ``NewsFeed`` from ``MockNewsSource`` or ``ExaNewsSource``
        as_of  : the 'current' timestamp
        """
        returns = data.returns()
        feat_df = self._market_features(returns, as_of)

        # Default fallback if we can't train
        def _fallback() -> ChaosSignal:
            avg_s = self._aggregate_sentiment(news)
            mult  = self._news_multiplier(avg_s)
            p     = float(np.clip(0.5 * mult, 0.0, 1.0))
            adjs  = {t: 1.0 for t in data.tickers}
            return ChaosSignal(
                as_of=pd.Timestamp(as_of),
                crash_probability=p,
                event_label="unknown",
                confidence=0.0,
                ticker_adjustments=adjs,
                reasoning=f"ChaosEngine: insufficient history — sentiment-only estimate P(crash)={p:.2f}",
            )

        if len(feat_df) < self.crash_horizon + 15:
            return _fallback()

        X, y = self._make_dataset(feat_df, returns)

        if len(X) < 10 or len(np.unique(y)) < 2:
            return _fallback()

        feat_now  = feat_df.iloc[-1].to_numpy(dtype=float)
        quantum_p = self._xpyq_classify(X, y, feat_now)

        # Blend with news sentiment
        avg_sentiment = self._aggregate_sentiment(news)
        multiplier    = self._news_multiplier(avg_sentiment)
        crash_prob    = float(np.clip(quantum_p * multiplier, 0.0, 1.0))

        # Model confidence: how far predict_proba is from 0.5
        confidence = float(abs(quantum_p - 0.5) * 2.0)

        # Event label
        if crash_prob >= self._HIGH:
            event_label = "market_crash"
        elif crash_prob >= self._MODERATE:
            event_label = "elevated_risk"
        else:
            event_label = "normal"

        # Per-ticker weight multipliers
        ticker_adjustments: dict[str, float] = {}
        for t in data.tickers:
            if crash_prob >= self._HIGH:
                ticker_adjustments[t] = -0.80   # strong short signal
            elif crash_prob >= self._MODERATE:
                ticker_adjustments[t] = 0.40    # cut position in half
            else:
                ticker_adjustments[t] = 1.00    # no change

        reasoning = self._build_reasoning(
            as_of, quantum_p, avg_sentiment, multiplier,
            crash_prob, feat_now, ticker_adjustments,
        )

        return ChaosSignal(
            as_of=pd.Timestamp(as_of),
            crash_probability=crash_prob,
            event_label=event_label,
            confidence=confidence,
            ticker_adjustments=ticker_adjustments,
            reasoning=reasoning,
        )

    def adjust_forecast(self, forecast: Forecast, signal: ChaosSignal) -> Forecast:
        """
        Dampen or reverse a Forecast based on the ChaosSignal.

        - HIGH crash probability  → flip direction to -1, scale expected_returns
        - MODERATE                → dampen expected_returns by (1 − crash_prob)
        - LOW                     → return forecast unchanged
        """
        p = signal.crash_probability

        if p < self._MODERATE:
            return forecast

        new_expected: dict[str, float] = {}
        new_direction: dict[str, int]  = {}
        new_confidence: dict[str, float] = {}

        for t in forecast.expected_returns:
            if p >= self._HIGH:
                # Flip: expect the crash → short signal
                new_direction[t]   = -1
                new_confidence[t]  = float(signal.confidence)
                new_expected[t]    = -abs(forecast.expected_returns[t]) * p
            else:
                # Dampen
                scale = 1.0 - p
                new_direction[t]   = forecast.direction.get(t, 1)
                new_confidence[t]  = float(forecast.confidence.get(t, 0.0)) * scale
                new_expected[t]    = forecast.expected_returns[t] * scale

        return Forecast(
            as_of=forecast.as_of,
            horizon_days=forecast.horizon_days,
            expected_returns=new_expected,
            direction=new_direction,
            confidence=new_confidence,
        )

    def adjust_portfolio(
        self, portfolio: TargetPortfolio, signal: ChaosSignal
    ) -> TargetPortfolio:
        """
        Apply ``signal.ticker_adjustments`` as multipliers to portfolio weights.

        A multiplier of −0.80 on a long position turns it into a short; 0.40
        halves it; 1.0 leaves it unchanged.
        """
        if signal.crash_probability < self._MODERATE:
            return portfolio

        new_weights = {
            t: w * signal.ticker_adjustments.get(t, 1.0)
            for t, w in portfolio.weights.items()
        }
        return TargetPortfolio(as_of=portfolio.as_of, weights=new_weights)
