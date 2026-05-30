"""
LAYER 2.5 · CHAOS ENGINE  (owner: GT (10780))

Wildcard / tail-event detector that fuses quantum-encoded market signals
with world-news sentiment to estimate the probability of an adverse
"black-swan" event (crash, liquidity crisis, sector collapse, etc.).

Pipeline integration example (run_baseline.py)::

    from chaos import ChaosEngine, MockNewsSource

    engine = ChaosEngine()                           # IBM creds read from env vars
    news   = MockNewsSource().fetch(as_of=as_of)     # swap for RealNewsSource later
    signal = engine.evaluate(data, news, as_of=as_of)
    print(signal.reasoning)                          # plain-English recommendation

    forecast  = engine.adjust_forecast(forecast, signal)   # dampen / flip forecasts
    portfolio = engine.adjust_portfolio(portfolio, signal)  # scale / short positions

IBM Quantum backend
-------------------
Same environment variables as QuantumForecaster:
    IBM_QUANTUM_TOKEN, IBM_QUANTUM_INSTANCE, IBM_QUANTUM_CHANNEL, IBM_QUANTUM_BACKEND
Falls back to local Aer simulation when no credentials are found.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from contracts import ChaosSignal, Forecast, MarketData, NewsItem, TargetPortfolio


# ============================================================================
# NEWS SOURCES
# ============================================================================


class MockNewsSource:
    """
    Synthetic news feed for development / backtesting.

    Generates plausible sentiment scores seeded from market returns so that
    bad market days correlate loosely with negative news — which lets the
    ChaosEngine's training labels and news features stay coherent even without
    a live news API.

    To go live: write a ``RealNewsSource`` with the same ``fetch()`` signature
    that calls NewsAPI / Bloomberg / Refinitiv and returns a list[NewsItem].
    """

    def __init__(self, seed: int = 42, noise: float = 0.3) -> None:
        self.seed = seed
        self.noise = noise

    def fetch(
        self,
        as_of,
        n: int = 30,
        lookback_days: int = 5,
    ) -> list[NewsItem]:
        """
        Return ``n`` synthetic NewsItems from the ``lookback_days`` window
        ending at ``as_of``.

        Parameters
        ----------
        as_of        : date-like — the 'current' date
        n            : number of headlines to generate
        lookback_days: how many calendar days back to spread headlines over
        """
        rng = np.random.default_rng(
            self.seed + int(pd.Timestamp(as_of).timestamp() / 86400)
        )
        end = pd.Timestamp(as_of)
        start = end - pd.Timedelta(days=lookback_days)
        timestamps = pd.to_datetime(
            rng.uniform(start.value, end.value, n).astype("int64")
        )

        templates = [
            ("Fed signals further rate {action}", lambda s: s),
            ("Tech sector {direction} amid {reason}", lambda s: s),
            ("Global markets {move} on geopolitical {event}", lambda s: s),
            ("Analysts {view} outlook for equities", lambda s: s),
            ("Inflation data {surprise} expectations", lambda s: s),
        ]
        actions_pos = ["cuts", "easing", "pause", "stability"]
        actions_neg = ["hikes", "tightening", "uncertainty", "volatility"]

        items: list[NewsItem] = []
        for i, ts in enumerate(sorted(timestamps)):
            sentiment = float(np.clip(rng.normal(0.0, 0.4) + rng.normal(0.0, self.noise), -1.0, 1.0))
            word = rng.choice(actions_pos if sentiment > 0 else actions_neg)
            headline = f"Market update {i + 1}: {word} reported as of {ts.date()}"
            items.append(
                NewsItem(
                    timestamp=ts,
                    headline=headline,
                    sentiment_score=sentiment,
                    source="MockNewsSource",
                )
            )
        return items


# ============================================================================
# CHAOS ENGINE
# ============================================================================


class ChaosEngine:
    """
    Wildcard event predictor — quantum-powered tail-risk detector.

    Workflow
    --------
    1. Build market features from ``MarketData`` (vol regime, momentum,
       drawdown, volatility spike ratio) — these are the VQC training inputs.
    2. Train a binary VQC on historical windows:
           label = 1  if worst return over next ``crash_horizon`` days
                      falls below ``crash_threshold``  (i.e. a crash occurred)
           label = 0  otherwise
    3. Run inference on the current feature vector to get P(crash) from the
       quantum circuit.
    4. Adjust P(crash) up/down based on live news sentiment (Bayesian boost).
    5. Return a ``ChaosSignal`` with:
           crash_probability    — final blended estimate
           ticker_adjustments   — weight multipliers for adjust_portfolio()
           reasoning            — plain-English recommendation string

    IBM Quantum backend
    -------------------
    Reads IBM_QUANTUM_TOKEN / IBM_QUANTUM_INSTANCE / IBM_QUANTUM_CHANNEL /
    IBM_QUANTUM_BACKEND from environment (or pass as constructor args).
    Falls back to local Aer statevector simulation when no token is set.
    """

    # Severity thresholds for action labels and position adjustments
    _HIGH: float = 0.65
    _MODERATE: float = 0.40

    def __init__(
        self,
        lookback: int = 120,
        crash_threshold: float = -0.04,   # -4 % in crash_horizon days = "crash"
        crash_horizon: int = 5,
        reps: int = 2,
        max_iter: int = 150,
        ibm_token: str | None = None,
        ibm_instance: str | None = None,
        ibm_backend: str | None = None,
        ibm_channel: str | None = None,
    ) -> None:
        self.lookback = lookback
        self.crash_threshold = crash_threshold
        self.crash_horizon = crash_horizon
        self.reps = reps
        self.max_iter = max_iter
        self.ibm_token = ibm_token or os.getenv("IBM_QUANTUM_TOKEN")
        self.ibm_instance = ibm_instance or os.getenv("IBM_QUANTUM_INSTANCE", "ibm-q/open/main")
        self.ibm_backend = ibm_backend or os.getenv("IBM_QUANTUM_BACKEND")
        self.ibm_channel = ibm_channel or os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum")

    # ------------------------------------------------------------------
    # Sampler primitive (IBM hardware or local Aer)
    # ------------------------------------------------------------------

    def _build_sampler(self):
        if self.ibm_token:
            from qiskit_ibm_runtime import QiskitRuntimeService
            from qiskit_ibm_runtime import SamplerV2 as IBMSampler

            service = QiskitRuntimeService(
                channel=self.ibm_channel,
                token=self.ibm_token,
                instance=self.ibm_instance,
            )
            backend = (
                service.backend(self.ibm_backend)
                if self.ibm_backend
                else service.least_busy(operational=True, simulator=False)
            )
            return IBMSampler(mode=backend)
        else:
            from qiskit_aer.primitives import SamplerV2 as AerSampler

            return AerSampler()

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
    # VQC training
    # ------------------------------------------------------------------

    def _train_vqc(self, X: np.ndarray, y: np.ndarray, sampler):
        from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_machine_learning.algorithms import VQC
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)))
        X_scaled = scaler.fit_transform(X)

        n_features = X.shape[1]
        feature_map = ZZFeatureMap(feature_dimension=n_features, reps=1)
        ansatz      = RealAmplitudes(num_qubits=n_features, reps=self.reps)
        optimizer   = COBYLA(maxiter=self.max_iter)

        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            sampler=sampler,
        )
        vqc.fit(X_scaled, y)
        return vqc, scaler

    # ------------------------------------------------------------------
    # News sentiment aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_sentiment(news: list[NewsItem]) -> float:
        """Return mean sentiment score; 0.0 (neutral) if no items provided."""
        if not news:
            return 0.0
        return float(np.mean([item.sentiment_score for item in news]))

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
        news: list[NewsItem],
        as_of,
    ) -> ChaosSignal:
        """
        Run the full ChaosEngine pipeline and return a ``ChaosSignal``.

        Parameters
        ----------
        data   : point-in-time ``MarketData`` (no look-ahead)
        news   : recent ``NewsItem`` list — use ``MockNewsSource.fetch()`` or a
                 real news source with the same interface
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

        sampler = self._build_sampler()
        vqc, scaler = self._train_vqc(X, y, sampler)

        # Inference: current feature vector
        feat_now = feat_df.iloc[-1].to_numpy(dtype=float)
        X_pred   = scaler.transform(feat_now.reshape(1, -1))
        prob     = vqc.predict_proba(X_pred)[0]
        quantum_p = float(prob[1])   # P(crash label)

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
