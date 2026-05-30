"""
LAYER 2 · FORECAST   (owner: ____)

BASELINE (no ML, no quantum): MomentumForecaster.
QUANTUM SWAP: QuantumForecaster (QSVM / VQC) — same interface, you fill the body.
You could also drop in a Chronos or LSTM forecaster here; the contract is identical.
"""
from __future__ import annotations

import pandas as pd

from .contracts import Forecast, MarketData


class MomentumForecaster:
    """
    Expected return = average daily return over `lookback`, scaled to the horizon.
    Dumb on purpose — it is the bar everything else must beat.
    Implements Forecaster: predict(data, as_of, horizon_days) -> Forecast.
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def predict(self, data: MarketData, as_of, horizon_days: int) -> Forecast:
        rets = data.returns().loc[:as_of].tail(self.lookback)
        exp_daily = rets.mean()

        expected = {t: float(exp_daily[t] * horizon_days) for t in data.tickers}
        direction = {t: int(1 if expected[t] >= 0 else -1) for t in data.tickers}
        confidence = {}
        for t in data.tickers:
            s = float(rets[t].std())
            confidence[t] = float(min(1.0, abs(exp_daily[t]) / s)) if s > 0 else 0.0

        return Forecast(
            as_of=pd.Timestamp(as_of),
            horizon_days=horizon_days,
            expected_returns=expected,
            direction=direction,
            confidence=confidence,
        )


class QuantumForecaster:
    """
    QSVM / VQC swap — SAME interface, different brain.

    TEAM: build features from `data` (e.g. recent returns, volume change), encode
    them onto qubits, run a Qiskit QSVM or PennyLane VQC, and return a Forecast.
    Map the classifier output to the contract:
        direction[ticker]        = +1 / -1   (up / down vote)
        confidence[ticker]       = P(up)     (0..1)
        expected_returns[ticker] = direction * confidence * small_scale
    Do NOT change the signature — that is what keeps it a drop-in.
    """

    def predict(self, data: MarketData, as_of, horizon_days: int) -> Forecast:  # noqa: D401
        raise NotImplementedError(
            "QSVM/VQC goes here. Return a Forecast with the same fields as MomentumForecaster."
        )
