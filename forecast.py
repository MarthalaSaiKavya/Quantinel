"""
LAYER 2 · FORECAST   (owner: GT (10780))

BASELINE (no ML, no quantum): MomentumForecaster.
QUANTUM SWAP: QuantumForecaster (QSVM / VQC) — same interface, you fill the body.
You could also drop in a Chronos or LSTM forecaster here; the contract is identical.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from contracts import Forecast, MarketData


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
    VQC forecaster — SAME interface as MomentumForecaster, quantum brain.

    For each ticker a labelled dataset is built from historical sliding windows:
        feature  = last ``feature_window`` normalised daily returns  (→ qubits)
        label    = 1 if next ``horizon_days`` cumulative return > 0, else 0
    A Variational Quantum Classifier (ZZFeatureMap + RealAmplitudes ansatz,
    COBYLA optimiser) is trained, then the inferred class probabilities are
    mapped back to the Forecast contract:
        direction[t]        = +1 / -1
        confidence[t]       = P(winning class) ∈ [0.5, 1]
        expected_returns[t] = direction[t] * confidence[t] * scale

    IBM Quantum backend
    -------------------
    Pass ``ibm_token`` (and optionally ``ibm_instance``, ``ibm_backend``,
    ``ibm_channel``) to run on real hardware via ``QiskitRuntimeService``.
    Alternatively set the environment variables:
        IBM_QUANTUM_TOKEN     – API token from quantum.ibm.com
        IBM_QUANTUM_INSTANCE  – hub/group/project (default: ibm-q/open/main)
        IBM_QUANTUM_BACKEND   – backend name; omit to use least-busy
        IBM_QUANTUM_CHANNEL   – "ibm_quantum" or "ibm_cloud" (default: ibm_quantum)
    When no credentials are found, a local Aer statevector sampler is used.
    """

    def __init__(
        self,
        lookback: int = 60,
        feature_window: int = 4,
        reps: int = 2,
        max_iter: int = 100,
        ibm_token: str | None = None,
        ibm_instance: str | None = None,
        ibm_backend: str | None = None,
        ibm_channel: str | None = None,
        scale: float = 0.05,
    ) -> None:
        self.lookback = lookback
        self.feature_window = feature_window
        self.reps = reps
        self.max_iter = max_iter
        self.ibm_token = ibm_token or os.getenv("IBM_QUANTUM_TOKEN")
        self.ibm_instance = ibm_instance or os.getenv("IBM_QUANTUM_INSTANCE", "ibm-q/open/main")
        self.ibm_backend = ibm_backend or os.getenv("IBM_QUANTUM_BACKEND")
        self.ibm_channel = ibm_channel or os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum")
        self.scale = scale

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sampler(self):
        """Return a Qiskit Sampler primitive — IBM Quantum hardware or local Aer."""
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

    def _make_dataset(
        self, col: pd.Series, horizon_days: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sliding-window supervised dataset for one ticker.

        Returns X of shape (n_samples, feature_window) and y of shape (n_samples,)
        where y=1 means the next horizon_days cumulative return was positive.
        """
        # Cap training history to avoid training on very old (irrelevant) data.
        window_needed = self.lookback + self.feature_window + horizon_days
        col = col.tail(window_needed)
        vals = col.to_numpy(dtype=float)
        n = len(vals)

        X, y = [], []
        for i in range(self.feature_window, n - horizon_days):
            X.append(vals[i - self.feature_window : i])
            future_ret = float(np.prod(1.0 + vals[i : i + horizon_days]) - 1.0)
            y.append(1 if future_ret > 0.0 else 0)

        return np.array(X, dtype=float), np.array(y, dtype=int)

    def _train_vqc(self, X: np.ndarray, y: np.ndarray, sampler):
        """
        Scale features, build circuit, train VQC, return (vqc, scaler).

        Features are rescaled to [0, π] so they work as rotation angles on the
        ZZFeatureMap; the scaler must be retained for inference.
        """
        from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_machine_learning.algorithms import VQC
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)))
        X_scaled = scaler.fit_transform(X)

        n_features = X.shape[1]
        feature_map = ZZFeatureMap(feature_dimension=n_features, reps=1)
        ansatz = RealAmplitudes(num_qubits=n_features, reps=self.reps)
        optimizer = COBYLA(maxiter=self.max_iter)

        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            sampler=sampler,
        )
        vqc.fit(X_scaled, y)
        return vqc, scaler

    # ------------------------------------------------------------------
    # Public interface (same signature as MomentumForecaster)
    # ------------------------------------------------------------------

    def predict(self, data: MarketData, as_of, horizon_days: int) -> Forecast:
        returns = data.returns()
        sampler = self._build_sampler()

        expected_returns: dict[str, float] = {}
        direction: dict[str, int] = {}
        confidence: dict[str, float] = {}

        for ticker in data.tickers:
            col = returns[ticker].dropna().loc[:as_of]

            X, y = self._make_dataset(col, horizon_days)

            # Need enough samples and at least one example of each class to train.
            if len(X) < 10 or len(np.unique(y)) < 2:
                direction[ticker] = 1
                confidence[ticker] = 0.0
                expected_returns[ticker] = 0.0
                continue

            vqc, scaler = self._train_vqc(X, y, sampler)

            # Inference: most recent feature_window returns as the input vector.
            recent = col.tail(self.feature_window).to_numpy(dtype=float)
            if len(recent) < self.feature_window:
                direction[ticker] = 1
                confidence[ticker] = 0.0
                expected_returns[ticker] = 0.0
                continue

            X_pred = scaler.transform(recent.reshape(1, -1))
            # predict_proba returns shape (1, n_classes); index 1 == P(up).
            prob = vqc.predict_proba(X_pred)[0]
            p_up = float(prob[1])

            dir_val: int = 1 if p_up >= 0.5 else -1
            conf: float = p_up if p_up >= 0.5 else (1.0 - p_up)

            direction[ticker] = dir_val
            confidence[ticker] = conf
            expected_returns[ticker] = dir_val * conf * self.scale

        return Forecast(
            as_of=pd.Timestamp(as_of),
            horizon_days=horizon_days,
            expected_returns=expected_returns,
            direction=direction,
            confidence=confidence,
        )


class LSTMForecaster:
    """
    LSTM regression forecaster — SAME interface as MomentumForecaster.

    For each ticker a sliding-window dataset is built from historical daily
    returns.  A small two-layer LSTM is trained (via PyTorch) to predict the
    next ``horizon_days`` cumulative return from a sequence of ``seq_len``
    past returns.  The raw regression output maps directly to the contract:
        expected_returns[t] = predicted cumulative return  (signed)
        direction[t]        = sign of the prediction  (+1 / -1)
        confidence[t]       = min(1, |pred| / train_std)   (0..1)

    Usage::

        forecaster = LSTMForecaster()
        # or in run_baseline.py:
        forecaster = LSTMForecaster(seq_len=20, epochs=100)
    """

    def __init__(
        self,
        lookback: int = 120,
        seq_len: int = 20,
        hidden_size: int = 32,
        num_layers: int = 2,
        epochs: int = 50,
        lr: float = 1e-3,
    ) -> None:
        self.lookback = lookback
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_dataset(
        self, col: pd.Series, horizon_days: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sliding-window supervised dataset for one ticker.

        Returns X of shape (n_samples, seq_len) and y of shape (n_samples,)
        where y is the realized cumulative return over the next horizon_days.
        """
        window_needed = self.lookback + self.seq_len + horizon_days
        col = col.tail(window_needed)
        vals = col.to_numpy(dtype=float)
        n = len(vals)

        X, y = [], []
        for i in range(self.seq_len, n - horizon_days):
            X.append(vals[i - self.seq_len : i])
            future_ret = float(np.prod(1.0 + vals[i : i + horizon_days]) - 1.0)
            y.append(future_ret)

        return np.array(X, dtype=float), np.array(y, dtype=float)

    def _train(self, X: np.ndarray, y: np.ndarray):
        """
        Scale features and target, build and train the LSTM, return
        (model, x_scaler, y_scaler) ready for inference.
        """
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_scaled = x_scaler.fit_transform(X)                          # (n, seq_len)
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()   # (n,)

        X_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(-1)  # (n, seq_len, 1)
        y_t = torch.tensor(y_scaled, dtype=torch.float32)

        class _LSTMNet(nn.Module):
            def __init__(self, hidden_size: int, num_layers: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        model = _LSTMNet(self.hidden_size, self.num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_fn(model(X_t), y_t)
            loss.backward()
            optimizer.step()

        model.eval()
        return model, x_scaler, y_scaler

    # ------------------------------------------------------------------
    # Public interface (same signature as MomentumForecaster)
    # ------------------------------------------------------------------

    def predict(self, data: MarketData, as_of, horizon_days: int) -> Forecast:
        import torch

        returns = data.returns()
        expected_returns: dict[str, float] = {}
        direction: dict[str, int] = {}
        confidence: dict[str, float] = {}

        for ticker in data.tickers:
            col = returns[ticker].dropna().loc[:as_of]
            X, y = self._make_dataset(col, horizon_days)

            if len(X) < 10:
                direction[ticker] = 1
                confidence[ticker] = 0.0
                expected_returns[ticker] = 0.0
                continue

            model, x_scaler, y_scaler = self._train(X, y)

            recent = col.tail(self.seq_len).to_numpy(dtype=float)
            if len(recent) < self.seq_len:
                direction[ticker] = 1
                confidence[ticker] = 0.0
                expected_returns[ticker] = 0.0
                continue

            recent_scaled = x_scaler.transform(recent.reshape(1, -1))
            X_pred = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(-1)

            with torch.no_grad():
                pred_scaled = model(X_pred).item()

            pred = float(y_scaler.inverse_transform([[pred_scaled]])[0, 0])
            train_std = float(np.std(y)) if len(y) > 1 else 1e-6

            dir_val: int = 1 if pred >= 0.0 else -1
            conf: float = float(min(1.0, abs(pred) / (train_std + 1e-9)))

            direction[ticker] = dir_val
            confidence[ticker] = conf
            expected_returns[ticker] = pred

        return Forecast(
            as_of=pd.Timestamp(as_of),
            horizon_days=horizon_days,
            expected_returns=expected_returns,
            direction=direction,
            confidence=confidence,
        )
