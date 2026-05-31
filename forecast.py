"""
LAYER 2 · FORECAST   (owner: GT (10780))

BASELINE (no ML, no quantum): MomentumForecaster.
QUANTUM SWAP: QuantumForecaster (QSVM / VQC) — same interface, you fill the body.
You could also drop in a Chronos or LSTM forecaster here; the contract is identical.

QuantumForecaster is designed to run on IBM Quantum hardware via Qiskit Runtime, but it can also run locally on a simulator if no credentials are provided.  See the docstring for details and usage instructions.
VCQ has been chosen as the quantum model for this challenge because it is relatively lightweight to train and infer, and it has a nice probabilistic output that maps well to the Forecast contract.  However, feel free to experiment with other quantum algorithms or models if you prefer.
Chronos forecasting is also added as a third option.  It uses Amazon's pretrained Chronos-T5 transformer, which is a zero-shot probabilistic time-series model — no training loop required.  The model is downloaded from HuggingFace on first use and cached locally.

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
        self.ibm_instance = ibm_instance or os.getenv("IBM_QUANTUM_INSTANCE") or None
        self.ibm_backend = ibm_backend or os.getenv("IBM_QUANTUM_BACKEND")
        self.ibm_channel = ibm_channel or os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform")
        self.scale = scale

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sampler(self):
        """Return (sampler, pass_manager) — IBM Quantum hardware or local Aer."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        if self.ibm_token:
            from qiskit_ibm_runtime import QiskitRuntimeService
            from qiskit_ibm_runtime import SamplerV2 as IBMSampler

            service = QiskitRuntimeService(
                channel=self.ibm_channel,
                token=self.ibm_token,
                **({"instance": self.ibm_instance} if self.ibm_instance else {}),
            )
            backend = (
                service.backend(self.ibm_backend)
                if self.ibm_backend
                else service.least_busy(operational=True, simulator=False)
            )
            pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
            return IBMSampler(mode=backend), pass_manager
        else:
            from qiskit_aer import AerSimulator
            from qiskit_aer.primitives import SamplerV2 as AerSampler

            backend = AerSimulator()
            pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend)
            return AerSampler(), pass_manager

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

    def _train_vqc(self, X: np.ndarray, y: np.ndarray, sampler, pass_manager):
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
            pass_manager=pass_manager,
        )
        vqc.fit(X_scaled, y)
        return vqc, scaler

    # ------------------------------------------------------------------
    # Public interface (same signature as MomentumForecaster)
    # ------------------------------------------------------------------

    def predict(self, data: MarketData, as_of, horizon_days: int) -> Forecast:
        returns = data.returns()
        sampler, pass_manager = self._build_sampler()

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

            vqc, scaler = self._train_vqc(X, y, sampler, pass_manager)

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


class ChronosForecaster:
    """
    Chronos pretrained transformer — SAME interface as MomentumForecaster.

    Zero-shot: no training loop.  The pretrained Chronos-T5 model (downloaded
    from HuggingFace on first use and cached locally) receives the raw return
    series as context and returns a predictive distribution over the next
    ``horizon_days`` steps.  The median of that distribution becomes
    ``expected_returns``; the inter-sample spread drives ``confidence``.

        expected_returns[t] = median predicted cumulative return  (signed)
        direction[t]        = sign of the median  (+1 / -1)
        confidence[t]       = min(1, |median| / std_of_samples)  (0..1)

    Usage::

        forecaster = ChronosForecaster()                            # tiny, CPU-friendly
        forecaster = ChronosForecaster(model_name="amazon/chronos-t5-small")  # more accurate
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-tiny",
        context_len: int = 60,
        num_samples: int = 20,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.context_len = context_len
        self.num_samples = num_samples
        self.device = device
        self._pipeline = None   # lazy-loaded on first predict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pipeline(self):
        """Lazy-load the ChronosPipeline (downloads weights on first call)."""
        if self._pipeline is None:
            import torch
            from chronos import ChronosPipeline

            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                dtype=torch.float32,
            )
        return self._pipeline

    # ------------------------------------------------------------------
    # Public interface (same signature as MomentumForecaster)
    # ------------------------------------------------------------------

    def predict(self, data: MarketData, as_of, horizon_days: int) -> Forecast:
        import torch

        pipeline = self._get_pipeline()
        returns = data.returns()

        expected_returns: dict[str, float] = {}
        direction: dict[str, int] = {}
        confidence: dict[str, float] = {}

        for ticker in data.tickers:
            col = returns[ticker].dropna().loc[:as_of].tail(self.context_len)

            if len(col) < 10:
                direction[ticker] = 1
                confidence[ticker] = 0.0
                expected_returns[ticker] = 0.0
                continue

            context = torch.tensor(col.to_numpy(dtype=float), dtype=torch.float32)

            # predict() returns a single samples tensor
            # shape: (batch=1, num_samples, horizon_days)
            forecast_samples = pipeline.predict(
                context.unsqueeze(0),
                prediction_length=horizon_days,
                num_samples=self.num_samples,
            )

            # Compound each sample's per-step returns into a single horizon return.
            sample_returns = (
                (1.0 + forecast_samples[0]).prod(dim=-1) - 1.0
            ).numpy()

            pred = float(np.median(sample_returns))
            spread = float(np.std(sample_returns))

            dir_val: int = 1 if pred >= 0.0 else -1
            conf: float = float(min(1.0, abs(pred) / (spread + 1e-9)))

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
