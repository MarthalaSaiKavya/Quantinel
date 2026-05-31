"""
LAYER 2 · FORECAST   (owner: GT (10780))

BASELINE (no ML, no quantum): MomentumForecaster.
QUANTUM SWAP: QuantumForecaster — submits Python to the xpyq compute API.
  xpyq runs SVD on its purpose-built hardware; we read back U/S/Vt and
  extract factor-momentum signals per ticker. Same interface — drop it in.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from contracts import CrystalBallPrediction, Forecast, MarketData, NewsFeed

_XPYQ_BASE = "https://xpyq-lib-production.up.railway.app"


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
    Factor-momentum forecaster backed by the xpyq compute API.

    How it works:
      1. Build a returns matrix R (lookback x n_tickers) from recent history.
      2. POST Python code to xpyq that calls linalg.svd(R) on hardware.
         xpyq returns U (time factors), S (singular values), Vt (ticker loadings).
      3. Compute factor scores F = U * S  — the time series of each market factor.
      4. Factor momentum = F[-1, 0] - F[-horizon_days, 0]  (dominant factor trend).
      5. Each ticker's direction = sign(factor_momentum * Vt[0, ticker_index]).
      6. Falls back to MomentumForecaster if the API is unreachable or fails.

    Args:
        api_key:    xpyq Bearer token.
        lookback:   rows of return history fed into SVD (default 40).
        poll_secs:  polling interval while waiting for xpyq result (default 0.4s).
        timeout:    max seconds to wait per run before falling back (default 20s).
    """

    def __init__(
        self,
        api_key: str | None = None,
        lookback: int = 40,
        poll_secs: float = 0.4,
        timeout: float = 20.0,
    ):
        self.api_key = api_key or os.environ.get("XPYQ_KEY", "")
        self.lookback = lookback
        self.poll_secs = poll_secs
        self.timeout = timeout
        self._fallback = MomentumForecaster()
        self._disabled = not bool(self.api_key)
        self._stats = {
            "calls": 0,
            "xpyq_completed": 0,
            "fallbacks": 0,
            "status_counts": {},
        }

    # ------------------------------------------------------------------
    # xpyq helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _run_code(self, code: str, name: str = "forecast") -> dict:
        """Submit code to xpyq and block until terminal status."""
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
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, data: MarketData, as_of, horizon_days: int) -> Forecast:
        self._stats["calls"] += 1
        if self._disabled:
            self._stats["fallbacks"] += 1
            return self._fallback.predict(data, as_of, horizon_days)

        rets = data.returns().loc[:as_of].tail(self.lookback)

        if len(rets) < horizon_days + 2:
            self._stats["fallbacks"] += 1
            return self._fallback.predict(data, as_of, horizon_days)

        tickers = data.tickers
        R_list = rets[tickers].values.astype(float).tolist()

        # Code that runs on xpyq hardware
        code = f"""
import numpy as _np, json
R = from_numpy(_np.array({R_list}, dtype=_np.float32))
U_mat, S_mat, Vt_mat = linalg.svd(R)
U_arr, S_arr, Vt_arr = U_mat.numpy()
factor_scores = U_arr * S_arr          # (lookback x n_factors)
ticker_vols = _np.array({[float(rets[t].std()) for t in tickers]})
print(json.dumps({{
    "factor_scores_col0": factor_scores[:, 0].tolist(),
    "Vt_row0": Vt_arr[0].tolist(),
    "ticker_vols": ticker_vols.tolist(),
}}))
"""

        try:
            result = self._run_code(code)
            status = result.get("status", "unknown")
            self._stats["status_counts"][status] = (
                self._stats["status_counts"].get(status, 0) + 1
            )
            if result["status"] != "completed" or not result.get("stdout", "").strip():
                if result["status"] in ("failed", "timed_out", "cancelled"):
                    self._disabled = True
                self._stats["fallbacks"] += 1
                return self._fallback.predict(data, as_of, horizon_days)

            out = self._parse_json_stdout(result["stdout"])
            self._stats["xpyq_completed"] += 1
        except Exception:
            self._disabled = True
            self._stats["fallbacks"] += 1
            return self._fallback.predict(data, as_of, horizon_days)

        factor_scores_col0 = np.array(out["factor_scores_col0"])
        Vt_row0 = np.array(out["Vt_row0"])
        ticker_vols = np.array(out["ticker_vols"])

        factor_vol = float(factor_scores_col0.std()) + 1e-8
        momentum = float(factor_scores_col0[-1] - factor_scores_col0[-horizon_days])

        expected: dict[str, float] = {}
        direction: dict[str, int] = {}
        confidence: dict[str, float] = {}

        for i, ticker in enumerate(tickers):
            loading = float(Vt_row0[i])
            signal = momentum * loading
            scale = float(ticker_vols[i] * np.sqrt(horizon_days))

            direction[ticker] = 1 if signal >= 0 else -1
            confidence[ticker] = float(min(1.0, abs(momentum) / factor_vol))
            expected[ticker] = float(signal * scale)

        return Forecast(
            as_of=pd.Timestamp(as_of),
            horizon_days=horizon_days,
            expected_returns=expected,
            direction=direction,
            confidence=confidence,
        )

    def diagnostics(self) -> dict:
        return {
            "calls": self._stats["calls"],
            "xpyq_completed": self._stats["xpyq_completed"],
            "fallbacks": self._stats["fallbacks"],
            "status_counts": dict(self._stats["status_counts"]),
            "disabled": self._disabled,
        }


class CrystalBall:
    """
    1-year scenario forecaster that fuses the Chaos Engine's tail-risk signal
    with a short-horizon Forecast and xpyq eigendecomposition of the returns
    covariance matrix — the same quantum path used by ChaosEngine.

    Produces a ``CrystalBallPrediction`` with three scenarios per ticker:
      - bull  : base + 1.5 × annual_vol  (optimistic)
      - base  : short-horizon expected return compounded over ``horizon_days``
      - bear  : base − 1.5 × annual_vol  (pessimistic)
    plus a crash-adjusted return that applies the ChaosEngine's per-ticker
    weight multipliers to the base estimate.

    The per-ticker annual volatility is derived from the leading eigenvalues of
    the returns covariance matrix submitted to xpyq ``linalg.eig``.  The
    dominant factor variance (leading eigenvalue × 252) measures how strongly
    a single market-wide risk factor drives co-movement.

    Args:
        forecaster:    Any Forecaster (``MomentumForecaster`` or ``QuantumForecaster``).
        chaos_engine:  An initialised ``ChaosEngine`` instance.
        lookback:      Trading days of return history for the covariance matrix (default 60).
        short_horizon: Days for the underlying Forecast before compounding (default 5).
        horizon_days:  Prediction target in trading days (default 252 ≈ 1 year).
        api_key:       xpyq Bearer token (falls back to XPYQ_KEY env var).
        poll_secs:     Polling interval while waiting for xpyq (default 0.4 s).
        timeout:       Max seconds per xpyq job before classical fallback (default 20 s).
    """

    def __init__(
        self,
        forecaster,
        chaos_engine,
        lookback: int = 60,
        short_horizon: int = 5,
        horizon_days: int = 252,
        api_key: str | None = None,
        poll_secs: float = 0.4,
        timeout: float = 20.0,
    ) -> None:
        self.forecaster = forecaster
        self.chaos_engine = chaos_engine
        self.lookback = lookback
        self.short_horizon = short_horizon
        self.horizon_days = horizon_days
        self.api_key = api_key or os.environ.get("XPYQ_KEY", "")
        self.poll_secs = poll_secs
        self.timeout = timeout
        self._disabled = not bool(self.api_key)

    # ------------------------------------------------------------------
    # xpyq helpers (same pattern as QuantumForecaster and ChaosEngine)
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _run_code(self, code: str, name: str = "crystal_ball") -> dict:
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
    # Factor volatility extraction via xpyq eigendecomposition
    # ------------------------------------------------------------------

    def _factor_vols(
        self,
        cov_list: list[list[float]],
        n_tickers: int,
    ) -> tuple[list[float], float]:
        """
        Submit the returns covariance matrix to xpyq ``linalg.eig``.

        Returns
        -------
        annual_vol_per_ticker : list[float]
            Per-ticker annualised vol derived from the factor model:
            ``vol_i = sqrt(sum_k  loading_ik^2 * eigenvalue_k * 252)``.
        dominant_factor_var : float
            Leading eigenvalue × 252 — the annualised variance of the strongest
            market-wide factor.

        Falls back to ``_classical_factor_vols`` if xpyq is unavailable.
        """
        code = f"""
import numpy as _np, json

cov = from_numpy(_np.array({cov_list}, dtype=_np.float32))
eigvals_mat, eigvecs_mat = linalg.eig(cov)
eigvals_arr, eigvecs_arr = eigvals_mat.numpy()

# Sort descending: factor 0 is the dominant market factor
idx = _np.argsort(eigvals_arr)[::-1]
eigvals_sorted = _np.maximum(eigvals_arr[idx], 0.0)
eigvecs_sorted = eigvecs_arr[:, idx]

# Per-ticker annual variance via factor model
factor_var_annual = eigvals_sorted * 252.0
annual_var = (eigvecs_sorted ** 2) @ factor_var_annual
annual_vol = _np.sqrt(annual_var).tolist()
dominant = float(eigvals_sorted[0] * 252.0)

print(json.dumps({{
    "annual_vol": annual_vol,
    "dominant_factor_var": dominant,
}}))
"""
        try:
            result = self._run_code(code, name="crystal_eig")
            if result["status"] == "completed" and result.get("stdout", "").strip():
                out = self._parse_json_stdout(result["stdout"])
                return out["annual_vol"], float(out["dominant_factor_var"])
            if result["status"] in ("failed", "timed_out", "cancelled"):
                self._disabled = True
        except Exception:
            self._disabled = True

        return self._classical_factor_vols(cov_list)

    @staticmethod
    def _classical_factor_vols(cov_list: list[list[float]]) -> tuple[list[float], float]:
        """Classical fallback using numpy.linalg.eigh (symmetric, numerically stable)."""
        cov = np.array(cov_list, dtype=float)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[idx], 0.0)
        eigvecs = eigvecs[:, idx]
        factor_var_annual = eigvals * 252.0
        annual_var = (eigvecs ** 2) @ factor_var_annual
        annual_vol = np.sqrt(annual_var).tolist()
        dominant = float(eigvals[0] * 252.0)
        return annual_vol, dominant

    # ------------------------------------------------------------------
    # Reasoning string
    # ------------------------------------------------------------------

    def _build_reasoning(
        self,
        as_of,
        tickers: list[str],
        base_returns: dict[str, float],
        bull_returns: dict[str, float],
        bear_returns: dict[str, float],
        crash_adjusted_returns: dict[str, float],
        annual_vol: dict[str, float],
        chaos_signal,
        dominant_factor_var: float,
    ) -> str:
        level = (
            "HIGH RISK"    if chaos_signal.crash_probability >= 0.65
            else "CAUTION" if chaos_signal.crash_probability >= 0.40
            else "NORMAL"
        )
        lines = [
            f"CrystalBall [{pd.Timestamp(as_of).date()}] — 1-YEAR OUTLOOK"
            f" ({self.horizon_days} trading days)",
            f"  Risk regime              : {level}",
            f"  Crash probability        : {chaos_signal.crash_probability:.3f}",
            f"  Dominant factor variance : {dominant_factor_var:.4f} (annualised)",
            f"  Ticker scenarios:",
        ]
        for t in tickers:
            lines.append(
                f"    {t:6s}  base: {base_returns[t]:+.1%}  "
                f"bull: {bull_returns[t]:+.1%}  "
                f"bear: {bear_returns[t]:+.1%}  "
                f"vol: {annual_vol[t]:.1%}  "
                f"crash-adj: {crash_adjusted_returns[t]:+.1%}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, data: MarketData, news: NewsFeed, as_of) -> CrystalBallPrediction:
        """
        Produce a 1-year ``CrystalBallPrediction``.

        Parameters
        ----------
        data   : point-in-time ``MarketData`` (no look-ahead).
        news   : ``NewsFeed`` passed through to the ``ChaosEngine``.
        as_of  : the 'current' timestamp.
        """
        tickers = data.tickers
        rets = data.returns().loc[:as_of].tail(self.lookback)

        # 1. Short-horizon forecast
        short_forecast = self.forecaster.predict(data, as_of, self.short_horizon)

        # 2. Tail-risk signal from ChaosEngine
        chaos_signal = self.chaos_engine.evaluate(data, news, as_of)

        # 3. Covariance matrix → xpyq for factor volatilities
        cov_list = rets[tickers].cov().values.tolist()
        annual_vol_list, dominant_factor_var = self._factor_vols(cov_list, len(tickers))
        annual_vol = {t: float(annual_vol_list[i]) for i, t in enumerate(tickers)}

        # 4. Compound short-horizon expected return to 1-year
        base_returns: dict[str, float] = {}
        bull_returns: dict[str, float] = {}
        bear_returns: dict[str, float] = {}
        crash_adjusted_returns: dict[str, float] = {}

        for t in tickers:
            daily_exp = short_forecast.expected_returns.get(t, 0.0) / self.short_horizon
            base = float((1.0 + daily_exp) ** self.horizon_days - 1.0)
            vol  = annual_vol[t]
            base_returns[t]           = base
            bull_returns[t]           = base + 1.5 * vol
            bear_returns[t]           = base - 1.5 * vol
            crash_adjusted_returns[t] = base * chaos_signal.ticker_adjustments.get(t, 1.0)

        # 5. Confidence inherited from short-horizon forecast
        confidence = {t: short_forecast.confidence.get(t, 0.0) for t in tickers}

        reasoning = self._build_reasoning(
            as_of, tickers, base_returns, bull_returns, bear_returns,
            crash_adjusted_returns, annual_vol, chaos_signal, dominant_factor_var,
        )

        return CrystalBallPrediction(
            as_of=pd.Timestamp(as_of),
            horizon_days=self.horizon_days,
            base_returns=base_returns,
            bull_returns=bull_returns,
            bear_returns=bear_returns,
            crash_adjusted_returns=crash_adjusted_returns,
            annual_volatility=annual_vol,
            crash_probability=chaos_signal.crash_probability,
            dominant_factor_var=dominant_factor_var,
            confidence=confidence,
            reasoning=reasoning,
        )
