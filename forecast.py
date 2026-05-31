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

from contracts import Forecast, MarketData

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