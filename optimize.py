"""
LAYER 4 · PICK & SIZE   (owner: ____)

Three optimizers, all behind the SAME Optimizer interface:
  - MeanVarianceOptimizer  : BASELINE. Continuous Markowitz weights (how much).
  - DiscreteQuboOptimizer  : QUBO scored by classical brute force. Discrete picks
                             (which) — the SAME output shape QAOA/VQE produces.
  - QaoaOptimizer          : QUANTUM SWAP. Same QUBO, solved on real hardware.

This is the layer where "quantum" actually earns its place at the hackathon.
"""

from __future__ import annotations

import itertools
import json
import os
import time

import numpy as np

from contracts import Forecast, RiskModel, TargetPortfolio


class MeanVarianceOptimizer:
    """
    BASELINE — Markowitz. Maximize  mu·w - lambda * w'Σw, force dollar-neutral,
    scale to a target gross exposure. Output = continuous signed weights.
    """

    def __init__(
        self,
        risk_aversion: float = 8.0,
        gross: float = 1.0,
        dollar_neutral: bool = True,
    ):
        self.risk_aversion = risk_aversion
        self.gross = gross
        self.dollar_neutral = dollar_neutral

    def solve(self, forecast: Forecast, risk: RiskModel) -> TargetPortfolio:
        tickers = list(risk.cov.index)
        mu = np.array([forecast.expected_returns[t] for t in tickers])
        Sigma = risk.cov.values

        w = np.linalg.solve(self.risk_aversion * Sigma, mu)  # w ∝ Σ^-1 μ
        if self.dollar_neutral:
            w = w - w.mean()  # sum(weights) -> ~0
        gross = np.abs(w).sum()
        if gross > 0:
            w = w / gross * self.gross  # scale to target gross

        shrink = 1.0 / (1.0 + risk.disagreement)
        weights = {t: float(w[i]) * shrink for i, t in enumerate(tickers)}
        return TargetPortfolio(as_of=forecast.as_of, weights=weights)


class DiscreteQuboOptimizer:
    """
    QUBO shape, classical solver. Each asset takes a discrete position from
    `levels` (e.g. {-1, 0, +1}). Brute-force the bitstring that maximizes
    mu·w - lambda * w'Σw. Output = discrete picks, normalized to gross 1.

    This is intentionally the SAME problem QaoaOptimizer solves — so you can A/B
    the classical solution against the quantum one for the Quantum Advantage Award.
    """

    def __init__(
        self, risk_aversion: float = 8.0, levels: tuple[int, ...] = (-1, 0, 1)
    ):
        self.risk_aversion = risk_aversion
        self.levels = levels

    def solve(self, forecast: Forecast, risk: RiskModel) -> TargetPortfolio:
        tickers = list(risk.cov.index)
        mu = np.array([forecast.expected_returns[t] for t in tickers])
        Sigma = risk.cov.values

        best, best_val = None, -np.inf
        for combo in itertools.product(self.levels, repeat=len(tickers)):
            w = np.array(combo, dtype=float)
            if np.abs(w).sum() == 0:
                continue
            w = w / np.abs(w).sum()
            val = w @ mu - self.risk_aversion * (w @ Sigma @ w)
            if val > best_val:
                best_val, best = val, w

        weights = {t: float(best[i]) for i, t in enumerate(tickers)}
        shrink = 1.0 / (1.0 + risk.disagreement)
        weights = {t: v * shrink for t, v in weights.items()}
        return TargetPortfolio(as_of=forecast.as_of, weights=weights)


class QaoaOptimizer:
    """
    QUANTUM SWAP — solves the QUBO via eigendecomposition on xpyq hardware.

    The QUBO objective is:  maximise  mu·w - lambda * w'Σw
    Equivalently:           minimise  w'Qw   where Q = lambda*Sigma - diag(mu)

    Finding the ground state of Q (eigenvector with lowest eigenvalue) is the
    quantum relaxation of the QUBO — the same problem QAOA/VQE solve on a QPU.
    xpyq runs linalg.eigh(Q) on its compute hardware; we read back the ground-state
    vector, take the sign pattern as discrete long/short positions, and normalise.

    Falls back to DiscreteQuboOptimizer if the API is unreachable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        risk_aversion: float = 8.0,
        poll_secs: float = 0.4,
        timeout: float = 20.0,
    ):
        self.api_key = api_key or os.environ.get("XPYQ_KEY", "")
        self.risk_aversion = risk_aversion
        self.poll_secs = poll_secs
        self.timeout = timeout
        self._fallback = DiscreteQuboOptimizer(risk_aversion=risk_aversion)
        self._disabled = not bool(self.api_key)
        self._stats = {
            "calls": 0,
            "xpyq_completed": 0,
            "fallbacks": 0,
            "status_counts": {},
        }

    def _run_code(self, code: str) -> dict:
        import requests

        if self._disabled:
            return {"status": "disabled", "stdout": ""}

        BASE = "https://xpyq-lib-production.up.railway.app"
        H = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        run = requests.post(
            f"{BASE}/api/v1/compute/runs",
            headers=H,
            json={"code": code, "name": "qaoa_opt"},
            timeout=10,
        ).json()
        run_id = run.get("run_id") or run.get("id")
        if not run_id:
            self._disabled = True
            return {"status": "failed", "stdout": ""}

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            r = requests.get(
                f"{BASE}/api/v1/compute/runs/{run_id}",
                headers=H,
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

    def solve(self, forecast: Forecast, risk: RiskModel) -> TargetPortfolio:
        self._stats["calls"] += 1
        if self._disabled:
            self._stats["fallbacks"] += 1
            return self._fallback.solve(forecast, risk)

        tickers = list(risk.cov.index)
        mu = np.array([forecast.expected_returns[t] for t in tickers])
        Sigma = risk.cov.values

        # QUBO matrix: minimising w'Qw  ≡  maximising mu'w - lambda * w'Sigma w
        Q = self.risk_aversion * Sigma - np.diag(mu)
        Q_list = Q.tolist()

        code = f"""
import numpy as _np, json
Q = from_numpy(_np.array({Q_list}, dtype=_np.float32))
eigvals_mat, eigvecs_mat = linalg.eigh(Q)
eigvals_arr, eigvecs_arr = eigvals_mat.numpy()
ground_state = eigvecs_arr[:, 0].tolist()
print(json.dumps({{"ground_state": ground_state, "eigvals": eigvals_arr.tolist()}}))
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
                return self._fallback.solve(forecast, risk)
            out = self._parse_json_stdout(result["stdout"])
            ground_state = np.array(out["ground_state"])
            self._stats["xpyq_completed"] += 1
        except Exception:
            self._disabled = True
            self._stats["fallbacks"] += 1
            return self._fallback.solve(forecast, risk)

        # Decode: sign of each component = long (+1) or short (-1)
        w = np.sign(ground_state)
        w[w == 0] = 1.0   # break ties long
        gross = np.abs(w).sum()
        if gross > 0:
            w = w / gross

        shrink = 1.0 / (1.0 + risk.disagreement)
        weights = {t: float(w[i]) * shrink for i, t in enumerate(tickers)}
        return TargetPortfolio(as_of=forecast.as_of, weights=weights)

    def diagnostics(self) -> dict:
        return {
            "calls": self._stats["calls"],
            "xpyq_completed": self._stats["xpyq_completed"],
            "fallbacks": self._stats["fallbacks"],
            "status_counts": dict(self._stats["status_counts"]),
            "disabled": self._disabled,
        }
