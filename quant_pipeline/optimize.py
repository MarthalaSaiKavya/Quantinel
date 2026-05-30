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

import numpy as np

from .contracts import Forecast, RiskModel, TargetPortfolio


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
    QUANTUM SWAP — same Optimizer interface.

    TEAM: build the QUBO matrix Q from (mu, Sigma) exactly as DiscreteQuboOptimizer
    scores it, run QAOA (or VQE) on Qiskit, take the most-measured / lowest-energy
    bitstring, decode it to weights, and return a TargetPortfolio. Same signature.
    """

    def solve(self, forecast: Forecast, risk: RiskModel) -> TargetPortfolio:
        raise NotImplementedError(
            "Build QUBO from (mu, Sigma), run QAOA/VQE, decode best bitstring -> weights."
        )
