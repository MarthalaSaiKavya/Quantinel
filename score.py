"""
LAYER 6 · SCORE   (owner: ____)

BacktestScorer turns the per-step records into the numbers judges care about:
Sharpe, total return, directional accuracy, information coefficient (IC), and the
Sharpe edge over a plain 50/50 buy-and-hold baseline. Transaction cost is charged
on turnover so the score is honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from contracts import Scorecard

PERIODS_PER_YEAR = 52   # weekly rebalances


def _sharpe(returns: pd.Series) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(PERIODS_PER_YEAR))


class BacktestScorer:
    """Implements Scorer: score(records, baseline_records) -> Scorecard."""

    def __init__(self, cost_bps: float = 2.0):
        self.cost = cost_bps / 1e4

    def _equity(self, records):
        prev: dict[str, float] = {}
        pnl, idx = [], []
        pred, real = [], []
        dir_hits = dir_tot = 0

        for r in records:
            gross_pnl = sum(r.weights.get(t, 0.0) * r.forward_returns[t] for t in r.forward_returns)
            turnover = sum(abs(r.weights.get(t, 0.0) - prev.get(t, 0.0)) for t in r.weights)
            pnl.append(gross_pnl - turnover * self.cost)
            idx.append(r.as_of)
            prev = r.weights

            if r.forecast is not None:
                for t in r.forward_returns:
                    pred.append(r.forecast.expected_returns.get(t, 0.0))
                    real.append(r.forward_returns[t])
                    if t in r.forecast.direction:
                        dir_tot += 1
                        dir_hits += int(np.sign(r.forecast.direction[t]) == np.sign(r.forward_returns[t]))

        ret = pd.Series(pnl, index=pd.to_datetime(idx))
        equity = (1 + ret).cumprod()
        ic = float(np.corrcoef(pred, real)[0, 1]) if len(pred) > 1 and np.std(pred) > 0 else 0.0
        da = dir_hits / dir_tot if dir_tot else 0.0
        return ret, equity, ic, da

    def score(self, records, baseline_records) -> Scorecard:
        ret, equity, ic, da = self._equity(records)
        b_ret, _, _, _ = self._equity(baseline_records)
        return Scorecard(
            sharpe=_sharpe(ret),
            total_return=float(equity.iloc[-1] - 1) if len(equity) else 0.0,
            directional_accuracy=da,
            information_coefficient=ic,
            vs_baseline_sharpe=_sharpe(ret) - _sharpe(b_ret),
            equity_curve=equity,
        )
