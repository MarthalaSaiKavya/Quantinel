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

from contracts import RiskReport, Scorecard, SubAgentReport

PERIODS_PER_YEAR = 52  # weekly rebalances


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
            gross_pnl = sum(
                r.weights.get(t, 0.0) * r.forward_returns[t] for t in r.forward_returns
            )
            turnover = sum(
                abs(r.weights.get(t, 0.0) - prev.get(t, 0.0)) for t in r.weights
            )
            pnl.append(gross_pnl - turnover * self.cost)
            idx.append(r.as_of)
            prev = r.weights

            if r.forecast is not None:
                for t in r.forward_returns:
                    pred.append(r.forecast.expected_returns.get(t, 0.0))
                    real.append(r.forward_returns[t])
                    if t in r.forecast.direction:
                        dir_tot += 1
                        dir_hits += int(
                            np.sign(r.forecast.direction[t])
                            == np.sign(r.forward_returns[t])
                        )

        ret = pd.Series(pnl, index=pd.to_datetime(idx))
        equity = (1 + ret).cumprod()
        ic = (
            float(np.corrcoef(pred, real)[0, 1])
            if len(pred) > 1 and np.std(pred) > 0
            else 0.0
        )
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


class RiskScorer:
    """Computes RiskReport from backtest records with stored RiskModel data."""

    def score(self, records) -> RiskReport:
        breaches = 0
        total_checks = 0
        disagreements: list[float] = []
        per_agent_breaches: dict[str, int] = {}
        per_agent_total: dict[str, int] = {}

        for r in records:
            if r.risk_model is None:
                continue
            rm = r.risk_model
            disagreements.append(rm.disagreement)

            # Check VaR breaches: did any ticker's return fall below -VaR?
            for t in r.forward_returns:
                if t not in rm.var_95:
                    continue
                fwd = r.forward_returns[t]
                total_checks += 1
                if fwd < rm.var_95[t]:
                    breaches += 1

                # Per-sub-agent breach check
                for sa in rm.sub_agent_results:
                    if t not in sa.var_95:
                        continue
                    per_agent_total.setdefault(sa.agent_label, 0)
                    per_agent_breaches.setdefault(sa.agent_label, 0)
                    per_agent_total[sa.agent_label] += 1
                    if fwd < sa.var_95[t]:
                        per_agent_breaches[sa.agent_label] += 1

        breach_rate = breaches / total_checks if total_checks else 0.0
        avg_dis = float(np.mean(disagreements)) if disagreements else 0.0
        max_dis = float(max(disagreements)) if disagreements else 0.0

        sub_reports = []
        for label in sorted(per_agent_total):
            total = per_agent_total[label]
            br = per_agent_breaches.get(label, 0)
            avg_var = 0.0  # could compute from records, simplified for now
            sub_reports.append(
                SubAgentReport(
                    agent_label=label,
                    avg_var_95=avg_var,
                    var_breach_rate=br / total if total else 0.0,
                )
            )

        return RiskReport(
            var_breaches=breaches,
            var_breach_rate=breach_rate,
            avg_disagreement=avg_dis,
            max_disagreement=max_dis,
            sub_agent_reports=sub_reports,
        )