"""
run_quantum.py  —  three-way comparison: baseline vs quantum forecast vs full quantum.

  1. Baseline      : MomentumForecaster  + MeanVarianceOptimizer
  2. Quantum Fcst  : QuantumForecaster   + MeanVarianceOptimizer  (xpyq SVD)
  3. Full Quantum  : QuantumForecaster   + QaoaOptimizer           (xpyq SVD + eigh)

Run:  python run_quantum.py
"""
import os

from backtest import Backtest
from data import MockDataSource
from execute import PaperExecutor
from forecast import MomentumForecaster, QuantumForecaster
from news import MockNewsSource
from optimize import MeanVarianceOptimizer, QaoaOptimizer
from risk import SampleCovRisk
from score import BacktestScorer, RiskScorer

API_KEY = os.environ.get("XPYQ_KEY", "")
XPYQ_TIMEOUT = float(os.environ.get("XPYQ_TIMEOUT", "20"))
RISK_N_PATHS = int(os.environ.get("QUANTINEL_N_PATHS", "10000"))
N_DAYS = int(os.environ.get("QUANTINEL_N_DAYS", "504"))
REBALANCE_EVERY = int(os.environ.get("QUANTINEL_REBALANCE_EVERY", "5"))


def run(forecaster, optimizer, label):
    bt = Backtest(
        source=MockDataSource(n_days=N_DAYS),
        news_source=MockNewsSource(),
        forecaster=forecaster,
        risk=SampleCovRisk(n_paths=RISK_N_PATHS),
        optimizer=optimizer,
        executor=PaperExecutor(),
        rebalance_every=REBALANCE_EVERY,
    )
    data, records, baseline = bt.run()
    card = BacktestScorer().score(records, baseline)
    risk_report = RiskScorer().score(records)

    print(f"\n{'=' * 56}")
    print(f"  {label}")
    print(f"{'=' * 56}")
    print(f"  rebalances             : {len(records)}")
    print(f"  Sharpe                 : {card.sharpe:6.2f}")
    print(f"  total return           : {card.total_return * 100:6.2f}%")
    print(f"  directional accuracy   : {card.directional_accuracy * 100:6.1f}%")
    print(f"  IC                     : {card.information_coefficient:6.2f}")
    print(f"  Sharpe vs 50/50 hold   : {card.vs_baseline_sharpe:+6.2f}")
    print(f"  final equity           : {card.equity_curve.iloc[-1]:6.3f}")
    print(f"  VaR breaches           : {risk_report.var_breaches}")
    print(f"  avg disagreement       : {risk_report.avg_disagreement:.3f}")
    print(f"{'=' * 56}")
    return card


def delta(label, a, b):
    print(f"\n  {label}")
    print(f"  Sharpe        : {b.sharpe - a.sharpe:+6.2f}")
    print(f"  total return  : {(b.total_return - a.total_return) * 100:+6.2f}%")
    print(f"  dir accuracy  : {(b.directional_accuracy - a.directional_accuracy) * 100:+6.1f}%")
    print(f"  Sharpe edge   : {b.vs_baseline_sharpe - a.vs_baseline_sharpe:+6.2f}")


if __name__ == "__main__":
    print("1/3  Baseline (momentum + Markowitz)...")
    c_base = run(MomentumForecaster(), MeanVarianceOptimizer(), "BASELINE — momentum + Markowitz")

    print("\n2/3  Quantum forecast + classical optimizer (xpyq SVD)...")
    print("     (submitting SVD workloads to xpyq — ~1 min)")
    c_qfcst = run(QuantumForecaster(api_key=API_KEY, timeout=XPYQ_TIMEOUT), MeanVarianceOptimizer(),
                  "QUANTUM FORECAST — xpyq SVD + Markowitz")

    print("\n3/3  Full quantum: forecast + optimizer (xpyq SVD + eigh)...")
    print("     (submitting SVD + eigh workloads to xpyq — ~2 min)")
    c_full = run(
        QuantumForecaster(api_key=API_KEY, timeout=XPYQ_TIMEOUT),
        QaoaOptimizer(api_key=API_KEY, timeout=XPYQ_TIMEOUT),
                 "FULL QUANTUM — xpyq SVD + xpyq QUBO eigh")

    print(f"\n{'=' * 56}")
    print("  DELTAS")
    print(f"{'=' * 56}")
    delta("Quantum fcst vs baseline       :", c_base, c_qfcst)
    delta("Full quantum vs baseline       :", c_base, c_full)
    delta("Full quantum vs quantum fcst   :", c_qfcst, c_full)
    print(f"{'=' * 56}")