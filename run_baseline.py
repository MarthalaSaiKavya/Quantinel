"""
run_baseline.py  —  the no-quantum end-to-end run.

This is the integration test for the whole team: if everyone's layer honors its
contract, this prints a scorecard and risk report. To go quantum, change ONE line each:
    forecaster = QuantumForecaster()    # instead of MomentumForecaster()
    optimizer  = QaoaOptimizer()        # instead of MeanVarianceOptimizer()
Nothing else moves.

Run:  python run_baseline.py
"""

import os

from backtest import Backtest
from data import MockDataSource
from execute import PaperExecutor
from forecast import MomentumForecaster
from news import make_news_source
from optimize import MeanVarianceOptimizer
from risk import SampleCovRisk
from score import BacktestScorer, RiskScorer

EXA_KEY = os.environ.get("EXA_KEY", "")


def main():
    bt = Backtest(
        # TODO: switch to YFinanceDataSource for real NVDA/GOOG data
        # from data import YFinanceDataSource
        # source=YFinanceDataSource(tickers=["NVDA", "GOOG"], start="2023-01-01"),
        source=MockDataSource(),  # Layer 1 · Data
        news_source=make_news_source(EXA_KEY),  # Layer 1 · News (Exa or mock fallback)
        forecaster=MomentumForecaster(),  # Layer 2 · Forecast
        risk=SampleCovRisk(),  # Layer 3 · Risk
        optimizer=MeanVarianceOptimizer(),  # Layer 4 · Pick & size
        executor=PaperExecutor(),  # Layer 5 · Execute
    )
    data, records, baseline = bt.run()
    card = BacktestScorer().score(records, baseline)  # Layer 6 · Score
    report = RiskScorer().score(records)  # Layer 6 · Risk Report

    print("=" * 52)
    print("  NVDA & GOOG pair  —  BASELINE (no quantum)")
    print("=" * 52)
    print(f"  rebalances             : {len(records)}")
    print(f"  Sharpe                 : {card.sharpe:6.2f}")
    print(f"  total return           : {card.total_return * 100:6.2f}%")
    print(f"  directional accuracy   : {card.directional_accuracy * 100:6.1f}%")
    print(f"  information coeff (IC) : {card.information_coefficient:6.2f}")
    print(f"  Sharpe vs 50/50 hold   : {card.vs_baseline_sharpe:+6.2f}")
    print(f"  final equity (1.0=flat): {card.equity_curve.iloc[-1]:6.3f}")
    print("=" * 52)

    last = records[-1]
    print("  last position (realized weights):")
    for t, w in last.weights.items():
        print(f"    {t:5s} {w:+.3f}")
    print("=" * 52)

    print()
    print("=" * 52)
    print("  RISK REPORT")
    print("=" * 52)
    print(f"  VaR breaches (agg)     : {report.var_breaches}")
    print(f"  avg disagreement       : {report.avg_disagreement:.3f}")
    print(f"  max disagreement       : {report.max_disagreement:.3f}")
    for sa in report.sub_agent_reports:
        print(f"  {sa.agent_label:12s} breach rate: {sa.var_breach_rate:.3f}")
    print("=" * 52)


if __name__ == "__main__":
    main()
