"""
run_baseline.py  —  the no-quantum end-to-end run.

This is the integration test for the whole team: if everyone's layer honors its
contract, this prints a scorecard and risk report. To go quantum, change ONE line each:
    forecaster = QuantumForecaster()    # instead of MomentumForecaster()
    optimizer  = QaoaOptimizer()        # instead of MeanVarianceOptimizer()
Nothing else moves.

Run:  python run_baseline.py
"""
from dotenv import load_dotenv
load_dotenv()  # loads IBM_QUANTUM_TOKEN etc. from .env before any imports use them
from backtest import Backtest
from data import MockDataSource
from execute import PaperExecutor
from forecast import ChronosForecaster, QuantumForecaster
from optimize import MeanVarianceOptimizer, QaoaOptimizer
from risk import SampleCovRisk
from score import BacktestScorer, RiskScorer


def main():
    bt = Backtest(
        source=MockDataSource(),              # Layer 1 · Data
        forecaster=QuantumForecaster(),      # Layer 2 · Forecast   <-- swap QuantumForecaster() here
        risk=SampleCovRisk(),                 # Layer 3 · Risk
        optimizer=MeanVarianceOptimizer(),    # Layer 4 · Pick & size <-- swap QaoaOptimizer() here
        executor=PaperExecutor(),             # Layer 5 · Execute
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
