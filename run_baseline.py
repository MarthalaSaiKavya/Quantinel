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
from news import ExaNewsSource, MockNewsSource
from optimize import MeanVarianceOptimizer
from risk import SampleCovRisk
from score import BacktestScorer, RiskScorer


def _load_dotenv():
    """Load .env file into os.environ (no extra dependencies)."""
    env_path = os.path.join(os.path.dirname(__file__) or ".", ".env")
    if not os.path.exists(env_path):
        return
    for line in open(env_path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _get_news_source():
    """Use Exa if API key is available, otherwise fall back to mock."""
    _load_dotenv()
    key = os.environ.get("EXA_API_KEY", "")
    if key:
        print(f"  News source: Exa (live)  —  key {key[:6]}...\n")
        return ExaNewsSource(api_key=key)
    print("  News source: Mock (offline)  —  set EXA_API_KEY in .env for real news\n")
    return MockNewsSource()


def main():
    news_source = _get_news_source()
    bt = Backtest(
        source=MockDataSource(),              # Layer 1 · Data
        news_source=news_source,              # Layer 1 · News (Exa or mock)
        forecaster=MomentumForecaster(),      # Layer 2 · Forecast
        risk=SampleCovRisk(),                 # Layer 3 · Risk
        optimizer=MeanVarianceOptimizer(),    # Layer 4 · Pick & size
        executor=PaperExecutor(),             # Layer 5 · Execute
    )
    data, records, baseline = bt.run()
    card = BacktestScorer().score(records, baseline)   # Layer 6 · Score
    report = RiskScorer().score(records)               # Layer 6 · Risk Report

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
