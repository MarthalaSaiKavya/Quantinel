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
from chaos import ChaosEngine, MockNewsSource
from data import MockDataSource
from execute import PaperExecutor
from forecast import MomentumForecaster
from optimize import MeanVarianceOptimizer
from risk import SampleCovRisk
from score import BacktestScorer


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


def main():
    # Load .env so IBM_QUANTUM_TOKEN and friends are available before anything
    # constructs a sampler.  Safe to call even if .env doesn't exist.
    _load_dotenv()

    # ── ChaosEngine backend selection ────────────────────────────────────────
    # LOCAL  (default): leave IBM_QUANTUM_TOKEN blank in .env — runs entirely
    #                   on your CPU via Aer statevector simulation. No queue.
    # CLOUD           : set IBM_QUANTUM_TOKEN (and optionally
    #                   IBM_QUANTUM_CHANNEL / IBM_QUANTUM_BACKEND) in .env to
    #                   route VQC training to real IBM Quantum hardware.
    # No code change needed — ChaosEngine reads the env vars automatically.
    # ─────────────────────────────────────────────────────────────────────────
    engine = ChaosEngine()
    news_src = MockNewsSource()   # swap for a RealNewsSource for live news

    bt = Backtest(
        source=MockDataSource(),              # Layer 1 · Data
        forecaster=MomentumForecaster(),      # Layer 2 · Forecast
        risk=SampleCovRisk(),                 # Layer 3 · Risk
        optimizer=MeanVarianceOptimizer(),    # Layer 4 · Pick & size
        executor=PaperExecutor(),             # Layer 5 · Execute
    )
    data, records, baseline = bt.run()
    card = BacktestScorer().score(records, baseline)   # Layer 6 · Score

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

    # ── ChaosEngine evaluation on the final date ──────────────────────────
    as_of = last.as_of
    news  = news_src.fetch(as_of=as_of)
    signal = engine.evaluate(data, news, as_of=as_of)

    print()
    print("=" * 52)
    print("  CHAOS ENGINE SIGNAL")
    print("=" * 52)
    for line in signal.reasoning.splitlines():
        print(f"  {line}")
    print("=" * 52)


if __name__ == "__main__":
    main()
