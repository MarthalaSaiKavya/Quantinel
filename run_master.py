"""
run_master.py  —  normal vs quantum comparison + final agent.

Runs two simulations on the same mock market:
  1. Normal  : MomentumForecaster + MeanVarianceOptimizer
  2. Quantum : QuantumForecaster + QaoaOptimizer

Both runs are sent to MasterAgent, which compares them in simple terms and
recommends what to use next.

Run:
  set -a; source .env; set +a; .venv/bin/python run_master.py
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from backtest import Backtest
from data import YFinanceDataSource
from execute import PaperExecutor
from forecast import MomentumForecaster, QuantumForecaster
from intelligence import intelligence_from_news_source
from master_agent import MasterAgent
from news import make_news_source
from optimize import MeanVarianceOptimizer, QaoaOptimizer
from risk import SampleCovRisk
from score import BacktestScorer, RiskScorer

XPYQ_KEY = os.environ.get("XPYQ_KEY", "")
EXA_KEY = os.environ.get("EXA_API_KEY", os.environ.get("EXA_KEY", ""))
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
XPYQ_TIMEOUT = float(os.environ.get("XPYQ_TIMEOUT", "60"))
RISK_N_PATHS = int(os.environ.get("QUANTINEL_N_PATHS", "10000"))
START = os.environ.get("QUANTINEL_START", "2023-01-01")
REBALANCE_EVERY = int(os.environ.get("QUANTINEL_REBALANCE_EVERY", "5"))


@dataclass
class PipelineResult:
    name: str
    label: str
    data: object
    records: list
    baseline: list
    scorecard: object
    risk_report: object
    engine_diagnostics: dict


def run_pipeline(
    name,
    label,
    forecaster,
    optimizer,
    *,
    n_days: int | None = None,
    rebalance_every: int | None = None,
    risk_n_paths: int | None = None,
    news_source=None,
) -> PipelineResult:
    print(f"Starting {name} simulation: {label}")
    bt = Backtest(
        source=YFinanceDataSource(tickers=["NVDA", "GOOG"], start=START),
        news_source=news_source or make_news_source(EXA_KEY),
        forecaster=forecaster,
        risk=SampleCovRisk(
            n_paths=risk_n_paths if risk_n_paths is not None else RISK_N_PATHS
        ),
        optimizer=optimizer,
        executor=PaperExecutor(),
        rebalance_every=rebalance_every
        if rebalance_every is not None
        else REBALANCE_EVERY,
    )
    data, records, baseline = bt.run()
    scorecard = BacktestScorer().score(records, baseline)
    risk_report = RiskScorer().score(records)
    engine_diagnostics = {}
    for role, component in (("forecaster", forecaster), ("optimizer", optimizer)):
        if hasattr(component, "diagnostics"):
            engine_diagnostics[role] = component.diagnostics()
    print(f"Finished {name} simulation: {len(records)} rebalances.")
    return PipelineResult(
        name,
        label,
        data,
        records,
        baseline,
        scorecard,
        risk_report,
        engine_diagnostics,
    )


def summarize(result: PipelineResult) -> dict:
    last = result.records[-1]
    forecast = last.forecast.expected_returns if last.forecast is not None else {}
    equity = result.scorecard.equity_curve
    return {
        "name": result.name,
        "label": result.label,
        "as_of": str(last.as_of.date()),
        "rebalances": len(result.records),
        "sharpe": round(result.scorecard.sharpe, 3),
        "total_return_pct": round(result.scorecard.total_return * 100, 2),
        "directional_accuracy_pct": round(
            result.scorecard.directional_accuracy * 100, 1
        ),
        "information_coefficient": round(result.scorecard.information_coefficient, 3),
        "vs_50_50_sharpe": round(result.scorecard.vs_baseline_sharpe, 3),
        "final_equity": round(float(equity.iloc[-1]), 3),
        "var_breaches": result.risk_report.var_breaches,
        "avg_disagreement": round(result.risk_report.avg_disagreement, 3),
        "max_disagreement": round(result.risk_report.max_disagreement, 3),
        "last_weights": {t: round(w, 3) for t, w in last.weights.items()},
        "last_forecast": {t: round(float(v), 4) for t, v in forecast.items()},
        "engine_diagnostics": result.engine_diagnostics,
    }


def print_run(summary: dict):
    print(f"\n  {summary['name'].upper()} — {summary['label']}")
    print(f"  rebalances      : {summary['rebalances']}")
    print(f"  Sharpe          : {summary['sharpe']:6.2f}")
    print(f"  total return    : {summary['total_return_pct']:6.2f}%")
    print(f"  dir accuracy    : {summary['directional_accuracy_pct']:6.1f}%")
    print(f"  IC              : {summary['information_coefficient']:6.2f}")
    print(f"  vs 50/50 hold   : {summary['vs_50_50_sharpe']:+6.2f}")
    print(f"  final equity    : {summary['final_equity']:6.3f}")
    print(f"  VaR breaches    : {summary['var_breaches']}")
    print(f"  avg disagreem   : {summary['avg_disagreement']:.3f}")
    print(
        "  last weights    : "
        + ", ".join(f"{t} {w:+.3f}" for t, w in summary["last_weights"].items())
    )
    if summary["engine_diagnostics"]:
        print("  engine trace    :")
        for role, diag in summary["engine_diagnostics"].items():
            print(
                f"    {role:10s} calls={diag.get('calls', 0)} "
                f"xpyq_completed={diag.get('xpyq_completed', 0)} "
                f"fallbacks={diag.get('fallbacks', 0)} "
                f"statuses={diag.get('status_counts', {})}"
            )


def build_trace(normal: dict, quantum: dict) -> tuple[dict, list[str]]:
    deltas = {
        "quantum_minus_normal_total_return_pct": round(
            quantum["total_return_pct"] - normal["total_return_pct"], 2
        ),
        "quantum_minus_normal_sharpe": round(quantum["sharpe"] - normal["sharpe"], 3),
        "quantum_minus_normal_directional_accuracy_pct": round(
            quantum["directional_accuracy_pct"] - normal["directional_accuracy_pct"], 1
        ),
        "quantum_minus_normal_var_breaches": quantum["var_breaches"]
        - normal["var_breaches"],
        "quantum_minus_normal_final_equity": round(
            quantum["final_equity"] - normal["final_equity"], 3
        ),
    }

    trace = [
        "Both simulations used the same real NVDA/GOOG market data (yfinance), news source, risk model, executor, and scorer.",
        (
            f"Normal returned {normal['total_return_pct']:.2f}% while quantum returned "
            f"{quantum['total_return_pct']:.2f}%, a quantum-minus-normal gap of "
            f"{deltas['quantum_minus_normal_total_return_pct']:+.2f} percentage points."
        ),
        (
            f"Normal risk-adjusted score was {normal['sharpe']:.2f}; quantum was "
            f"{quantum['sharpe']:.2f}, a gap of {deltas['quantum_minus_normal_sharpe']:+.2f}."
        ),
        (
            f"Directional accuracy was {normal['directional_accuracy_pct']:.1f}% for normal "
            f"and {quantum['directional_accuracy_pct']:.1f}% for quantum."
        ),
        (
            f"Risk breaches were {normal['var_breaches']} for normal and "
            f"{quantum['var_breaches']} for quantum."
        ),
        (
            "Final weights were "
            + (
                "the same"
                if normal["last_weights"] == quantum["last_weights"]
                else "different"
            )
            + f": normal {normal['last_weights']}, quantum {quantum['last_weights']}."
        ),
    ]

    quantum_diag = quantum.get("engine_diagnostics", {})
    for role, diag in quantum_diag.items():
        trace.append(
            f"Quantum {role} trace: {diag.get('xpyq_completed', 0)} xpyq completions, "
            f"{diag.get('fallbacks', 0)} local fallbacks, statuses {diag.get('status_counts', {})}."
        )

    return deltas, trace


def main():
    normal_label = "MomentumForecaster + MeanVarianceOptimizer"
    quantum_label = "QuantumForecaster + QaoaOptimizer (xpyq)"

    normal_job = (
        "normal",
        normal_label,
        MomentumForecaster(),
        MeanVarianceOptimizer(),
    )
    quantum_job = (
        "quantum",
        quantum_label,
        QuantumForecaster(api_key=XPYQ_KEY, timeout=XPYQ_TIMEOUT),
        QaoaOptimizer(api_key=XPYQ_KEY, timeout=XPYQ_TIMEOUT),
    )

    print("Running normal and quantum simulations in parallel...")
    if XPYQ_KEY:
        print(
            "(quantum branch will submit xpyq workloads, with local fallback on timeout/failure)"
        )
    else:
        print("(XPYQ_KEY is missing, so the quantum branch will use local fallbacks)")

    news_source = make_news_source(EXA_KEY)
    if EXA_KEY:
        print("(using Exa API for news during backtest and intelligence feed)")
    else:
        print("(EXA_KEY is missing, falling back to MockNewsSource)")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run_pipeline, *job, news_source=news_source)
            for job in (normal_job, quantum_job)
        ]
        results = [future.result() for future in futures]

    by_name = {result.name: result for result in results}
    normal = summarize(by_name["normal"])
    quantum = summarize(by_name["quantum"])
    metric_deltas, decision_trace = build_trace(normal, quantum)

    print("\nBuilding market intelligence from Exa news...")
    intel = intelligence_from_news_source(
        news_source,
        by_name["normal"].data.tickers,
        as_of=by_name["normal"].records[-1].as_of,
    )
    print("Intelligence ready.")

    print("\nSending both simulations to final agent...")
    report = MasterAgent(openrouter_key=OPENROUTER_KEY).compare(
        normal,
        quantum,
        intel,
        metric_deltas=metric_deltas,
        decision_trace=decision_trace,
    )
    print("Final comparison ready.")

    width = 66
    print("\n" + "=" * width)
    print("  QUANTINEL COMPARISON REPORT")
    print("=" * width)
    print(f"  as_of      : {report.as_of.date()}")
    print(f"  winner     : {report.winner}")
    print(f"  next step  : {report.recommendation}")

    print(f"\n  {'-' * 24} SIMULATION RESULTS {'-' * 22}")
    print_run(normal)
    print_run(quantum)

    print(f"\n  {'-' * 22} MARKET INTELLIGENCE {'-' * 22}")
    for ticker in by_name["normal"].data.tickers:
        print(f"  {ticker} sentiment : {report.sentiment.get(ticker, 0.0):+.2f}")
    print(f"  themes        : {', '.join(report.key_themes) or 'N/A'}")
    for ticker in by_name["normal"].data.tickers:
        heads = report.headlines.get(ticker, [])[:2]
        label = f"  {ticker}:"
        if not heads:
            print(f"{label} (no headlines)")
            continue
        for i, headline in enumerate(heads):
            prefix = label if i == 0 else " " * len(label)
            print(f"{prefix} {headline[:76]}")

    print(f"\n  {'-' * 24} DECISION TRACE {'-' * 24}")
    for key, value in report.metric_deltas.items():
        print(
            f"  {key}: {value:+}"
            if isinstance(value, (int, float))
            else f"  {key}: {value}"
        )
    for item in report.decision_trace:
        print(f"  - {item}")

    print(f"\n  {'-' * 26} FINAL AGENT {'-' * 25}")
    for line in report.rationale.split(". "):
        line = line.strip()
        if line:
            if not line.endswith("."):
                line += "."
            print(f"  {line}")
    print("=" * width)


if __name__ == "__main__":
    main()
