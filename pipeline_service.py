"""
pipeline_service.py — Run the Quantinel pipeline and build dashboard JSON.

The dashboard (dashboard/dashboard.js) expects a single payload with keys:
  tickers, decision, intelligence, risk, normal, quantum, execution_logs
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from forecast import MomentumForecaster, QuantumForecaster
from intelligence import intelligence_from_news_source
from master_agent import MasterAgent
from news import make_news_source
from optimize import MeanVarianceOptimizer, QaoaOptimizer
from pipeline_agent import PipelineAgent
from run_master import PipelineResult, build_trace, run_pipeline, summarize

XPYQ_KEY = os.environ.get("XPYQ_KEY", "")
EXA_KEY = os.environ.get("EXA_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
XPYQ_TIMEOUT = float(os.environ.get("XPYQ_TIMEOUT", "60"))

_FAST = os.environ.get("QUANTINEL_API_FAST", "").lower() in {"1", "true", "yes"}
RISK_N_PATHS = int(os.environ.get("QUANTINEL_N_PATHS", "1000" if _FAST else "10000"))
N_DAYS = int(os.environ.get("QUANTINEL_N_DAYS", "252" if _FAST else "504"))
REBALANCE_EVERY = int(os.environ.get("QUANTINEL_REBALANCE_EVERY", "20" if _FAST else "5"))


@dataclass
class PipelineCache:
    payload: dict[str, Any] | None = None
    updated_at: float = 0.0
    running: bool = False
    error: str | None = None
    logs: list[str] = field(default_factory=list)


_cache = PipelineCache()
_lock = threading.Lock()


def _log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    _cache.logs.append(line)


def _scorecard_dict(scorecard) -> dict[str, Any]:
    return {
        "sharpe": float(scorecard.sharpe),
        "total_return": float(scorecard.total_return),
        "directional_accuracy": float(scorecard.directional_accuracy),
        "information_coefficient": float(scorecard.information_coefficient),
        "vs_baseline_sharpe": float(scorecard.vs_baseline_sharpe),
        "equity_curve": {"values": [float(v) for v in scorecard.equity_curve.tolist()]},
    }


def _records_dict(records) -> list[dict[str, Any]]:
    return [
        {
            "as_of": str(r.as_of.date()),
            "weights": {t: float(w) for t, w in r.weights.items()},
        }
        for r in records
    ]


def _risk_panel(last_record, risk_report) -> dict[str, Any]:
    rm = last_record.risk_model
    var_95: dict[str, float] = {}
    cvar_95: dict[str, float] = {}
    disagreement = 0.0
    models: list[str] = []
    breach_rates: dict[str, float] = {}

    if rm is not None:
        var_95 = {t: float(v) for t, v in rm.var_95.items()}
        cvar_95 = {t: float(v) for t, v in rm.cvar_95.items()}
        disagreement = float(rm.disagreement)
        models = [sa.agent_label for sa in rm.sub_agent_results]

    for sa in risk_report.sub_agent_reports:
        breach_rates[sa.agent_label] = float(sa.var_breach_rate)

    return {
        "var_95": var_95,
        "cvar_95": cvar_95,
        "disagreement": disagreement,
        "models": models or list(breach_rates.keys()),
        "breach_rates": breach_rates,
    }


def _stances_from_weights(weights: dict[str, float]) -> dict[str, str]:
    stances = {}
    for t, w in weights.items():
        if w > 0.01:
            stances[t] = "LONG"
        elif w < -0.01:
            stances[t] = "SHORT"
        else:
            stances[t] = "FLAT"
    return stances


def _build_payload(
    normal_result: PipelineResult,
    quantum_result: PipelineResult,
    comparison,
    forward,
    logs: list[str],
) -> dict[str, Any]:
    tickers = list(normal_result.data.tickers)
    winner_key = comparison.winner if comparison.winner in {"normal", "quantum"} else "normal"
    winner_result = quantum_result if winner_key == "quantum" else normal_result
    last = winner_result.records[-1]

    winner = comparison.winner.upper() if comparison.winner else "TIE"
    stances = _stances_from_weights(last.weights)
    for t in tickers:
        stances.setdefault(t, "FLAT")

    return {
        "tickers": tickers,
        "as_of": str(comparison.as_of.date()),
        "active_branch": winner_key,
        "decision": {
            "winner": winner,
            "confidence": forward.confidence,
            "rationale": comparison.rationale or forward.reasoning,
            "stances": {t: str(stances.get(t, "FLAT")).upper() for t in tickers},
            "gross_exposure": round(float(forward.recommended_gross_exposure), 2),
            "risk_aversion": round(float(forward.recommended_risk_aversion), 1),
            "decision_trace": list(comparison.decision_trace or []),
        },
        "intelligence": {
            "headlines": {t: list(comparison.headlines.get(t, [])) for t in tickers},
            "sentiment": {t: float(comparison.sentiment.get(t, 0.0)) for t in tickers},
            "themes": list(comparison.key_themes or []),
        },
        "risk": _risk_panel(last, winner_result.risk_report),
        "normal": {
            "records": _records_dict(normal_result.records),
            "scorecard": _scorecard_dict(normal_result.scorecard),
            "engine_diagnostics": normal_result.engine_diagnostics,
        },
        "quantum": {
            "records": _records_dict(quantum_result.records),
            "scorecard": {
                "sharpe": float(quantum_result.scorecard.sharpe),
                "total_return": float(quantum_result.scorecard.total_return),
                "directional_accuracy": float(quantum_result.scorecard.directional_accuracy),
                "equity_curve": {
                    "values": [
                        float(v)
                        for v in quantum_result.scorecard.equity_curve.tolist()
                    ],
                },
            },
            "engine_diagnostics": quantum_result.engine_diagnostics,
        },
        "execution_logs": logs,
    }


def _run_pipeline_impl() -> dict[str, Any]:
    global _cache
    _cache.logs = []
    _cache.error = None
    _cache.running = True

    try:
        _log(
            f"Starting pipeline (n_days={N_DAYS}, rebalance={REBALANCE_EVERY}, "
            f"risk_paths={RISK_N_PATHS})"
        )

        normal_job = (
            "normal",
            "MomentumForecaster + MeanVarianceOptimizer",
            MomentumForecaster(),
            MeanVarianceOptimizer(),
        )
        quantum_job = (
            "quantum",
            "QuantumForecaster + QaoaOptimizer",
            QuantumForecaster(api_key=XPYQ_KEY, timeout=XPYQ_TIMEOUT),
            QaoaOptimizer(api_key=XPYQ_KEY, timeout=XPYQ_TIMEOUT),
        )

        run_kw = {
            "n_days": N_DAYS,
            "rebalance_every": REBALANCE_EVERY,
            "risk_n_paths": RISK_N_PATHS,
        }
        news_source = make_news_source(EXA_KEY)
        if EXA_KEY:
            _log("Using Exa API for news")
        else:
            _log("EXA_KEY missing — using MockNewsSource")

        def _run_named(job):
            name, label, forecaster, optimizer = job
            _log(f"{name} simulation started")
            result = run_pipeline(
                name, label, forecaster, optimizer, news_source=news_source, **run_kw
            )
            _log(f"{name} simulation finished ({len(result.records)} rebalances)")
            return result

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(_run_named, job) for job in (normal_job, quantum_job)]
            results = [f.result() for f in futures]

        by_name = {r.name: r for r in results}
        normal_result = by_name["normal"]
        quantum_result = by_name["quantum"]

        normal_summary = summarize(normal_result)
        quantum_summary = summarize(quantum_result)
        metric_deltas, decision_trace = build_trace(normal_summary, quantum_summary)

        _log("Building market intelligence from Exa news")
        intel = intelligence_from_news_source(
            news_source,
            normal_result.data.tickers,
            as_of=normal_result.records[-1].as_of,
        )
        _log("Intelligence ready")

        _log("Running master agent comparison")
        comparison = MasterAgent(openrouter_key=OPENROUTER_KEY).compare(
            normal_summary,
            quantum_summary,
            intel,
            metric_deltas=metric_deltas,
            decision_trace=decision_trace,
        )
        _log(f"Winner: {comparison.winner}")

        winner_result = quantum_result if comparison.winner == "quantum" else normal_result
        winner_label = (
            "QuantumForecaster + QaoaOptimizer"
            if comparison.winner == "quantum"
            else "MomentumForecaster + MeanVarianceOptimizer"
        )

        _log("Running forward pipeline agent")
        forward = PipelineAgent(openrouter_key=OPENROUTER_KEY).decide(
            winner_result.records,
            winner_result.scorecard,
            winner_result.risk_report,
            intel,
            winner_label,
        )
        _log("Pipeline complete")

        payload = _build_payload(
            normal_result,
            quantum_result,
            comparison,
            forward,
            list(_cache.logs),
        )
        for role, diag in quantum_result.engine_diagnostics.items():
            _log(
                f"Quantum {role}: xpyq_completed={diag.get('xpyq_completed', 0)} "
                f"fallbacks={diag.get('fallbacks', 0)}"
            )
        _cache.payload = payload
        _cache.updated_at = time.time()
        return payload
    except Exception as exc:
        _cache.error = str(exc)
        _log(f"ERROR: {exc}")
        raise
    finally:
        _cache.running = False


def get_pipeline_payload(*, refresh: bool = False) -> dict[str, Any]:
    """Return cached dashboard payload, running the pipeline if needed."""
    with _lock:
        if refresh or _cache.payload is None:
            return _run_pipeline_impl()
        return _cache.payload


def pipeline_status() -> dict[str, Any]:
    return {
        "ready": _cache.payload is not None,
        "running": _cache.running,
        "updated_at": _cache.updated_at,
        "error": _cache.error,
    }
