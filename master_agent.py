"""
MASTER AGENT  (owner: Adithya Kalidindi)

Compiles all pipeline layer outputs + market intelligence into a MasterReport.
Uses an LLM (via OpenRouter) to synthesise quantitative signals with live
news context into a concise trading rationale.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import requests

from contracts import MarketIntelligence, Scorecard

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_MODEL = "anthropic/claude-sonnet-4-5"


@dataclass
class MasterReport:
    as_of: pd.Timestamp
    # quant pipeline
    scorecard: Scorecard
    last_weights: dict[str, float]
    last_forecast: dict[str, float]
    last_risk_vols: dict[str, float]
    # intelligence
    sentiment: dict[str, float]
    headlines: dict[str, list[str]]
    key_themes: list[str]
    # synthesis
    conviction: dict[str, float]
    recommendation: dict[str, str]
    rationale: str


@dataclass
class ComparisonReport:
    as_of: pd.Timestamp
    winner: str
    recommendation: str
    rationale: str
    metric_deltas: dict
    decision_trace: list[str]
    sentiment: dict[str, float]
    headlines: dict[str, list[str]]
    key_themes: list[str]


class MasterAgent:
    """Compiles all layer outputs into a MasterReport with LLM-generated rationale."""

    def __init__(self, openrouter_key: str):
        self.openrouter_key = openrouter_key

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _conviction(
        self,
        tickers: list[str],
        forecast: dict[str, float],
        sentiment: dict[str, float],
        vols: dict[str, float],
    ) -> dict[str, float]:
        result = {}
        for t in tickers:
            exp = abs(forecast.get(t, 0.0))
            vol = vols.get(t, 1.0) + 1e-8
            quant_signal = min(1.0, exp / vol)
            news_signal = abs(sentiment.get(t, 0.0))
            result[t] = round(min(1.0, (quant_signal + news_signal) / 2), 3)
        return result

    def _recommendation(
        self,
        tickers: list[str],
        forecast: dict[str, float],
        sentiment: dict[str, float],
    ) -> dict[str, str]:
        recs = {}
        for t in tickers:
            f_sign = 1 if forecast.get(t, 0.0) >= 0 else -1
            s_sign = 1 if sentiment.get(t, 0.0) >= 0 else -1
            if f_sign > 0 and s_sign > 0:
                recs[t] = "LONG"
            elif f_sign < 0 and s_sign < 0:
                recs[t] = "SHORT"
            else:
                recs[t] = "FLAT"
        return recs

    def _llm_rationale(self, context: dict) -> str:
        prompt = f"""Explain these trading-pipeline results in simple terms for a non-technical teammate. Write 3-4 short sentences. Keep the important numbers, but avoid jargon like "IC", "Sharpe", "conviction", "alpha", or "model degradation" unless you immediately explain what it means. Do not use bullet points.

Pipeline results:
- Rebalances: {context['n_rebalances']}
- Sharpe: {context['sharpe']:.2f}
- Total return: {context['total_return']:.1f}%
- Directional accuracy: {context['dir_acc']:.1f}%
- IC: {context['ic']:.2f}
- Sharpe vs 50/50: {context['vs_baseline']:+.2f}

Last positions: {context['weights']}
Forecast expected returns: {context['forecast']}

Market intelligence:
- NVDA sentiment: {context['nvda_sent']:+.2f}
- GOOG sentiment: {context['goog_sent']:+.2f}
- Key themes: {', '.join(context['themes'])}
- NVDA headlines: {context['nvda_headlines']}
- GOOG headlines: {context['goog_headlines']}

Recommendations: {context['recommendations']}
Conviction scores: {context['conviction']}

Write the simple explanation:"""

        try:
            resp = requests.post(
                _OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": _MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            sharpe = context["sharpe"]
            ret = context["total_return"]
            da = context["dir_acc"]
            nvda = context["nvda_sent"]
            goog = context["goog_sent"]
            themes = ", ".join(context["themes"]) or "N/A"
            return (
                f"The pipeline ran {context['n_rebalances']} trading checks and ended with a "
                f"{ret:.1f}% return. Its risk-adjusted score was {sharpe:.2f}, and it picked the "
                f"right direction {da:.0f}% of the time. News looks {nvda:+.2f} for NVDA and "
                f"{goog:+.2f} for GOOG, with themes around {themes}."
            )

    def _llm_comparison(self, context: dict) -> dict[str, str]:
        prompt = f"""You are the final decision agent. Compare these two trading simulations in simple terms for a non-technical teammate.

Choose which setup should be used next:
- "normal" if the normal pipeline is better
- "quantum" if the quantum pipeline is better
- "tie" if neither clearly wins

Use plain English. Keep important numbers, but avoid jargon unless you explain it immediately.

Normal simulation:
{json.dumps(context['normal'], indent=2)}

Quantum simulation:
{json.dumps(context['quantum'], indent=2)}

Market intelligence:
- Sentiment: {context['sentiment']}
- Key themes: {context['themes']}
- Headlines: {context['headlines']}

System-computed comparison facts:
{json.dumps(context['metric_deltas'], indent=2)}

Decision trace to consider:
{json.dumps(context['decision_trace'], indent=2)}

Respond ONLY with valid JSON:
{{
  "winner": "normal" | "quantum" | "tie",
  "recommendation": "<one short sentence saying what to run next>",
  "rationale": "<4-6 short sentences explaining the comparison in simple terms>",
  "decision_trace": ["<short evidence item>", "<short evidence item>", "<short evidence item>"]
}}"""

        try:
            resp = requests.post(
                _OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": _MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 450,
                    "temperature": 0.1,
                },
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            start = raw.index("{")
            end = raw.rindex("}") + 1
            parsed = json.loads(raw[start:end])
            return {
                "winner": str(parsed.get("winner", "tie")),
                "recommendation": str(parsed.get("recommendation", "")),
                "rationale": str(parsed.get("rationale", "")),
                "decision_trace": [
                    str(item) for item in parsed.get("decision_trace", [])[:6]
                ],
            }
        except Exception:
            normal = context["normal"]
            quantum = context["quantum"]
            if quantum["sharpe"] > normal["sharpe"]:
                winner = "quantum"
            elif normal["sharpe"] > quantum["sharpe"]:
                winner = "normal"
            else:
                winner = "tie"

            return {
                "winner": winner,
                "recommendation": f"Use {winner} next." if winner != "tie" else "Treat this as a tie and keep testing.",
                "rationale": (
                    f"Normal returned {normal['total_return_pct']:.1f}% with a risk-adjusted score of "
                    f"{normal['sharpe']:.2f}. Quantum returned {quantum['total_return_pct']:.1f}% with a "
                    f"risk-adjusted score of {quantum['sharpe']:.2f}. The better choice is {winner} based "
                    f"on the side-by-side score."
                ),
                "decision_trace": list(context["decision_trace"]),
            }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compile(
        self,
        records,
        scorecard: Scorecard,
        intelligence: MarketIntelligence,
        n_rebalances: int | None = None,
    ) -> MasterReport:
        last = records[-1]
        as_of = pd.Timestamp(last.as_of)
        tickers = list(last.weights.keys())

        last_weights = dict(last.weights)
        last_forecast = (
            dict(last.forecast.expected_returns)
            if last.forecast is not None
            else {t: 0.0 for t in tickers}
        )
        # use |weight| as a vol proxy when no explicit risk model is passed
        last_risk_vols = {t: abs(last_weights.get(t, 0.0)) + 1e-4 for t in tickers}

        conviction = self._conviction(tickers, last_forecast, intelligence.sentiment, last_risk_vols)
        recommendation = self._recommendation(tickers, last_forecast, intelligence.sentiment)

        context = {
            "n_rebalances": n_rebalances or len(records),
            "sharpe": scorecard.sharpe,
            "total_return": scorecard.total_return * 100,
            "dir_acc": scorecard.directional_accuracy * 100,
            "ic": scorecard.information_coefficient,
            "vs_baseline": scorecard.vs_baseline_sharpe,
            "weights": {t: round(v, 3) for t, v in last_weights.items()},
            "forecast": {t: round(v, 4) for t, v in last_forecast.items()},
            "nvda_sent": intelligence.sentiment.get("NVDA", 0.0),
            "goog_sent": intelligence.sentiment.get("GOOG", 0.0),
            "themes": intelligence.key_themes,
            "nvda_headlines": intelligence.headlines.get("NVDA", [])[:2],
            "goog_headlines": intelligence.headlines.get("GOOG", [])[:2],
            "recommendations": recommendation,
            "conviction": conviction,
        }

        rationale = self._llm_rationale(context)

        return MasterReport(
            as_of=as_of,
            scorecard=scorecard,
            last_weights=last_weights,
            last_forecast=last_forecast,
            last_risk_vols=last_risk_vols,
            sentiment=dict(intelligence.sentiment),
            headlines=dict(intelligence.headlines),
            key_themes=list(intelligence.key_themes),
            conviction=conviction,
            recommendation=recommendation,
            rationale=rationale,
        )

    def compare(
        self,
        normal: dict,
        quantum: dict,
        intelligence: MarketIntelligence,
        metric_deltas: dict | None = None,
        decision_trace: list[str] | None = None,
    ) -> ComparisonReport:
        context = {
            "normal": normal,
            "quantum": quantum,
            "metric_deltas": metric_deltas or {},
            "decision_trace": decision_trace or [],
            "sentiment": intelligence.sentiment,
            "themes": intelligence.key_themes,
            "headlines": {
                ticker: intelligence.headlines.get(ticker, [])[:2]
                for ticker in set(normal.get("last_weights", {})) | set(quantum.get("last_weights", {}))
            },
        }
        result = self._llm_comparison(context)
        as_of = pd.Timestamp(quantum.get("as_of") or normal.get("as_of"))

        return ComparisonReport(
            as_of=as_of,
            winner=result["winner"],
            recommendation=result["recommendation"],
            rationale=result["rationale"],
            metric_deltas=dict(metric_deltas or {}),
            decision_trace=list(result.get("decision_trace") or decision_trace or []),
            sentiment=dict(intelligence.sentiment),
            headlines=dict(intelligence.headlines),
            key_themes=list(intelligence.key_themes),
        )
