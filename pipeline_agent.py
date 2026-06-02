"""
PIPELINE AGENT  (owner: Adithya Kalidindi)

Runs ONCE after the full backtest completes. Reads the complete picture —
scorecard, risk report, all records, live market intelligence — and produces
a forward-looking ForwardDecision: what to do on the NEXT real rebalance.

This is the only place where LLM reasoning touches the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_MODEL = "anthropic/claude-haiku-4-5"


@dataclass
class ForwardDecision:
    recommended_forecaster: str  # "quantum" | "momentum"
    recommended_risk_aversion: float  # for next run
    recommended_gross_exposure: float  # for next run
    next_position: dict[str, str]  # ticker -> "LONG" | "SHORT" | "FLAT"
    confidence: str  # "HIGH" | "MEDIUM" | "LOW"
    reasoning: str  # full LLM rationale


class PipelineAgent:
    """
    Post-run agent. Call decide() once after backtest + score + intelligence.
    Returns a ForwardDecision with actionable recommendations for the next period.
    """

    def __init__(self, openrouter_key: str):
        self.openrouter_key = openrouter_key

    def decide(
        self, records, scorecard, risk_report, intelligence, forecaster_label: str
    ) -> ForwardDecision:
        context = self._build_context(
            records, scorecard, risk_report, intelligence, forecaster_label
        )
        raw = self._call_llm(context)
        return self._parse(raw)

    # ------------------------------------------------------------------

    def _build_context(
        self, records, scorecard, risk_report, intelligence, forecaster_label
    ) -> str:
        # Equity curve trend — last 20 rebalances vs first 20
        equity = scorecard.equity_curve
        early = float(equity.iloc[:20].mean()) if len(equity) >= 20 else 1.0
        late = (
            float(equity.iloc[-20:].mean())
            if len(equity) >= 20
            else float(equity.mean())
        )
        trend = "improving" if late > early else "deteriorating"

        # Recent IC trend (last 20 records)
        recent = records[-20:]
        ic_vals = []
        for r in recent:
            if r.forecast:
                for t in r.forward_returns:
                    ic_vals.append(
                        (r.forecast.expected_returns.get(t, 0), r.forward_returns[t])
                    )
        recent_ic = 0.0
        if len(ic_vals) > 1:
            pred, real = zip(*ic_vals)
            recent_ic = (
                float(np.corrcoef(pred, real)[0, 1]) if np.std(pred) > 0 else 0.0
            )

        # Last position
        last = records[-1]
        last_weights = {t: round(w, 3) for t, w in last.weights.items()}

        # Risk sub-agent breach rates
        breach_lines = "\n".join(
            f"  {sa.agent_label}: breach rate {sa.var_breach_rate:.3f}"
            for sa in risk_report.sub_agent_reports
        )

        # Intelligence
        sent = intelligence.sentiment
        heads = {t: intelligence.headlines.get(t, [])[:2] for t in sent}

        return f"""You are reviewing a completed trading test and deciding what to do next. Explain your decision in simple terms for a non-technical teammate. Keep the JSON fields exactly as requested, but make the "reasoning" value plain English. Avoid jargon like "IC", "Sharpe", "VaR", "conviction", "signal degradation", or "model disagreement" unless you immediately explain it in simple words.

COMPLETED BACKTEST SUMMARY
Forecaster used: {forecaster_label}
Total rebalances: {len(records)}

Full-run performance:
  Sharpe:               {scorecard.sharpe:.2f}
  Total return:         {scorecard.total_return * 100:.1f}%
  Directional accuracy: {scorecard.directional_accuracy * 100:.1f}%
  IC (full run):        {scorecard.information_coefficient:.3f}
  IC (last 20 steps):   {recent_ic:.3f}
  vs 50/50 hold:        {scorecard.vs_baseline_sharpe:+.2f}
  Equity curve trend:   {trend} (early avg {early:.3f} → late avg {late:.3f})

Risk model:
  VaR breaches: {risk_report.var_breaches} / {len(records)}
  Avg disagreement: {risk_report.avg_disagreement:.3f}
  Max disagreement: {risk_report.max_disagreement:.3f}
{breach_lines}

Last realized position: {last_weights}

LIVE MARKET INTELLIGENCE (from Exa)
Sentiment: {", ".join(f"{t}: {v:+.2f}" for t, v in sent.items())}
Key themes: {", ".join(intelligence.key_themes)}
Headlines:
{chr(10).join(f"  {t}: {h[:80]}" for t, hs in heads.items() for h in hs)}

DECISION REQUIRED
Based on the above, decide for the NEXT real rebalance:
1. Which forecaster to use: "quantum" (SVD factor, better IC) or "momentum" (simpler, faster)
2. risk_aversion (2-20): how conservative to be
3. gross_exposure (0.3-1.5): total portfolio leverage
4. Position per ticker: "LONG", "SHORT", or "FLAT"
5. Confidence: "HIGH", "MEDIUM", or "LOW"

Rules:
- If recent IC > full-run IC: signal is strengthening → use quantum, lower risk_aversion
- If recent IC < 0 and disagreement > 0.5: stay defensive
- If sentiment and forecast direction agree: higher conviction → raise gross_exposure
- GBM breach rate > 0.2 means GBM is overconfident → raise risk_aversion

Respond ONLY with valid JSON, no other text:
{{
  "recommended_forecaster": "quantum" | "momentum",
  "recommended_risk_aversion": <float>,
  "recommended_gross_exposure": <float>,
  "next_position": {{"NVDA": "LONG"|"SHORT"|"FLAT", "GOOG": "LONG"|"SHORT"|"FLAT"}},
  "confidence": "HIGH"|"MEDIUM"|"LOW",
  "reasoning": "<2-3 short sentences in simple terms explaining the decision>"
}}"""

    def _call_llm(self, prompt: str) -> str:
        if not self.openrouter_key:
            return ""
        import requests

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
                    "max_tokens": 500,
                    "temperature": 0.1,
                },
                timeout=20,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            import sys

            print(
                f"[pipeline_agent] LLM call failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return ""

    def _parse(self, raw: str) -> ForwardDecision:
        try:
            # extract the first {...} block regardless of markdown fencing
            start = raw.index("{")
            end = raw.rindex("}") + 1
            text = raw[start:end]
            d = json.loads(text)
            return ForwardDecision(
                recommended_forecaster=str(d.get("recommended_forecaster", "momentum")),
                recommended_risk_aversion=float(
                    np.clip(d.get("recommended_risk_aversion", 8.0), 2.0, 20.0)
                ),
                recommended_gross_exposure=float(
                    np.clip(d.get("recommended_gross_exposure", 1.0), 0.3, 1.5)
                ),
                next_position=dict(d.get("next_position", {})),
                confidence=str(d.get("confidence", "MEDIUM")),
                reasoning=str(d.get("reasoning", "")),
            )
        except Exception:
            return ForwardDecision(
                recommended_forecaster="momentum",
                recommended_risk_aversion=8.0,
                recommended_gross_exposure=1.0,
                next_position={},
                confidence="LOW",
                reasoning="The agent response could not be read, so the safer choice is to use the simpler forecast and keep risk at a normal level.",
            )
