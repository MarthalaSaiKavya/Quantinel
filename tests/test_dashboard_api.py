"""Tests for dashboard API payload shape."""

from pipeline_service import _build_payload
from pipeline_agent import ForwardDecision


class _FakeComparison:
    winner = "normal"
    as_of = __import__("pandas").Timestamp("2024-06-01")
    rationale = "Normal beat quantum on risk-adjusted return."
    decision_trace = ["Normal Sharpe was higher.", "Quantum had more fallbacks."]
    headlines = {"NVDA": ["AI demand strong"], "GOOG": ["Cloud growth steady"]}
    sentiment = {"NVDA": 0.4, "GOOG": 0.1}
    key_themes = ["AI", "cloud"]


class _FakeScorecard:
    sharpe = 1.2
    total_return = 0.15
    directional_accuracy = 0.55
    information_coefficient = 0.08
    vs_baseline_sharpe = 0.3

    class _Eq:
        def tolist(self):
            return [1.0, 1.02, 1.05, 1.08]

    equity_curve = _Eq()


class _FakeRiskModel:
    var_95 = {"NVDA": -0.03, "GOOG": -0.02}
    cvar_95 = {"NVDA": -0.05, "GOOG": -0.04}
    disagreement = 0.25

    sub_agent_results = [
        type("SA", (), {"agent_label": "gbm"})(),
        type("SA", (), {"agent_label": "markov"})(),
        type("SA", (), {"agent_label": "bootstrap"})(),
    ]


class _FakeRecord:
    as_of = __import__("pandas").Timestamp("2024-06-01")
    weights = {"NVDA": 0.3, "GOOG": -0.3}
    risk_model = _FakeRiskModel()


class _FakeRiskReport:
    sub_agent_reports = [
        type("R", (), {"agent_label": "gbm", "var_breach_rate": 0.04})(),
        type("R", (), {"agent_label": "markov", "var_breach_rate": 0.06})(),
        type("R", (), {"agent_label": "bootstrap", "var_breach_rate": 0.05})(),
    ]


class _FakeResult:
    def __init__(self, name):
        self.name = name
        self.data = type("D", (), {"tickers": ["NVDA", "GOOG"]})()
        self.records = [_FakeRecord()]
        self.scorecard = _FakeScorecard()
        self.risk_report = _FakeRiskReport()
        self.engine_diagnostics = {}


def test_dashboard_payload_shape():
    forward = ForwardDecision(
        recommended_forecaster="momentum",
        recommended_risk_aversion=8.0,
        recommended_gross_exposure=1.0,
        next_position={"NVDA": "LONG", "GOOG": "SHORT"},
        confidence="MEDIUM",
        reasoning="Stay with momentum.",
    )
    payload = _build_payload(
        _FakeResult("normal"),
        _FakeResult("quantum"),
        _FakeComparison(),
        forward,
        ["log line 1"],
    )

    assert payload["tickers"] == ["NVDA", "GOOG"]
    assert payload["decision"]["winner"] == "NORMAL"
    assert payload["decision"]["confidence"] == "MEDIUM"
    assert payload["decision"]["stances"]["NVDA"] == "LONG"
    assert payload["intelligence"]["themes"] == ["AI", "cloud"]
    assert "NVDA" in payload["risk"]["var_95"]
    assert payload["normal"]["records"][-1]["weights"]["NVDA"] == 0.3
    assert len(payload["normal"]["scorecard"]["equity_curve"]["values"]) == 4
    assert payload["quantum"]["scorecard"]["sharpe"] == 1.2
    assert payload["execution_logs"] == ["log line 1"]
