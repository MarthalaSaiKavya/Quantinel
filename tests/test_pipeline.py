"""TDD tests for the Quantinel pipeline — news, multi-agent risk, VaR/CVaR."""

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.contracts import (
    Forecast,
    MarketData,
    NewsArticle,
    NewsFeed,
    RiskModel,
)
from quant_pipeline.data import MockDataSource
from quant_pipeline.forecast import MomentumForecaster
from quant_pipeline.risk import SampleCovRisk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_news(
    tickers: list[str], as_of: pd.Timestamp, sentiment: float = 0.0
) -> NewsFeed:
    """Build a minimal NewsFeed with one article per ticker."""
    articles = [
        NewsArticle(
            ticker=t,
            title=f"Fake news about {t}",
            snippet="...",
            url=f"https://example.com/{t}",
            published_date=as_of,
            sentiment_score=sentiment,
        )
        for t in tickers
    ]
    return NewsFeed(as_of=as_of, articles=articles)


def _make_data_and_forecast(seed: int = 42, lookback: int = 60, horizon_days: int = 5):
    data = MockDataSource(seed=seed).load()
    as_of = data.bars["NVDA"].index[-1]
    window = data.slice_until(as_of)
    forecast = MomentumForecaster(lookback=lookback).predict(
        window,
        as_of,
        horizon_days,
    )
    return window, forecast, as_of


# ============================================================================
# SLICE 1: GBM sub-agent produces VaR/CVaR in RiskModel
# ============================================================================


def test_gbm_sub_agent_produces_var_cvar():
    """SampleCovRisk returns var_95 and cvar_95 for each ticker from GBM."""
    window, forecast, as_of = _make_data_and_forecast()
    news = _make_mock_news(window.tickers, as_of)

    risk = SampleCovRisk().estimate(window, news, forecast, as_of)

    assert isinstance(risk, RiskModel)
    assert "NVDA" in risk.var_95
    assert "GOOG" in risk.var_95
    assert "NVDA" in risk.cvar_95
    assert "GOOG" in risk.cvar_95

    # VaR and CVaR are losses — should be negative
    assert risk.var_95["NVDA"] < 0
    assert risk.var_95["GOOG"] < 0
    # CVaR is more extreme than VaR (more negative, or equal)
    assert risk.cvar_95["NVDA"] <= risk.var_95["NVDA"]
    assert risk.cvar_95["GOOG"] <= risk.var_95["GOOG"]


def test_gbm_var_cvar_deterministic_with_seed():
    """Same seed produces identical VaR/CVaR (reproducibility)."""
    window, forecast, as_of = _make_data_and_forecast(seed=99)
    news = _make_mock_news(window.tickers, as_of, sentiment=0.3)

    r1 = SampleCovRisk(n_paths=5000).estimate(window, news, forecast, as_of)
    r2 = SampleCovRisk(n_paths=5000).estimate(window, news, forecast, as_of)

    assert r1.var_95 == r2.var_95
    assert r1.cvar_95 == r2.cvar_95


# ============================================================================
# SLICE 2: Ensemble of 3 sub-agents with disagreement
# ============================================================================


def test_ensemble_three_sub_agents():
    """RiskModel carries per-agent breakdown from all 3 sub-agents."""
    window, forecast, as_of = _make_data_and_forecast()
    news = _make_mock_news(window.tickers, as_of)

    risk = SampleCovRisk().estimate(window, news, forecast, as_of)

    assert len(risk.sub_agent_results) == 3
    labels = {r.agent_label for r in risk.sub_agent_results}
    assert labels == {"gbm", "markov", "bootstrap"}


def test_ensemble_disagreement_bounded():
    """Disagreement is a scalar in [0, 1]."""
    window, forecast, as_of = _make_data_and_forecast()
    news = _make_mock_news(window.tickers, as_of)

    risk = SampleCovRisk().estimate(window, news, forecast, as_of)

    assert 0.0 <= risk.disagreement <= 1.0


def test_ensemble_cvar_more_extreme_than_var():
    """Aggregated CVaR is more extreme (more negative) than any sub-agent's VaR."""
    window, forecast, as_of = _make_data_and_forecast()
    news = _make_mock_news(window.tickers, as_of)

    risk = SampleCovRisk().estimate(window, news, forecast, as_of)

    for t in window.tickers:
        assert risk.cvar_95[t] <= risk.var_95[t]
        # Check: worst CVaR means aggregated CVaR ≤ any sub-agent's CVaR
        for r in risk.sub_agent_results:
            assert risk.cvar_95[t] <= r.cvar_95[t]


# ============================================================================
# SLICE 3: News sentiment adjusts Markov transitions
# ============================================================================


def test_negative_sentiment_increases_markov_risk():
    """Negative news sentiment makes Markov VaR more negative (riskier)."""
    window, forecast, as_of = _make_data_and_forecast(seed=42)

    neutral = _make_mock_news(window.tickers, as_of, sentiment=0.0)
    negative = _make_mock_news(window.tickers, as_of, sentiment=-0.8)

    r_neutral = SampleCovRisk(n_paths=5000).estimate(
        window,
        neutral,
        forecast,
        as_of,
    )
    r_negative = SampleCovRisk(n_paths=5000).estimate(
        window,
        negative,
        forecast,
        as_of,
    )

    mv_neutral = [r for r in r_neutral.sub_agent_results if r.agent_label == "markov"][
        0
    ]
    mv_negative = [
        r for r in r_negative.sub_agent_results if r.agent_label == "markov"
    ][0]

    for t in window.tickers:
        assert mv_negative.var_95[t] <= mv_neutral.var_95[t], (
            f"{t}: negative={mv_negative.var_95[t]}, neutral={mv_neutral.var_95[t]}"
        )


def test_sentiment_does_not_affect_gbm_or_bootstrap():
    """GBM and bootstrap outputs are identical regardless of sentiment."""
    window, forecast, as_of = _make_data_and_forecast(seed=42)

    neutral = _make_mock_news(window.tickers, as_of, sentiment=0.0)
    negative = _make_mock_news(window.tickers, as_of, sentiment=-0.8)

    r_neutral = SampleCovRisk(n_paths=5000).estimate(
        window,
        neutral,
        forecast,
        as_of,
    )
    r_negative = SampleCovRisk(n_paths=5000).estimate(
        window,
        negative,
        forecast,
        as_of,
    )

    gbm_n = [r for r in r_neutral.sub_agent_results if r.agent_label == "gbm"][0]
    gbm_neg = [r for r in r_negative.sub_agent_results if r.agent_label == "gbm"][0]
    bs_n = [r for r in r_neutral.sub_agent_results if r.agent_label == "bootstrap"][0]
    bs_neg = [r for r in r_negative.sub_agent_results if r.agent_label == "bootstrap"][
        0
    ]

    assert gbm_n.var_95 == gbm_neg.var_95
    assert bs_n.var_95 == bs_neg.var_95


# ============================================================================
# SLICE 4: Optimizer shrinks positions by disagreement
# ============================================================================


def test_disagreement_shrinks_mean_variance_positions():
    """High disagreement produces smaller gross exposure in MeanVarianceOptimizer."""
    window, forecast, as_of = _make_data_and_forecast(seed=42)
    news = _make_mock_news(window.tickers, as_of)

    risk_real = SampleCovRisk(n_paths=3000).estimate(window, news, forecast, as_of)

    # Build low- and high-disagreement RiskModels with same cov/vol
    risk_low = RiskModel(
        as_of=risk_real.as_of,
        cov=risk_real.cov,
        vol=risk_real.vol,
        var_95=risk_real.var_95,
        cvar_95=risk_real.cvar_95,
        sub_agent_results=risk_real.sub_agent_results,
        disagreement=0.0,
    )
    risk_high = RiskModel(
        as_of=risk_real.as_of,
        cov=risk_real.cov,
        vol=risk_real.vol,
        var_95=risk_real.var_95,
        cvar_95=risk_real.cvar_95,
        sub_agent_results=risk_real.sub_agent_results,
        disagreement=0.9,
    )

    from quant_pipeline.optimize import MeanVarianceOptimizer

    opt = MeanVarianceOptimizer()

    target_low = opt.solve(forecast, risk_low)
    target_high = opt.solve(forecast, risk_high)

    # High disagreement → smaller positions
    assert target_high.gross < target_low.gross, (
        f"high disagreement gross={target_high.gross}, "
        f"low disagreement gross={target_low.gross}"
    )


def test_disagreement_shrinks_discrete_qubo_positions():
    """High disagreement produces smaller gross exposure in DiscreteQuboOptimizer."""
    window, forecast, as_of = _make_data_and_forecast(seed=42)
    news = _make_mock_news(window.tickers, as_of)

    risk_real = SampleCovRisk(n_paths=3000).estimate(window, news, forecast, as_of)

    risk_low = RiskModel(
        as_of=risk_real.as_of,
        cov=risk_real.cov,
        vol=risk_real.vol,
        var_95=risk_real.var_95,
        cvar_95=risk_real.cvar_95,
        sub_agent_results=risk_real.sub_agent_results,
        disagreement=0.0,
    )
    risk_high = RiskModel(
        as_of=risk_real.as_of,
        cov=risk_real.cov,
        vol=risk_real.vol,
        var_95=risk_real.var_95,
        cvar_95=risk_real.cvar_95,
        sub_agent_results=risk_real.sub_agent_results,
        disagreement=0.9,
    )

    from quant_pipeline.optimize import DiscreteQuboOptimizer

    opt = DiscreteQuboOptimizer()

    target_low = opt.solve(forecast, risk_low)
    target_high = opt.solve(forecast, risk_high)

    assert target_high.gross < target_low.gross, (
        f"high disagreement gross={target_high.gross}, "
        f"low disagreement gross={target_low.gross}"
    )


# ============================================================================
# SLICE 5: Backtest wires news and stores RiskModel on StepRecord
# ============================================================================


def test_backtest_stores_risk_model_on_steps():
    """StepRecord carries RiskModel after backtest run with news source."""
    from quant_pipeline.backtest import Backtest
    from quant_pipeline.data import MockDataSource
    from quant_pipeline.execute import PaperExecutor
    from quant_pipeline.forecast import MomentumForecaster
    from quant_pipeline.news import MockNewsSource
    from quant_pipeline.optimize import MeanVarianceOptimizer
    from quant_pipeline.risk import SampleCovRisk

    bt = Backtest(
        source=MockDataSource(seed=99),
        news_source=MockNewsSource(seed=77),
        forecaster=MomentumForecaster(),
        risk=SampleCovRisk(),
        optimizer=MeanVarianceOptimizer(),
        executor=PaperExecutor(),
        rebalance_every=20,
        lookback_days=60,
        horizon_days=5,
    )
    _, records, _ = bt.run()

    assert len(records) > 0
    for r in records:
        assert r.risk_model is not None, f"StepRecord at {r.as_of} lacks risk_model"
        assert "NVDA" in r.risk_model.var_95
        assert "GOOG" in r.risk_model.var_95


# ============================================================================
# SLICE 6: RiskReport computes VaR breaches
# ============================================================================


def test_risk_report_computes_var_breaches():
    """RiskScorer counts periods where actual return exceeded VaR."""
    from quant_pipeline.backtest import StepRecord
    from quant_pipeline.contracts import (
        Forecast,
        PerSubAgentRisk,
        RiskModel,
        RiskReport,
        SubAgentReport,
    )
    from quant_pipeline.score import RiskScorer

    tickers = ["NVDA", "GOOG"]
    as_of = pd.Timestamp("2024-01-05")

    # Scenario: NVDA forward return is -3%, VaR was -2% (breach!)
    #           GOOG forward return is +1%, VaR was -2% (no breach)
    risk1 = RiskModel(
        as_of=as_of,
        cov=pd.DataFrame(
            {"NVDA": [0.1, 0.0], "GOOG": [0.0, 0.1]}, index=["NVDA", "GOOG"]
        ),
        vol={"NVDA": 0.3, "GOOG": 0.2},
        var_95={"NVDA": -0.02, "GOOG": -0.02},
        cvar_95={"NVDA": -0.04, "GOOG": -0.03},
        sub_agent_results=[
            PerSubAgentRisk(
                "gbm", {"NVDA": -0.02, "GOOG": -0.02}, {"NVDA": -0.04, "GOOG": -0.03}
            ),
        ],
        disagreement=0.1,
    )

    rec1 = StepRecord(
        as_of=as_of,
        weights={"NVDA": 0.5, "GOOG": 0.5},
        forward_returns={"NVDA": -0.03, "GOOG": 0.01},
        forecast=Forecast(
            as_of=as_of, horizon_days=5, expected_returns={"NVDA": 0.01, "GOOG": 0.01}
        ),
        risk_model=risk1,
    )

    # Second record: no breach (returns stay within VaR)
    risk2 = RiskModel(
        as_of=as_of + pd.Timedelta(days=5),
        cov=pd.DataFrame(
            {"NVDA": [0.1, 0.0], "GOOG": [0.0, 0.1]}, index=["NVDA", "GOOG"]
        ),
        vol={"NVDA": 0.3, "GOOG": 0.2},
        var_95={"NVDA": -0.02, "GOOG": -0.02},
        cvar_95={"NVDA": -0.04, "GOOG": -0.03},
        sub_agent_results=[
            PerSubAgentRisk(
                "gbm", {"NVDA": -0.02, "GOOG": -0.02}, {"NVDA": -0.04, "GOOG": -0.03}
            ),
        ],
        disagreement=0.3,
    )

    rec2 = StepRecord(
        as_of=as_of + pd.Timedelta(days=5),
        weights={"NVDA": 0.5, "GOOG": 0.5},
        forward_returns={"NVDA": -0.01, "GOOG": 0.005},
        forecast=Forecast(
            as_of=as_of + pd.Timedelta(days=5),
            horizon_days=5,
            expected_returns={"NVDA": 0.01, "GOOG": 0.01},
        ),
        risk_model=risk2,
    )

    report = RiskScorer().score([rec1, rec2])

    assert isinstance(report, RiskReport)
    assert report.var_breaches == 1  # Only NVDA in rec1 breached
    assert report.avg_disagreement == pytest.approx(0.2)
    assert report.max_disagreement == 0.3
    assert len(report.sub_agent_reports) == 1  # gbm
    assert report.sub_agent_reports[0].agent_label == "gbm"
    assert report.sub_agent_reports[0].var_breach_rate == 0.25  # 1 breach / 4 checks


# ============================================================================
# SLICE 7: End-to-end baseline runs and prints both reports
# ============================================================================

def test_baseline_runs_with_news_and_risk_report(capsys):
    """Baseline pipeline runs without error, prints Scorecard and RiskReport."""
    from quant_pipeline.backtest import Backtest
    from quant_pipeline.data import MockDataSource
    from quant_pipeline.execute import PaperExecutor
    from quant_pipeline.forecast import MomentumForecaster
    from quant_pipeline.news import MockNewsSource
    from quant_pipeline.optimize import MeanVarianceOptimizer
    from quant_pipeline.risk import SampleCovRisk
    from quant_pipeline.score import BacktestScorer, RiskScorer

    bt = Backtest(
        source=MockDataSource(seed=42, n_days=252),
        news_source=MockNewsSource(seed=7),
        forecaster=MomentumForecaster(),
        risk=SampleCovRisk(n_paths=1000),
        optimizer=MeanVarianceOptimizer(),
        executor=PaperExecutor(),
        rebalance_every=20,
        lookback_days=60,
        horizon_days=5,
    )
    data, records, baseline = bt.run()

    card = BacktestScorer().score(records, baseline)
    report = RiskScorer().score(records)

    assert card.sharpe is not None
    assert report.var_breaches >= 0
    assert isinstance(report.avg_disagreement, float)
    assert len(report.sub_agent_reports) == 3  # gbm, markov, bootstrap

    # Printed output
    print("=" * 52)
    print("  NVDA & GOOG pair  —  BASELINE (no quantum)")
    print("=" * 52)
    print(f"  Sharpe                 : {card.sharpe:6.2f}")
    print(f"  total return           : {card.total_return * 100:6.2f}%")
    print(f"  directional accuracy   : {card.directional_accuracy * 100:6.1f}%")
    print(f"  information coeff (IC) : {card.information_coefficient:6.2f}")
    print(f"  Sharpe vs 50/50 hold   : {card.vs_baseline_sharpe:+6.2f}")
    print("=" * 52)
    print("  RISK REPORT")
    print("=" * 52)
    print(f"  VaR breaches           : {report.var_breaches}")
    print(f"  avg disagreement       : {report.avg_disagreement:.3f}")
    print(f"  max disagreement       : {report.max_disagreement:.3f}")
    for sa in report.sub_agent_reports:
        print(f"  {sa.agent_label:12s} breach rate: {sa.var_breach_rate:.3f}")
    print("=" * 52)
