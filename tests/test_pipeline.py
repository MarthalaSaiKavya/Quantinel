"""TDD tests for the Quantinel pipeline — news, multi-agent risk, VaR/CVaR."""

import numpy as np
import pandas as pd
import pytest

from contracts import (
    Forecast,
    MarketData,
    NewsArticle,
    NewsFeed,
    PerSubAgentRisk,
    RiskModel,
    RiskReport,
    SubAgentReport,
)
from data import MockDataSource
from forecast import MomentumForecaster
from risk import SampleCovRisk

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

    from optimize import MeanVarianceOptimizer

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

    from optimize import DiscreteQuboOptimizer

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
    from backtest import Backtest
    from data import MockDataSource
    from execute import PaperExecutor
    from forecast import MomentumForecaster
    from news import MockNewsSource
    from optimize import MeanVarianceOptimizer
    from risk import SampleCovRisk

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
    from backtest import StepRecord
    from contracts import (
        Forecast,
        PerSubAgentRisk,
        RiskModel,
        RiskReport,
        SubAgentReport,
    )
    from score import RiskScorer

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
    from backtest import Backtest
    from data import MockDataSource
    from execute import PaperExecutor
    from forecast import MomentumForecaster
    from news import MockNewsSource
    from optimize import MeanVarianceOptimizer
    from risk import SampleCovRisk
    from score import BacktestScorer, RiskScorer

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


# ============================================================================
# SLICE 8: ChaosEngine — classical fallback (no xpyq key required)
# ============================================================================


def _make_chaos_fixtures(seed: int = 42, n_days: int = 252):
    """Return (data, news, as_of) with enough history for ChaosEngine."""
    data = MockDataSource(seed=seed, n_days=n_days).load()
    as_of = data.bars["NVDA"].index[-1]
    window = data.slice_until(as_of)
    news = _make_mock_news(window.tickers, as_of, sentiment=0.0)
    return window, news, as_of


def test_chaos_engine_returns_chaos_signal():
    """ChaosEngine.evaluate() returns a well-formed ChaosSignal (classical fallback)."""
    from chaos import ChaosEngine
    from contracts import ChaosSignal

    window, news, as_of = _make_chaos_fixtures()
    # No api_key → forces classical centroid-distance path
    engine = ChaosEngine(api_key="")
    signal = engine.evaluate(window, news, as_of)

    assert isinstance(signal, ChaosSignal)
    assert 0.0 <= signal.crash_probability <= 1.0
    assert 0.0 <= signal.confidence <= 1.0
    assert signal.event_label in ("normal", "elevated_risk", "market_crash", "unknown")
    assert set(signal.ticker_adjustments.keys()) == set(window.tickers)
    assert isinstance(signal.reasoning, str) and len(signal.reasoning) > 0


def test_chaos_engine_negative_news_raises_crash_prob():
    """Strongly negative news produces a higher crash probability than neutral news."""
    from chaos import ChaosEngine

    window, _, as_of = _make_chaos_fixtures()
    engine = ChaosEngine(api_key="")

    neutral  = _make_mock_news(window.tickers, as_of, sentiment=0.0)
    negative = _make_mock_news(window.tickers, as_of, sentiment=-0.9)

    p_neutral  = engine.evaluate(window, neutral,  as_of).crash_probability
    p_negative = engine.evaluate(window, negative, as_of).crash_probability

    assert p_negative >= p_neutral, (
        f"Expected negative news to raise crash_probability: "
        f"neutral={p_neutral:.3f}, negative={p_negative:.3f}"
    )


def test_chaos_engine_positive_news_lowers_crash_prob():
    """Positive news produces a lower crash probability than neutral news."""
    from chaos import ChaosEngine

    window, _, as_of = _make_chaos_fixtures()
    engine = ChaosEngine(api_key="")

    neutral  = _make_mock_news(window.tickers, as_of, sentiment=0.0)
    positive = _make_mock_news(window.tickers, as_of, sentiment=0.8)

    p_neutral  = engine.evaluate(window, neutral,  as_of).crash_probability
    p_positive = engine.evaluate(window, positive, as_of).crash_probability

    assert p_positive <= p_neutral, (
        f"Expected positive news to lower crash_probability: "
        f"neutral={p_neutral:.3f}, positive={p_positive:.3f}"
    )


def test_chaos_engine_adjust_forecast_high_prob_flips_direction():
    """adjust_forecast() flips direction to -1 when crash_probability >= 0.65."""
    from chaos import ChaosEngine
    from contracts import ChaosSignal, Forecast

    engine = ChaosEngine(api_key="")
    as_of = pd.Timestamp("2025-01-10")

    signal = ChaosSignal(
        as_of=as_of,
        crash_probability=0.80,
        event_label="market_crash",
        confidence=0.60,
        ticker_adjustments={"NVDA": -0.80, "GOOG": -0.80},
        reasoning="test",
    )
    forecast = Forecast(
        as_of=as_of,
        horizon_days=5,
        expected_returns={"NVDA": 0.05, "GOOG": 0.03},
        direction={"NVDA": 1, "GOOG": 1},
        confidence={"NVDA": 0.7, "GOOG": 0.6},
    )

    adjusted = engine.adjust_forecast(forecast, signal)

    for t in ["NVDA", "GOOG"]:
        assert adjusted.direction[t] == -1, f"{t} direction should be flipped to -1"
        assert adjusted.expected_returns[t] < 0, f"{t} expected_return should be negative"


def test_chaos_engine_adjust_forecast_moderate_prob_dampens():
    """adjust_forecast() dampens (but does not flip) at moderate crash probability."""
    from chaos import ChaosEngine
    from contracts import ChaosSignal, Forecast

    engine = ChaosEngine(api_key="")
    as_of = pd.Timestamp("2025-01-10")

    signal = ChaosSignal(
        as_of=as_of,
        crash_probability=0.50,
        event_label="elevated_risk",
        confidence=0.40,
        ticker_adjustments={"NVDA": 0.40, "GOOG": 0.40},
        reasoning="test",
    )
    forecast = Forecast(
        as_of=as_of,
        horizon_days=5,
        expected_returns={"NVDA": 0.05, "GOOG": 0.03},
        direction={"NVDA": 1, "GOOG": 1},
        confidence={"NVDA": 0.7, "GOOG": 0.6},
    )

    adjusted = engine.adjust_forecast(forecast, signal)

    for t in ["NVDA", "GOOG"]:
        assert adjusted.direction[t] == 1, f"{t} direction should stay positive"
        assert adjusted.expected_returns[t] < forecast.expected_returns[t], (
            f"{t} expected_return should be dampened"
        )


def test_chaos_engine_low_prob_leaves_forecast_unchanged():
    """adjust_forecast() is a no-op when crash_probability < 0.40."""
    from chaos import ChaosEngine
    from contracts import ChaosSignal, Forecast

    engine = ChaosEngine(api_key="")
    as_of = pd.Timestamp("2025-01-10")

    signal = ChaosSignal(
        as_of=as_of,
        crash_probability=0.20,
        event_label="normal",
        confidence=0.60,
        ticker_adjustments={"NVDA": 1.0, "GOOG": 1.0},
        reasoning="test",
    )
    forecast = Forecast(
        as_of=as_of,
        horizon_days=5,
        expected_returns={"NVDA": 0.05, "GOOG": 0.03},
        direction={"NVDA": 1, "GOOG": 1},
        confidence={"NVDA": 0.7, "GOOG": 0.6},
    )

    adjusted = engine.adjust_forecast(forecast, signal)

    assert adjusted is forecast  # identical object — no copy made


def test_chaos_engine_insufficient_history_returns_fallback():
    """ChaosEngine returns a sentiment-only fallback when history < 25 samples."""
    from chaos import ChaosEngine

    # Only 30 days — not enough to build the crash-label dataset
    data = MockDataSource(seed=42, n_days=30).load()
    as_of = data.bars["NVDA"].index[-1]
    window = data.slice_until(as_of)
    news = _make_mock_news(window.tickers, as_of, sentiment=0.0)

    engine = ChaosEngine(api_key="")
    signal = engine.evaluate(window, news, as_of)

    assert signal.event_label == "unknown"
    assert signal.confidence == 0.0


# ============================================================================
# SLICE 9: CrystalBall — signal detection, backcasting, Two Curves
# ============================================================================


def test_crystal_ball_predict_1year_default():
    """CrystalBall.predict() returns a CrystalBallPrediction with the 1-year horizon."""
    from chaos import ChaosEngine
    from contracts import CrystalBallPrediction
    from forecast import CrystalBall, MomentumForecaster

    window, news, as_of = _make_chaos_fixtures()
    engine = ChaosEngine(api_key="")
    cb = CrystalBall(MomentumForecaster(), engine)

    pred = cb.predict(window, news, as_of)

    assert isinstance(pred, CrystalBallPrediction)
    assert pred.horizon_days == 252
    for t in window.tickers:
        assert t in pred.base_returns
        assert t in pred.bull_returns
        assert t in pred.bear_returns
        assert t in pred.crash_adjusted_returns
        assert t in pred.annual_volatility
        assert 0.0 <= pred.confidence[t] <= 1.0
    assert 0.0 <= pred.crash_probability <= 1.0
    assert pred.dominant_factor_var >= 0.0


def test_crystal_ball_predict_2year_horizon():
    """CrystalBall.predict() with TWO_YEAR_DAYS returns horizon_days=504."""
    from chaos import ChaosEngine
    from forecast import CrystalBall, MomentumForecaster

    window, news, as_of = _make_chaos_fixtures()
    engine = ChaosEngine(api_key="")
    cb = CrystalBall(MomentumForecaster(), engine)

    pred = cb.predict(window, news, as_of, horizon_days=CrystalBall.TWO_YEAR_DAYS)

    assert pred.horizon_days == 504


def test_crystal_ball_2year_base_larger_than_1year():
    """2-year compounded base return is larger in magnitude than the 1-year base return."""
    from chaos import ChaosEngine
    from forecast import CrystalBall, MomentumForecaster

    window, news, as_of = _make_chaos_fixtures()
    engine = ChaosEngine(api_key="")
    cb = CrystalBall(MomentumForecaster(), engine)

    pred_1y = cb.predict(window, news, as_of)
    pred_2y = cb.predict(window, news, as_of, horizon_days=CrystalBall.TWO_YEAR_DAYS)

    for t in window.tickers:
        assert abs(pred_2y.base_returns[t]) >= abs(pred_1y.base_returns[t]), (
            f"{t}: 2-year base ({pred_2y.base_returns[t]:+.4f}) should have "
            f"larger magnitude than 1-year base ({pred_1y.base_returns[t]:+.4f})"
        )


def test_crystal_ball_bull_above_base_above_bear():
    """bull_return > base_return > bear_return for every ticker."""
    from chaos import ChaosEngine
    from forecast import CrystalBall, MomentumForecaster

    window, news, as_of = _make_chaos_fixtures()
    engine = ChaosEngine(api_key="")
    cb = CrystalBall(MomentumForecaster(), engine)
    pred = cb.predict(window, news, as_of)

    for t in window.tickers:
        assert pred.bull_returns[t] > pred.base_returns[t], f"{t}: bull <= base"
        assert pred.base_returns[t] > pred.bear_returns[t], f"{t}: base <= bear"


def test_crystal_ball_reasoning_contains_all_sections():
    """reasoning string contains the three IFTF principle section headers."""
    from chaos import ChaosEngine
    from forecast import CrystalBall, MomentumForecaster

    window, news, as_of = _make_chaos_fixtures()
    engine = ChaosEngine(api_key="")
    cb = CrystalBall(MomentumForecaster(), engine)
    pred = cb.predict(window, news, as_of)

    assert "PRINCIPLE 2" in pred.reasoning
    assert "PRINCIPLE 3" in pred.reasoning
    assert "PRINCIPLE 4" in pred.reasoning
    assert "SCENARIO PROJECTIONS" in pred.reasoning


def test_crystal_ball_reasoning_labels_horizon():
    """reasoning string says '1-YEAR' for default and '2-YEAR' for TWO_YEAR_DAYS."""
    from chaos import ChaosEngine
    from forecast import CrystalBall, MomentumForecaster

    window, news, as_of = _make_chaos_fixtures()
    engine = ChaosEngine(api_key="")
    cb = CrystalBall(MomentumForecaster(), engine)

    pred_1y = cb.predict(window, news, as_of)
    pred_2y = cb.predict(window, news, as_of, horizon_days=CrystalBall.TWO_YEAR_DAYS)

    assert "1-YEAR" in pred_1y.reasoning
    assert "2-YEAR" in pred_2y.reasoning


def test_detect_signals_returns_dict_for_all_tickers():
    """_detect_signals returns a key for every ticker, even with no signals fired."""
    from forecast import CrystalBall

    window, news, as_of = _make_chaos_fixtures()
    rets = window.returns().loc[:as_of]

    signals = CrystalBall._detect_signals(rets, window.tickers)

    assert set(signals.keys()) == set(window.tickers)
    for t, sigs in signals.items():
        assert isinstance(sigs, list)


def test_backcast_regimes_returns_required_keys():
    """_backcast_regimes always returns the four required keys."""
    from forecast import CrystalBall

    window, news, as_of = _make_chaos_fixtures()
    rets = window.returns().loc[:as_of]

    result = CrystalBall._backcast_regimes(rets, window.tickers)

    assert "analog_count" in result
    assert "median_fwd_return" in result
    assert "pct_positive" in result
    assert "regime_label" in result
    if result["analog_count"] > 0:
        assert 0.0 <= result["pct_positive"] <= 1.0


def test_backcast_regimes_insufficient_history():
    """_backcast_regimes gracefully handles < 40 rows of history."""
    from forecast import CrystalBall

    data = MockDataSource(seed=42, n_days=30).load()
    as_of = data.bars["NVDA"].index[-1]
    window = data.slice_until(as_of)
    rets = window.returns().loc[:as_of]

    result = CrystalBall._backcast_regimes(rets, window.tickers)

    assert result["analog_count"] == 0
    assert "insufficient" in result["regime_label"]


def test_two_curves_classify_returns_valid_labels():
    """_two_curves_classify returns a valid label for every ticker."""
    from forecast import CrystalBall

    window, news, as_of = _make_chaos_fixtures()
    rets = window.returns().loc[:as_of]

    valid_labels = {
        "first_curve_ascending",
        "first_curve_peak",
        "first_curve_declining",
        "second_curve_emerging",
        "transition",
        "indeterminate",
    }

    result = CrystalBall._two_curves_classify(rets, window.tickers)

    assert set(result.keys()) == set(window.tickers)
    for t, label in result.items():
        assert label in valid_labels, f"{t}: unexpected label '{label}'"
