"""
Behavioral integration tests for the full pipeline.
Tests verify observable outputs through public interfaces only.
"""
import os
import pytest
import pandas as pd

# ── helpers ──────────────────────────────────────────────────────────────────

EXA_KEY = os.environ.get("EXA_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
TICKERS = ["NVDA", "GOOG"]


# ── Layer 1: Data ─────────────────────────────────────────────────────────────

def test_yfinance_returns_real_ohlcv():
    from data import YFinanceDataSource
    md = YFinanceDataSource(tickers=TICKERS, start="2024-01-01").load()
    assert md.tickers == TICKERS
    for t in TICKERS:
        df = md.bars[t]
        assert len(df) > 50, f"{t} has too few rows"
        assert set(df.columns) >= {"open", "high", "low", "close", "volume"}
        assert df["close"].notna().all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["close"]).all()


# ── Layer 1: News ─────────────────────────────────────────────────────────────

def test_mock_news_returns_valid_feed():
    from news import MockNewsSource
    source = MockNewsSource()
    feed = source.fetch(TICKERS, pd.Timestamp("2024-06-01"))
    assert feed.as_of == pd.Timestamp("2024-06-01")
    assert len(feed.articles) > 0
    for a in feed.articles:
        assert a.ticker in TICKERS
        assert -1.0 <= a.sentiment_score <= 1.0


@pytest.mark.skipif(not EXA_KEY, reason="EXA_KEY not set")
def test_exa_news_returns_real_articles():
    from news import ExaNewsSource
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        source = ExaNewsSource(api_key=EXA_KEY, n_results=2, cache_dir=tmp)
        feed = source.fetch(TICKERS, pd.Timestamp("2025-01-15"))
        assert len(feed.articles) > 0
        assert all(a.ticker in TICKERS for a in feed.articles)
        assert all(-1.0 <= a.sentiment_score <= 1.0 for a in feed.articles)
    finally:
        shutil.rmtree(tmp)


@pytest.mark.skipif(not EXA_KEY, reason="EXA_KEY not set")
def test_exa_news_disk_cache_avoids_second_fetch():
    from news import ExaNewsSource
    import tempfile, shutil, time
    tmp = tempfile.mkdtemp()
    try:
        source = ExaNewsSource(api_key=EXA_KEY, n_results=2, cache_dir=tmp)
        as_of = pd.Timestamp("2025-01-15")
        source.fetch(TICKERS, as_of)                  # populates cache
        t0 = time.monotonic()
        source.fetch(TICKERS, as_of)                  # should be instant from disk
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, f"Cache hit took {elapsed:.2f}s — expected <0.5s"
    finally:
        shutil.rmtree(tmp)


# ── Layer 2: Forecast ─────────────────────────────────────────────────────────

def test_momentum_forecast_returns_valid_contract():
    from data import YFinanceDataSource
    from forecast import MomentumForecaster
    md = YFinanceDataSource(tickers=TICKERS, start="2024-01-01").load()
    as_of = md.bars["NVDA"].index[80]
    fc = MomentumForecaster().predict(md, as_of, horizon_days=5)
    assert fc.horizon_days == 5
    assert set(fc.expected_returns.keys()) == set(TICKERS)
    assert set(fc.direction.keys()) == set(TICKERS)
    assert all(v in (-1, 1) for v in fc.direction.values())


# ── Layer 3: Risk ─────────────────────────────────────────────────────────────

def test_risk_model_produces_ensemble_and_disagreement():
    from data import YFinanceDataSource
    from forecast import MomentumForecaster
    from news import MockNewsSource
    from risk import SampleCovRisk
    md = YFinanceDataSource(tickers=TICKERS, start="2024-01-01").load()
    as_of = md.bars["NVDA"].index[80]
    window = md.slice_until(as_of)
    fc = MomentumForecaster().predict(window, as_of, 5)
    news = MockNewsSource().fetch(TICKERS, as_of)
    rm = SampleCovRisk(n_paths=500).estimate(window, news, fc, as_of)
    assert set(rm.var_95.keys()) == set(TICKERS)
    assert set(rm.cvar_95.keys()) == set(TICKERS)
    assert len(rm.sub_agent_results) == 3
    assert 0.0 <= rm.disagreement <= 5.0
    labels = {r.agent_label for r in rm.sub_agent_results}
    assert labels == {"gbm", "markov", "bootstrap"}


# ── Layer 4: Optimize ─────────────────────────────────────────────────────────

def test_optimizer_shrinks_positions_under_high_disagreement():
    from data import YFinanceDataSource
    from forecast import MomentumForecaster
    from news import MockNewsSource
    from risk import SampleCovRisk
    from contracts import RiskModel
    from optimize import MeanVarianceOptimizer
    import numpy as np
    md = YFinanceDataSource(tickers=TICKERS, start="2024-01-01").load()
    as_of = md.bars["NVDA"].index[80]
    window = md.slice_until(as_of)
    fc = MomentumForecaster().predict(window, as_of, 5)
    news = MockNewsSource().fetch(TICKERS, as_of)

    low_disagreement_risk = SampleCovRisk(n_paths=500).estimate(window, news, fc, as_of)
    cov = low_disagreement_risk.cov
    vol = low_disagreement_risk.vol

    high_disagreement_risk = RiskModel(
        as_of=as_of, cov=cov, vol=vol,
        var_95=low_disagreement_risk.var_95,
        cvar_95=low_disagreement_risk.cvar_95,
        sub_agent_results=low_disagreement_risk.sub_agent_results,
        disagreement=2.0,
    )

    low_tp = MeanVarianceOptimizer().solve(fc, low_disagreement_risk)
    high_tp = MeanVarianceOptimizer().solve(fc, high_disagreement_risk)
    assert high_tp.gross < low_tp.gross, "High disagreement should shrink gross exposure"


# ── Layer 5+6: End-to-end short backtest ─────────────────────────────────────

def test_short_backtest_produces_scorecard_and_risk_report():
    from backtest import Backtest
    from data import YFinanceDataSource
    from execute import PaperExecutor
    from forecast import MomentumForecaster
    from news import MockNewsSource
    from optimize import MeanVarianceOptimizer
    from risk import SampleCovRisk
    from score import BacktestScorer, RiskScorer

    bt = Backtest(
        source=YFinanceDataSource(tickers=TICKERS, start="2024-01-01"),
        news_source=MockNewsSource(),
        forecaster=MomentumForecaster(),
        risk=SampleCovRisk(n_paths=200),
        optimizer=MeanVarianceOptimizer(),
        executor=PaperExecutor(),
        rebalance_every=20,
    )
    data, records, baseline = bt.run()
    assert len(records) >= 5

    card = BacktestScorer().score(records, baseline)
    assert isinstance(card.sharpe, float)
    assert 0.0 < card.directional_accuracy < 1.0
    assert len(card.equity_curve) == len(records)

    report = RiskScorer().score(records)
    assert report.var_breach_rate >= 0
    assert len(report.sub_agent_reports) == 3


# ── Intelligence layer ────────────────────────────────────────────────────────

@pytest.mark.skipif(not EXA_KEY, reason="EXA_KEY not set")
def test_market_intelligence_returns_sentiment_and_headlines():
    from intelligence import MarketIntelligenceAgent
    intel = MarketIntelligenceAgent(api_key=EXA_KEY, n_results=3).fetch(TICKERS)
    assert set(intel.sentiment.keys()) == set(TICKERS)
    assert all(-1.0 <= v <= 1.0 for v in intel.sentiment.values())
    assert any(len(intel.headlines.get(t, [])) > 0 for t in TICKERS)


# ── Master agent ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(not OPENROUTER_KEY, reason="OPENROUTER_KEY not set")
def test_master_agent_returns_valid_comparison_report():
    from intelligence import MarketIntelligenceAgent
    from master_agent import MasterAgent

    intel = MarketIntelligenceAgent(api_key=EXA_KEY, n_results=2).fetch(TICKERS)

    dummy = {
        "name": "normal", "label": "test", "as_of": "2025-01-01",
        "rebalances": 10, "sharpe": 0.5, "total_return_pct": 5.0,
        "directional_accuracy_pct": 55.0, "information_coefficient": 0.1,
        "vs_50_50_sharpe": 0.2, "final_equity": 1.05,
        "var_breaches": 1, "avg_disagreement": 0.2, "max_disagreement": 0.4,
        "last_weights": {"NVDA": 0.5, "GOOG": 0.5}, "last_forecast": {},
        "engine_diagnostics": {},
    }
    quantum = {**dummy, "name": "quantum", "sharpe": 0.6, "total_return_pct": 6.0}

    report = MasterAgent(openrouter_key=OPENROUTER_KEY).compare(
        dummy, quantum, intel,
        metric_deltas={"quantum_minus_normal_sharpe": 0.1},
        decision_trace=["quantum Sharpe was higher"],
    )
    assert report.winner in ("normal", "quantum", "tie")
    assert len(report.rationale) > 20
    assert isinstance(report.metric_deltas, dict)
