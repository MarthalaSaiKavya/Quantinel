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
# SLICE 8: ExaNewsSource — real news via Exa API
# ============================================================================

class _FakeExaSource:
    """ExaNewsSource with mocked HTTP — overrides _search_exa to return fake results."""

    def __init__(self, api_key: str = "test-key"):
        from news import ExaNewsSource
        self._real = ExaNewsSource(api_key=api_key)
        self._real._search_exa = self._fake_search

    def _fake_search(self, query: str, as_of):
        """Return fake Exa-style results for a query."""
        import pandas as pd
        date_str = str(as_of.date()) if hasattr(as_of, 'date') else str(as_of)
        if "NVDA" in query or "NVIDIA" in query:
            return [
                {
                    "title": "NVIDIA stock surges on AI chip demand",
                    "text": "NVIDIA shares rallied today as demand for AI chips continues to surge, driving record revenue growth.",
                    "url": "https://example.com/nvda-surge",
                    "publishedDate": date_str,
                },
                {
                    "title": "Analyst downgrades NVDA on valuation concerns",
                    "text": "A prominent analyst downgraded NVIDIA citing valuation risk and potential slowdown in demand.",
                    "url": "https://example.com/nvda-downgrade",
                    "publishedDate": date_str,
                },
            ]
        if "GOOG" in query or "GOOGL" in query or "Alphabet" in query:
            return [
                {
                    "title": "Google beats earnings estimates",
                    "text": "Alphabet reported strong quarterly results, beating analyst estimates on cloud and ad revenue.",
                    "url": "https://example.com/goog-beat",
                    "publishedDate": date_str,
                },
            ]
        return []

    def fetch(self, tickers: list[str], as_of) -> "NewsFeed":
        return self._real.fetch(tickers, as_of)


def test_exa_news_source_produces_valid_newsfeed():
    """ExaNewsSource generates a NewsFeed from mocked Exa API results."""
    import pandas as pd
    from contracts import NewsFeed

    source = _FakeExaSource(api_key="test-key")
    as_of = pd.Timestamp("2025-06-15")
    feed = source.fetch(["NVDA", "GOOG"], as_of)

    assert isinstance(feed, NewsFeed)
    assert feed.as_of == as_of
    assert len(feed.articles) >= 1

    for a in feed.articles:
        assert isinstance(a.ticker, str)
        assert isinstance(a.title, str)
        assert isinstance(a.snippet, str)
        assert isinstance(a.url, str)
        assert isinstance(a.sentiment_score, float)
        assert -1.0 <= a.sentiment_score <= 1.0

    tickers_found = {a.ticker for a in feed.articles}
    assert "NVDA" in tickers_found
    assert "GOOG" in tickers_found


def test_exa_news_sentiment_scores_are_directional():
    """Bullish text gets positive sentiment, bearish gets negative."""
    import pandas as pd

    source = _FakeExaSource(api_key="test-key")
    as_of = pd.Timestamp("2025-06-15")
    feed = source.fetch(["NVDA"], as_of)

    nvda_articles = [a for a in feed.articles if a.ticker == "NVDA"]
    sentiments = [a.sentiment_score for a in nvda_articles]

    assert any(s > 0 for s in sentiments), f"Expected a positive sentiment: {sentiments}"
    assert any(s < 0 for s in sentiments), f"Expected a negative sentiment: {sentiments}"


def test_keyword_sentiment_extractor_bullish():
    """KeywordSentimentExtractor returns positive for bullish text."""
    from news import KeywordSentimentExtractor
    ext = KeywordSentimentExtractor()
    score = ext.extract("NVIDIA profit surges on record revenue growth and strong demand", "NVDA")
    assert score > 0, f"Expected bullish score, got {score}"


def test_keyword_sentiment_extractor_bearish():
    """KeywordSentimentExtractor returns negative for bearish text."""
    from news import KeywordSentimentExtractor
    ext = KeywordSentimentExtractor()
    score = ext.extract("Analyst downgrades stock on weak outlook and concerns of decline", "NVDA")
    assert score < 0, f"Expected bearish score, got {score}"


def test_keyword_sentiment_extractor_neutral():
    """KeywordSentimentExtractor returns near-zero for neutral text."""
    from news import KeywordSentimentExtractor
    ext = KeywordSentimentExtractor()
    score = ext.extract("The company announced regular quarterly earnings today", "NVDA")
    assert -0.2 <= score <= 0.2, f"Expected neutral, got {score}"


def test_exa_news_feeds_into_risk_layer():
    """ExaNewsSource NewsFeed is accepted by SampleCovRisk.estimate()."""
    import pandas as pd
    from contracts import RiskModel
    from data import MockDataSource
    from forecast import MomentumForecaster
    from risk import SampleCovRisk

    data = MockDataSource(seed=42, n_days=120).load()
    as_of = data.bars["NVDA"].index[-1]
    window = data.slice_until(as_of)
    forecast = MomentumForecaster(lookback=30).predict(window, as_of, 5)

    source = _FakeExaSource(api_key="test-key")
    news = source.fetch(window.tickers, as_of)

    risk = SampleCovRisk(n_paths=2000).estimate(window, news, forecast, as_of)

    assert isinstance(risk, RiskModel)
    assert "NVDA" in risk.var_95
    assert "GOOG" in risk.var_95
    assert risk.disagreement >= 0
    assert len(risk.sub_agent_results) == 3


def test_baseline_runs_with_exa_news_source():
    """End-to-end baseline run with ExaNewsSource (mocked)."""
    from backtest import Backtest
    from contracts import RiskReport, Scorecard
    from data import MockDataSource
    from execute import PaperExecutor
    from forecast import MomentumForecaster
    from optimize import MeanVarianceOptimizer
    from risk import SampleCovRisk
    from score import BacktestScorer, RiskScorer

    exa_source = _FakeExaSource(api_key="test-key")

    bt = Backtest(
        source=MockDataSource(seed=42, n_days=200),
        news_source=exa_source,
        forecaster=MomentumForecaster(),
        risk=SampleCovRisk(n_paths=500),
        optimizer=MeanVarianceOptimizer(),
        executor=PaperExecutor(),
        rebalance_every=25,
        lookback_days=50,
        horizon_days=5,
    )
    data, records, baseline = bt.run()

    card = BacktestScorer().score(records, baseline)
    report = RiskScorer().score(records)

    assert isinstance(card, Scorecard)
    assert isinstance(report, RiskReport)
    assert card.sharpe is not None
    assert report.var_breaches >= 0
    assert len(report.sub_agent_reports) == 3
