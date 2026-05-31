"""Tests for Exa-backed news source and intelligence conversion."""

import pandas as pd

from contracts import NewsArticle, NewsFeed
from news import ExaNewsSource, MockNewsSource, feed_to_market_intelligence, make_news_source


def test_make_news_source_uses_exa_when_key_present(monkeypatch):
    monkeypatch.setenv("EXA_KEY", "test-key")
    source = make_news_source()
    assert isinstance(source, ExaNewsSource)


def test_make_news_source_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("EXA_KEY", raising=False)
    source = make_news_source("")
    assert isinstance(source, MockNewsSource)


def test_feed_to_market_intelligence_from_articles():
    as_of = pd.Timestamp("2024-06-01")
    feed = NewsFeed(
        as_of=as_of,
        articles=[
            NewsArticle(
                ticker="NVDA",
                title="Nvidia beats earnings expectations",
                snippet="Strong data center growth continues",
                url="https://example.com/nvda",
                published_date=as_of,
                sentiment_score=0.5,
            ),
            NewsArticle(
                ticker="GOOG",
                title="Alphabet warns on cloud spending",
                snippet="Analysts cut outlook after miss",
                url="https://example.com/goog",
                published_date=as_of,
                sentiment_score=-0.3,
            ),
        ],
    )
    intel = feed_to_market_intelligence(feed)
    assert intel.headlines["NVDA"] == ["Nvidia beats earnings expectations"]
    assert intel.headlines["GOOG"] == ["Alphabet warns on cloud spending"]
    assert "NVDA" in intel.sentiment
    assert intel.key_themes
