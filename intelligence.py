"""
MARKET INTELLIGENCE LAYER  (owner: Adithya Kalidindi)

Thin wrapper over ExaNewsSource for agents and the dashboard intelligence panel.
"""
from __future__ import annotations

import pandas as pd

from contracts import MarketIntelligence
from news import ExaNewsSource, feed_to_market_intelligence, make_news_source


class MarketIntelligenceAgent:
    """Fetches news + sentiment for a list of tickers via Exa neural search."""

    def __init__(self, api_key: str, n_results: int = 5):
        self.api_key = api_key
        self.n_results = n_results

    def fetch(self, tickers: list[str], as_of=None) -> MarketIntelligence:
        as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
        source = ExaNewsSource(api_key=self.api_key, n_results=self.n_results)
        feed = source.fetch(tickers, as_of)
        return feed_to_market_intelligence(feed)


def intelligence_from_news_source(
    news_source, tickers: list[str], as_of=None
) -> MarketIntelligence:
    """Build MarketIntelligence from an existing news source (avoids duplicate Exa calls)."""
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
    feed = news_source.latest_feed
    if feed is None or not feed.articles:
        feed = news_source.fetch(tickers, as_of)
    return feed_to_market_intelligence(feed)
