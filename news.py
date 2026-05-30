"""
LAYER 1 · NEWS (owner: ____)

MockNewsSource generates fake Exa-style news articles with controllable per-ticker
sentiment scores. Seeds ensure reproducibility, matching the MockDataSource pattern.

Swap for a real ExaNewsSource later — same NewsSource Protocol, no downstream changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from contracts import NewsArticle, NewsFeed


class MockNewsSource:
    """Implements NewsSource: fetch(tickers, as_of) -> NewsFeed."""

    def __init__(
        self,
        articles_per_day: int = 3,
        seed: int = 42,
        sentiment_bias: float = 0.0,
        ticker_corr: float = 0.5,
    ):
        self.articles_per_day = articles_per_day
        self.seed = seed
        self.sentiment_bias = sentiment_bias
        self.ticker_corr = ticker_corr

    def fetch(self, tickers: list[str], as_of: pd.Timestamp) -> NewsFeed:
        rng = np.random.default_rng(self.seed ^ hash(as_of) % 2**32)
        articles: list[NewsArticle] = []

        # Correlated sentiment: same underlying "market mood" per day
        market_mood = float(rng.normal(self.sentiment_bias, 0.3))

        for day_offset in range(5):
            date = as_of - pd.Timedelta(days=day_offset)
            for _ in range(self.articles_per_day):
                for t in tickers:
                    # Ticker-specific noise on top of market mood
                    mood = self.ticker_corr * market_mood + (
                        1 - self.ticker_corr
                    ) * rng.normal(self.sentiment_bias, 0.3)
                    sentiment = float(np.clip(mood, -1.0, 1.0))

                    articles.append(
                        NewsArticle(
                            ticker=t,
                            title=f"Fake news about {t} on {date.date()}",
                            snippet=f"Mock article snippet for {t} with sentiment {sentiment:.2f}",
                            url=f"https://mock.example.com/{t}/{date.date()}",
                            published_date=date,
                            sentiment_score=sentiment,
                        )
                    )

        return NewsFeed(as_of=as_of, articles=articles)
