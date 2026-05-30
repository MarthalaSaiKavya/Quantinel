"""
LAYER 1 · NEWS (owner: ____)

MockNewsSource: fake Exa-style news with controllable sentiment (seeded).
ExaNewsSource:  real Exa API integration with keyword sentiment extraction.

Swap between them — same NewsSource Protocol, no downstream changes.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from contracts import NewsArticle, NewsFeed


# ============================================================================
# Sentiment extraction
# ============================================================================

BULLISH_WORDS = [
    "surge", "rallied", "rally", "beat", "beats", "upgrade", "upgraded",
    "breakthrough", "profit", "growth", "rise", "gain", "bullish",
    "outperform", "positive", "strong", "record", "soar", "soars",
    "boost", "jump", "climb", "expansion", "optimistic", "outlook",
]

BEARISH_WORDS = [
    "plunge", "plunges", "drop", "miss", "misses", "downgrade", "downgraded",
    "loss", "decline", "fall", "bearish", "underperform", "negative",
    "weak", "concern", "risk", "probe", "investigation", "lawsuit",
    "sanction", "tariff", "crash", "sell-off", "selloff", "fears",
]


class KeywordSentimentExtractor:
    """Simple word-count sentiment: (bullish - bearish) / max(total, 1)."""

    def extract(self, text: str, ticker: str) -> float:
        lower = text.lower()
        bull = sum(1 for w in BULLISH_WORDS if w in lower)
        bear = sum(1 for w in BEARISH_WORDS if w in lower)
        total = bull + bear
        if total == 0:
            return 0.0
        return float(np.clip((bull - bear) / total, -1.0, 1.0))


@runtime_checkable
class SentimentExtractor(Protocol):
    def extract(self, text: str, ticker: str) -> float: ...


# ============================================================================
# Mock news source (seeded, no network)
# ============================================================================

class MockNewsSource:
    """Implements NewsSource: fetch(tickers, as_of) -> NewsFeed."""

    def __init__(self, articles_per_day: int = 3, seed: int = 42,
                 sentiment_bias: float = 0.0, ticker_corr: float = 0.5):
        self.articles_per_day = articles_per_day
        self.seed = seed
        self.sentiment_bias = sentiment_bias
        self.ticker_corr = ticker_corr

    def fetch(self, tickers: list[str], as_of: pd.Timestamp) -> NewsFeed:
        rng = np.random.default_rng(self.seed ^ hash(as_of) % 2**32)
        articles: list[NewsArticle] = []

        market_mood = float(rng.normal(self.sentiment_bias, 0.3))

        for day_offset in range(5):
            date = as_of - pd.Timedelta(days=day_offset)
            for _ in range(self.articles_per_day):
                for t in tickers:
                    mood = (self.ticker_corr * market_mood
                            + (1 - self.ticker_corr) * rng.normal(self.sentiment_bias, 0.3))
                    sentiment = float(np.clip(mood, -1.0, 1.0))

                    articles.append(NewsArticle(
                        ticker=t,
                        title=f"Fake news about {t} on {date.date()}",
                        snippet=f"Mock article snippet for {t} with sentiment {sentiment:.2f}",
                        url=f"https://mock.example.com/{t}/{date.date()}",
                        published_date=date,
                        sentiment_score=sentiment,
                    ))

        return NewsFeed(as_of=as_of, articles=articles)


# ============================================================================
# Real Exa news source
# ============================================================================

EXA_SEARCH_URL = "https://api.exa.ai/search"


class ExaNewsSource:
    """Implements NewsSource: fetch(tickers, as_of) -> NewsFeed using Exa API.

    Requires EXA_API_KEY environment variable.
    Override _search_exa() in tests to mock the network call.
    """

    def __init__(self, api_key: str | None = None,
                 sentiment_extractor: SentimentExtractor | None = None,
                 num_results: int = 5):
        self.api_key = api_key or os.environ.get("EXA_API_KEY", "")
        self.sentiment = sentiment_extractor or KeywordSentimentExtractor()
        self.num_results = num_results

    def fetch(self, tickers: list[str], as_of: pd.Timestamp) -> NewsFeed:
        articles: list[NewsArticle] = []

        for ticker in tickers:
            query = self._build_query(ticker)
            raw_results = self._search_exa(query, as_of)

            for raw in raw_results:
                title = raw.get("title", "")
                text = raw.get("text", "")
                sentiment = self.sentiment.extract(title + " " + text, ticker)

                pub_date = as_of
                if raw.get("publishedDate"):
                    try:
                        pub_date = pd.Timestamp(raw["publishedDate"])
                    except (ValueError, TypeError):
                        pass

                articles.append(NewsArticle(
                    ticker=ticker,
                    title=title,
                    snippet=text[:500] if text else "",
                    url=raw.get("url", ""),
                    published_date=pub_date,
                    sentiment_score=sentiment,
                ))

        return NewsFeed(as_of=as_of, articles=articles)

    # ------------------------------------------------------------------
    # Override point for tests (mock the network)
    # ------------------------------------------------------------------

    def _search_exa(self, query: str, as_of: pd.Timestamp) -> list[dict]:
        """Call Exa search API. Override in tests to return fake results."""
        if not self.api_key:
            raise RuntimeError(
                "EXA_API_KEY not set. Set the environment variable or pass api_key=."
            )
        payload = json.dumps({
            "query": query,
            "numResults": self.num_results,
            "useAutoprompt": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            EXA_SEARCH_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("results", [])

    # ------------------------------------------------------------------
    # Query builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_query(ticker: str) -> str:
        name_map = {
            "NVDA": "NVIDIA stock",
            "GOOG": "Alphabet Google stock",
            "GOOGL": "Alphabet Google stock",
        }
        name = name_map.get(ticker, ticker)
        return f"{name} latest news financial"
