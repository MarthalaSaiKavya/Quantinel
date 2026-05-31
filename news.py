"""
LAYER 1 · NEWS (owner: ____)

ExaNewsSource fetches live headlines via the Exa API for backtest + dashboard use.
MockNewsSource remains for deterministic tests and offline runs.

Both implement NewsSource: fetch(tickers, as_of) -> NewsFeed.
"""

from __future__ import annotations

import os
import sys
import threading
from collections import Counter

import numpy as np
import pandas as pd

from contracts import MarketIntelligence, NewsArticle, NewsFeed

_POSITIVE = {
    "beat", "surge", "growth", "upgrade", "bullish", "strong", "record",
    "raises", "rally", "soar", "profit", "gain",
}
_NEGATIVE = {
    "miss", "decline", "cut", "downgrade", "bearish", "weak", "loss",
    "lowers", "fall", "drop", "risk", "warn",
}
_STOPWORDS = {
    "stock", "shares", "market", "company", "quarter", "fiscal",
    "nvda", "goog", "google", "nvidia", "the", "and", "for", "with",
    "from", "that", "this", "will", "its",
}


def sentiment_score(texts: list[str]) -> float:
    if not texts:
        return 0.0
    combined = " ".join(texts).lower()
    words = combined.split()
    pos = sum(1 for w in words if w in _POSITIVE)
    neg = sum(1 for w in words if w in _NEGATIVE)
    denom = max(1, len(texts) * 2)
    return max(-1.0, min(1.0, (pos - neg) / denom))


def top_themes(all_texts: list[str], n: int = 3) -> list[str]:
    words = " ".join(all_texts).lower().split()
    counts = Counter(
        w for w in words if len(w) > 5 and w.isalpha() and w not in _STOPWORDS
    )
    return [w for w, _ in counts.most_common(n)]


def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def _parse_published(value, fallback: pd.Timestamp) -> pd.Timestamp:
    if value is None:
        return _to_naive(fallback)
    try:
        return _to_naive(pd.Timestamp(value))
    except (TypeError, ValueError):
        return _to_naive(fallback)


def fetch_exa_articles(
    tickers: list[str],
    api_key: str,
    *,
    n_results: int = 5,
    as_of: pd.Timestamp | None = None,
) -> list[NewsArticle]:
    """Fetch recent articles for each ticker from Exa."""
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
    articles: list[NewsArticle] = []

    from exa_py import Exa

    exa = Exa(api_key=api_key)
    for ticker in tickers:
        results = exa.search_and_contents(
            f"{ticker} stock news earnings analyst outlook",
            num_results=n_results,
            text=True,
        )
        for result in results.results:
            title = result.title or ""
            snippet = result.text[:300] if result.text else ""
            if not title and not snippet:
                continue
            texts = [t for t in (title, snippet) if t]
            articles.append(
                NewsArticle(
                    ticker=ticker,
                    title=title or snippet[:80],
                    snippet=snippet,
                    url=result.url or "",
                    published_date=_parse_published(
                        getattr(result, "published_date", None), as_of
                    ),
                    sentiment_score=sentiment_score(texts),
                )
            )

    return articles


def feed_to_market_intelligence(feed: NewsFeed) -> MarketIntelligence:
    """Convert a NewsFeed into the dashboard/agent MarketIntelligence contract."""
    tickers = sorted({a.ticker for a in feed.articles})
    headlines: dict[str, list[str]] = {t: [] for t in tickers}
    urls: dict[str, list[str]] = {t: [] for t in tickers}
    all_texts: list[str] = []

    for article in feed.articles:
        if article.title:
            headlines[article.ticker].append(article.title)
        if article.url:
            urls[article.ticker].append(article.url)
        all_texts.extend([article.title, article.snippet])

    sentiment = feed.sentiment_scores()
    for ticker in tickers:
        sentiment.setdefault(ticker, 0.0)

    return MarketIntelligence(
        as_of=feed.as_of,
        headlines=headlines,
        sentiment=sentiment,
        key_themes=top_themes(all_texts),
        urls=urls,
    )


def make_news_source(api_key: str | None = None):
    """Return ExaNewsSource when a key is available, else MockNewsSource."""
    key = api_key if api_key is not None else os.environ.get("EXA_KEY", "")
    if key:
        return ExaNewsSource(api_key=key)
    return MockNewsSource()


class ExaNewsSource:
    """Implements NewsSource using live Exa headlines (cached per run)."""

    def __init__(self, api_key: str, n_results: int = 5):
        self.api_key = api_key
        self.n_results = n_results
        self._articles: list[NewsArticle] | None = None
        self._last_feed: NewsFeed | None = None
        self._lock = threading.Lock()

    @property
    def latest_feed(self) -> NewsFeed | None:
        return self._last_feed

    def _load_articles(self, tickers: list[str], as_of: pd.Timestamp) -> list[NewsArticle]:
        with self._lock:
            if self._articles is not None:
                return self._articles
            try:
                self._articles = fetch_exa_articles(
                    tickers,
                    self.api_key,
                    n_results=self.n_results,
                    as_of=as_of,
                )
            except Exception as exc:
                print(f"[news] Exa fetch failed: {exc}", file=sys.stderr)
                self._articles = []
            return self._articles

    def fetch(self, tickers: list[str], as_of: pd.Timestamp) -> NewsFeed:
        as_of = _to_naive(as_of)
        articles = [
            a for a in self._load_articles(tickers, as_of) if a.ticker in tickers
        ]
        feed = NewsFeed(as_of=as_of, articles=articles)
        self._last_feed = feed
        return feed


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
        self._last_feed: NewsFeed | None = None

    @property
    def latest_feed(self) -> NewsFeed | None:
        return self._last_feed

    def fetch(self, tickers: list[str], as_of: pd.Timestamp) -> NewsFeed:
        rng = np.random.default_rng(self.seed ^ hash(as_of) % 2**32)
        articles: list[NewsArticle] = []

        market_mood = float(rng.normal(self.sentiment_bias, 0.3))

        for day_offset in range(5):
            date = as_of - pd.Timedelta(days=day_offset)
            for _ in range(self.articles_per_day):
                for t in tickers:
                    mood = self.ticker_corr * market_mood + (
                        1 - self.ticker_corr
                    ) * rng.normal(self.sentiment_bias, 0.3)
                    sentiment = float(np.clip(mood, -1.0, 1.0))

                    articles.append(
                        NewsArticle(
                            ticker=t,
                            title=f"Fake news about {t} on {date.date()}",
                            snippet=(
                                f"Mock article snippet for {t} "
                                f"with sentiment {sentiment:.2f}"
                            ),
                            url=f"https://mock.example.com/{t}/{date.date()}",
                            published_date=date,
                            sentiment_score=sentiment,
                        )
                    )

        feed = NewsFeed(as_of=as_of, articles=articles)
        self._last_feed = feed
        return feed
