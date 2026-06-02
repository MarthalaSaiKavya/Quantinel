"""
LAYER 1 · NEWS (owner: ____)

ExaNewsSource   — real point-in-time news via Exa neural search API, with a
                  disk cache so Exa is only called once per (date, tickers) pair.
                  Subsequent backtest runs are instant — zero API calls.
MockNewsSource  — synthetic articles for fast local runs and offline tests.

Both implement NewsSource: fetch(tickers, as_of) -> NewsFeed.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from contracts import MarketIntelligence, NewsArticle, NewsFeed

_POSITIVE = {
    "beat",
    "surge",
    "growth",
    "upgrade",
    "bullish",
    "strong",
    "record",
    "raises",
    "rally",
    "soar",
    "profit",
    "gain",
}
_NEGATIVE = {
    "miss",
    "decline",
    "cut",
    "downgrade",
    "bearish",
    "weak",
    "loss",
    "lowers",
    "fall",
    "drop",
    "risk",
    "warn",
}
_STOPWORDS = {
    "stock",
    "shares",
    "market",
    "company",
    "quarter",
    "fiscal",
    "nvda",
    "goog",
    "google",
    "nvidia",
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "will",
    "its",
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
    key = (
        api_key
        if api_key is not None
        else os.environ.get("EXA_API_KEY", os.environ.get("EXA_KEY", ""))
    )
    if key:
        return ExaNewsSource(api_key=key)
    return MockNewsSource()


class ExaNewsSource:
    """Real point-in-time news via Exa, with a disk cache.

    Each unique (date, tickers) pair is fetched once from Exa and saved to
    .news_cache/<date>_<tickers>.json. Every subsequent call for that pair
    is served from disk — zero API calls, no added runtime.

    Falls back to empty sentiment (s=0.0) if Exa is unavailable, so the
    Markov sub-agent still runs without crashing.
    """

    def __init__(
        self,
        api_key: str,
        n_results: int = 5,
        lookback_days: int = 7,
        cache_dir: str = ".news_cache",
    ):
        self.api_key = api_key
        self.n_results = n_results
        self.lookback_days = lookback_days
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._last_feed: NewsFeed | None = None

    @property
    def latest_feed(self) -> NewsFeed | None:
        return self._last_feed

    def fetch(self, tickers: list[str], as_of: pd.Timestamp) -> NewsFeed:
        as_of = _to_naive(as_of)
        cache_key = f"{as_of.date().isoformat()}_{'_'.join(sorted(tickers))}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists() and cache_file.stat().st_size > 10:
            feed = self._load(cache_file, as_of)
            if not feed.articles:  # empty cache from a prior failed fetch
                cache_file.unlink()
                feed = None
        else:
            feed = None

        if feed is None:
            articles = self._fetch_exa(tickers, as_of)
            self._save(cache_file, articles)
            feed = NewsFeed(as_of=as_of, articles=articles)

        self._last_feed = feed
        return feed

    # ------------------------------------------------------------------

    def _fetch_exa(self, tickers: list[str], as_of: pd.Timestamp) -> list[NewsArticle]:
        start_date = (as_of - pd.Timedelta(days=self.lookback_days)).strftime(
            "%Y-%m-%dT00:00:00Z"
        )
        end_date = as_of.strftime("%Y-%m-%dT23:59:59Z")
        articles: list[NewsArticle] = []
        try:
            from exa_py import Exa

            exa = Exa(api_key=self.api_key)
            for ticker in tickers:
                results = exa.search(
                    f"{ticker} stock news earnings analyst outlook",
                    num_results=self.n_results,
                    start_published_date=start_date,
                    end_published_date=end_date,
                )
                for r in results.results:
                    title = r.title or ""
                    snippet = r.text[:300] if r.text else ""
                    if not title and not snippet:
                        continue
                    texts = [t for t in (title, snippet) if t]
                    articles.append(
                        NewsArticle(
                            ticker=ticker,
                            title=title or snippet[:80],
                            snippet=snippet,
                            url=r.url or "",
                            published_date=_parse_published(
                                getattr(r, "published_date", None), as_of
                            ),
                            sentiment_score=sentiment_score(texts),
                        )
                    )
        except Exception as e:
            print(f"[news] Exa fetch failed for {as_of.date()}: {e}", file=sys.stderr)
        return articles

    def _save(self, path: Path, articles: list[NewsArticle]) -> None:
        data = [
            {
                "ticker": a.ticker,
                "title": a.title,
                "snippet": a.snippet,
                "url": a.url,
                "published_date": a.published_date.isoformat(),
                "sentiment_score": a.sentiment_score,
            }
            for a in articles
        ]
        path.write_text(json.dumps(data), encoding="utf-8")

    def _load(self, path: Path, as_of: pd.Timestamp) -> NewsFeed:
        data = json.loads(path.read_text(encoding="utf-8"))
        articles = [
            NewsArticle(
                ticker=d["ticker"],
                title=d["title"],
                snippet=d["snippet"],
                url=d["url"],
                published_date=_parse_published(d["published_date"], as_of),
                sentiment_score=d["sentiment_score"],
            )
            for d in data
        ]
        return NewsFeed(as_of=as_of, articles=articles)


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
