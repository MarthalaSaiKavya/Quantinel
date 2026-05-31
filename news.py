"""
LAYER 1 · NEWS (owner: ____)

MockNewsSource  — synthetic articles for fast local runs.
ExaNewsSource   — real point-in-time news via Exa neural search API, with a
                  disk cache so Exa is only called once per (date, tickers) pair.
                  Subsequent backtest runs are instant — zero API calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from contracts import NewsArticle, NewsFeed

# Sentiment keyword lists (same as intelligence.py so scores are consistent)
_POSITIVE = {"beat", "surge", "growth", "upgrade", "bullish", "strong", "record",
             "raises", "rally", "soar", "profit", "gain"}
_NEGATIVE = {"miss", "decline", "cut", "downgrade", "bearish", "weak", "loss",
             "lowers", "fall", "drop", "risk", "warn"}


def _score(texts: list[str]) -> float:
    combined = " ".join(texts).lower().split()
    pos = sum(1 for w in combined if w in _POSITIVE)
    neg = sum(1 for w in combined if w in _NEGATIVE)
    denom = max(1, len(texts) * 2)
    return max(-1.0, min(1.0, (pos - neg) / denom))


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

    def fetch(self, tickers: list[str], as_of: pd.Timestamp) -> NewsFeed:
        cache_key = f"{as_of.date().isoformat()}_{'_'.join(sorted(tickers))}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            return self._load(cache_file, as_of)

        articles = self._fetch_exa(tickers, as_of)
        self._save(cache_file, articles)
        return NewsFeed(as_of=as_of, articles=articles)

    # ------------------------------------------------------------------

    def _fetch_exa(self, tickers: list[str], as_of: pd.Timestamp) -> list[NewsArticle]:
        start_date = (as_of - pd.Timedelta(days=self.lookback_days)).strftime("%Y-%m-%dT00:00:00Z")
        end_date = as_of.strftime("%Y-%m-%dT23:59:59Z")
        articles: list[NewsArticle] = []
        try:
            from exa_py import Exa
            exa = Exa(api_key=self.api_key)
            for ticker in tickers:
                results = exa.search_and_contents(
                    f"{ticker} stock news earnings analyst",
                    num_results=self.n_results,
                    text=True,
                    start_published_date=start_date,
                    end_published_date=end_date,
                )
                for r in results.results:
                    texts = [r.title or "", r.text[:300] if r.text else ""]
                    articles.append(NewsArticle(
                        ticker=ticker,
                        title=r.title or "",
                        snippet=r.text[:300] if r.text else "",
                        url=r.url or "",
                        published_date=pd.Timestamp(r.published_date) if r.published_date else as_of,
                        sentiment_score=_score(texts),
                    ))
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
                published_date=pd.Timestamp(d["published_date"]),
                sentiment_score=d["sentiment_score"],
            )
            for d in data
        ]
        return NewsFeed(as_of=as_of, articles=articles)
