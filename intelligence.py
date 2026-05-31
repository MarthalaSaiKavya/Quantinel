"""
MARKET INTELLIGENCE LAYER  (owner: Adithya Kalidindi)

MarketIntelligenceAgent uses the Exa neural search API to fetch recent
news for each ticker, score sentiment, and extract key themes.
Produces a MarketIntelligence contract consumed by MasterAgent.
"""
from __future__ import annotations

import sys
from collections import Counter

import pandas as pd

from contracts import MarketIntelligence

_POSITIVE = {"beat", "surge", "growth", "upgrade", "bullish", "strong", "record", "raises", "rally", "soar", "profit", "gain"}
_NEGATIVE = {"miss", "decline", "cut", "downgrade", "bearish", "weak", "loss", "lowers", "fall", "drop", "risk", "warn"}
_STOPWORDS = {"stock", "shares", "market", "company", "quarter", "fiscal", "nvda", "goog", "google", "nvidia", "the", "and", "for", "with", "from", "that", "this", "will", "its"}


def _sentiment_score(texts: list[str]) -> float:
    if not texts:
        return 0.0
    combined = " ".join(texts).lower()
    words = combined.split()
    pos = sum(1 for w in words if w in _POSITIVE)
    neg = sum(1 for w in words if w in _NEGATIVE)
    denom = max(1, len(texts) * 2)
    return max(-1.0, min(1.0, (pos - neg) / denom))


def _top_themes(all_texts: list[str], n: int = 3) -> list[str]:
    words = " ".join(all_texts).lower().split()
    counts = Counter(
        w for w in words
        if len(w) > 5 and w.isalpha() and w not in _STOPWORDS
    )
    return [w for w, _ in counts.most_common(n)]


class MarketIntelligenceAgent:
    """Fetches news + sentiment for a list of tickers via Exa neural search."""

    def __init__(self, api_key: str, n_results: int = 5):
        self.api_key = api_key
        self.n_results = n_results

    def fetch(self, tickers: list[str], as_of=None) -> MarketIntelligence:
        as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()

        headlines: dict[str, list[str]] = {}
        sentiment: dict[str, float] = {}
        urls: dict[str, list[str]] = {}
        all_texts: list[str] = []

        try:
            from exa_py import Exa

            exa = Exa(api_key=self.api_key)
            for ticker in tickers:
                results = exa.search_and_contents(
                    f"{ticker} stock news earnings analyst outlook",
                    num_results=self.n_results,
                    text=True,
                )
                titles = [r.title or "" for r in results.results]
                snippets = [r.text[:300] if r.text else "" for r in results.results]
                sources = [r.url or "" for r in results.results]

                headlines[ticker] = [t for t in titles if t]
                urls[ticker] = [u for u in sources if u]
                texts = titles + snippets
                sentiment[ticker] = _sentiment_score(texts)
                all_texts.extend(texts)

        except Exception as e:
            print(f"[intelligence] Exa fetch failed: {e}", file=sys.stderr)
            for ticker in tickers:
                headlines.setdefault(ticker, [])
                sentiment.setdefault(ticker, 0.0)
                urls.setdefault(ticker, [])

        return MarketIntelligence(
            as_of=as_of,
            headlines=headlines,
            sentiment=sentiment,
            key_themes=_top_themes(all_texts),
            urls=urls,
        )
