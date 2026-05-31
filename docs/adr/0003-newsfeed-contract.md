# ADR 0003: Separate NewsSource Protocol and NewsFeed contract

## Status
Accepted (2026-05-30)

## Context

News data (Exa articles) needs to flow from Layer 1 to Layer 3. Two integration
patterns were considered: embedding news inside the existing `MarketData` contract,
or creating a parallel contract with its own Protocol.

## Decision

**A separate `NewsSource` Protocol and `NewsFeed` contract.** `DataSource.load()`
returns `MarketData` (OHLCV bars). `NewsSource.fetch()` returns `NewsFeed` (articles
with ticker association and sentiment scores). Both are wired in the Backtest
orchestrator.

The `NewsFeed` dataclass contains:
- Raw articles (`NewsArticle` list: title, snippet, URL, date, ticker, sentiment)
- Helper methods: `sentiment_scores(lookback_days)` for per-ticker rolling
  sentiment, `article_count(ticker)`

The `NewsArticle.sentiment_score` is produced by Layer 1's NLP step (mock
heuristic now, swap to LLM later).

## Alternatives considered

- **Nested inside MarketData (`MarketData.news`):** Breaks the existing contract
  for every layer. Risk would have to dig into MarketData for news, while Forecast
  would see news fields it's explicitly not supposed to use. Violates
  least-privilege.
- **NewsFetcher called inside Forecaster/RiskEstimator directly:** No contract at
  all. Layers have hidden I/O dependencies. Can't mock news independently from
  prices. Violates the pipeline's "swap components without touching neighbors"
  philosophy.
- **Extend DataSource.load() to return a tuple:** `load() → (MarketData,
  NewsFeed)`. Breaks every existing DataSource implementation. The Protocol says
  `load() -> MarketData`, period.

## Consequences

- Backtest wiring creates both a `DataSource` and a `NewsSource`, passes the latter
  to `RiskEstimator`.
- `RiskEstimator.estimate()` signature becomes: `estimate(data: MarketData, news:
  NewsFeed, forecast: Forecast, as_of: Timestamp) -> RiskModel`.
- MockNewsSource generates fake articles independently from MockDataSource,
  allowing testing of news-on/off scenarios.
