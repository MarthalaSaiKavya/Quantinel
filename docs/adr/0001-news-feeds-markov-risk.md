# ADR 0001: News feeds into Risk via Markov regime-switching

## Status
Accepted (2026-05-30)

## Context

The pipeline needs to incorporate world news (fetched via Exa) to assess how risky
a given Forecast is. The Forecast itself is pure price-history — it does not see
news. Two design questions arose:

1. Which layer consumes news?
2. How does news actually change risk estimates?

## Decision

**News is consumed only by Layer 3 (Risk).** It feeds into exactly one of the three
risk sub-agents: the Markov regime-switching model. GBM and bootstrap remain
price-only.

**How it works:** News articles are converted to per-ticker sentiment scores
[-1, +1] in Layer 1. The Markov sub-agent uses a base transition matrix estimated
from historical return clustering (bull/bear regimes). Sentiment scores adjust the
transition probabilities: negative sentiment raises P(bear | bull) and lowers
P(bull | bear), making bear regimes stickier under bad news.

## Alternatives considered

- **News feeds all sub-agents:** Would muddy GBM (which assumes constant parameters)
  and bootstrap (which is non-parametric by design). Overcomplicates without clear
  benefit.
- **News creates a standalone risk score separate from simulations:** Would not
  integrate with the sub-agent ensemble framework. A standalone score would be an
  extra input the Optimizer must weigh, rather than flowing through the same
  VaR/CVaR aggregation pipeline.
- **News goes into Forecast instead:** Rejected early. Forecast is intentionally
  price-only so it can be compared with and without news-augmented risk. Separating
  prediction quality from risk assessment makes each independently testable.

## Consequences

- RiskEstimator interface expands to accept `NewsFeed` alongside `MarketData` and
  `Forecast`.
- A MockNewsSource generates fake articles with correlated sentiment, keeping the
  baseline runnable without Exa or LLM dependencies.
- The Markov sub-agent has a dependency on sentiment scores; GBM and bootstrap do
  not. Future sub-agents can choose whether to consume news.
