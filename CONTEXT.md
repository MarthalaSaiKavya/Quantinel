# CONTEXT — Quantinel domain glossary

This file defines the canonical language of the Quantinel trading pipeline. It is a
glossary, not a spec. No implementation details live here.

---

## Pipeline layers

| Layer | Name | Responsibility |
|-------|------|----------------|
| 1 | Data | Fetch price bars and news articles. Produce `MarketData` and `NewsFeed`. |
| 2 | Forecast | Predict future returns from price history. Produce `Forecast`. |
| 3 | Risk | Estimate risk from prices, news, and the forecast. Produce `RiskModel`. |
| 4 | Pick & Size | Convert forecast and risk into target weights. Produce `TargetPortfolio`. |
| 5 | Execute | Turn target weights into fills. Produce `ExecutionResult`. |
| 6 | Score | Measure trading performance and risk model honesty. Produce `Scorecard` and `RiskReport`. |

---

## Contracts (data that flows between layers)

### MarketData
OHLCV bars per ticker, with a DatetimeIndex. Produced by Layer 1. Consumed by
Forecast (Layer 2) and Risk (Layer 3).

### NewsFeed
A batch of news articles with per-article ticker association and sentiment scores.
Produced by Layer 1. Consumed only by Risk (Layer 3).

### NewsArticle
A single news item: title, snippet, URL, publish date, associated ticker, and a
sentiment score in [-1, +1] (+1 = bullish, -1 = bearish).

### Sentiment Score
A float in [-1, +1] assigned to each article-ticker pair. Positive means the
article is bullish for that ticker; negative means bearish.

### Forecast
Expected returns, direction (+1/-1), and confidence (0..1) per ticker over a fixed
horizon. Produced by Layer 2. Consumed by Layer 4 (Optimizer) and Layer 3 (Risk).

### RiskModel
Covariance matrix, per-ticker volatility, aggregated VaR/CVaR, per-sub-agent
breakdown, and an ensemble disagreement score. Produced by Layer 3. Consumed by
Layer 4 (Optimizer).

### VaR (Value at Risk)
The maximum loss not exceeded with a given confidence level over the forecast
horizon. In Quantinel: 95% VaR over the horizon (default 5 days), per ticker.

### CVaR (Conditional Value at Risk)
The expected loss given that the loss exceeds VaR. The "average of the worst 5%."

### Disagreement
A scalar [0..1] measuring how much the sub-agent risk models diverge. 0 = all
agree; 1 = maximum divergence. Used by the Optimizer to shrink positions when the
models fight.

### Sub-agent
One of several independent models that simulate return distributions. Each produces
its own VaR/CVaR. The RiskModel aggregates across sub-agents.

### Ensemble
The combination of sub-agent outputs: median VaR (consensus) and worst CVaR
(conservative tail).

### Regime
A latent market state (bull, bear) used by the Markov regime-switching sub-agent.
News sentiment shifts the transition probabilities between regimes.

### TargetPortfolio
Signed target weights per ticker. Produced by Layer 4. Consumed by Layer 5.

### ExecutionResult
Realized fills and actual post-execution weights. Produced by Layer 5. Consumed by
Layer 6.

### Scorecard
Trading performance metrics: Sharpe, total return, directional accuracy,
information coefficient, equity curve. Produced by Layer 6.

### RiskReport
Risk model honesty metrics: VaR breach count and rate, CVaR exceedances,
disagreement timeline, per-sub-agent calibration. Produced by Layer 6 from the
per-step risk data stored during the backtest.

---

## Relationships

```
NewsFeed ──► Risk (Layer 3) ── Markov regime transitions
MarketData ──► Forecast (Layer 2) ──► Risk (Layer 3)
                                    ──► Optimizer (Layer 4)
Forecast ──► Risk (Layer 3)
          ──► Optimizer (Layer 4)
RiskModel ──► Optimizer (Layer 4)
TargetPortfolio ──► Execute (Layer 5)
ExecutionResult ──► Score (Layer 6)
```

- Forecast is **pure price-history**. It does not consume news.
- Risk consumes **both prices and news**. News only affects the Markov
  regime-switching sub-agent; GBM and bootstrap are price-only.
- The Optimizer uses **disagreement** from the RiskModel to shrink position sizes.
