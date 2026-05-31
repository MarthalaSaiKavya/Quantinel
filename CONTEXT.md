# CONTEXT — Quantinel domain glossary

This file defines the canonical language of the Quantinel trading pipeline. It is a
glossary, not a spec. No implementation details live here.

---

## Pipeline layers

| Layer | Name | Responsibility |
|-------|------|----------------|
| 1 | Data | Fetch price bars and news articles. Produce `MarketData` and `NewsFeed`. |
| 2 | Forecast | Predict future returns from price history. Produce `Forecast`. |
| 2.5 | Chaos Engine | Detect tail-risk events by fusing market features with news sentiment. Produce `ChaosSignal`. |
| 2.6 | Crystal Ball | Fuse a short-horizon `Forecast` with a `ChaosSignal` to produce a 1- or 2-year scenario prediction enriched with IFTF futures thinking signals, backcasting, and Two Curves pattern analysis. Produce `CrystalBallPrediction`. |
| 3 | Risk | Estimate risk from prices, news, and the forecast. Produce `RiskModel`. |
| 4 | Pick & Size | Convert forecast and risk into target weights. Produce `TargetPortfolio`. |
| 5 | Execute | Turn target weights into fills. Produce `ExecutionResult`. |
| 6 | Score | Measure trading performance and risk model honesty. Produce `Scorecard` and `RiskReport`. |

---

## Contracts (data that flows between layers)

### ChaosSignal
The output of the Chaos Engine (Layer 2.5). Contains:
- `crash_probability`: float in [0, 1] — blended estimate of an adverse tail event.
- `event_label`: one of `normal`, `elevated_risk`, or `market_crash`.
- `confidence`: float in [0, 1] — how far the probability estimate is from 0.5.
- `ticker_adjustments`: per-ticker weight multipliers applied by `adjust_portfolio()`.
- `reasoning`: plain-English explanation of what signals drove the estimate.

Produced by Layer 2.5. Consumed optionally by Forecast (to dampen/flip signals) and
by the Optimizer (to scale or short positions).

### CrystalBallPrediction
The output of Crystal Ball (Layer 2.6). Contains:
- `base_returns`: per-ticker compounded expected return over the forecast horizon (default 252 days ≈ 1 year).
- `bull_returns`: optimistic scenario — `base + 1.5 × annual_vol`.
- `bear_returns`: pessimistic scenario — `base − 1.5 × annual_vol`.
- `crash_adjusted_returns`: base returns scaled by the `ChaosSignal.ticker_adjustments` multipliers.
- `annual_volatility`: per-ticker annualised volatility derived from the factor model (leading eigenvalues × 252).
- `crash_probability`: forwarded from the `ChaosSignal`.
- `dominant_factor_var`: leading eigenvalue × 252 — the annualised variance of the strongest market-wide factor.
- `confidence`: per-ticker, inherited from the short-horizon `Forecast`.
- `reasoning`: plain-English narrative structured around three IFTF futures thinking principles:
  - **Principle 2 — Focus on signals**: per-ticker anomalous deviations (volatility surges, momentum breaks, counter-trend bounces, drawdown warnings).
  - **Principle 3 — Look back to see forward**: backcasting across historical vol-regime analogues to surface recurrent patterns and median forward return.
  - **Principle 4 — Uncover patterns**: Two Curves classification per ticker (`first_curve_ascending`, `first_curve_peak`, `first_curve_declining`, `second_curve_emerging`, `transition`, `indeterminate`).

Produced by Layer 2.6. Intended for external reporting and decision support; not consumed by downstream pipeline layers.

### Crash Probability
A float in [0, 1] output by the Chaos Engine representing the estimated likelihood
of an adverse tail event (crash, liquidity crisis, sector collapse) within the
forecast horizon. Above 0.65 triggers a CRASH ALERT; above 0.40 triggers CAUTION.

### Tail Event
An adverse market move that exceeds the crash threshold (default: cumulative return
below −4 % over 5 days). The Chaos Engine labels historical windows as tail events
to train its classifier.

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
NewsFeed ──► Chaos Engine (Layer 2.5) ── news sentiment boost
MarketData ──► Chaos Engine (Layer 2.5) ── market feature extraction
ChaosSignal ──► Forecast (Layer 2) ── optional dampen / flip
             ──► Optimizer (Layer 4) ── optional position scaling / shorting
             ──► Crystal Ball (Layer 2.6) ── crash-adjusted scenario returns
Forecast ──► Crystal Ball (Layer 2.6) ── compounded base return
CrystalBallPrediction ──► (reporting / external consumers)
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
- The Chaos Engine consumes **both prices and news** independently of the Risk layer.
  It runs after the Forecast and can adjust both the `Forecast` and `TargetPortfolio`
  before they reach the Optimizer and Executor.
- Risk consumes **both prices and news**. News only affects the Markov
  regime-switching sub-agent; GBM and bootstrap are price-only.
- The Optimizer uses **disagreement** from the RiskModel to shrink position sizes.