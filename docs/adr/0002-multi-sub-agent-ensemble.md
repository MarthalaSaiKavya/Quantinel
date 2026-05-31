# ADR 0002: Multi-sub-agent ensemble for risk estimation

## Status
Accepted (2026-05-30)

## Context

A single risk model (e.g., sample covariance or GBM) captures one view of the
return distribution. Different models capture different features — constant vol
(GBM), regime changes (Markov), and distribution-free tail behavior (bootstrap).
The pipeline needs a way to combine them.

## Decision

**Three sub-agents run independently, each producing the same output shape:**
simulated return paths over the forecast horizon (5 days, 10,000 paths each).

| Sub-agent | Method | Price data | News data |
|-----------|--------|------------|-----------|
| GBM | Geometric Brownian Motion | Drift from Forecast, vol from historical | No |
| Markov | 2-regime (bull/bear) switching | Transition matrix from return clustering | Sentiment adjusts transitions |
| Bootstrap | Block resampling of historical returns | Yes | No |

**Aggregation:** Median VaR across agents (consensus) and worst CVaR across agents
(conservative tail). A `disagreement` scalar [0..1] measures model divergence.

## Alternatives considered

- **Single sub-agent:** Defeats the purpose — no ensemble signal, no disagreement
  metric. One model's blind spots become the pipeline's blind spots.
- **Weighted ensemble by recent calibration:** Attractive in theory (better model
  gets more weight), but 2 tickers × 80 rebalances gives too little data to
  reliably rank models. Overfitting risk.
- **More than 3 sub-agents:** Jump-diffusion, GARCH, and Heston were considered.
  Deferred to keep the baseline lean. Adding a sub-agent is a single new
  `_run_xxx()` method in the RiskEstimator — low-friction extension later.

## Consequences

- RiskModel carries per-sub-agent breakdown, aggregate VaR/CVaR, and disagreement.
- The Optimizer uses disagreement to shrink position sizes: `w_adjusted = w / (1 +
  disagreement)`.
- Adding a fourth sub-agent requires: (1) implementing its simulation, (2) adding
  its label to the PerSubAgentRisk list. No contract changes needed.
