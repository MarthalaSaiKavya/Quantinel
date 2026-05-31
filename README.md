# Quantinel: normal vs quantum trading pipeline

Quantinel is a modular NVDA/GOOG backtesting pipeline that compares two trading
systems on the same market data:

- **Normal pipeline**: recent-return momentum forecast + classical Markowitz optimizer.
- **Quantum pipeline**: xpyq SVD factor forecast + xpyq/QUBO-style optimizer.

Both simulations run through the same data, news, risk, execution, and scoring
layers. The final `MasterAgent` receives both result summaries, Exa market
intelligence, and a decision trace, then explains which branch won in simple
terms.

## Current workflow

```text
Mock NVDA/GOOG OHLCV data
        |
        | run in parallel
        |
   -----------------------------
   |                           |
NORMAL PIPELINE           QUANTUM PIPELINE
   |                           |
MomentumForecaster        QuantumForecaster
recent return signal      xpyq SVD factor signal
   |                           |
SampleCovRisk             SampleCovRisk
same risk engine          same risk engine
   |                           |
MeanVarianceOptimizer     QaoaOptimizer
classic Markowitz         xpyq eig/QUBO path
   |                           |
PaperExecutor             PaperExecutor
same execution            same execution
   |                           |
BacktestScorer            BacktestScorer
same scoring              same scoring
   \                           /
    -------- comparison summary
                 |
       Exa headlines + sentiment
                 |
        OpenRouter MasterAgent
                 |
       plain-English final decision
```

## What "88 rebalances" means

The default mock dataset has about two years of business-day prices:

```text
504 trading days
- 60 days lookback history
- 5 days forward-return scoring window
rebalance every 5 trading days
= 88 rebalance decisions
```

So `88` means the strategy made 88 weekly trading decisions. In the full quantum
branch, that can mean up to:

```text
88 xpyq forecast jobs
+ 88 xpyq optimizer jobs
= up to 176 remote xpyq jobs
```

For quick debugging, use a larger rebalance interval to reduce the number of
remote jobs.

## Main commands

Create a local `.env` file with your keys:

```bash
XPYQ_KEY=...
EXA_KEY=...
OPENROUTER_KEY=...
```

`.env` is ignored by git.

Run the full comparison:

```bash
set -a; source .env; set +a; .venv/bin/python run_master.py
```

Run a faster comparison while debugging xpyq:

```bash
set -a; source .env; set +a; \
QUANTINEL_REBALANCE_EVERY=20 \
XPYQ_TIMEOUT=10 \
.venv/bin/python run_master.py
```

Run the normal baseline only:

```bash
.venv/bin/python run_baseline.py
```

Run the older three-way quantum comparison:

```bash
set -a; source .env; set +a; .venv/bin/python run_quantum.py
```

## Environment knobs

| Variable | Default | Purpose |
|----------|---------|---------|
| `XPYQ_KEY` | empty | xpyq bearer token for remote compute. |
| `EXA_KEY` | empty | Exa API key for market intelligence. |
| `OPENROUTER_KEY` | empty | OpenRouter key for final agent reasoning. |
| `XPYQ_TIMEOUT` | `20` | Seconds to wait per xpyq job before fallback. |
| `QUANTINEL_REBALANCE_EVERY` | `5` | Trading-day step between decisions. Higher means fewer remote jobs. |
| `QUANTINEL_N_DAYS` | `504` | Number of mock business days to generate. |
| `QUANTINEL_N_PATHS` | `10000` | Monte Carlo paths per risk estimate. |

## Layer map

| Layer | File | Normal branch | Quantum branch | Output |
|-------|------|---------------|----------------|--------|
| Data | `data.py` | `MockDataSource` | same | `MarketData` |
| News | `news.py` | `MockNewsSource` during backtest | same | `NewsFeed` |
| Forecast | `forecast.py` | `MomentumForecaster` | `QuantumForecaster` | `Forecast` |
| Risk | `risk.py` | `SampleCovRisk` | same | `RiskModel` |
| Optimize | `optimize.py` | `MeanVarianceOptimizer` | `QaoaOptimizer` | `TargetPortfolio` |
| Execute | `execute.py` | `PaperExecutor` | same | `ExecutionResult` |
| Score | `score.py` | `BacktestScorer`, `RiskScorer` | same | `Scorecard`, `RiskReport` |
| Intelligence | `intelligence.py` | Exa after both backtests | same | `MarketIntelligence` |
| Final agent | `master_agent.py` | compares both summaries | compares both summaries | `ComparisonReport` |

The layer boundaries are defined in `contracts.py`. Each branch must return the
same contract objects, which is what makes the comparison fair.

## Normal branch

The normal branch is the fully local baseline.

1. `MomentumForecaster` looks at recent returns.
2. It predicts expected return over the next horizon.
3. `SampleCovRisk` estimates covariance, VaR, CVaR, and model disagreement.
4. `MeanVarianceOptimizer` uses classical Markowitz sizing.
5. `PaperExecutor` converts target weights into simulated holdings.
6. `BacktestScorer` calculates return, Sharpe, directional accuracy, IC, and
   edge versus 50/50 buy-and-hold.

This branch is fast, deterministic enough for comparisons, and is the benchmark
the quantum branch must beat.

## Quantum branch

The quantum branch keeps the same outer pipeline but swaps the forecast and
optimizer.

### Quantum forecast

`QuantumForecaster` builds a recent returns matrix:

```text
          NVDA     GOOG
day 1    0.012   0.004
day 2   -0.006  -0.002
...
```

It submits Python to xpyq:

```python
R = from_numpy(...)
U_mat, S_mat, Vt_mat = linalg.svd(R)
U_arr, S_arr, Vt_arr = U_mat.numpy()
```

SVD breaks returns into hidden market factors. The code uses the strongest
factor to estimate whether each ticker should move up or down.

### Quantum optimizer

`QaoaOptimizer` receives forecasted returns and the risk covariance matrix. It
builds a QUBO-style matrix:

```python
Q = risk_aversion * Sigma - diag(mu)
```

Plain English:

```text
reward higher expected returns
penalize risky combinations
```

Then it submits the optimization problem to xpyq:

```python
Q = from_numpy(...)
eigvals_mat, eigvecs_mat = linalg.eigh(Q)
eigvals_arr, eigvecs_arr = eigvals_mat.numpy()
```

The lowest-energy vector is decoded into long/short weights.

## Fallback and traceability

The quantum branch is designed to finish even if xpyq fails or queues too long.

- `QuantumForecaster` falls back to `MomentumForecaster`.
- `QaoaOptimizer` falls back to `DiscreteQuboOptimizer`.

The comparison report prints engine diagnostics for the quantum branch:

```text
engine trace:
  forecaster calls=88 xpyq_completed=... fallbacks=... statuses={...}
  optimizer  calls=88 xpyq_completed=... fallbacks=... statuses={...}
```

The final report also prints a `DECISION TRACE`, including:

- return gap between quantum and normal
- Sharpe gap
- directional accuracy gap
- risk breach difference
- final weight comparison
- xpyq completion/fallback counts

This makes the final agent's recommendation auditable instead of just a black-box
LLM answer.

## Final agent

`MasterAgent.compare(...)` receives:

- normal simulation summary
- quantum simulation summary
- metric deltas
- decision trace
- Exa headlines and sentiment

It returns:

- `winner`: `normal`, `quantum`, or `tie`
- `recommendation`: what to run next
- `rationale`: simple explanation for a non-technical teammate
- `decision_trace`: evidence items the decision used

The final agent is intentionally not allowed to hide the score comparison. The
numbers are computed in code first, then the agent explains them.

## Important files

| File | Purpose |
|------|---------|
| `run_master.py` | Main normal-vs-quantum comparison runner. |
| `run_baseline.py` | Normal-only baseline run. |
| `run_quantum.py` | Older three-way baseline/quantum/full-quantum comparison. |
| `contracts.py` | Shared dataclasses and Protocol interfaces. |
| `forecast.py` | Momentum and xpyq SVD forecast logic. |
| `optimize.py` | Markowitz, discrete QUBO, and xpyq eig optimizer logic. |
| `risk.py` | Covariance + multi-agent VaR/CVaR risk simulation. |
| `score.py` | Performance and risk scoring. |
| `intelligence.py` | Exa search, sentiment, and theme extraction. |
| `master_agent.py` | Final report and comparison reasoning. |

## Current interpretation

Use the normal branch as the benchmark. Use the quantum branch to test whether
xpyq-backed factor extraction and QUBO-style optimization can beat that benchmark.

If the quantum branch has many fallbacks, the result is not a clean quantum
advantage test. First verify xpyq completions in the engine trace, then compare
returns and risk-adjusted scores.