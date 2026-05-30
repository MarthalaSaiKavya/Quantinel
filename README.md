# NVDA / GOOG modular trading pipeline

A six-layer pipeline split so four people can work in parallel. Layers talk **only**
through the typed contracts in `contracts.py`. Anyone can rewrite the inside of their
layer — momentum → Chronos → QSVM, or Markowitz → QAOA — and as long as the contract
holds, nothing else breaks. The whole thing runs today with **no quantum and no ML**,
on **mock NVDA + GOOG data**, so you have a working baseline before anyone integrates.

## Data flow

```
 Layer 1        Layer 2          Layer 3        Layer 4           Layer 5        Layer 6
  DATA    -->   FORECAST   -->    RISK    -->   PICK & SIZE  -->  EXECUTE   -->  SCORE
MarketData     Forecast        RiskModel     TargetPortfolio  ExecutionResult Scorecard
   |              |  ^             |  ^            ^   ^                              |
   +--------------+  |             +--+            |   |                              |
        (data feeds both forecast and risk)       |   |                              |
                                  forecast + risk -+   +- (also reads MarketData)     |
                                                                                       v
                                                              <----  loop: score refines next forecast
```

## Contracts (the hand-off points)

| Layer | Owner | Consumes | Produces | Interface (in `contracts.py`) |
|-------|-------|----------|----------|-------------------------------|
| 1 · Data | A | — | `MarketData` | `DataSource.load()` |
| 2 · Forecast | B | `MarketData` | `Forecast` | `Forecaster.predict()` |
| 3 · Risk | C | `MarketData` | `RiskModel` | `RiskEstimator.estimate()` |
| 4 · Pick & size | C | `Forecast`, `RiskModel` | `TargetPortfolio` | `Optimizer.solve()` |
| 5 · Execute | D | `TargetPortfolio`, `MarketData` | `ExecutionResult` | `Executor.execute()` |
| 6 · Score | D | step records + baseline | `Scorecard` | `BacktestScorer.score()` |

Each contract is a frozen dataclass. The important fields:

- **`MarketData`** — `tickers`, `bars{ticker: OHLCV DataFrame}`; helpers `close_prices()`, `returns()`, `slice_until(as_of)`.
- **`Forecast`** — `expected_returns{ticker: float}`, `direction{ticker: +1/-1}`, `confidence{ticker: 0..1}`.
- **`RiskModel`** — `cov` (annualized covariance DataFrame), `vol{ticker: float}`; method `portfolio_vol(weights)`.
- **`TargetPortfolio`** — `weights{ticker: signed float}`; properties `gross`, `net`.
- **`ExecutionResult`** — `fills[Fill]`, `realized_weights{ticker: float}`.
- **`Scorecard`** — `sharpe`, `total_return`, `directional_accuracy`, `information_coefficient`, `vs_baseline_sharpe`, `equity_curve`.

## Suggested team split

- **Person A — Data** (`data.py`): owns `MockDataSource` now; writes a real `AlpacaDataSource` later. Just return a `MarketData`.
- **Person B — Forecast** (`forecast.py`): owns `MomentumForecaster` (baseline) and the `QuantumForecaster` (QSVM/VQC) swap. Also where a Chronos/LSTM forecaster would go.
- **Person C — Risk + Optimize** (`risk.py`, `optimize.py`): owns `SampleCovRisk`, `MeanVarianceOptimizer` (Markowitz baseline), `DiscreteQuboOptimizer`, and the `QaoaOptimizer` swap.
- **Adithya kalidindi — Execute + Score + wiring** (`execute.py`, `score.py`, `backtest.py`, `run_baseline.py`): owns the paper executor, the scorer, and the orchestrator that calls everyone.

Because the contracts are fixed, B and C can build against the baseline data while A is still wiring the real API, and D can score against a dummy forecaster while B trains models.

## Run it

```bash
python run_baseline.py
```

Prints a scorecard for the no-quantum baseline (Sharpe, return, directional accuracy, IC, and the edge over a 50/50 buy-and-hold).

## Swap in quantum (one line each)

In `run_baseline.py`:

```python
forecaster = QuantumForecaster()    # QSVM / VQC  instead of MomentumForecaster()
optimizer  = QaoaOptimizer()        # QAOA / VQE  instead of MeanVarianceOptimizer()
```

Both quantum classes already exist with the correct signatures and raise
`NotImplementedError` with a note describing exactly what to return. Fill the body,
keep the signature, and the pipeline runs unchanged. For the **Quantum Advantage
Award**: run `DiscreteQuboOptimizer` (classical) and `QaoaOptimizer` (quantum) on the
*same* QUBO and compare the two `Scorecard`s — that is your honest, measured lift.

## Mock data

`MockDataSource` makes ~2 years of daily OHLCV for **NVDA** and **GOOG** as two
correlated return streams (default correlation 0.6, NVDA higher drift/vol than GOOG),
seeded for reproducibility. Tune drift, vol, correlation, length, and seed in the
constructor. Swap the whole class for a real feed without touching any other layer.
