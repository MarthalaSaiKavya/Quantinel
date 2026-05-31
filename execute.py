"""
LAYER 5 · EXECUTE   (owner: ____)

PaperExecutor turns target weights into whole-share holdings against a fixed
capital base, at the close on `as_of`. Whole-share rounding means realized
weights differ slightly from target — that gap is the first taste of real-world
execution friction. Swap later for an AlpacaExecutor that places live paper orders.
"""
from __future__ import annotations

import pandas as pd

from contracts import ExecutionResult, Fill, MarketData, TargetPortfolio


class PaperExecutor:
    """Implements Executor: execute(target, data, as_of) -> ExecutionResult."""

    def __init__(self, capital: float = 100_000.0):
        self.capital = capital

    def execute(self, target: TargetPortfolio, data: MarketData, as_of) -> ExecutionResult:
        fills: list[Fill] = []
        realized: dict[str, float] = {}

        for t, w in target.weights.items():
            price = float(data.bars[t]["close"].loc[:as_of].iloc[-1])
            shares = int((w * self.capital) / price)        # whole-share rounding
            realized[t] = (shares * price) / self.capital
            if shares != 0:
                fills.append(
                    Fill(ticker=t, side="buy" if shares > 0 else "sell", qty=abs(shares), price=price)
                )

        return ExecutionResult(as_of=pd.Timestamp(as_of), fills=fills, realized_weights=realized)