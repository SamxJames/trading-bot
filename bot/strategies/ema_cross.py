"""
EMA Crossover strategy (reference implementation).

Generates a BUY signal when the fast EMA crosses above the slow EMA, and a
SELL signal when it crosses below.  Uses incremental O(1) EMA updates so
backtesting over hundreds of thousands of bars stays fast.

Configuration keys (read from Settings):
  fast_period  (default: 9)
  slow_period  (default: 21)

The strategy will not emit any signal until it has accumulated at least
slow_period bars for a stable EMA seed.  EMAs are seeded with the SMA of
the first slow_period closes, then updated recursively:
  EMA(t) = alpha * close(t) + (1 - alpha) * EMA(t-1)
  where alpha = 2 / (period + 1)
"""

from __future__ import annotations

from collections import deque
from typing import Deque

from bot.data.feed import Bar
from bot.logging.logger import get_logger
from bot.strategies.base import Signal, SignalType


class EmaCrossStrategy:
    """Exponential Moving Average crossover strategy."""

    name = "ema_cross"

    def __init__(self, fast_period: int = 20, slow_period: int = 50, **kwargs) -> None:  # noqa: ARG002
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be less than slow_period ({slow_period})"
            )
        self.fast_period = fast_period
        self.slow_period = slow_period

        self._alpha_fast = 2.0 / (fast_period + 1)
        self._alpha_slow = 2.0 / (slow_period + 1)

        # Seed buffer — only needed until we have slow_period bars
        self._seed: Deque[float] = deque(maxlen=slow_period)
        self._seeded = False

        # Running EMA values
        self._fast_ema: float | None = None
        self._slow_ema: float | None = None

        # Previous bar's EMAs for crossover detection
        self._prev_fast: float | None = None
        self._prev_slow: float | None = None

        self._log = get_logger(__name__)

    def on_start(self) -> None:
        self._log.info("strategy_started", name=self.name, fast=self.fast_period, slow=self.slow_period)

    def on_stop(self) -> None:
        self._log.info("strategy_stopped", name=self.name)

    def on_bar(self, bar: Bar) -> Signal | None:
        """Process one bar.  Returns a Signal on crossover, None otherwise."""
        close = bar.close

        if not self._seeded:
            self._seed.append(close)
            if len(self._seed) < self.slow_period:
                return None
            # Seed: SMA of first slow_period closes
            prices = list(self._seed)
            self._slow_ema = sum(prices) / self.slow_period
            self._fast_ema = sum(prices[-self.fast_period :]) / self.fast_period
            self._seeded = True
            return None  # no signal on the seeding bar itself

        # Snapshot previous values before updating
        self._prev_fast = self._fast_ema
        self._prev_slow = self._slow_ema

        # Incremental EMA update
        self._fast_ema = self._alpha_fast * close + (1 - self._alpha_fast) * self._fast_ema
        self._slow_ema = self._alpha_slow * close + (1 - self._alpha_slow) * self._slow_ema

        if self._prev_fast is None or self._prev_slow is None:
            return None

        # Golden cross: fast crosses above slow
        if self._prev_fast <= self._prev_slow and self._fast_ema > self._slow_ema:
            signal = Signal(
                type=SignalType.BUY,
                ticker=bar.ticker,
                reason=f"golden_cross fast={self._fast_ema:.4f} slow={self._slow_ema:.4f}",
            )
            self._log.info(
                "signal_generated",
                ticker=bar.ticker,
                signal="BUY",
                fast_ema=round(self._fast_ema, 4),
                slow_ema=round(self._slow_ema, 4),
            )
            return signal

        # Death cross: fast crosses below slow
        if self._prev_fast >= self._prev_slow and self._fast_ema < self._slow_ema:
            signal = Signal(
                type=SignalType.SELL,
                ticker=bar.ticker,
                reason=f"death_cross fast={self._fast_ema:.4f} slow={self._slow_ema:.4f}",
            )
            self._log.info(
                "signal_generated",
                ticker=bar.ticker,
                signal="SELL",
                fast_ema=round(self._fast_ema, 4),
                slow_ema=round(self._slow_ema, 4),
            )
            return signal

        return None

    def _compute_ema(self, prices: Deque[float], period: int) -> float | None:
        """Compute EMA from a price buffer — used only during seeding."""
        if len(prices) < period:
            return None
        alpha = 2.0 / (period + 1)
        it = iter(prices)
        ema = next(it)
        for price in it:
            ema = alpha * price + (1 - alpha) * ema
        return ema
