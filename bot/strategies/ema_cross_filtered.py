"""
EMA Crossover — filtered variant.

Extends the baseline EMA crossover strategy with three signal-quality
checks that gate BUY entries only (SELL signals and stop-loss exits are
never suppressed):

  FILTER 1 — Trend filter
    Only emit BUY if close > SMA(trend_sma_period).
    Rationale: price below its long-run average = downtrend; EMA crossovers
    in a downtrend are statistically unreliable.

  FILTER 2 — RSI confirmation
    Only emit BUY if RSI(rsi_period) < rsi_overbought.
    Rationale: entering when RSI is already overbought risks buying an
    exhausted move.  Calculated via pandas-ta (not hand-rolled).

  FILTER 3 — Per-trade stop loss
    Tracked inside on_bar(), not in the risk manager.
    After a BUY is emitted, record entry_price.  On every subsequent bar,
    if close < entry_price * (1 - stop_loss_pct / 100) emit SELL("stop_loss").

Blocked-signal log events (for observability):
  "buy_signal_blocked", reason="trend_filter", close=x, sma200=y
  "buy_signal_blocked", reason="rsi_overbought", rsi=x
  "stop_loss_triggered", entry=x, current=y, loss_pct=z
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import pandas as pd
import pandas_ta as ta

from bot.data.feed import Bar
from bot.strategies.base import Signal, SignalType
from bot.strategies.ema_cross import EmaCrossStrategy


class EmaCrossFilteredStrategy(EmaCrossStrategy):
    """EMA crossover with trend filter, RSI confirmation, and per-trade stop loss."""

    name = "ema_cross_filtered"

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        trend_sma_period: int = 200,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        stop_loss_pct: float = 1.5,
        **kwargs,
    ) -> None:
        super().__init__(fast_period=fast_period, slow_period=slow_period)
        self.trend_sma_period = trend_sma_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.stop_loss_pct = stop_loss_pct

        # Indicator buffers (separate from EMA buffers in parent)
        self._trend_prices: Deque[float] = deque(maxlen=trend_sma_period)
        self._rsi_prices: Deque[float] = deque(maxlen=rsi_period * 3 + 1)

        # Position tracking for stop loss.
        # _pending_entry is True for exactly one bar after the BUY signal fires.
        # On that next bar we set _entry_price = bar.open, which matches the
        # backtester's fill price (fills at next bar's open after the signal).
        self._in_position: bool = False
        self._entry_price: float = 0.0
        self._pending_entry: bool = False

    def on_start(self) -> None:
        self._in_position = False
        self._entry_price = 0.0
        self._pending_entry = False
        super().on_start()

    def on_bar(self, bar: Bar) -> Signal | None:
        close = bar.close

        # Always update all indicator buffers first
        self._trend_prices.append(close)
        self._rsi_prices.append(close)

        # Always update EMA (even when stop loss fires, so future signals stay accurate)
        base_signal = super().on_bar(bar)

        # Resolve the fill price on the bar immediately following the BUY signal.
        # The backtester fills at the NEXT bar's open after the signal fires, so
        # bar.open here is exactly the price we paid — anchor the stop to that.
        if self._pending_entry:
            self._entry_price = bar.open
            self._pending_entry = False

        # ── FILTER 3: Stop loss (highest priority — overrides any EMA signal) ──
        if self._in_position:
            stop_price = self._entry_price * (1.0 - self.stop_loss_pct / 100.0)
            if close < stop_price:
                loss_pct = (self._entry_price - close) / self._entry_price * 100.0
                self._log.info(
                    "stop_loss_triggered",
                    entry=round(self._entry_price, 4),
                    current=round(close, 4),
                    loss_pct=round(loss_pct, 2),
                )
                self._in_position = False
                return Signal(type=SignalType.SELL, ticker=bar.ticker, reason="stop_loss")

        if base_signal is None:
            return None

        # ── SELL signals always pass through — never suppress an exit ──
        if base_signal.type == SignalType.SELL:
            self._in_position = False
            return base_signal

        # ── BUY signal: apply entry filters ──
        if base_signal.type == SignalType.BUY:

            # Filter 1 — Trend filter
            if len(self._trend_prices) >= self.trend_sma_period:
                sma = sum(self._trend_prices) / self.trend_sma_period
                if close <= sma:
                    self._log.info(
                        "buy_signal_blocked",
                        reason="trend_filter",
                        close=round(close, 4),
                        sma200=round(sma, 4),
                    )
                    return None

            # Filter 2 — RSI confirmation
            rsi_val = self._compute_rsi()
            if rsi_val is not None and rsi_val >= self.rsi_overbought:
                self._log.info(
                    "buy_signal_blocked",
                    reason="rsi_overbought",
                    rsi=round(rsi_val, 2),
                )
                return None

            # All filters passed — emit BUY.
            # Set _pending_entry so that on the *next* bar (the fill bar) we
            # update _entry_price to bar.open, matching the backtester's fill price.
            self._in_position = True
            self._entry_price = close   # provisional; overwritten on next bar
            self._pending_entry = True
            return base_signal

        return None

    def _compute_rsi(self) -> float | None:
        """Compute RSI using pandas-ta on the rolling price buffer."""
        if len(self._rsi_prices) < self.rsi_period + 1:
            return None
        series = pd.Series(list(self._rsi_prices))
        rsi_series = ta.rsi(series, length=self.rsi_period)
        if rsi_series is None or rsi_series.empty:
            return None
        last_val = rsi_series.iloc[-1]
        if pd.isna(last_val):
            return None
        return float(last_val)
