"""
RSI Crossover Strategy.

Generates signals based on RSI crossing key threshold levels:

  BUY  — RSI was below rsi_oversold on the previous bar and is at or above
          it on the current bar (oversold → recovery cross).
          Gated by the same trend filter as ema_cross_filtered:
          close must be above SMA(trend_sma_period).

  SELL — RSI was at or above rsi_overbought on the previous bar and is
          below it on the current bar (overbought → retreat cross).
          Also exits on per-trade stop loss (same logic as ema_cross_filtered).

Configuration params (all have defaults, all pulled from Settings via kwargs):
    rsi_period        (default 14)
    rsi_oversold      (default 30.0)
    rsi_overbought    (default 70.0)
    trend_sma_period  (default 200)
    stop_loss_pct     (default 1.5)

RSI is calculated using pandas-ta (Wilder smoothing — not hand-rolled).
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import pandas as pd
import pandas_ta as ta

from bot.data.feed import Bar
from bot.logging.logger import get_logger
from bot.strategies.base import Signal, SignalType


class RSIStrategy:
    """RSI threshold-crossing strategy with trend filter and per-trade stop loss."""

    name = "rsi"

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        trend_sma_period: int = 200,
        stop_loss_pct: float = 1.5,
        **kwargs,                       # absorb extra config params gracefully
    ) -> None:
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.trend_sma_period = trend_sma_period
        self.stop_loss_pct = stop_loss_pct

        # Rolling price buffer for RSI — needs at least rsi_period*3+1 bars
        # for pandas-ta's Wilder EMA to be stable.
        self._rsi_prices: Deque[float] = deque(maxlen=rsi_period * 3 + 1)
        self._trend_prices: Deque[float] = deque(maxlen=trend_sma_period)

        # RSI state for crossover detection
        self._prev_rsi: float | None = None

        # Position tracking (same _pending_entry pattern as ema_cross_filtered)
        self._in_position: bool = False
        self._entry_price: float = 0.0
        self._pending_entry: bool = False

        self._log = get_logger(__name__)

    # ------------------------------------------------------------------
    # Strategy protocol
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        self._in_position = False
        self._entry_price = 0.0
        self._pending_entry = False
        self._prev_rsi = None
        self._log.info(
            "strategy_started",
            name=self.name,
            rsi_period=self.rsi_period,
            rsi_oversold=self.rsi_oversold,
            rsi_overbought=self.rsi_overbought,
        )

    def on_stop(self) -> None:
        self._log.info("strategy_stopped", name=self.name)

    def on_bar(self, bar: Bar) -> Signal | None:
        close = bar.close

        # 1. Always update price buffers first
        self._rsi_prices.append(close)
        self._trend_prices.append(close)

        # 2. Finalise fill price on the bar after a BUY signal
        if self._pending_entry:
            self._entry_price = bar.open   # matches backtester fill price
            self._pending_entry = False

        # 3. Stop loss — highest priority, checked every bar while in position
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

        # 4. Compute RSI; skip until buffer is warm
        curr_rsi = self._compute_rsi()
        if curr_rsi is None:
            self._prev_rsi = None
            return None

        prev_rsi = self._prev_rsi
        self._prev_rsi = curr_rsi

        if prev_rsi is None:
            return None     # need two RSI values to detect a cross

        # 5. RSI crosses UP through oversold → BUY candidate
        if prev_rsi < self.rsi_oversold <= curr_rsi and not self._in_position:
            # Trend filter: close must be above SMA(trend_sma_period)
            if len(self._trend_prices) >= self.trend_sma_period:
                sma = sum(self._trend_prices) / self.trend_sma_period
                if close <= sma:
                    self._log.info(
                        "buy_signal_blocked",
                        reason="trend_filter",
                        close=round(close, 4),
                        sma=round(sma, 4),
                    )
                    return None

            self._in_position = True
            self._entry_price = close   # provisional; overwritten on next bar
            self._pending_entry = True
            self._log.info(
                "signal_generated",
                ticker=bar.ticker,
                signal="BUY",
                rsi=round(curr_rsi, 2),
            )
            return Signal(
                type=SignalType.BUY,
                ticker=bar.ticker,
                reason=f"rsi_oversold_cross rsi={curr_rsi:.2f}",
            )

        # 6. RSI crosses DOWN through overbought → SELL (only when in position)
        if prev_rsi >= self.rsi_overbought > curr_rsi and self._in_position:
            self._in_position = False
            self._log.info(
                "signal_generated",
                ticker=bar.ticker,
                signal="SELL",
                rsi=round(curr_rsi, 2),
            )
            return Signal(
                type=SignalType.SELL,
                ticker=bar.ticker,
                reason=f"rsi_overbought_cross rsi={curr_rsi:.2f}",
            )

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
