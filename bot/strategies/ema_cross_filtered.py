"""
EMA Crossover — filtered variant (v2).

Extends v1 with three new improvements:

  FILTER 4 — Volume confirmation
    Only emit BUY if current bar volume >= volume_multiplier × N-bar average.
    Rationale: low-volume crossovers are statistically less reliable breakouts.
    Log event: "buy_signal_blocked", reason="low_volume"

  EXIT 1 — Trailing stop loss (replaces static stop)
    Initial floor = entry_price × (1 - stop_loss_pct / 100).
    As price rises the floor ratchets up:
        current_stop = max(initial_stop, highest_price × (1 - trailing_stop_pct / 100))
    The stop only ever moves up — never down.
    Rationale: locks in profit as a trade moves in our favour.
    Log event: "stop_loss_triggered", trailing=True/False

  EXIT 2 — Take-profit target (set take_profit_rr=0 to disable)
    target = entry_price + (entry_price × stop_loss_pct/100) × take_profit_rr
    Exits at a fixed R:R multiple rather than waiting for EMA crossover reversal.
    Rationale: captures gains before a mean-reversion eats them.
    Log event: "take_profit_triggered"

  FILTER 9 — Weekly EMA confirmation (set weekly_ema_filter=False to disable)
    Only emit BUY if the ticker's weekly EMA(20) > weekly EMA(50).
    Fetches ~60 weeks of weekly closes via yfinance (lazily, once per ticker
    per strategy instance, then cached).
    Rationale: avoid taking daily-chart entries against the weekly trend.
    Fails permissively — if yfinance is unavailable or returns insufficient
    history, the check is skipped and the BUY is allowed.
    Log event: "buy_signal_blocked", reason="weekly_ema_bearish"

All v1 filters remain unchanged:
  FILTER 1 — Trend SMA gate (close > SMA(trend_sma_period))
  FILTER 2 — RSI overbought gate (RSI < rsi_overbought)
  FILTER 3 — Stop loss (now superseded by EXIT 1, kept for log compatibility)
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
    """EMA crossover with trend/RSI/volume filters, trailing stop, and take-profit."""

    name = "ema_cross_filtered"

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        trend_sma_period: int = 150,
        rsi_period: int = 14,
        rsi_overbought: float = 75.0,
        stop_loss_pct: float = 2.5,
        trailing_stop_pct: float = 2.0,
        take_profit_rr: float = 3.0,
        volume_lookback: int = 20,
        volume_multiplier: float = 1.0,
        weekly_ema_filter: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(fast_period=fast_period, slow_period=slow_period)

        self.trend_sma_period = trend_sma_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_rr = take_profit_rr
        self.volume_lookback = volume_lookback
        self.volume_multiplier = volume_multiplier
        self.weekly_ema_filter = weekly_ema_filter

        # Indicator buffers
        self._trend_prices: Deque[float] = deque(maxlen=trend_sma_period)
        self._rsi_prices: Deque[float] = deque(maxlen=rsi_period * 3 + 1)
        self._volume_buffer: Deque[float] = deque(maxlen=volume_lookback)

        # Per-ticker cache for the weekly EMA confirmation check (Filter 9) —
        # fetched at most once per ticker per strategy instance.
        self._weekly_ema_cache: dict[str, bool] = {}

        # Position state
        # _pending_entry: True for exactly one bar after BUY fires.
        # On that next bar we set _entry_price = bar.open (actual fill price).
        self._in_position: bool = False
        self._entry_price: float = 0.0
        self._pending_entry: bool = False
        self._highest_price: float = 0.0       # trailing stop anchor
        self._take_profit_price: float = 0.0   # 0 = disabled

    def on_start(self) -> None:
        self._in_position = False
        self._entry_price = 0.0
        self._pending_entry = False
        self._highest_price = 0.0
        self._take_profit_price = 0.0
        super().on_start()

    def _reset_position_state(self) -> None:
        self._in_position = False
        self._highest_price = 0.0
        self._take_profit_price = 0.0

    def on_bar(self, bar: Bar) -> Signal | None:
        close = bar.close

        # Always update all indicator buffers first
        self._trend_prices.append(close)
        self._rsi_prices.append(close)
        volume = getattr(bar, "volume", None)
        if volume and float(volume) > 0:
            self._volume_buffer.append(float(volume))

        # Always update EMA (even when stop fires, so future signals stay accurate)
        base_signal = super().on_bar(bar)

        # ── Resolve fill price on the bar immediately following the BUY ──────
        # Backtester fills at NEXT bar's open — anchor stops to that price.
        if self._pending_entry:
            self._entry_price = bar.open
            self._pending_entry = False
            self._highest_price = self._entry_price
            # Set take-profit target (disabled if take_profit_rr == 0)
            if self.take_profit_rr > 0:
                initial_risk = self._entry_price * self.stop_loss_pct / 100.0
                self._take_profit_price = self._entry_price + initial_risk * self.take_profit_rr
            else:
                self._take_profit_price = 0.0
            self._log.info(
                "position_opened",
                entry=round(self._entry_price, 4),
                initial_stop=round(self._entry_price * (1 - self.stop_loss_pct / 100), 4),
                take_profit=round(self._take_profit_price, 4) if self._take_profit_price else "disabled",
                trailing_stop_pct=self.trailing_stop_pct,
            )

        # ── EXIT CHECKS (highest priority — checked before any new signal) ───
        if self._in_position:
            # Ratchet trailing high
            if close > self._highest_price:
                self._highest_price = close

            # Current stop = max of initial floor and trailing stop
            initial_stop  = self._entry_price * (1.0 - self.stop_loss_pct / 100.0)
            trailing_stop = self._highest_price * (1.0 - self.trailing_stop_pct / 100.0)
            current_stop  = max(initial_stop, trailing_stop)
            trailing_active = trailing_stop > initial_stop

            # Take-profit (check before stop — upside exit takes priority)
            if self._take_profit_price > 0 and close >= self._take_profit_price:
                gain_pct = (close - self._entry_price) / self._entry_price * 100.0
                self._log.info(
                    "take_profit_triggered",
                    entry=round(self._entry_price, 4),
                    current=round(close, 4),
                    target=round(self._take_profit_price, 4),
                    gain_pct=round(gain_pct, 2),
                )
                self._reset_position_state()
                return Signal(type=SignalType.SELL, ticker=bar.ticker, reason="take_profit")

            # Trailing / initial stop
            if close < current_stop:
                loss_pct = (self._entry_price - close) / self._entry_price * 100.0
                self._log.info(
                    "stop_loss_triggered",
                    entry=round(self._entry_price, 4),
                    current=round(close, 4),
                    stop=round(current_stop, 4),
                    loss_pct=round(loss_pct, 2),
                    trailing=trailing_active,
                    highest=round(self._highest_price, 4),
                )
                self._reset_position_state()
                return Signal(type=SignalType.SELL, ticker=bar.ticker, reason="stop_loss")

        if base_signal is None:
            return None

        # ── SELL signals always pass through — never suppress an exit ─────────
        if base_signal.type == SignalType.SELL:
            self._reset_position_state()
            return base_signal

        # ── BUY signal: apply entry filters ───────────────────────────────────
        if base_signal.type == SignalType.BUY:

            # Filter 1 — Trend filter
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

            # Filter 2 — RSI confirmation
            rsi_val = self._compute_rsi()
            if rsi_val is not None and rsi_val >= self.rsi_overbought:
                self._log.info(
                    "buy_signal_blocked",
                    reason="rsi_overbought",
                    rsi=round(rsi_val, 2),
                )
                return None

            # Filter 4 — Volume confirmation
            if (
                self.volume_multiplier > 0
                and len(self._volume_buffer) >= self.volume_lookback
            ):
                # Current bar is already in the buffer — compare against prior bars
                prior_bars = list(self._volume_buffer)[:-1]
                if prior_bars:
                    avg_volume = sum(prior_bars) / len(prior_bars)
                    current_volume = self._volume_buffer[-1]
                    if avg_volume > 0 and current_volume < avg_volume * self.volume_multiplier:
                        self._log.info(
                            "buy_signal_blocked",
                            reason="low_volume",
                            volume=round(current_volume),
                            avg_volume=round(avg_volume),
                            ratio=round(current_volume / avg_volume, 2),
                        )
                        return None

            # Filter 9 — Weekly EMA confirmation
            if self.weekly_ema_filter and not self._weekly_ema_bullish(bar.ticker):
                self._log.info(
                    "buy_signal_blocked",
                    reason="weekly_ema_bearish",
                    ticker=bar.ticker,
                )
                return None

            # All filters passed — emit BUY
            self._in_position = True
            self._entry_price = close   # provisional; overwritten on next bar open
            self._pending_entry = True
            return base_signal

        return None

    def _weekly_ema_bullish(self, ticker: str) -> bool:
        """
        Return True if the ticker's weekly EMA(20) > EMA(50), cached per ticker
        for the lifetime of this strategy instance.

        Fails permissively (returns True) if data is unavailable.
        """
        if ticker in self._weekly_ema_cache:
            return self._weekly_ema_cache[ticker]

        bullish = _fetch_weekly_ema_bullish(ticker)
        if bullish is None:
            self._log.warning("weekly_ema_data_unavailable", ticker=ticker)
            bullish = True  # fail permissive

        self._weekly_ema_cache[ticker] = bullish
        return bullish

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


def _fetch_weekly_ema_bullish(ticker: str) -> bool | None:
    """
    Fetch ~60 weeks of weekly closes via yfinance and compute EMA(20)/EMA(50).

    Returns:
        True  — weekly EMA(20) > EMA(50) (bullish)
        False — weekly EMA(20) <= EMA(50) (bearish)
        None  — data unavailable (insufficient history, yfinance error, etc.)

    Module-level so it can be monkeypatched in tests without any network access.
    """
    try:
        import yfinance as yf

        df = yf.Ticker(ticker).history(period="60wk", interval="1wk", auto_adjust=True)
        if df is None or df.empty or len(df) < 50:
            return None

        closes = df["Close"]
        ema20 = ta.ema(closes, length=20)
        ema50 = ta.ema(closes, length=50)
        if ema20 is None or ema50 is None or ema20.empty or ema50.empty:
            return None

        last20, last50 = ema20.iloc[-1], ema50.iloc[-1]
        if pd.isna(last20) or pd.isna(last50):
            return None

        return bool(last20 > last50)
    except Exception:
        return None
