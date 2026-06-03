"""
Unit tests for RSIStrategy.

All tests use synthetic bar sequences — no live connection or API calls.
Small rsi_period (3) keeps warmup short while still exercising real pandas-ta
RSI logic (Wilder smoothing).

Price engineering notes
-----------------------
With rsi_period=3:
  * A sustained decline (many down bars) drives RSI toward 0.
  * A sustained rise (many up bars) drives RSI toward 100.
  * A reversal after strong decline causes RSI to cross up through 30.
  * A reversal after strong rise causes RSI to cross down through 70.

The 'warm-up' period is rsi_period+1 bars before the first RSI value
is available; one more bar provides the prev_rsi needed for crossover
detection.  For rsi_period=3 that is ~5+ bars before any signal.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pytest

from bot.data.feed import Bar
from bot.strategies.base import SignalType
from bot.strategies.rsi_strategy import RSIStrategy


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_ema_cross.py)
# ---------------------------------------------------------------------------


def _bar(close: float, ticker: str = "TEST") -> Bar:
    return Bar(
        ticker=ticker,
        timestamp=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc),
        open=close, high=close, low=close, close=close, volume=1000,
    )


def _feed(strategy, prices: List[float]) -> List:
    return [s for p in prices if (s := strategy.on_bar(_bar(p))) is not None]


def _make_rsi(
    rsi_period: int = 3,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    trend_sma_period: int = 3,   # small so trend filter is easy to satisfy
    stop_loss_pct: float = 99.0, # never triggers by default
) -> RSIStrategy:
    return RSIStrategy(
        rsi_period=rsi_period,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        trend_sma_period=trend_sma_period,
        stop_loss_pct=stop_loss_pct,
    )


# ---------------------------------------------------------------------------
# Test 1 — BUY fires when RSI crosses UP through oversold
# ---------------------------------------------------------------------------

def test_buy_on_rsi_oversold_cross():
    """
    A strong decline drives RSI well below 30; recovery causes the oversold
    cross and a BUY signal should fire.
    Price engineering (rsi_period=3, trend_sma_period=3):
      - 10 bars falling 100→10: RSI → ~0 (all losses)
      - 3 rising bars to 60, 70, 80: RSI recovers, crosses up through 30
      - trend_sma will be the last 3 bars which are rising, so close > SMA
    """
    strat = _make_rsi()

    # Strong decline → RSI goes to near 0
    _feed(strat, [100.0 - i * 9 for i in range(11)])   # 100, 91, 82, ..., 10

    # Recovery — RSI should cross up through 30 somewhere here
    signals = _feed(strat, [20.0, 35.0, 50.0, 65.0, 80.0])
    buy_signals = [s for s in signals if s.type == SignalType.BUY]
    assert len(buy_signals) >= 1, (
        f"Expected at least one BUY on RSI oversold cross; got {signals}"
    )


# ---------------------------------------------------------------------------
# Test 2 — SELL fires when RSI crosses DOWN through overbought
# ---------------------------------------------------------------------------

def test_sell_on_rsi_overbought_cross():
    """
    Strong rise drives RSI above 70; a pullback causes the overbought cross
    and a SELL signal should fire.
    Strategy must be in a position (BUY must have fired first).
    """
    strat = _make_rsi()

    # First get into a position via the oversold path
    _feed(strat, [100.0 - i * 9 for i in range(11)])   # drive RSI low
    _feed(strat, [20.0, 35.0, 50.0, 65.0, 80.0])       # BUY fires somewhere

    # Now drive prices strongly upward so RSI goes above 70
    _feed(strat, [80.0 + i * 5 for i in range(10)])     # 80, 85, 90, ... 125

    # A pullback causes RSI to drop below 70 → SELL
    signals = _feed(strat, [120.0, 115.0, 108.0])
    sell_signals = [s for s in signals if s.type == SignalType.SELL]
    assert len(sell_signals) >= 1, (
        f"Expected at least one SELL on RSI overbought cross; got {signals}"
    )


# ---------------------------------------------------------------------------
# Test 3 — BUY is blocked when price is below trend SMA
# ---------------------------------------------------------------------------

def test_buy_blocked_below_trend_sma():
    """
    RSI crosses up through oversold but price is below SMA(trend_sma_period).
    The trend filter should suppress the BUY.

    Engineering (trend_sma_period=10):
      - 7 bars at 200 → SMA anchored high (~200)
      - Crash to 20, 10, 5, 3 → RSI → near 0; SMA(10) still ~144
      - Recovery to 15, 25, 40: RSI crosses up through 30; price (~25-40) < SMA(144)
    """
    strat = _make_rsi(trend_sma_period=10)

    _feed(strat, [200.0] * 7)                              # anchor SMA at 200
    _feed(strat, [20.0, 10.0, 5.0, 3.0])                  # RSI → ~0; SMA drops slowly
    signals = _feed(strat, [8.0, 15.0, 25.0, 40.0, 55.0]) # RSI recovery cross below SMA

    buy_signals = [s for s in signals if s.type == SignalType.BUY]
    assert buy_signals == [], (
        f"Expected no BUY when price is below trend SMA; got {buy_signals}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Stop loss fires while in position
# ---------------------------------------------------------------------------

def test_stop_loss_fires():
    """After a BUY, a bar 10% below the fill price should trigger stop_loss SELL."""
    strat = _make_rsi(stop_loss_pct=5.0)

    # Get into a position
    _feed(strat, [100.0 - i * 9 for i in range(11)])
    buy_sigs = _feed(strat, [20.0, 35.0, 50.0, 65.0, 80.0])
    assert any(s.type == SignalType.BUY for s in buy_sigs), "Need BUY to test stop loss"

    # Feed fill bar so _entry_price is finalised (open = close in _bar())
    _feed(strat, [80.0])

    entry = strat._entry_price
    stop_price = entry * 0.90   # 10% below — well past the 5% stop
    sell_sigs = _feed(strat, [stop_price])

    assert len(sell_sigs) == 1
    assert sell_sigs[0].reason == "stop_loss"


# ---------------------------------------------------------------------------
# Test 5 — No API calls on instantiation
# ---------------------------------------------------------------------------

def test_no_api_calls_on_instantiation():
    """Constructing an RSIStrategy must never make any network call."""
    RSIStrategy(rsi_period=14, rsi_oversold=30, rsi_overbought=70,
                trend_sma_period=200, stop_loss_pct=1.5)
