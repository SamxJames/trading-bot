"""
Unit tests for EMA crossover strategies (baseline and filtered).

All tests use synthetic bar sequences — no live connection or API calls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pytest

from bot.data.feed import Bar
from bot.strategies.base import SignalType
from bot.strategies.ema_cross import EmaCrossStrategy
from bot.strategies.ema_cross_filtered import EmaCrossFilteredStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bar(close: float, ticker: str = "TEST") -> Bar:
    return Bar(
        ticker=ticker,
        timestamp=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc),
        open=close, high=close, low=close, close=close, volume=1000,
    )


def _feed(strategy, prices: List[float]) -> List:
    return [s for p in prices if (s := strategy.on_bar(_bar(p))) is not None]


# ---------------------------------------------------------------------------
# Baseline EMA crossover tests
# ---------------------------------------------------------------------------


def test_no_signal_before_warmup():
    strat = EmaCrossStrategy(fast_period=3, slow_period=5)
    for price in [10.0, 10.1, 10.2, 10.3]:           # slow_period - 1 bars
        assert strat.on_bar(_bar(price)) is None


def test_no_signal_on_seed_bar():
    strat = EmaCrossStrategy(fast_period=3, slow_period=5)
    assert _feed(strat, [10.0, 10.1, 10.2, 10.3, 10.4]) == []


def test_buy_signal_on_golden_cross():
    strat = EmaCrossStrategy(fast_period=3, slow_period=5)
    _feed(strat, [100.0, 99.0, 98.0, 97.0, 96.0])    # seed with decline
    signals = _feed(strat, [96.0, 96.0, 120.0, 130.0, 140.0])
    assert any(s.type == SignalType.BUY for s in signals)


def test_sell_signal_on_death_cross():
    strat = EmaCrossStrategy(fast_period=3, slow_period=5)
    _feed(strat, [100.0, 101.0, 102.0, 103.0, 104.0])
    signals = _feed(strat, [104.0, 104.0, 80.0, 70.0, 60.0])
    assert any(s.type == SignalType.SELL for s in signals)


def test_no_signal_on_flat_prices():
    strat = EmaCrossStrategy(fast_period=3, slow_period=5)
    assert _feed(strat, [100.0] * 30) == []


def test_signal_carries_ticker():
    strat = EmaCrossStrategy(fast_period=3, slow_period=5)
    _feed(strat, [100.0, 99.0, 98.0, 97.0, 96.0])
    for sig in _feed(strat, [96.0, 96.0, 130.0, 140.0, 150.0]):
        assert sig.ticker == "TEST"


def test_fast_must_be_less_than_slow():
    with pytest.raises(ValueError):
        EmaCrossStrategy(fast_period=21, slow_period=9)
    with pytest.raises(ValueError):
        EmaCrossStrategy(fast_period=9, slow_period=9)


def test_signal_reason_not_empty():
    strat = EmaCrossStrategy(fast_period=3, slow_period=5)
    _feed(strat, [100.0, 99.0, 98.0, 97.0, 96.0])
    for sig in _feed(strat, [96.0, 96.0, 130.0, 140.0, 150.0]):
        assert sig.reason


# ---------------------------------------------------------------------------
# Filtered strategy tests
# ---------------------------------------------------------------------------
#
# We use small periods throughout so tests run in < 1 s with no warmup pain.
#
# Price engineering for trend-filter test (fast=3, slow=5, trend_sma_period=10):
#   Step 1: 10 bars at 200 → SMA=200, EMAs seeded at 200
#   Step 2:  5 bars at  50 → after decay: fast≈54.7, slow≈69.8 (fast < slow)
#   Step 3: bar at 90 (fast≈72.4, slow≈76.4 — still no cross)
#           bar at 95 → fast≈83.7, slow≈82.8  ← GOLDEN CROSS
#   At the golden cross: last-10 SMA ≈ 103.5, price=95 < SMA → BLOCKED
# ---------------------------------------------------------------------------


def _make_filtered(
    fast=3, slow=5, trend_sma=10,
    rsi_period=3, rsi_overbought=100.0,   # RSI=100 → never blocks by default
    stop_loss_pct=99.0,                   # 99 % → never triggers by default
):
    return EmaCrossFilteredStrategy(
        fast_period=fast,
        slow_period=slow,
        trend_sma_period=trend_sma,
        rsi_period=rsi_period,
        rsi_overbought=rsi_overbought,
        stop_loss_pct=stop_loss_pct,
    )


def test_filtered_passes_buy_above_sma():
    """BUY is NOT blocked when price is above the trend SMA."""
    strat = _make_filtered()
    # All prices at 100 → SMA=100; then spike to 150 which is above SMA
    _feed(strat, [100.0] * 10)                         # seed trend + EMA
    _feed(strat, [50.0] * 5)                           # force death cross
    signals = _feed(strat, [150.0, 160.0, 170.0])      # recovery above SMA
    # At 150, SMA of last 10 = (5×100 + 5×50 + some recovery) / 10 ≤ 150
    # So at least some BUY should pass through
    buy_signals = [s for s in signals if s.type == SignalType.BUY]
    assert len(buy_signals) >= 1


def test_filtered_blocks_buy_below_trend_sma():
    """BUY is blocked when price is below SMA(trend_sma_period)."""
    strat = _make_filtered()
    _feed(strat, [200.0] * 10)   # SMA anchored at 200
    _feed(strat, [50.0] * 5)     # force fast < slow; prices crash to 50
    # Golden cross will fire at ~95 (see engineering note above) but 95 < SMA≈103.5
    signals = _feed(strat, [90.0, 95.0, 100.0])
    buy_signals = [s for s in signals if s.type == SignalType.BUY]
    assert buy_signals == [], f"Expected no BUY below SMA, got {buy_signals}"


def test_filtered_blocks_buy_rsi_overbought():
    """BUY is blocked when RSI is at or above rsi_overbought threshold."""
    # Use a very low rsi_overbought (30) and rising prices to guarantee RSI > 30.
    # trend_sma_period=3 so trend filter is easily satisfied.
    strat = _make_filtered(
        fast=3, slow=5, trend_sma=3,
        rsi_period=3, rsi_overbought=30.0,   # very aggressive threshold
        stop_loss_pct=99.0,
    )
    # Seed with low prices so SMA stays low (trend filter passes)
    _feed(strat, [50.0] * 5)                  # seed EMA at 50, SMA=50
    _feed(strat, [30.0] * 5)                  # death cross; prices fall
    # Recovery with very strong gains → RSI will be high → BUY blocked
    signals = _feed(strat, [80.0, 100.0, 120.0, 140.0, 160.0])
    buy_signals = [s for s in signals if s.type == SignalType.BUY]
    # RSI on all-upward bars (30→80→100→120→140→160) >> 30 → all blocked
    assert buy_signals == [], f"Expected BUYs blocked by RSI, got {buy_signals}"


def test_filtered_stop_loss_triggers():
    """
    SELL with reason='stop_loss' fires once the trailing stop has ratcheted
    up and price pulls back below it — even while price remains above the
    original fixed-entry stop level.
    """
    strat = _make_filtered(
        fast=3, slow=5, trend_sma=3,
        rsi_period=3, rsi_overbought=100.0,  # RSI never blocks
        stop_loss_pct=5.0,                   # 5% initial stop
    )
    # Create conditions for a BUY: SMA low (3 bars), then golden cross
    _feed(strat, [100.0] * 5)                # seed at 100
    _feed(strat, [80.0] * 5)                 # death cross; SMA drops

    # Golden cross fires on the bar at 90; the fill is resolved on the next
    # bar's open (100), which becomes the finalised entry price.
    signals_before = _feed(strat, [90.0, 100.0])
    buy_signals = [s for s in signals_before if s.type == SignalType.BUY]
    assert len(buy_signals) >= 1, "Need at least one BUY to test the trailing stop"
    assert strat._entry_price == 100.0

    # Initial (fixed) floor = 100 * (1 - 5%) = 95.
    initial_stop = strat._entry_price * (1 - strat.stop_loss_pct / 100.0)

    # Price rallies to 110 — the trailing stop ratchets up to 110 * (1 - 2%)
    # = 107.8, well above the initial fixed floor of 95.
    _feed(strat, [110.0])
    trailing_stop = strat._highest_price * (1 - strat.trailing_stop_pct / 100.0)
    assert trailing_stop > initial_stop, "Trailing stop should ratchet above the initial floor"

    # Price pulls back to 105 — still above the initial fixed floor (95) but
    # below the ratcheted trailing stop (107.8) → the trailing stop fires.
    stop_price = 105.0
    assert initial_stop < stop_price < trailing_stop
    signals_after = _feed(strat, [stop_price])

    sell_signals = [s for s in signals_after if s.type == SignalType.SELL]
    assert len(sell_signals) == 1
    assert sell_signals[0].reason == "stop_loss"


def test_filtered_sell_passthrough():
    """SELL from death cross always passes through regardless of filters."""
    strat = _make_filtered(
        fast=3, slow=5, trend_sma=3,
        rsi_period=3, rsi_overbought=0.0,   # RSI blocks ALL BUYs
        stop_loss_pct=99.0,
    )
    _feed(strat, [100.0] * 5)
    _feed(strat, [50.0] * 5)               # death cross
    signals = _feed(strat, [100.0, 110.0, 120.0])  # recovery
    # BUYs blocked by RSI=0 threshold; but if price then crashes...
    _feed(strat, [120.0] * 5)             # hold high
    sell_signals = _feed(strat, [40.0, 30.0, 20.0])  # crash → death cross
    assert any(s.type == SignalType.SELL for s in sell_signals)


def test_no_api_calls_in_strategy():
    """Strategies must work with no network access — instantiation is enough."""
    EmaCrossStrategy(fast_period=20, slow_period=50)
    EmaCrossFilteredStrategy(
        fast_period=20, slow_period=50,
        trend_sma_period=200, rsi_period=14,
        rsi_overbought=70.0, stop_loss_pct=1.5,
    )
