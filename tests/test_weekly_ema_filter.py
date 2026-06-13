"""
Unit tests for FILTER 9 — weekly EMA confirmation
(bot/strategies/ema_cross_filtered.py).

All tests use synthetic daily bars and monkeypatch
``_fetch_weekly_ema_bullish`` so no network access ever occurs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import bot.strategies.ema_cross_filtered as ecf
from bot.data.feed import Bar
from bot.strategies.base import SignalType
from bot.strategies.ema_cross_filtered import EmaCrossFilteredStrategy


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/test_ema_cross.py)
# ---------------------------------------------------------------------------


def _bar(close: float, ticker: str = "TEST") -> Bar:
    return Bar(
        ticker=ticker,
        timestamp=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc),
        open=close, high=close, low=close, close=close, volume=1000,
    )


def _feed(strategy, prices: List[float], ticker: str = "TEST") -> List:
    return [s for p in prices if (s := strategy.on_bar(_bar(p, ticker))) is not None]


def _make_filtered(weekly_ema_filter: bool, **overrides) -> EmaCrossFilteredStrategy:
    params = dict(
        fast_period=3, slow_period=5, trend_sma_period=10,
        rsi_period=3, rsi_overbought=100.0,   # RSI never blocks
        stop_loss_pct=99.0,                   # stop never triggers
        volume_multiplier=0.0,                # volume filter never blocks
        take_profit_rr=0.0,                   # take-profit disabled
        weekly_ema_filter=weekly_ema_filter,
    )
    params.update(overrides)
    return EmaCrossFilteredStrategy(**params)


def _feed_to_buy(strategy, ticker: str = "TEST") -> List:
    """Drive the strategy through a golden cross + trend recovery -> BUY attempt."""
    _feed(strategy, [100.0] * 10, ticker)   # seed trend SMA + EMA at 100
    _feed(strategy, [50.0] * 5, ticker)     # force death cross; SMA drops
    return _feed(strategy, [150.0, 160.0, 170.0], ticker)  # recovery above SMA


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_weekly_ema_bearish_blocks_buy(monkeypatch):
    """BUY is blocked when the weekly EMA(20) < EMA(50) (bearish)."""
    monkeypatch.setattr(ecf, "_fetch_weekly_ema_bullish", lambda ticker: False)

    strat = _make_filtered(weekly_ema_filter=True)
    signals = _feed_to_buy(strat)

    buy_signals = [s for s in signals if s.type == SignalType.BUY]
    assert buy_signals == [], f"Expected BUY blocked by weekly EMA filter, got {buy_signals}"


def test_weekly_ema_bullish_passes_buy(monkeypatch):
    """BUY passes through when the weekly EMA(20) > EMA(50) (bullish)."""
    monkeypatch.setattr(ecf, "_fetch_weekly_ema_bullish", lambda ticker: True)

    strat = _make_filtered(weekly_ema_filter=True)
    signals = _feed_to_buy(strat)

    buy_signals = [s for s in signals if s.type == SignalType.BUY]
    assert len(buy_signals) >= 1, "Expected BUY to pass when weekly EMA is bullish"


def test_weekly_ema_filter_disabled_skips_check(monkeypatch):
    """When weekly_ema_filter=False, BUY passes and the fetch is never called."""
    calls: list[str] = []

    def _fail_if_called(ticker: str):
        calls.append(ticker)
        return False

    monkeypatch.setattr(ecf, "_fetch_weekly_ema_bullish", _fail_if_called)

    strat = _make_filtered(weekly_ema_filter=False)
    signals = _feed_to_buy(strat)

    buy_signals = [s for s in signals if s.type == SignalType.BUY]
    assert len(buy_signals) >= 1, "Expected BUY to pass when the filter is disabled"
    assert calls == [], "Weekly EMA data should never be fetched when the filter is disabled"


def test_weekly_ema_unavailable_fails_permissive(monkeypatch):
    """If data is unavailable (None), the BUY is allowed (fail permissive)."""
    monkeypatch.setattr(ecf, "_fetch_weekly_ema_bullish", lambda ticker: None)

    strat = _make_filtered(weekly_ema_filter=True)
    signals = _feed_to_buy(strat)

    buy_signals = [s for s in signals if s.type == SignalType.BUY]
    assert len(buy_signals) >= 1, "Expected BUY to pass when weekly EMA data is unavailable"
    # Permissive result is cached as bullish for this ticker
    assert strat._weekly_ema_cache.get("TEST") is True


def test_weekly_ema_result_is_cached_per_ticker(monkeypatch):
    """The weekly EMA check should be fetched at most once per ticker."""
    calls: list[str] = []

    def _counting_fetch(ticker: str):
        calls.append(ticker)
        return True

    monkeypatch.setattr(ecf, "_fetch_weekly_ema_bullish", _counting_fetch)

    strat = _make_filtered(weekly_ema_filter=True)
    # Two full BUY/cycle attempts on the same ticker
    _feed_to_buy(strat)
    _feed(strat, [50.0] * 5)          # death cross to allow a second BUY attempt
    _feed(strat, [150.0, 160.0, 170.0])

    assert calls.count("TEST") <= 1, f"Expected at most one fetch per ticker, got {calls}"


def test_fetch_weekly_ema_bullish_no_network_dependency():
    """The fetch helper itself must exist and be importable without making a call."""
    assert callable(ecf._fetch_weekly_ema_bullish)
