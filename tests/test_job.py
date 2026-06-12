"""
Unit tests for bot/job.py.

All Alpaca API calls are mocked — no real network requests are made.
Tests cover the three key job exit paths:
  1. Market closed today  → exits cleanly, no order, no notification.
  2. Market open, no signal → exits cleanly, sends heartbeat notification.
  3. Market open, BUY signal approved → places market order, sends trade notification.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from bot.execution.broker import AccountInfo, Position
from bot.strategies.base import Signal, SignalType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_settings() -> MagicMock:
    """Minimal Settings stand-in that satisfies every field job.py reads."""
    s = MagicMock()
    s.apca_api_key_id        = "fake_key"
    s.apca_api_secret_key    = "fake_secret"
    s.apca_base_url          = "https://paper-api.alpaca.markets"
    s.tickers                = ["AAPL"]
    s.strategy               = "ema_cross_filtered"
    s.fast_period            = 20
    s.slow_period            = 50
    s.trend_sma_period       = 200
    s.rsi_period             = 14
    s.rsi_oversold           = 30.0
    s.rsi_overbought         = 70.0
    s.stop_loss_pct          = 1.5
    s.max_positions          = 3
    s.max_notional_per_trade = 500.0
    s.drawdown_halt_pct      = 5.0
    s.vix_threshold          = 25.0
    s.vix_tight_threshold    = 20.0
    s.vix_tight_stop_pct     = 1.5
    s.spy_macro_filter       = True
    s.spy_sma_period         = 200
    s.earnings_blackout_days = 1
    s.max_correlation        = 0.8
    s.dynamic_correlation    = True
    s.correlation_lookback   = 60
    s.atr_sizing             = False
    s.atr_period             = 14
    s.atr_target_pct         = 2.0
    s.discord_webhook_url    = ""
    return s


def _fake_broker(*, is_trading: bool = True, positions: list | None = None) -> MagicMock:
    """Mock BrokerClient with sensible defaults for every method job.py calls."""
    broker = MagicMock()
    broker.get_account    = AsyncMock(return_value=AccountInfo(
        equity=10_000.0, buying_power=10_000.0, status="ACTIVE"
    ))
    broker.is_trading_day = AsyncMock(return_value=is_trading)
    broker.get_positions  = AsyncMock(return_value=positions if positions is not None else [])
    broker.place_market_order = AsyncMock()
    return broker


def _fake_df(today: date) -> pd.DataFrame:
    """
    Two-row split-adjusted daily DataFrame with today as the last bar.
    Provides just enough history for the strategy warm-up path in job.py.
    """
    yesterday = today - timedelta(days=1)
    idx = pd.to_datetime([
        datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=timezone.utc),
        datetime(today.year,     today.month,     today.day,     tzinfo=timezone.utc),
    ])
    return pd.DataFrame(
        {
            "open":   [150.0, 152.0],
            "high":   [151.5, 153.5],
            "low":    [149.0, 151.0],
            "close":  [150.5, 152.5],
            "volume": [1_000_000, 1_100_000],
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_market_closed_exits_cleanly():
    """
    Job exits with code 0 and places no orders when today is not a trading day
    (weekend or market holiday).
    """
    mock_broker = _fake_broker(is_trading=False)

    with (
        patch("bot.job.get_settings", return_value=_fake_settings()),
        patch("bot.job.BrokerClient", return_value=mock_broker),
        patch("bot.notify.send", new_callable=AsyncMock) as mock_notify,
    ):
        from bot.job import run_job
        await run_job()   # must not raise

    # No order should be placed on a non-trading day
    mock_broker.place_market_order.assert_not_called()

    # No notification needed for a non-trading day (per spec)
    mock_notify.assert_not_called()


@pytest.mark.asyncio
async def test_no_signal_sends_heartbeat():
    """
    Job sends a 'Daily Heartbeat' notification when the strategy produces no
    signal for today's bar, then exits cleanly without placing an order.
    """
    today = date.today()
    mock_broker = _fake_broker(is_trading=True)

    mock_strategy = MagicMock()
    mock_strategy.on_bar.return_value = None   # strategy never fires
    mock_strategy.on_start = MagicMock()

    with (
        patch("bot.job.get_settings",   return_value=_fake_settings()),
        patch("bot.job.BrokerClient",   return_value=mock_broker),
        patch("bot.job.fetch_bars",     new_callable=AsyncMock, return_value=_fake_df(today)),
        patch("bot.job.get_strategy",   return_value=mock_strategy),
        patch("bot.notify.send",        new_callable=AsyncMock) as mock_notify,
    ):
        from bot.job import run_job
        await run_job()

    # No order placed
    mock_broker.place_market_order.assert_not_called()

    # Heartbeat notification must have been sent exactly once
    mock_notify.assert_called_once()
    title_sent = mock_notify.call_args[1].get("title") or mock_notify.call_args[0][0]
    assert title_sent == "Daily Heartbeat"


@pytest.mark.asyncio
async def test_buy_signal_approved_places_order_and_notifies():
    """
    Job places a BUY market order and sends a 'Trade Opened' notification when:
      - Market is open today
      - Strategy returns a BUY signal
      - Risk manager approves (no existing positions, well within limits)
    """
    today = date.today()
    mock_broker = _fake_broker(is_trading=True, positions=[])

    buy_signal = Signal(type=SignalType.BUY, ticker="AAPL", reason="test_signal")
    mock_strategy = MagicMock()
    mock_strategy.on_bar.return_value = buy_signal
    mock_strategy.on_start = MagicMock()

    with (
        patch("bot.job.get_settings",   return_value=_fake_settings()),
        patch("bot.job.BrokerClient",   return_value=mock_broker),
        patch("bot.job.fetch_bars",     new_callable=AsyncMock, return_value=_fake_df(today)),
        patch("bot.job.get_strategy",   return_value=mock_strategy),
        patch("bot.notify.send",        new_callable=AsyncMock) as mock_notify,
    ):
        from bot.job import run_job
        await run_job()

    # A market order must have been submitted
    mock_broker.place_market_order.assert_called_once()
    order_kwargs = mock_broker.place_market_order.call_args[1]
    assert order_kwargs["ticker"] == "AAPL"
    assert order_kwargs["side"]   == "buy"
    assert order_kwargs["qty"]    >= 1

    # 'Trade Opened' notification must have been sent
    mock_notify.assert_called_once()
    title_sent = mock_notify.call_args[1].get("title") or mock_notify.call_args[0][0]
    assert "Trade Opened" in title_sent


@pytest.mark.asyncio
async def test_buy_signal_blocked_by_regime_filter():
    """
    Job places no order and sends a 'Signal Blocked' notification when the
    regime filter rejects a BUY entry (e.g. VIX too high or SPY < SMA200).
    """
    today = date.today()
    mock_broker = _fake_broker(is_trading=True, positions=[])

    buy_signal = Signal(type=SignalType.BUY, ticker="AAPL", reason="test_signal")
    mock_strategy = MagicMock()
    mock_strategy.on_bar.return_value = buy_signal
    mock_strategy.on_start = MagicMock()

    mock_regime = MagicMock()
    mock_regime.allow_buy.return_value = False

    with (
        patch("bot.job.get_settings",   return_value=_fake_settings()),
        patch("bot.job.BrokerClient",   return_value=mock_broker),
        patch("bot.job.fetch_bars",     new_callable=AsyncMock, return_value=_fake_df(today)),
        patch("bot.job.get_strategy",   return_value=mock_strategy),
        patch("bot.job.RegimeFilter.from_config", return_value=mock_regime),
        patch("bot.notify.send",        new_callable=AsyncMock) as mock_notify,
    ):
        from bot.job import run_job
        await run_job()

    # No order should be placed when the regime filter blocks the entry
    mock_broker.place_market_order.assert_not_called()

    # 'Signal Blocked' notification must have been sent
    mock_notify.assert_called_once()
    title_sent = mock_notify.call_args[1].get("title") or mock_notify.call_args[0][0]
    assert title_sent == "Signal Blocked"


@pytest.mark.asyncio
async def test_buy_signal_blocked_by_earnings_filter():
    """
    Job places no order and sends a 'Signal Blocked' notification when the
    earnings filter reports a blackout for the ticker (earnings imminent).
    """
    today = date.today()
    mock_broker = _fake_broker(is_trading=True, positions=[])

    buy_signal = Signal(type=SignalType.BUY, ticker="AAPL", reason="test_signal")
    mock_strategy = MagicMock()
    mock_strategy.on_bar.return_value = buy_signal
    mock_strategy.on_start = MagicMock()

    mock_earnings = MagicMock()
    mock_earnings.is_blackout.return_value = True

    with (
        patch("bot.job.get_settings",   return_value=_fake_settings()),
        patch("bot.job.BrokerClient",   return_value=mock_broker),
        patch("bot.job.fetch_bars",     new_callable=AsyncMock, return_value=_fake_df(today)),
        patch("bot.job.get_strategy",   return_value=mock_strategy),
        patch("bot.job.EarningsFilter.from_config", return_value=mock_earnings),
        patch("bot.notify.send",        new_callable=AsyncMock) as mock_notify,
    ):
        from bot.job import run_job
        await run_job()

    # No order should be placed during an earnings blackout
    mock_broker.place_market_order.assert_not_called()

    # 'Signal Blocked' notification must have been sent
    mock_notify.assert_called_once()
    title_sent = mock_notify.call_args[1].get("title") or mock_notify.call_args[0][0]
    assert title_sent == "Signal Blocked"


@pytest.mark.asyncio
async def test_buy_signal_blocked_by_correlation_guard():
    """
    Job places no order and sends a 'Signal Blocked' notification when the
    correlation guard reports that the new entry is too correlated with an
    existing open position.
    """
    today = date.today()
    existing_position = Position(
        ticker="QQQ", qty=1, avg_entry_price=400.0,
        current_price=410.0, unrealized_pnl=10.0,
    )
    mock_broker = _fake_broker(is_trading=True, positions=[existing_position])

    buy_signal = Signal(type=SignalType.BUY, ticker="AAPL", reason="test_signal")
    mock_strategy = MagicMock()
    mock_strategy.on_bar.return_value = buy_signal
    mock_strategy.on_start = MagicMock()

    mock_corr_guard = MagicMock()
    mock_corr_guard.is_blocked.return_value = True

    with (
        patch("bot.job.get_settings",   return_value=_fake_settings()),
        patch("bot.job.BrokerClient",   return_value=mock_broker),
        patch("bot.job.fetch_bars",     new_callable=AsyncMock, return_value=_fake_df(today)),
        patch("bot.job.get_strategy",   return_value=mock_strategy),
        patch("bot.job.CorrelationGuard.from_config", return_value=mock_corr_guard),
        patch("bot.notify.send",        new_callable=AsyncMock) as mock_notify,
    ):
        from bot.job import run_job
        await run_job()

    # No order should be placed when the correlation guard blocks the entry
    mock_broker.place_market_order.assert_not_called()

    # Guard must have been consulted with the held ticker
    mock_corr_guard.is_blocked.assert_called_once_with("AAPL", {"QQQ"})

    # 'Signal Blocked' notification must have been sent
    mock_notify.assert_called_once()
    title_sent = mock_notify.call_args[1].get("title") or mock_notify.call_args[0][0]
    assert title_sent == "Signal Blocked"


@pytest.mark.asyncio
async def test_atr_sizing_scales_order_quantity():
    """
    When atr_sizing is enabled, the order quantity is derived from the
    volatility-scaled notional returned by atr_position_size rather than
    a flat max_notional_per_trade / price calculation.
    """
    today = date.today()
    mock_broker = _fake_broker(is_trading=True, positions=[])

    buy_signal = Signal(type=SignalType.BUY, ticker="AAPL", reason="test_signal")
    mock_strategy = MagicMock()
    mock_strategy.on_bar.return_value = buy_signal
    mock_strategy.on_start = MagicMock()

    settings = _fake_settings()
    settings.atr_sizing = True

    with (
        patch("bot.job.get_settings",      return_value=settings),
        patch("bot.job.BrokerClient",      return_value=mock_broker),
        patch("bot.job.fetch_bars",        new_callable=AsyncMock, return_value=_fake_df(today)),
        patch("bot.job.get_strategy",      return_value=mock_strategy),
        patch("bot.job.atr_position_size", return_value=250.0) as mock_atr,
        patch("bot.notify.send",           new_callable=AsyncMock),
    ):
        from bot.job import run_job
        await run_job()

    # atr_position_size must be consulted to size the order
    mock_atr.assert_called_once()

    # today's close is 152.5 → 250.0 / 152.5 = 1 share
    order_kwargs = mock_broker.place_market_order.call_args[1]
    assert order_kwargs["qty"] == max(1, int(250.0 / 152.5))
