"""
Unit tests for bot/rebalance.py (the monthly GEM rebalance job).

All Alpaca and yfinance calls are mocked — no network, no orders. Tests cover:
  - a first-ever rebalance (no prior holding) places a single BUY
  - a rotation (target differs from logged holding) sells then buys
  - a no-change run places no orders
  - the non-rebalance-day guard short-circuits before touching the broker
  - dry-run never places orders
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.execution.broker import AccountInfo


def _fake_settings(tmp_alloc: float = 50_000.0) -> MagicMock:
    s = MagicMock()
    s.apca_api_key_id     = "fake_key"
    s.apca_api_secret_key = "fake_secret"
    s.apca_base_url       = "https://paper-api.alpaca.markets"
    s.gem_enabled         = True
    s.gem_lookback_months = 12
    s.gem_assets          = {"us": "SPY", "intl": "VEU", "bonds": "AGG"}
    s.gem_rebalance       = "monthly"
    s.gem_allocation_usd  = tmp_alloc
    s.discord_webhook_url = ""
    return s


def _fake_broker() -> MagicMock:
    broker = MagicMock()
    broker.get_account = AsyncMock(return_value=AccountInfo(
        equity=100_000.0, buying_power=100_000.0, status="ACTIVE"
    ))
    broker.place_market_order = AsyncMock()
    return broker


def _orders(broker: MagicMock) -> list[tuple]:
    """(ticker, side) for each placed order, in call order."""
    return [
        (c.kwargs.get("ticker"), c.kwargs.get("side"))
        for c in broker.place_market_order.call_args_list
    ]


@pytest.mark.asyncio
async def test_first_rebalance_buys_target(tmp_path):
    broker = _fake_broker()
    log_path = tmp_path / "rebalance_log.jsonl"
    from bot.rebalance import run_rebalance

    with patch("bot.rebalance.get_settings", return_value=_fake_settings()), \
         patch("bot.rebalance.BrokerClient", return_value=broker), \
         patch("bot.rebalance.REBALANCE_LOG_PATH", log_path), \
         patch("bot.rebalance._fetch_monthly_closes",
               side_effect=lambda sym: {"SPY": [100.0]*12+[120.0],
                                        "VEU": [100.0]*12+[110.0],
                                        "AGG": [50.0]*13}[sym]), \
         patch("bot.rebalance._fetch_tbill_12m", return_value=0.05), \
         patch("bot.notify.send", new=AsyncMock()):
        await run_rebalance(dry_run=False, force=True)

    # No prior holding -> a single BUY of SPY (US momentum strongest).
    assert _orders(broker) == [("SPY", "buy")]
    record = json.loads(log_path.read_text().strip().splitlines()[-1])
    assert record["holding"] == "SPY"
    assert record["previous_holding"] is None
    assert record["changed"] is True


@pytest.mark.asyncio
async def test_rotation_sells_then_buys(tmp_path):
    broker = _fake_broker()
    log_path = tmp_path / "rebalance_log.jsonl"
    # Seed a prior holding of AGG (qty 100).
    log_path.write_text(json.dumps({
        "event": "gem_rebalance", "holding": "AGG", "qty": 100,
    }) + "\n")
    from bot.rebalance import run_rebalance

    with patch("bot.rebalance.get_settings", return_value=_fake_settings()), \
         patch("bot.rebalance.BrokerClient", return_value=broker), \
         patch("bot.rebalance.REBALANCE_LOG_PATH", log_path), \
         patch("bot.rebalance._fetch_monthly_closes",
               side_effect=lambda sym: {"SPY": [100.0]*12+[120.0],
                                        "VEU": [100.0]*12+[110.0],
                                        "AGG": [50.0]*13}[sym]), \
         patch("bot.rebalance._fetch_tbill_12m", return_value=0.05), \
         patch("bot.notify.send", new=AsyncMock()):
        await run_rebalance(dry_run=False, force=True)

    # Target is SPY now -> SELL the old AGG, then BUY SPY (sell before buy).
    assert _orders(broker) == [("AGG", "sell"), ("SPY", "buy")]


@pytest.mark.asyncio
async def test_no_change_places_no_orders(tmp_path):
    broker = _fake_broker()
    log_path = tmp_path / "rebalance_log.jsonl"
    log_path.write_text(json.dumps({
        "event": "gem_rebalance", "holding": "SPY", "qty": 10,
    }) + "\n")
    from bot.rebalance import run_rebalance

    with patch("bot.rebalance.get_settings", return_value=_fake_settings()), \
         patch("bot.rebalance.BrokerClient", return_value=broker), \
         patch("bot.rebalance.REBALANCE_LOG_PATH", log_path), \
         patch("bot.rebalance._fetch_monthly_closes",
               side_effect=lambda sym: {"SPY": [100.0]*12+[120.0],
                                        "VEU": [100.0]*12+[110.0],
                                        "AGG": [50.0]*13}[sym]), \
         patch("bot.rebalance._fetch_tbill_12m", return_value=0.05), \
         patch("bot.notify.send", new=AsyncMock()):
        await run_rebalance(dry_run=False, force=True)

    assert _orders(broker) == []  # already holding SPY


@pytest.mark.asyncio
async def test_not_rebalance_day_skips(tmp_path):
    broker = _fake_broker()
    log_path = tmp_path / "rebalance_log.jsonl"
    from bot.rebalance import run_rebalance

    # force=False and should_rebalance False (mid-month) -> never touches broker.
    with patch("bot.rebalance.get_settings", return_value=_fake_settings()), \
         patch("bot.rebalance.BrokerClient", return_value=broker), \
         patch("bot.rebalance.REBALANCE_LOG_PATH", log_path), \
         patch("bot.strategies.dual_momentum.DualMomentumStrategy.should_rebalance",
               return_value=False), \
         patch("bot.notify.send", new=AsyncMock()):
        await run_rebalance(dry_run=False, force=False)

    broker.get_account.assert_not_called()
    broker.place_market_order.assert_not_called()
    assert not log_path.exists()


@pytest.mark.asyncio
async def test_dry_run_places_no_orders(tmp_path):
    broker = _fake_broker()
    log_path = tmp_path / "rebalance_log.jsonl"
    from bot.rebalance import run_rebalance

    with patch("bot.rebalance.get_settings", return_value=_fake_settings()), \
         patch("bot.rebalance.BrokerClient", return_value=broker), \
         patch("bot.rebalance.REBALANCE_LOG_PATH", log_path), \
         patch("bot.rebalance._fetch_monthly_closes",
               side_effect=lambda sym: {"SPY": [100.0]*12+[120.0],
                                        "VEU": [100.0]*12+[110.0],
                                        "AGG": [50.0]*13}[sym]), \
         patch("bot.rebalance._fetch_tbill_12m", return_value=0.05), \
         patch("bot.notify.send", new=AsyncMock()):
        await run_rebalance(dry_run=True, force=True)

    assert _orders(broker) == []                 # no orders in dry-run
    record = json.loads(log_path.read_text().strip().splitlines()[-1])
    assert record["dry_run"] is True
    assert record["holding"] == "SPY"
