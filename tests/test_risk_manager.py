"""
Unit tests for RiskManager — Phase 3.

All tests use in-memory state; no live broker or API calls.
The broker is mocked with a simple async stub where needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.risk.manager import RiskDecision, RiskManager
from bot.strategies.base import Signal, SignalType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _buy(ticker: str = "AAPL") -> Signal:
    return Signal(type=SignalType.BUY, ticker=ticker, reason="test")

def _sell(ticker: str = "AAPL") -> Signal:
    return Signal(type=SignalType.SELL, ticker=ticker, reason="test")

def _rm(**kwargs) -> RiskManager:
    defaults = dict(max_positions=3, max_notional=500.0,
                    drawdown_halt_pct=5.0, initial_equity=10_000.0)
    defaults.update(kwargs)
    return RiskManager(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_approves_valid_trade():
    """Trade is approved when all risk rules pass."""
    rm = _rm()
    decision = rm.evaluate(_buy(), current_price=150.0)
    assert decision.approved is True
    assert decision.reason == "approved"


def test_rejects_when_max_positions_reached():
    """Trade is rejected when max_positions positions are already open."""
    rm = _rm(max_positions=2)
    rm.record_open("AAPL")
    rm.record_open("MSFT")
    decision = rm.evaluate(_buy("SPY"), current_price=400.0)
    assert decision.approved is False
    assert "max_positions" in decision.reason


def test_rejects_when_notional_exceeded():
    """Trade is rejected when max_notional is zero or negative."""
    rm = _rm(max_notional=0.0)
    decision = rm.evaluate(_buy(), current_price=150.0)
    assert decision.approved is False


def test_drawdown_halt_triggers():
    """Halt flag is set when equity drops by drawdown_halt_pct from session start."""
    rm = _rm(initial_equity=10_000.0, drawdown_halt_pct=5.0)
    assert rm.is_halted() is False

    # A 5% drop should trigger the halt
    rm.update_equity(9_500.0)
    assert rm.is_halted() is True


def test_halt_blocks_all_trades():
    """All subsequent evaluate() calls return approved=False after halt."""
    rm = _rm(initial_equity=10_000.0, drawdown_halt_pct=5.0)
    rm.update_equity(9_000.0)   # >5% drop — halted
    assert rm.is_halted() is True

    decision = rm.evaluate(_buy(), current_price=150.0)
    assert decision.approved is False
    assert decision.reason == "trading_halted"


@pytest.mark.asyncio
async def test_emergency_flatten_calls_broker():
    """emergency_flatten() calls broker.close_all_positions() and cancel_all_orders()."""
    mock_broker = MagicMock()
    mock_broker.close_all_positions = AsyncMock()
    mock_broker.cancel_all_orders   = AsyncMock()

    rm = _rm(broker=mock_broker)
    rm.record_open("AAPL")

    await rm.emergency_flatten()

    mock_broker.close_all_positions.assert_awaited_once()
    mock_broker.cancel_all_orders.assert_awaited_once()
