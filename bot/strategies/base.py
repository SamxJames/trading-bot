"""
Strategy base protocol and shared types.

Every concrete strategy must implement the Strategy protocol defined here.
The engine in main.py and backtest.py operates against this interface only
— it never imports a concrete strategy class directly, keeping the core
engine decoupled from strategy logic.

Adding a new strategy:
  1. Create a new file in bot/strategies/ (e.g. rsi.py)
  2. Implement the Strategy protocol
  3. Register the class in bot/strategies/registry.py
  4. Set strategy: <name> in config.yaml

Signal semantics:
  - BUY  → open a long position (or close an existing short)
  - SELL → open a short position (or close an existing long)
  - HOLD → no action; strategy is waiting for conditions to be met
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from bot.data.feed import Bar


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """Output of a strategy's on_bar() call."""

    type: SignalType
    ticker: str
    reason: str = ""


@runtime_checkable
class Strategy(Protocol):
    """
    Interface that every concrete strategy must satisfy.

    Strategies are stateful objects: they accumulate bar history internally
    and emit a Signal (or None) on each new bar.  The engine calls
    on_start() once before the first bar and on_stop() once after the last.
    """

    name: str

    def on_bar(self, bar: Bar) -> Signal | None:
        """Process one bar.  Return a Signal or None if no action is warranted."""
        ...

    def on_start(self) -> None:
        """Called once before the first bar.  Use for any warm-up logic."""
        ...

    def on_stop(self) -> None:
        """Called once after the session ends.  Use for cleanup / final logging."""
        ...
