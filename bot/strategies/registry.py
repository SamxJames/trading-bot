"""
Strategy registry.

Maps strategy name strings (as used in config.yaml) to their classes.
The engine calls get_strategy() with the configured name and receives a
ready-to-use Strategy instance — it never imports strategy classes directly.

To register a new strategy:
  1. Import its class at the top of this file
  2. Add an entry to REGISTRY
"""

from __future__ import annotations

from typing import Any, Dict, Type

from bot.strategies.base import Strategy
from bot.strategies.ema_cross import EmaCrossStrategy
from bot.strategies.ema_cross_filtered import EmaCrossFilteredStrategy
from bot.strategies.rsi_strategy import RSIStrategy

REGISTRY: Dict[str, Type] = {
    "ema_cross":          EmaCrossStrategy,
    "ema_cross_filtered": EmaCrossFilteredStrategy,
    "rsi":                RSIStrategy,
}


def get_strategy(name: str, **kwargs: Any) -> Strategy:
    """
    Instantiate and return the strategy registered under *name*.

    *kwargs* are forwarded to the strategy's __init__ — pass all config
    params and each strategy will pick up what it needs (extras are ignored
    via **kwargs in the constructor).
    """
    cls = REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(
            f"Unknown strategy '{name}'. Available: {available}"
        )
    return cls(**kwargs)
