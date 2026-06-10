"""
bot/risk/sizing.py

ATR-based position sizing.

Scales the per-trade notional inversely with the ticker's recent volatility,
so that dollar risk per trade stays approximately constant across tickers.

  High ATR ticker (e.g. NVDA) → smaller position
  Low ATR ticker  (e.g. GLD)  → larger position, capped at max_notional

The scaling formula:
    scaled_notional = base_notional × (target_atr_pct / current_atr_pct)

Where target_atr_pct is the reference volatility at which base_notional is used.
Result is clamped between min_notional and max_notional.

INTEGRATION — add to bot/job.py or wherever orders are placed:

    from bot.risk.sizing import atr_position_size
    import pandas as pd

    # Build a Series of recent close prices for this ticker
    # (already fetched from Alpaca or SQLite cache)
    recent_closes = pd.Series([bar.close for bar in recent_bars])

    notional = atr_position_size(
        close=bar.close,
        prices=recent_closes,
        base_notional=settings.max_notional_per_trade,
        atr_period=getattr(settings, 'atr_period', 14),
        target_atr_pct=getattr(settings, 'atr_target_pct', 2.0),
        max_notional=settings.max_notional_per_trade,
    )
    # Use notional instead of settings.max_notional_per_trade when placing the order
"""

from __future__ import annotations

import math

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def _close_to_close_atr(prices: pd.Series, period: int) -> float:
    """
    Compute ATR as a percentage of the last close using close-to-close ranges.
    This is a simplified ATR that doesn't require OHLC bars — only close prices.
    Returns 0.0 if insufficient data or computation fails.
    """
    if len(prices) < period + 1:
        return 0.0
    try:
        pct_changes = prices.pct_change().abs().dropna()
        if len(pct_changes) < period:
            return 0.0
        atr_pct = float(pct_changes.rolling(period).mean().iloc[-1]) * 100.0
        return atr_pct if not math.isnan(atr_pct) and atr_pct > 0 else 0.0
    except Exception:
        return 0.0


def atr_position_size(
    close: float,
    prices: pd.Series,
    base_notional: float = 500.0,
    atr_period: int = 14,
    target_atr_pct: float = 2.0,
    min_notional: float = 100.0,
    max_notional: float = 500.0,
) -> float:
    """
    Compute a volatility-scaled position size.

    Args:
        close:           Current bar close price (for logging context)
        prices:          Series of recent close prices. Needs at least
                         atr_period + 1 bars. More bars = better ATR estimate.
        base_notional:   Dollar amount to use when ATR == target_atr_pct.
        atr_period:      ATR lookback in bars (default 14).
        target_atr_pct:  Reference volatility level (default 2.0%).
                         At this ATR the position equals base_notional.
        min_notional:    Floor (default $100). Prevents trivially small orders.
        max_notional:    Ceiling (default $500). Caps exposure on calm tickers.

    Returns:
        Scaled notional as a float, clamped to [min_notional, max_notional].
        Falls back to base_notional if ATR cannot be computed.
    """
    atr_pct = _close_to_close_atr(prices, period=atr_period)

    if atr_pct <= 0:
        logger.debug(
            "atr_sizing_fallback",
            reason="atr_unavailable",
            close=round(close, 2),
            bars_available=len(prices),
        )
        return base_notional

    # Scale inversely with volatility
    scaled = base_notional * (target_atr_pct / atr_pct)
    result = max(min_notional, min(max_notional, scaled))

    logger.info(
        "atr_position_sized",
        close=round(close, 2),
        atr_pct=round(atr_pct, 3),
        target_atr_pct=target_atr_pct,
        base_notional=base_notional,
        scaled_notional=round(result, 2),
        clamped=result != scaled,
    )

    return result
