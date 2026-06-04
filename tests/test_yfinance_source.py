"""
Integration test for bot/data/yfinance_historical.py.

Makes a real network request to Yahoo Finance — no mocking.
yfinance requires no API credentials so this is safe to run in CI.

Validates that the returned DataFrame matches the exact contract the
backtest engine depends on:
  - Non-empty
  - Columns exactly: {"open", "high", "low", "close", "volume"}
  - UTC timezone-aware DatetimeIndex
  - Index is sorted ascending (monotonic)
  - All close prices > 0
  - All volumes >= 0
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from bot.data.yfinance_historical import fetch_bars


# Use a short, well-known window that is guaranteed to contain trading days.
_START = date(2023, 1, 3)   # first trading day of 2023
_END   = date(2023, 1, 10)  # a week later


@pytest.mark.asyncio
async def test_fetch_spy_daily_shape_and_columns():
    """Returned DataFrame has the five expected columns and at least one row."""
    df = await fetch_bars("SPY", _START, _END, timeframe="1Day", adjustment="split")

    assert isinstance(df, pd.DataFrame), "return type must be DataFrame"
    assert len(df) > 0, "DataFrame must not be empty"
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}, (
        f"Unexpected columns: {set(df.columns)}"
    )


@pytest.mark.asyncio
async def test_fetch_spy_daily_utc_index():
    """DatetimeIndex must be UTC timezone-aware and sorted ascending."""
    df = await fetch_bars("SPY", _START, _END, timeframe="1Day", adjustment="split")

    assert df.index.tz is not None, "Index must be timezone-aware"
    assert str(df.index.tz) == "UTC", f"Index must be UTC, got {df.index.tz}"
    assert df.index.is_monotonic_increasing, "Index must be sorted ascending"


@pytest.mark.asyncio
async def test_fetch_spy_daily_data_sanity():
    """Price and volume values must be physically plausible."""
    df = await fetch_bars("SPY", _START, _END, timeframe="1Day", adjustment="split")

    assert (df["close"] > 0).all(),  "All close prices must be positive"
    assert (df["volume"] >= 0).all(), "All volumes must be non-negative"
    # High >= Low is a basic OHLC sanity check
    assert (df["high"] >= df["low"]).all(), "High must be >= Low on every bar"


@pytest.mark.asyncio
async def test_cache_returns_same_dataframe(tmp_path, monkeypatch):
    """
    Second call for the same date range is served from the SQLite cache
    and returns a DataFrame with the same shape and columns.
    """
    import bot.data.yfinance_historical as mod

    # Redirect cache to a temp directory so tests don't pollute project cache
    monkeypatch.setattr(mod, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(mod, "_CACHE_DB",  tmp_path / "bars_yf_test.db")

    df1 = await fetch_bars("SPY", _START, _END, timeframe="1Day", adjustment="split")
    df2 = await fetch_bars("SPY", _START, _END, timeframe="1Day", adjustment="split")

    assert df1.shape == df2.shape, "Cached result must have same shape"
    assert list(df1.columns) == list(df2.columns), "Cached result must have same columns"
