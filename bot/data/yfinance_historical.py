"""
yfinance historical data fetcher — backtesting only.

Drop-in replacement for bot/data/historical.py when ``--data-source yfinance``
is passed on the CLI.  The public ``fetch_bars()`` function has an identical
signature and returns an identical DataFrame so the backtest engine needs zero
changes.

Why this exists alongside historical.py
----------------------------------------
- yfinance requires no API credentials → good for CI and quick exploration
- yfinance has free 20-year daily history for most US equities
- Alpaca stays as the authoritative source for live trading and the default
  for backtesting

Cache
-----
Results are stored in ``cache/bars_yf_v1.db`` (separate from Alpaca's
``bars_v2.db``).  Same schema, same PRIMARY KEY — different file so the two
sources never clash.

Adjustment
----------
yfinance uses ``auto_adjust=True`` to apply both split and dividend adjustment.
There is no split-only mode in yfinance, so both "split" and "all" map to
``auto_adjust=True``.  "raw" maps to ``auto_adjust=False``.

Limitations
-----------
- Intraday data (1Min, 1Hour) is limited by yfinance to ~30 / 730 days
  respectively.  Daily data is available for 20+ years on most US tickers.
- yfinance end-date is exclusive; this module adds one day internally.
- yfinance may throttle or fail on high-frequency API calls.
  The SQLite cache mitigates this for repeated runs.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from bot.logging.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR = Path("cache")
_CACHE_DB  = _CACHE_DIR / "bars_yf_v1.db"

# Map our internal timeframe strings → yfinance interval strings
_INTERVAL_MAP: dict[str, str] = {
    "1Min":  "1m",
    "1Hour": "1h",
    "1Day":  "1d",
    "1Week": "1wk",
    "1Month": "1mo",
}


# ---------------------------------------------------------------------------
# SQLite cache (identical schema to Alpaca cache; different file)
# ---------------------------------------------------------------------------

def _open_cache() -> sqlite3.Connection:
    _CACHE_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(_CACHE_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bars (
            ticker      TEXT NOT NULL,
            timeframe   TEXT NOT NULL,
            adjustment  TEXT NOT NULL,
            ts          TEXT NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      INTEGER,
            PRIMARY KEY (ticker, timeframe, adjustment, ts)
        )
    """)
    conn.commit()
    return conn


def _cache_read(
    ticker: str, timeframe: str, adjustment: str, start: date, end: date
) -> pd.DataFrame | None:
    """Return cached bars or None if the cache is empty for this range."""
    try:
        conn = _open_cache()
        # Use ts < (end + 1 day) rather than ts <= end because stored
        # timestamps include a time component ("2023-01-10T05:00:00+00:00").
        # SQLite string comparison sees that as *greater* than "2023-01-10",
        # so <= would silently drop the last date's bars.
        end_excl = (end + timedelta(days=1)).isoformat()
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM bars "
            "WHERE ticker=? AND timeframe=? AND adjustment=? "
            "AND ts>=? AND ts<? ORDER BY ts",
            (ticker, timeframe, adjustment, start.isoformat(), end_excl),
        ).fetchall()
        conn.close()
        if not rows:
            return None

        # Validate span: the first and last cached bars must be within
        # 5 calendar days of the requested start/end.  This catches the
        # case where the cache holds a *small* slice from a previous run
        # (e.g. a unit-test fetch) and a new wider range is requested —
        # we must fetch the full range rather than return stale partials.
        first_date = date.fromisoformat(rows[0][0][:10])
        last_date  = date.fromisoformat(rows[-1][0][:10])
        start_gap  = (first_date - start).days   # 0–2 on weekends/holidays
        end_gap    = (end - last_date).days       # 0–3 when end is Sat/Sun/Mon
        if start_gap > 5 or end_gap > 5:
            conn.close()
            log.info(
                "yf_cache_miss_incomplete",
                ticker=ticker,
                cached_start=str(first_date),
                cached_end=str(last_date),
                requested_start=str(start),
                requested_end=str(end),
            )
            return None

        df = pd.DataFrame(
            rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp")
    except Exception as exc:
        log.warning("yf_cache_read_failed", error=str(exc))
        return None


def _cache_write(ticker: str, timeframe: str, adjustment: str, df: pd.DataFrame) -> None:
    """Persist bars to the cache.  Failures are non-fatal."""
    try:
        conn = _open_cache()
        rows = [
            (
                ticker,
                timeframe,
                adjustment,
                ts.isoformat(),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["volume"]),
            )
            for ts, row in df.iterrows()
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?,?)", rows
        )
        conn.commit()
        conn.close()
        log.info(
            "yf_cache_written",
            ticker=ticker,
            timeframe=timeframe,
            adjustment=adjustment,
            bars=len(rows),
        )
    except Exception as exc:
        log.warning("yf_cache_write_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Synchronous fetch (runs in a thread via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _fetch_sync(
    ticker: str, start: date, end: date, timeframe: str, adjustment: str
) -> pd.DataFrame:
    """
    Download OHLCV data from yfinance and return a normalised DataFrame.

    The returned DataFrame has:
      - UTC timezone-aware DatetimeIndex sorted ascending
      - Columns exactly: open, high, low, close, volume
    """
    import yfinance as yf

    interval    = _INTERVAL_MAP.get(timeframe, "1d")
    auto_adjust = (adjustment != "raw")   # "split", "all", "dividend" all → True
    # yfinance end is exclusive — add one day to include the requested end date
    end_excl = (end + timedelta(days=1)).isoformat()

    t   = yf.Ticker(ticker)
    raw = t.history(
        start=start.isoformat(),
        end=end_excl,
        interval=interval,
        auto_adjust=auto_adjust,
        actions=False,   # drop Dividends and Stock Splits columns
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for {ticker} "
            f"from {start} to {end} "
            f"(timeframe={timeframe}, interval={interval})"
        )

    # ── Normalise columns ────────────────────────────────────────────────────
    # Defensive: flatten multi-level index if present (can occur in some builds)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    raw.columns = [c.lower() for c in raw.columns]

    # Keep exactly the five columns the backtest engine expects
    needed  = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(
            f"yfinance data for {ticker} is missing expected columns: {missing}. "
            f"Got: {list(raw.columns)}"
        )
    df = raw[needed].copy()

    # ── Normalise index to UTC ───────────────────────────────────────────────
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.sort_index()
    # Drop any rows where close is NaN (can occur on market holidays or gaps)
    df = df.dropna(subset=["close"])

    log.info(
        "yf_bars_fetched",
        ticker=ticker,
        bars=len(df),
        start=str(start),
        end=str(end),
        timeframe=timeframe,
        adjustment=adjustment,
    )
    return df


# ---------------------------------------------------------------------------
# Public async API (identical signature to historical.fetch_bars)
# ---------------------------------------------------------------------------

async def fetch_bars(
    ticker: str,
    start: date,
    end: date,
    timeframe: Optional[str] = None,
    adjustment: str = "split",
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from yfinance for *ticker* between *start* and *end*.

    Signature is identical to ``bot.data.historical.fetch_bars`` so the
    backtest engine can swap data sources without any other changes.

    Parameters
    ----------
    ticker:     Ticker symbol, e.g. "SPY".
    start:      First bar date (inclusive).
    end:        Last bar date (inclusive).
    timeframe:  Bar width — "1Day", "1Hour", "1Min", "1Week", "1Month".
                Defaults to the value in config.yaml.
    adjustment: "split" or "all" → auto_adjust=True (default).
                "raw"            → auto_adjust=False.

    Returns
    -------
    DataFrame with UTC DatetimeIndex sorted ascending and columns:
        open, high, low, close, volume

    Raises
    ------
    ValueError if yfinance returns an empty result.
    """
    if timeframe is None:
        from bot.config import get_settings
        timeframe = get_settings().timeframe

    # Cache hit?
    cached = _cache_read(ticker, timeframe, adjustment, start, end)
    if cached is not None and not cached.empty:
        log.info(
            "yf_bars_from_cache",
            ticker=ticker,
            bars=len(cached),
            timeframe=timeframe,
            adjustment=adjustment,
        )
        return cached

    # Cache miss — fetch from yfinance in a thread (it's sync)
    df = await asyncio.to_thread(_fetch_sync, ticker, start, end, timeframe, adjustment)

    _cache_write(ticker, timeframe, adjustment, df)
    return df
