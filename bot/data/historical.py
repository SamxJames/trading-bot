"""
Historical OHLCV data fetcher with local SQLite cache.

Fetches bar data from Alpaca's REST API for a given ticker and date range.
Results are cached in cache/bars.db so repeated backtest runs on the same
period do not re-hit the API.

Cache behaviour:
  - On first fetch for a (ticker, timeframe, date range): fetch from Alpaca, store.
  - On subsequent fetches for the same range: return from cache immediately.
  - To force a re-fetch, delete cache/bars.db.

Used exclusively by the backtester (backtest.py). The live trading loop
uses feed.py instead.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from bot.logging.logger import get_logger

log = get_logger(__name__)

_CACHE_DIR = Path("cache")
_CACHE_DB  = _CACHE_DIR / "bars_v2.db"   # v2: adjustment field added to key

_TIMEFRAME_MAP: dict[str, TimeFrame] = {
    "1Min":  TimeFrame.Minute,
    "1Hour": TimeFrame.Hour,
    "1Day":  TimeFrame.Day,
    "1Week": TimeFrame.Week,
    "1Month": TimeFrame.Month,
}

_ADJUSTMENT_MAP: dict[str, Adjustment] = {
    "raw":      Adjustment.RAW,
    "split":    Adjustment.SPLIT,
    "dividend": Adjustment.DIVIDEND,
    "all":      Adjustment.ALL,
}


# ---------------------------------------------------------------------------
# SQLite cache helpers
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
    """Return cached bars for the range, or None if the cache is empty."""
    try:
        conn = _open_cache()
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM bars "
            "WHERE ticker=? AND timeframe=? AND adjustment=? AND ts>=? AND ts<=? ORDER BY ts",
            (ticker, timeframe, adjustment, start.isoformat(), end.isoformat()),
        ).fetchall()
        conn.close()
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp")
    except Exception as exc:
        log.warning("cache_read_failed", error=str(exc))
        return None


def _cache_write(ticker: str, timeframe: str, adjustment: str, df: pd.DataFrame) -> None:
    """Persist fetched bars to the cache. Failures are non-fatal."""
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
        log.info("cache_written", ticker=ticker, timeframe=timeframe, adjustment=adjustment, bars=len(rows))
    except Exception as exc:
        log.warning("cache_write_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _fetch_sync(ticker: str, start: date, end: date, timeframe: str, adjustment: str) -> pd.DataFrame:
    """Synchronous inner fetch — runs in a thread via asyncio.to_thread."""
    from bot.config import get_settings
    s = get_settings()

    tf  = _TIMEFRAME_MAP.get(timeframe, TimeFrame.Day)
    adj = _ADJUSTMENT_MAP.get(adjustment, Adjustment.SPLIT)

    client = StockHistoricalDataClient(
        api_key=s.apca_api_key_id,
        secret_key=s.apca_api_secret_key,
    )

    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=tf,
        start=datetime(start.year, start.month, start.day, tzinfo=timezone.utc),
        end=datetime(end.year, end.month, end.day, tzinfo=timezone.utc),
        adjustment=adj,
    )
    bars = client.get_stock_bars(request)
    df: pd.DataFrame = bars.df

    # alpaca-py returns a MultiIndex (symbol, timestamp) — drop the symbol level
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level="symbol")

    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df.columns = [c.lower() for c in df.columns]

    log.info(
        "bars_fetched_from_api",
        ticker=ticker,
        bars=len(df),
        start=str(start),
        end=str(end),
        timeframe=timeframe,
        adjustment=adjustment,
    )
    return df[["open", "high", "low", "close", "volume"]]


async def fetch_bars(
    ticker: str,
    start: date,
    end: date,
    timeframe: str | None = None,
    adjustment: str = "split",
) -> pd.DataFrame:
    """
    Fetch split-adjusted OHLCV bars for *ticker* between *start* and *end* (inclusive).

    Uses the SQLite cache if data is already available for the (ticker, timeframe,
    adjustment, date-range) combination.  Falls back to the Alpaca API on miss and
    stores the result so subsequent runs are instant.

    Args:
        ticker:     Ticker symbol, e.g. "AAPL".
        start:      First bar date (inclusive).
        end:        Last bar date (inclusive).
        timeframe:  Bar width string, e.g. "1Day".  Defaults to config.yaml value.
        adjustment: Corporate-action adjustment.  One of "raw", "split", "dividend",
                    "all".  Defaults to "split" so stock-split discontinuities are
                    eliminated from all backtest price series.

    Returns:
        DataFrame with UTC DatetimeIndex sorted ascending and columns:
            open, high, low, close, volume

    Raises:
        ValueError: if no bars are returned for the requested range.
    """
    if timeframe is None:
        from bot.config import get_settings
        timeframe = get_settings().timeframe

    # Cache hit?
    cached = _cache_read(ticker, timeframe, adjustment, start, end)
    if cached is not None and not cached.empty:
        log.info(
            "bars_loaded_from_cache",
            ticker=ticker,
            bars=len(cached),
            timeframe=timeframe,
            adjustment=adjustment,
        )
        return cached

    # Cache miss — fetch from API
    df = await asyncio.to_thread(_fetch_sync, ticker, start, end, timeframe, adjustment)
    if df.empty:
        raise ValueError(
            f"No bars returned for {ticker} from {start} to {end} "
            f"(timeframe={timeframe}, adjustment={adjustment})"
        )

    _cache_write(ticker, timeframe, adjustment, df)
    return df
