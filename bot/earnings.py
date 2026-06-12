"""
bot/earnings.py

Earnings blackout filter — blocks new BUY entries within a configurable
window of a ticker's earnings announcement date.

Rationale: earnings releases cause overnight price gaps that EMA/RSI based
trend-following strategies can't react to intraday — a position opened the
day before earnings can gap straight through its stop loss.

Fails open — if earnings dates can't be fetched for a ticker, that ticker is
never blacked out. This ensures a data outage never silently halts trading.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Iterable

import structlog

logger = structlog.get_logger(__name__)


class EarningsFilter:
    """
    Fetches each ticker's nearby earnings dates once per session and blocks
    new BUY entries within `blackout_days` of any such date (before or after).
    """

    def __init__(self, blackout_days: int = 1) -> None:
        self.blackout_days = blackout_days
        self._blackout: dict[str, bool] = {}

    @classmethod
    def from_config(cls, config: Any) -> "EarningsFilter":
        """Construct from a settings/config object. Falls back to defaults for missing attrs."""
        return cls(blackout_days=getattr(config, "earnings_blackout_days", 1))

    def fetch(self, tickers: Iterable[str]) -> None:
        """
        Fetch nearby earnings dates for each ticker via yfinance and cache
        whether today falls within that ticker's blackout window.

        Call once at the start of each daily job run. Fails open per-ticker
        — a ticker whose earnings dates can't be fetched is never blacked out.
        """
        self._blackout = {}
        today = date.today()

        for ticker in tickers:
            try:
                import yfinance as yf

                earnings_dates = yf.Ticker(ticker).get_earnings_dates(limit=8)
                blocked = False
                if earnings_dates is not None and not earnings_dates.empty:
                    for ts in earnings_dates.index:
                        if abs((ts.date() - today).days) <= self.blackout_days:
                            blocked = True
                            break

                self._blackout[ticker] = blocked
                if blocked:
                    logger.info("earnings_blackout_active", ticker=ticker)

            except Exception as exc:
                logger.warning("earnings_fetch_failed", ticker=ticker, error=str(exc))
                self._blackout[ticker] = False  # fail open

    def is_blackout(self, ticker: str) -> bool:
        """
        Returns True if `ticker` is within its earnings blackout window today.
        Returns False (permissive) if no earnings data is available for it.
        """
        return self._blackout.get(ticker, False)
