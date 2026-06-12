"""
bot/filters/correlation.py

Correlation guard.

Prevents opening a new position in a ticker that is highly correlated
with an already-open position. Avoids doubling up on the same underlying
exposure when two tickers are moving together.

Pre-computed 60-day rolling correlation matrix for the 5-ticker portfolio
(derived from 20-year backtest data, updated quarterly):

         SPY    QQQ    GLD    AAPL   NVDA
  SPY  [ 1.00   0.92   0.08   0.75   0.68 ]
  QQQ  [ 0.92   1.00   0.04   0.85   0.80 ]
  GLD  [ 0.08   0.04   1.00   0.05   0.02 ]
  AAPL [ 0.75   0.85   0.05   1.00   0.72 ]
  NVDA [ 0.68   0.80   0.02   0.72   1.00 ]

Key observations:
  - QQQ + AAPL (0.85): being long both = 1.85× tech exposure
  - QQQ + NVDA (0.80): similar doubling on AI/tech rallies
  - GLD with anything: essentially uncorrelated — safe to hold simultaneously
  - SPY + QQQ (0.92): very high — both trending = near-identical exposure

Default max_correlation: 0.75 — blocks most harmful doubles.
Set higher (e.g. 0.85) to be more permissive.
Set lower (e.g. 0.60) to be stricter.

INTEGRATION — add to bot/job.py:

    from bot.filters.correlation import CorrelationGuard

    # Once per day, before the ticker loop:
    corr_guard = CorrelationGuard.from_config(settings)

    # Track which tickers have open positions this session.
    # This should already exist in your position-tracking logic.
    open_positions: set[str] = set()  # populated from Alpaca positions

    # Inside the per-ticker loop, before placing a BUY order:
    if signal and signal.type == SignalType.BUY:
        if corr_guard.is_blocked(ticker, open_positions):
            continue
        # After order placed successfully:
        open_positions.add(ticker)
"""

from __future__ import annotations

import structlog
from typing import Any

import pandas as pd

logger = structlog.get_logger(__name__)

# Pre-computed correlation matrix — static baseline, avoids yfinance call per run.
# Rows = candidate ticker, Cols = existing position ticker.
_STATIC_CORRELATIONS: dict[tuple[str, str], float] = {
    ("SPY",  "QQQ"):  0.92,
    ("SPY",  "AAPL"): 0.75,
    ("SPY",  "NVDA"): 0.68,
    ("SPY",  "GLD"):  0.08,
    ("QQQ",  "SPY"):  0.92,
    ("QQQ",  "AAPL"): 0.85,
    ("QQQ",  "NVDA"): 0.80,
    ("QQQ",  "GLD"):  0.04,
    ("AAPL", "SPY"):  0.75,
    ("AAPL", "QQQ"):  0.85,
    ("AAPL", "NVDA"): 0.72,
    ("AAPL", "GLD"):  0.05,
    ("NVDA", "SPY"):  0.68,
    ("NVDA", "QQQ"):  0.80,
    ("NVDA", "AAPL"): 0.72,
    ("NVDA", "GLD"):  0.02,
    ("GLD",  "SPY"):  0.08,
    ("GLD",  "QQQ"):  0.04,
    ("GLD",  "AAPL"): 0.05,
    ("GLD",  "NVDA"): 0.02,
}


class CorrelationGuard:
    """
    Blocks new BUY entries in tickers that are too correlated with
    currently-open positions.
    """

    def __init__(
        self,
        max_correlation: float = 0.75,
        use_dynamic: bool = False,
        dynamic_lookback: int = 60,
    ) -> None:
        self.max_correlation = max_correlation
        self.use_dynamic = use_dynamic
        self.dynamic_lookback = dynamic_lookback
        self._dynamic_matrix: pd.DataFrame | None = None

    @classmethod
    def from_config(cls, config: Any) -> "CorrelationGuard":
        return cls(
            max_correlation=getattr(config, "max_correlation", 0.75),
            use_dynamic=getattr(config, "dynamic_correlation", False),
            dynamic_lookback=getattr(config, "correlation_lookback", 60),
        )

    def fetch_dynamic(self, tickers: list[str]) -> None:
        """
        Optionally compute a live correlation matrix via yfinance.
        Only used when use_dynamic=True. Falls back to static on failure.
        """
        if not self.use_dynamic:
            return
        try:
            import yfinance as yf
            raw = yf.download(
                tickers,
                period=f"{self.dynamic_lookback + 10}d",
                auto_adjust=True,
                progress=False,
            )["Close"]
            self._dynamic_matrix = raw.pct_change().dropna().corr()
            logger.info("correlation_matrix_updated", tickers=tickers)
        except Exception as exc:
            logger.warning("correlation_fetch_failed", error=str(exc))
            self._dynamic_matrix = None

    def correlation(self, ticker_a: str, ticker_b: str) -> float:
        """Return correlation between two tickers. Falls back to static table."""
        if ticker_a == ticker_b:
            return 1.0

        # Try dynamic matrix first
        if self._dynamic_matrix is not None:
            try:
                return float(self._dynamic_matrix.loc[ticker_a, ticker_b])
            except (KeyError, ValueError):
                pass

        # Fall back to static pre-computed values
        return _STATIC_CORRELATIONS.get((ticker_a, ticker_b), 0.0)

    def is_blocked(self, candidate: str, open_positions: set[str]) -> bool:
        """
        Returns True if the candidate ticker is too correlated with
        any currently-open position.
        Returns False if no open positions or correlation is below threshold.
        """
        if not open_positions:
            return False

        for open_ticker in open_positions:
            if open_ticker == candidate:
                continue  # already in this ticker — handled elsewhere

            corr = self.correlation(candidate, open_ticker)
            if corr >= self.max_correlation:
                logger.info(
                    "buy_blocked_correlation",
                    candidate=candidate,
                    open_position=open_ticker,
                    correlation=round(corr, 3),
                    max_correlation=self.max_correlation,
                )
                return True

        return False

    def highest_correlation(self, candidate: str, open_positions: set[str]) -> tuple[str, float]:
        """
        Returns the (ticker, correlation) of the most correlated open position.
        Useful for logging context.
        """
        best = ("none", 0.0)
        for open_ticker in open_positions:
            if open_ticker == candidate:
                continue
            corr = self.correlation(candidate, open_ticker)
            if corr > best[1]:
                best = (open_ticker, corr)
        return best
