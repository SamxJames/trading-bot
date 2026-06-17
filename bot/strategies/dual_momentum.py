"""
Dual Momentum (GEM — Global Equities Momentum).

A monthly portfolio-allocation strategy, fundamentally different from the
per-bar signal strategies (EmaCrossFilteredStrategy etc.) in this package:

  - It runs on a MONTHLY cadence, evaluated on the last trading day of the
    month — not bar-by-bar.
  - It outputs a TARGET ALLOCATION (hold one of SPY / VEU / AGG at 100%), not
    a BUY/SELL/HOLD signal for a single ticker.
  - It does NOT use the 9-filter stack. Momentum IS the strategy here; layering
    trend/RSI/volume/regime filters on top would corrupt the academic edge that
    GEM's backtest depends on.

THE RULES (Gary Antonacci, "Dual Momentum Investing"), evaluated monthly with a
fixed 12-month lookback (set by academic precedent — never tuned):

  1. Absolute momentum: if US (SPY) trailing-12m total return > the T-bill
     trailing-12m return  ->  equities are "on". Otherwise hold bonds (AGG).
  2. Relative momentum (only when equities are on): hold whichever of US (SPY)
     or international ex-US (VEU) has the higher trailing-12m total return.

This class holds the decision logic and current target only. Data fetching
(prices / T-bill rate) and order placement live in bot/rebalance.py, which
feeds the trailing-12m readings into compute_target().

The backtest behind this strategy lives in scripts/backtest_dual_momentum.py;
it cleared the validation gate (CAGR >= SPY AND smaller max drawdown over the
full period) before this production strategy was written.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Sequence

from bot.logging.logger import get_logger
from bot.strategies.base import Signal

log = get_logger(__name__)

_DEFAULT_ASSETS: Dict[str, str] = {"us": "SPY", "intl": "VEU", "bonds": "AGG"}


class DualMomentumStrategy:
    """Monthly Global Equities Momentum allocation engine."""

    name = "dual_momentum"

    def __init__(
        self,
        gem_lookback_months: int = 12,
        gem_assets: Optional[Dict[str, str]] = None,
        gem_enabled: bool = True,
        gem_rebalance: str = "monthly",
        **kwargs,  # absorb the EMA-strategy kwargs the registry forwards
    ) -> None:
        self.gem_lookback_months = int(gem_lookback_months)
        self.gem_assets = dict(gem_assets) if gem_assets else dict(_DEFAULT_ASSETS)
        self.gem_enabled = bool(gem_enabled)
        self.gem_rebalance = gem_rebalance

        # Current target ticker (None until compute_target() runs) and the
        # momentum readings behind the last decision (for logging / Discord /
        # dashboard).
        self._target_allocation: Optional[str] = None
        self.last_readings: Dict[str, Optional[float]] = {
            "us_12m": None,
            "intl_12m": None,
            "tbill_12m": None,
        }

    # ── Convenience accessors for the three asset tickers ─────────────────────
    @property
    def us_ticker(self) -> str:
        return self.gem_assets["us"]

    @property
    def intl_ticker(self) -> str:
        return self.gem_assets["intl"]

    @property
    def bonds_ticker(self) -> str:
        return self.gem_assets["bonds"]

    @property
    def target_allocation(self) -> Optional[str]:
        """The ticker GEM currently wants to hold 100% (None before evaluation)."""
        return self._target_allocation

    # ── Core decision logic ───────────────────────────────────────────────────
    def trailing_return(self, prices: Sequence[float]) -> Optional[float]:
        """Trailing total return over `gem_lookback_months` from an ascending
        sequence of period closes. Returns None if there isn't enough history
        (need lookback + 1 points), so callers can degrade gracefully."""
        if prices is None or len(prices) < self.gem_lookback_months + 1:
            return None
        past = prices[-1 - self.gem_lookback_months]
        if past in (None, 0):
            return None
        return prices[-1] / past - 1.0

    def compute_target(
        self,
        us_12m: Optional[float],
        intl_12m: Optional[float],
        tbill_12m: Optional[float],
    ) -> str:
        """Apply the GEM rules and return the target ticker.

        Defensive: if the US or T-bill reading is missing (insufficient
        lookback / data outage), GEM cannot confirm equities are "on", so it
        preserves capital in bonds rather than guessing — the same risk-off
        default it would pick in a downtrend.
        """
        self.last_readings = {"us_12m": us_12m, "intl_12m": intl_12m, "tbill_12m": tbill_12m}

        if us_12m is None or tbill_12m is None:
            target = self.bonds_ticker
        elif us_12m > tbill_12m:
            # Absolute momentum positive — equities are "on". Relative momentum
            # picks the stronger of US / international.
            if intl_12m is None or us_12m >= intl_12m:
                target = self.us_ticker
            else:
                target = self.intl_ticker
        else:
            # Absolute momentum negative — risk off.
            target = self.bonds_ticker

        self._target_allocation = target
        log.info(
            "gem_target_computed",
            target=target,
            us_12m=_round(us_12m),
            intl_12m=_round(intl_12m),
            tbill_12m=_round(tbill_12m),
        )
        return target

    # ── Cadence ────────────────────────────────────────────────────────────────
    def should_rebalance(self, current_date) -> bool:
        """True only on the last trading day of the month.

        Trading-day approximation: the last weekday (Mon-Fri) of the calendar
        month. Exchange holidays falling on that weekday aren't modelled here —
        the monthly_rebalance.yml cron is scheduled for the last weekday and a
        one-day-early rebalance on a rare holiday is immaterial for a monthly
        strategy. `current_date` may be a date or datetime.
        """
        if self.gem_rebalance != "monthly":
            return False
        if isinstance(current_date, datetime):
            current_date = current_date.date()
        return current_date == self._last_trading_day_of_month(current_date)

    @staticmethod
    def _last_trading_day_of_month(d: date) -> date:
        last_dom = calendar.monthrange(d.year, d.month)[1]
        last = date(d.year, d.month, last_dom)
        while last.weekday() >= 5:  # 5=Sat, 6=Sun -> step back to Friday
            last -= timedelta(days=1)
        return last

    # ── Strategy protocol shims (this engine is not bar-driven) ────────────────
    def on_start(self) -> None:
        self._target_allocation = None

    def on_bar(self, bar) -> Optional[Signal]:  # noqa: ARG002 - intentional no-op
        """GEM is monthly/allocation-based, not bar-driven. No per-bar signal."""
        return None

    def on_stop(self) -> None:
        return None


def _round(v: Optional[float]) -> Optional[float]:
    return round(v, 4) if isinstance(v, (int, float)) else None
