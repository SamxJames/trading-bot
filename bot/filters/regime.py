"""
bot/filters/regime.py

Macro regime filters that gate all BUY entries at the job level.

  FILTER 5 — VIX regime gate
    Skip all BUY entries if VIX > vix_threshold (default 25).
    Tighten stop loss to vix_tight_stop_pct if VIX > vix_tight_threshold (default 20).
    Rationale: EMA crossovers in high-volatility markets produce whipsaw losses.
    Historical basis: 2008, 2020, and 2022 drawdowns all occurred during VIX > 25.

  FILTER 6 — SPY macro gate
    Skip BUY entries for all non-SPY tickers if SPY is below its SMA(200).
    SPY itself is exempt — it may still signal even in a bear market.
    Rationale: trend-following strategies dramatically underperform in bear markets.
    Historical basis: most losing streaks in backtest occur below SPY SMA(200).

INTEGRATION — add these lines to bot/job.py:

    # At the top of the file:
    from bot.filters.regime import RegimeFilter

    # Once per day, before the ticker loop (after account_ok check):
    regime = RegimeFilter.from_config(settings)
    regime.fetch()

    # Inside the per-ticker loop, before strategy.on_bar() or after a BUY signal:
    if signal and signal.type == SignalType.BUY:
        if not regime.allow_buy(ticker):
            continue  # skip this entry
        # Optionally tighten the stop on high-VIX days:
        effective_stop = regime.adjusted_stop_pct(settings.stop_loss_pct)
        # pass effective_stop to strategy or risk manager as needed

Fails silently — if VIX or SPY data is unavailable, all signals are permitted.
This ensures a data outage never silently halts trading.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass
from typing import Any

logger = structlog.get_logger(__name__)


@dataclass
class RegimeState:
    vix: float = 0.0
    spy_close: float = 0.0
    spy_sma200: float = 0.0
    available: bool = False

    @property
    def spy_above_sma(self) -> bool:
        return self.spy_close >= self.spy_sma200 if self.spy_sma200 > 0 else True


class RegimeFilter:
    """
    Fetches VIX and SPY data once per session and gates BUY entries
    based on current macro regime.
    """

    def __init__(
        self,
        vix_threshold: float = 25.0,
        vix_tight_threshold: float = 20.0,
        vix_tight_stop_pct: float = 1.5,
        spy_macro_filter: bool = True,
        spy_sma_period: int = 200,
    ) -> None:
        self.vix_threshold = vix_threshold
        self.vix_tight_threshold = vix_tight_threshold
        self.vix_tight_stop_pct = vix_tight_stop_pct
        self.spy_macro_filter = spy_macro_filter
        self.spy_sma_period = spy_sma_period
        self._state = RegimeState()

    @classmethod
    def from_config(cls, config: Any) -> "RegimeFilter":
        """Construct from a settings/config object. Falls back to defaults for missing attrs."""
        return cls(
            vix_threshold=getattr(config, "vix_threshold", 25.0),
            vix_tight_threshold=getattr(config, "vix_tight_threshold", 20.0),
            vix_tight_stop_pct=getattr(config, "vix_tight_stop_pct", 1.5),
            spy_macro_filter=getattr(config, "spy_macro_filter", True),
            spy_sma_period=getattr(config, "spy_sma_period", 200),
        )

    def fetch(self) -> None:
        """
        Fetch latest VIX and SPY data via yfinance.
        Call once at the start of each daily job run, after account_ok.
        Fails silently — regime defaults to permissive on any error.
        """
        try:
            import yfinance as yf

            # VIX — use 5d period in case of weekend/holiday gaps
            vix_raw = yf.download("^VIX", period="5d", auto_adjust=True, progress=False)
            if not vix_raw.empty:
                self._state.vix = float(vix_raw["Close"].iloc[-1])

            # SPY — enough bars for SMA plus a buffer for non-trading days
            lookback = f"{self.spy_sma_period + 80}d"
            spy_raw = yf.download("SPY", period=lookback, auto_adjust=True, progress=False)
            if not spy_raw.empty and len(spy_raw) >= self.spy_sma_period:
                closes = spy_raw["Close"]
                self._state.spy_close = float(closes.iloc[-1])
                self._state.spy_sma200 = float(
                    closes.rolling(self.spy_sma_period).mean().iloc[-1]
                )
                self._state.available = True

            logger.info(
                "regime_fetched",
                vix=round(self._state.vix, 2),
                spy_close=round(self._state.spy_close, 2),
                spy_sma200=round(self._state.spy_sma200, 2),
                spy_above_sma=self._state.spy_above_sma,
                vix_high=self._state.vix > self.vix_threshold,
            )

        except Exception as exc:
            logger.warning("regime_fetch_failed", error=str(exc))
            # _state.available stays False → all allow_buy() calls return True

    def allow_buy(self, ticker: str = "") -> bool:
        """
        Returns True if current macro regime permits a new BUY entry.
        Returns True (permissive) if regime data is unavailable.
        """
        if not self._state.available:
            return True

        # VIX gate — market too choppy for trend following
        if self._state.vix > self.vix_threshold:
            logger.info(
                "buy_blocked_regime",
                reason="vix_too_high",
                vix=round(self._state.vix, 2),
                threshold=self.vix_threshold,
                ticker=ticker,
            )
            return False

        # SPY macro gate — skip non-SPY entries in a bear market
        if (
            self.spy_macro_filter
            and ticker != "SPY"
            and self._state.spy_sma200 > 0
            and not self._state.spy_above_sma
        ):
            logger.info(
                "buy_blocked_regime",
                reason="spy_below_sma200",
                spy_close=round(self._state.spy_close, 2),
                spy_sma200=round(self._state.spy_sma200, 2),
                ticker=ticker,
            )
            return False

        return True

    def adjusted_stop_pct(self, base_stop_pct: float) -> float:
        """
        Returns a tightened stop loss % when VIX is elevated but below the
        full vix_threshold. Gives trades less room to breathe in choppy markets.
        Returns base_stop_pct unchanged if regime data is unavailable.
        """
        if not self._state.available:
            return base_stop_pct

        if self._state.vix > self.vix_tight_threshold:
            tightened = min(base_stop_pct, self.vix_tight_stop_pct)
            if tightened < base_stop_pct:
                logger.info(
                    "stop_tightened_vix",
                    vix=round(self._state.vix, 2),
                    base_stop_pct=base_stop_pct,
                    tightened_stop_pct=tightened,
                )
            return tightened

        return base_stop_pct

    @property
    def state(self) -> RegimeState:
        return self._state
