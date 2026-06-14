"""
Shadow trading mode.

Runs the configured strategy (all 9 filters) against real Alpaca daily bars
every day and records what *would* have happened — entries, exits, and
P&L — without ever placing a real order. This closes the backtest-to-live
gap: instead of waiting years for the live paper account to accumulate
enough trades to judge the strategy, shadow trading evaluates the same
signals against real market data daily and tracks a hypothetical position
book.

Data flow
---------
  - `fetch_bars()` — same data path as bot/job.py (Alpaca, split-adjusted,
    ~60-day warm-up + today's bar).
  - A fresh strategy instance is fed every bar except today's; the returned
    signal is the evaluation of *yesterday's* bar — exactly what the live
    job would have produced had it run one day earlier.
  - If that signal is a BUY (all of filters 1/2/4/9 already passed inside
    the strategy) and the job-level gates (VIX/SPY/earnings/correlation)
    also pass, a hypothetical position is opened at *today's open* — the
    fill the live job would get if it placed the order after yesterday's
    close.
  - Each day, any open hypothetical position is checked against today's
    close for the strategy's two documented exits (Exit 1 trailing stop,
    Exit 2 take-profit), using the *same* stop_loss_pct/trailing_stop_pct/
    take_profit_rr the strategy instance was built with.

State
-----
  - Open hypothetical positions persist across days in
    bot/trade_journal/shadow_state.json (gitignored).
  - Closed round trips are appended to bot/trade_journal/shadow_trades.csv.
  - Every per-ticker evaluation is appended to
    bot/trade_journal/shadow_log.jsonl using the same `signal_evaluation`
    schema as bot/job.py's signal_log.jsonl.

This module never imports anything that can place an order — there is no
broker dependency and `place_market_order`/`place_limit_order` are never
called.
"""

from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from bot.correlation import CorrelationGuard
from bot.data.feed import Bar
from bot.data.historical import fetch_bars
from bot.earnings import EarningsFilter
from bot.filters.regime import RegimeFilter
from bot.logging.logger import get_logger
from bot.risk.sizing import atr_position_size
from bot.strategies.base import SignalType
from bot.strategies.registry import get_strategy

log = get_logger(__name__)

SHADOW_TRADES_PATH = Path("bot/trade_journal/shadow_trades.csv")
SHADOW_LOG_PATH = Path("bot/trade_journal/shadow_log.jsonl")
SHADOW_STATE_PATH = Path("bot/trade_journal/shadow_state.json")

SHADOW_TRADES_FIELDS = [
    "id", "ticker", "side", "entry_price", "exit_price", "pnl_usd",
    "pnl_pct", "reason", "entry_date", "exit_date", "bars_held",
]


# ---------------------------------------------------------------------------
# State / log persistence helpers
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if not SHADOW_STATE_PATH.exists():
        return {}
    try:
        return json.loads(SHADOW_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    try:
        SHADOW_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SHADOW_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("shadow_state_write_failed", error=str(exc))


def _append_jsonl(path: Path, record: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        log.warning("shadow_log_write_failed", error=str(exc))


def _append_trade_row(row: dict) -> None:
    try:
        SHADOW_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
        is_new = not SHADOW_TRADES_PATH.exists()
        with SHADOW_TRADES_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SHADOW_TRADES_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:
        log.warning("shadow_trade_write_failed", error=str(exc))


# ---------------------------------------------------------------------------
# ShadowTrader
# ---------------------------------------------------------------------------

class ShadowTrader:
    """Simulates the live strategy against real bars without placing orders."""

    def __init__(
        self,
        settings: object,
        regime: RegimeFilter,
        earnings: EarningsFilter,
        corr_guard: CorrelationGuard,
    ) -> None:
        self.settings = settings
        self.regime = regime
        self.earnings = earnings
        self.corr_guard = corr_guard

    async def run(self, today: date) -> None:
        """Evaluate every configured ticker for *today* and persist state."""
        state = _load_state()
        for ticker in self.settings.tickers:
            await self._process_ticker(ticker, today, state)
        _save_state(state)

    async def _process_ticker(self, ticker: str, today: date, state: dict) -> None:
        settings = self.settings
        warmup_start = today - timedelta(days=61)
        try:
            df = await fetch_bars(
                ticker, warmup_start, today, timeframe="1Day", adjustment="split",
            )
        except Exception as exc:
            log.warning("shadow_fetch_failed", ticker=ticker, error=str(exc))
            return

        last_bar_date = df.index[-1].to_pydatetime().date()
        if last_bar_date != today:
            log.info("shadow_bar_not_ready", ticker=ticker, last_bar_date=str(last_bar_date))
            return

        bars: list[Bar] = [
            Bar(
                ticker=ticker,
                timestamp=ts.to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
            )
            for ts, row in df.iterrows()
        ]
        today_bar = bars[-1]

        strategy = get_strategy(
            settings.strategy,
            fast_period=settings.fast_period,
            slow_period=settings.slow_period,
            trend_sma_period=settings.trend_sma_period,
            rsi_period=settings.rsi_period,
            rsi_oversold=settings.rsi_oversold,
            rsi_overbought=settings.rsi_overbought,
            stop_loss_pct=settings.stop_loss_pct,
            weekly_ema_filter=settings.weekly_ema_filter,
        )
        strategy.on_start()

        # Feed everything except today's bar — `signal` is the evaluation of
        # *yesterday's* bar, the same signal the live job would have produced
        # one day earlier. Today's bar supplies the hypothetical fill price.
        signal = None
        for bar in bars[:-1]:
            signal = strategy.on_bar(bar)

        # Imported lazily to avoid a circular import (bot.job imports
        # ShadowTrader from this module).
        from bot.job import _build_filter_record

        shadow_open_tickers = set(state.keys())
        filters = _build_filter_record(
            strategy, ticker, self.regime, self.earnings, self.corr_guard,
            shadow_open_tickers, settings,
        )
        blocked_by = [name for name, f in filters.items() if f["evaluated"] and not f["passed"]]

        _append_jsonl(SHADOW_LOG_PATH, {
            "event": "signal_evaluation",
            "ts": datetime.now(timezone.utc).isoformat(),
            "date": str(today),
            "ticker": ticker,
            "signal": signal.type.value if signal else None,
            "signal_reason": signal.reason if signal else None,
            "filters": filters,
            "blocked_by": blocked_by,
        })

        position = state.get(ticker)
        if position is not None:
            self._check_exit(ticker, today, today_bar, position, state)
            return

        if signal is None or signal.type != SignalType.BUY:
            return

        # Job-level gates (5-8): same checks bot/job.py applies before
        # placing a real BUY order.
        if not self.regime.allow_buy(ticker):
            return
        if self.earnings.is_blackout(ticker):
            return
        if self.corr_guard.is_blocked(ticker, shadow_open_tickers):
            return

        self._open_position(ticker, today, today_bar, bars, strategy, state)

    def _open_position(
        self,
        ticker: str,
        today: date,
        today_bar: Bar,
        bars: list[Bar],
        strategy: object,
        state: dict,
    ) -> None:
        settings = self.settings
        entry_price = today_bar.open
        if entry_price <= 0:
            log.warning("shadow_pending_entry_skipped", ticker=ticker, reason="invalid_open_price")
            return

        if settings.atr_sizing:
            target_notional = atr_position_size(
                close=entry_price,
                prices=pd.Series([b.close for b in bars]),
                base_notional=settings.max_notional_per_trade,
                atr_period=settings.atr_period,
                target_atr_pct=settings.atr_target_pct,
                max_notional=settings.max_notional_per_trade,
            )
        else:
            target_notional = settings.max_notional_per_trade

        qty = max(1, int(target_notional / entry_price))

        take_profit_price = 0.0
        if strategy.take_profit_rr > 0:
            initial_risk = entry_price * strategy.stop_loss_pct / 100.0
            take_profit_price = entry_price + initial_risk * strategy.take_profit_rr

        state[ticker] = {
            "id": f"S{ticker}-{today.isoformat()}",
            "entry_price": entry_price,
            "entry_date": today.isoformat(),
            "highest_price": entry_price,
            "take_profit_price": take_profit_price,
            "stop_loss_pct": strategy.stop_loss_pct,
            "trailing_stop_pct": strategy.trailing_stop_pct,
            "qty": qty,
            "bars_held": 0,
        }
        log.info("shadow_position_opened", ticker=ticker, entry_price=round(entry_price, 4), qty=qty)

    def _check_exit(
        self,
        ticker: str,
        today: date,
        today_bar: Bar,
        position: dict,
        state: dict,
    ) -> None:
        close = today_bar.close
        highest_price = max(position["highest_price"], close)
        initial_stop = position["entry_price"] * (1.0 - position["stop_loss_pct"] / 100.0)
        trailing_stop = highest_price * (1.0 - position["trailing_stop_pct"] / 100.0)
        current_stop = max(initial_stop, trailing_stop)
        bars_held = position["bars_held"] + 1

        exit_price: float | None = None
        reason: str | None = None
        if position["take_profit_price"] > 0 and close >= position["take_profit_price"]:
            exit_price = close
            reason = "take_profit"
        elif close < current_stop:
            exit_price = close
            reason = "stop_loss"

        if exit_price is None:
            position["highest_price"] = highest_price
            position["bars_held"] = bars_held
            return

        entry_price = position["entry_price"]
        qty = position["qty"]
        pnl_usd = round((exit_price - entry_price) * qty, 2)
        pnl_pct = round((exit_price - entry_price) / entry_price * 100, 2)

        _append_trade_row({
            "id": position["id"],
            "ticker": ticker,
            "side": "BUY",
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "entry_date": position["entry_date"],
            "exit_date": today.isoformat(),
            "bars_held": bars_held,
        })
        log.info("shadow_position_closed", ticker=ticker, reason=reason, pnl_usd=pnl_usd)
        del state[ticker]
