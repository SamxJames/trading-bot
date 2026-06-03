"""
Daily scheduled job entry point.

Designed to run once per weekday at 4:15 pm ET (after market close), triggered
by GitHub Actions.  The job completes in under 60 seconds and always exits
with a clean return value unless an unhandled exception propagates (which
lets GitHub Actions mark the run as failed).

Execution steps
---------------
1. Validate environment — Alpaca API reachable, settings sane.
2. Check market calendar — skip silently on non-trading days.
3. Fetch today's completed daily bar + 60-day warm-up for each ticker.
4. Run the configured strategy through the warm-up, then read the signal
   from today's bar.
5. Risk check — positions, drawdown guard.
6. Place market order (or dry-run log).
7. Send Discord notification with outcome.
8. Log job_complete with elapsed time and exit.

Exit codes
----------
  0 — success, no-signal, market-closed, or bar-not-ready (all expected).
  1 — unhandled exception (GitHub Actions marks the run as failed).

Usage
-----
    python -m bot job              # live paper-trading mode
    python -m bot job --dry-run    # simulate without placing orders
"""

from __future__ import annotations

import time
from datetime import date, timedelta

from bot import notify
from bot.config import get_settings
from bot.data.feed import Bar
from bot.data.historical import fetch_bars
from bot.execution.broker import BrokerClient
from bot.logging.logger import get_logger
from bot.risk.manager import RiskManager
from bot.strategies.base import SignalType
from bot.strategies.registry import get_strategy

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_job(dry_run: bool = False) -> None:
    """
    Run the daily trading job.

    Returns normally (exit 0) on success and on expected non-events
    (market closed, no bar yet, no signal, risk rejected).

    Re-raises any unhandled exception after sending a Discord alert, so the
    process exits with a non-zero code and GitHub Actions marks the run red.
    """
    t_start = time.monotonic()
    try:
        await _run(dry_run=dry_run)
    except Exception as exc:
        log.exception("job_failed", error=str(exc))
        try:
            await notify.send(
                title="Job Failed",
                message=f"Error in daily job: {exc}",
                colour="red",
            )
        except Exception:
            pass  # do not mask the original exception
        raise
    duration = time.monotonic() - t_start
    log.info("job_complete", duration_seconds=round(duration, 2))


# ---------------------------------------------------------------------------
# Inner orchestration
# ---------------------------------------------------------------------------

async def _run(dry_run: bool) -> None:
    settings = get_settings()
    log.info(
        "job_started",
        dry_run=dry_run,
        strategy=settings.strategy,
        tickers=settings.tickers,
    )

    # ── Step 1: Validate environment ─────────────────────────────────────────
    broker = BrokerClient(
        api_key=settings.apca_api_key_id,
        secret_key=settings.apca_api_secret_key,
        base_url=settings.apca_base_url,
    )
    account = await broker.get_account()
    log.info("account_ok", equity=account.equity, status=account.status)

    # ── Step 2: Check if market was open today ────────────────────────────────
    today = date.today()
    if not await broker.is_trading_day(today):
        log.info("market_closed_today", date=str(today))
        return  # weekend / holiday — exit 0, no notification needed

    # ── Steps 3-6: Per-ticker pass ────────────────────────────────────────────
    for ticker in settings.tickers:
        await _process_ticker(
            ticker=ticker,
            today=today,
            broker=broker,
            account_equity=account.equity,
            settings=settings,
            dry_run=dry_run,
        )


async def _process_ticker(
    ticker: str,
    today: date,
    broker: BrokerClient,
    account_equity: float,
    settings: object,
    dry_run: bool,
) -> None:
    """Run the full signal → risk → order pipeline for a single ticker."""

    # ── Step 3: Fetch today's bar + warm-up ──────────────────────────────────
    warmup_start = today - timedelta(days=61)   # ~60 trading days of context
    df = await fetch_bars(
        ticker,
        warmup_start,
        today,
        timeframe="1Day",
        adjustment="split",   # mandatory — never omit
    )

    # Guard: Alpaca only returns completed bars, but confirm today is present.
    last_bar_date = df.index[-1].to_pydatetime().date()
    if last_bar_date != today:
        log.info(
            "bar_not_ready",
            ticker=ticker,
            last_bar_date=str(last_bar_date),
            today=str(today),
        )
        return  # market may have just closed; job will re-run tomorrow

    # Build Bar objects the strategy expects
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

    # ── Step 4: Run strategy ─────────────────────────────────────────────────
    strategy = get_strategy(
        settings.strategy,
        fast_period=settings.fast_period,
        slow_period=settings.slow_period,
        trend_sma_period=settings.trend_sma_period,
        rsi_period=settings.rsi_period,
        rsi_oversold=settings.rsi_oversold,
        rsi_overbought=settings.rsi_overbought,
        stop_loss_pct=settings.stop_loss_pct,
    )
    strategy.on_start()

    # Feed warm-up bars to prime indicators; today's bar produces the signal.
    signal = None
    for bar in bars:
        signal = strategy.on_bar(bar)
    # `signal` is now the result from today_bar — the only actionable one.

    if signal is None or signal.type == SignalType.HOLD:
        log.info("no_signal", ticker=ticker, date=str(today))
        await notify.send(
            title="Daily Heartbeat",
            message=(
                f"Job ran. No signal on {ticker}. "
                f"Market is open: True"
            ),
            colour="grey",
        )
        return

    log.info(
        "signal_generated",
        ticker=ticker,
        signal=signal.type.value,
        reason=signal.reason,
    )

    # ── Step 5: Risk check ───────────────────────────────────────────────────
    positions = await broker.get_positions()
    risk = RiskManager(
        max_positions=settings.max_positions,
        max_notional=settings.max_notional_per_trade,
        drawdown_halt_pct=settings.drawdown_halt_pct,
        broker=broker,
        initial_equity=account_equity,
    )
    # Seed risk manager with existing open positions so the position-count
    # guard is accurate before we attempt a new entry.
    for pos in positions:
        risk.record_open(pos.ticker)

    decision = risk.evaluate(signal, current_price=today_bar.close)
    if not decision.approved:
        log.info("risk_rejected", ticker=ticker, reason=decision.reason)
        await notify.send(
            title="Signal Blocked",
            message=(
                f"BUY signal on {ticker} blocked by risk manager\n"
                f"Reason: {decision.reason}"
            ),
            colour="amber",
        )
        return

    # ── Step 6: Place order ──────────────────────────────────────────────────
    qty = max(1, int(settings.max_notional_per_trade / today_bar.close))
    notional = qty * today_bar.close
    side = signal.type.value.lower()   # "buy" or "sell"

    if dry_run:
        log.info(
            "dry_run_would_place_order",
            ticker=ticker,
            side=side,
            qty=qty,
            price=today_bar.close,
        )
        await notify.send(
            title="Trade Opened (DRY RUN)",
            message=(
                f"BUY {qty} {ticker} @ ${today_bar.close:.2f}\n"
                f"Strategy: {settings.strategy}\n"
                f"Notional: ${notional:.2f}"
            ),
            colour="green",
        )
        return

    # Live mode — submit the order
    if signal.type == SignalType.BUY:
        await broker.place_market_order(ticker=ticker, qty=qty, side="buy")
        log.info(
            "order_placed",
            ticker=ticker,
            side="buy",
            qty=qty,
            price=today_bar.close,
            notional=round(notional, 2),
        )
        await notify.send(
            title="Trade Opened",
            message=(
                f"BUY {qty} {ticker} @ ${today_bar.close:.2f}\n"
                f"Strategy: {settings.strategy}\n"
                f"Notional: ${notional:.2f}"
            ),
            colour="green",
        )
    else:
        # SELL — report unrealised PnL from the existing position if available
        pnl = next(
            (pos.unrealized_pnl for pos in positions if pos.ticker == ticker),
            0.0,
        )
        await broker.place_market_order(ticker=ticker, qty=qty, side="sell")
        log.info(
            "order_placed",
            ticker=ticker,
            side="sell",
            qty=qty,
            price=today_bar.close,
            pnl=round(pnl, 2),
        )
        await notify.send(
            title="Trade Closed",
            message=(
                f"SELL {qty} {ticker} @ ${today_bar.close:.2f}\n"
                f"PnL: ${pnl:.2f}"
            ),
            colour="green",
        )
