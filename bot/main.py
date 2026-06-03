"""
Live trading entry point.

Orchestrates the full pipeline for a live paper-trading session:
  1. Startup checklist — validates credentials, data, strategy, config
  2. Connect to broker and verify account
  3. Instantiate the configured strategy and risk manager
  4. Start the live data feed (WebSocket) as a background task
  5. Run the trading loop: bar → strategy → risk check → order (or dry-run log)
  6. Every 5 minutes: log a health check (equity, positions, session counters)
  7. On shutdown (Ctrl+C / drawdown halt): flatten, print session summary,
     write the trade journal CSV

Dry-run mode (--dry-run):
  Full loop including signal generation and risk checks, but no orders are
  placed — "DRY RUN — would place order" is logged instead.

Usage:
    python -m bot live
    python -m bot live --dry-run
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict

from bot.config import get_settings
from bot.data.feed import Bar, LiveFeed
from bot.execution.broker import BrokerClient
from bot.logging.logger import TradeJournal, configure_logging, get_logger
from bot.risk.manager import RiskManager
from bot.strategies.registry import get_strategy, REGISTRY


# ---------------------------------------------------------------------------
# Session stats container — shared across coroutines via dict reference
# ---------------------------------------------------------------------------

def _new_stats(start_equity: float) -> Dict:
    return {
        "start_time":       datetime.now(tz=timezone.utc),
        "start_equity":     start_equity,
        "signals_fired":    0,
        "signals_approved": 0,
        "signals_rejected": 0,
        "orders_placed":    0,
    }


# ---------------------------------------------------------------------------
# 5c — Startup checklist
# ---------------------------------------------------------------------------

async def _startup_checklist(broker: BrokerClient, settings, log) -> float:
    """
    Validate that the bot is ready to trade.  Returns the current equity.
    Raises SystemExit with a clear message if any check fails.
    """
    print("\n── Startup checklist ──────────────────────────────────────")

    # 1. API credentials + connectivity
    try:
        account = await broker.get_account()
        print(f"  [OK] Alpaca API reachable  |  equity=${account.equity:,.2f}  status={account.status}")
    except Exception as exc:
        raise SystemExit(f"  [FAIL] FAILED: Alpaca API credentials invalid -- {exc}") from exc

    # 2. Market data accessible (fetch a tiny sample for the first configured ticker)
    try:
        from bot.data.historical import fetch_bars
        from datetime import date, timedelta
        test_end   = date.today()
        test_start = test_end - timedelta(days=10)
        await fetch_bars(settings.tickers[0], test_start, test_end)
        print(f"  [OK] Market data accessible  |  ticker={settings.tickers[0]}")
    except Exception as exc:
        raise SystemExit(f"  [FAIL] FAILED: Market data not accessible -- {exc}") from exc

    # 3. Strategy loads without error
    try:
        get_strategy(
            settings.strategy,
            fast_period=settings.fast_period,
            slow_period=settings.slow_period,
            trend_sma_period=settings.trend_sma_period,
            rsi_period=settings.rsi_period,
            rsi_oversold=settings.rsi_oversold,
            rsi_overbought=settings.rsi_overbought,
            stop_loss_pct=settings.stop_loss_pct,
        )
        print(f"  [OK] Strategy '{settings.strategy}' loaded  "
              f"|  registered: {sorted(REGISTRY.keys())}")
    except KeyError as exc:
        raise SystemExit(f"  [FAIL] FAILED: Strategy not found -- {exc}") from exc

    # 4. Config sanity bounds
    errors = []
    if settings.fast_period >= settings.slow_period:
        errors.append(
            f"fast_period ({settings.fast_period}) must be < slow_period ({settings.slow_period})"
        )
    if settings.stop_loss_pct <= 0:
        errors.append(f"stop_loss_pct must be positive, got {settings.stop_loss_pct}")
    if "paper" not in settings.apca_base_url.lower():
        errors.append(
            "apca_base_url must point to the paper trading API "
            "(https://paper-api.alpaca.markets)"
        )
    if errors:
        raise SystemExit("  [FAIL] FAILED config bounds:\n  " + "\n  ".join(errors))
    print(f"  [OK] Config bounds OK  |  fast={settings.fast_period}  "
          f"slow={settings.slow_period}  stop_loss={settings.stop_loss_pct}%")

    print("───────────────────────────────────────────────────────────\n")
    return account.equity


# ---------------------------------------------------------------------------
# 5a — Health check loop (every 5 minutes)
# ---------------------------------------------------------------------------

async def _health_check_loop(
    broker: BrokerClient,
    stats: Dict,
    log,
    interval: int = 300,
) -> None:
    """Log a health snapshot every *interval* seconds."""
    while True:
        await asyncio.sleep(interval)
        try:
            account   = await broker.get_account()
            positions = await broker.get_positions()
            log.info(
                "health_check",
                equity=account.equity,
                positions=[
                    {"ticker": p.ticker, "unrealised_pnl": round(p.unrealized_pnl, 2)}
                    for p in positions
                ],
                signals_fired=stats["signals_fired"],
                signals_approved=stats["signals_approved"],
                signals_rejected=stats["signals_rejected"],
                orders_placed=stats["orders_placed"],
            )
        except Exception as exc:
            log.warning("health_check_failed", error=str(exc))


# ---------------------------------------------------------------------------
# 5b — Session summary
# ---------------------------------------------------------------------------

def _print_session_summary(stats: Dict, final_equity: float, journal: TradeJournal) -> None:
    """Print a human-readable session summary to stdout."""
    now      = datetime.now(tz=timezone.utc)
    duration = now - stats["start_time"]
    hours, rem  = divmod(int(duration.total_seconds()), 3600)
    minutes, secs = divmod(rem, 60)
    pnl = final_equity - stats["start_equity"]

    sep = "=" * 54
    print(f"\n{sep}")
    print("  SESSION SUMMARY")
    print(sep)
    print(f"  Duration          : {hours:02d}h {minutes:02d}m {secs:02d}s")
    print(f"  Starting equity   : ${stats['start_equity']:>10,.2f}")
    print(f"  Ending equity     : ${final_equity:>10,.2f}")
    print(f"  Session PnL       : ${pnl:>+10,.2f}")
    print(f"  Signals fired     : {stats['signals_fired']}")
    print(f"  Signals approved  : {stats['signals_approved']}")
    print(f"  Signals rejected  : {stats['signals_rejected']}")
    print(f"  Orders placed     : {stats['orders_placed']}")
    trades = journal._trades
    if trades:
        print(f"\n  {'Ticker':<6} {'Side':<5} {'Entry':>8} {'Exit':>8} {'PnL':>9}")
        print(f"  {'-'*6} {'-'*5} {'-'*8} {'-'*8} {'-'*9}")
        for t in trades:
            exit_str  = f"{t.exit_price:.2f}"  if t.exit_price  else "  open"
            pnl_str   = f"${t.pnl:>+.2f}"      if t.pnl is not None else "    N/A"
            print(f"  {t.ticker:<6} {t.side:<5} {t.entry_price:>8.2f} {exit_str:>8} {pnl_str:>9}")
    else:
        print("\n  No trades this session.")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run(dry_run: bool = False) -> None:
    """Main async loop for live paper trading."""
    settings = get_settings()
    log      = get_logger(__name__)

    if dry_run:
        log.info("dry_run_mode_active",
                 note="Orders will be logged but NOT sent to Alpaca")

    broker = BrokerClient(
        api_key=settings.apca_api_key_id,
        secret_key=settings.apca_api_secret_key,
        base_url=settings.apca_base_url,
    )

    # ── 5c: Startup checklist ────────────────────────────────────────────────
    initial_equity = await _startup_checklist(broker, settings, log)

    # ── Initialise strategy, risk manager, journal ───────────────────────────
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
    risk = RiskManager(
        max_positions=settings.max_positions,
        max_notional=settings.max_notional_per_trade,
        drawdown_halt_pct=settings.drawdown_halt_pct,
        broker=broker,
        initial_equity=initial_equity,
    )
    journal = TradeJournal()
    stats   = _new_stats(initial_equity)
    bar_queue: asyncio.Queue[Bar] = asyncio.Queue()

    strategy.on_start()
    log.info("session_started",
             strategy=settings.strategy,
             tickers=settings.tickers,
             dry_run=dry_run)

    async with LiveFeed(settings.tickers, bar_queue) as feed:
        feed_task    = asyncio.create_task(feed.start())
        # ── 5a: health check every 5 minutes ──────────────────────────────
        health_task  = asyncio.create_task(
            _health_check_loop(broker, stats, log, interval=300)
        )
        try:
            await _trading_loop(
                bar_queue=bar_queue,
                strategy=strategy,
                risk=risk,
                broker=broker,
                journal=journal,
                settings=settings,
                stats=stats,
                log=log,
                dry_run=dry_run,
            )
        except asyncio.CancelledError:
            log.info("kill_switch_triggered")
        except Exception:
            log.exception("fatal_error")
        finally:
            health_task.cancel()
            feed_task.cancel()
            for task in (health_task, feed_task):
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            strategy.on_stop()
            log.info("flattening_positions")
            await risk.emergency_flatten()
            journal.write("trades_live.csv")

            # ── 5b: Session summary ────────────────────────────────────────
            try:
                final = await broker.get_account()
                final_equity = final.equity
            except Exception:
                final_equity = stats["start_equity"]
            log.info("session_ended", final_equity=final_equity)
            _print_session_summary(stats, final_equity, journal)


# ---------------------------------------------------------------------------
# Inner trading loop
# ---------------------------------------------------------------------------

async def _trading_loop(
    bar_queue: asyncio.Queue,
    strategy: object,
    risk: RiskManager,
    broker: BrokerClient,
    journal: TradeJournal,
    settings: object,
    stats: Dict,
    log: object,
    dry_run: bool = False,
) -> None:
    """
    Core event loop: consume bars from the queue, run strategy + risk, place orders.
    Runs until cancelled (Ctrl+C) or the risk manager halts trading.
    """
    while True:
        try:
            bar: Bar = await asyncio.wait_for(bar_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        log.info("bar_received",
                 ticker=bar.ticker, close=bar.close,
                 ts=bar.timestamp.isoformat())

        if risk.is_halted():
            log.warning("trading_halted", reason="drawdown_limit_reached")
            continue

        signal = strategy.on_bar(bar)
        if signal is None:
            continue

        stats["signals_fired"] += 1
        log.info("signal_generated",
                 ticker=bar.ticker,
                 signal=signal.type.value,
                 reason=signal.reason)

        decision = risk.evaluate(signal, current_price=bar.close)
        if not decision.approved:
            stats["signals_rejected"] += 1
            log.info("risk_rejected", ticker=bar.ticker, reason=decision.reason)
            continue

        stats["signals_approved"] += 1
        log.info("risk_approved", ticker=bar.ticker, signal=signal.type.value)

        qty = max(1, int(settings.max_notional_per_trade / bar.close))

        # ── 5d: Dry-run gate ──────────────────────────────────────────────
        if dry_run:
            log.info(
                "dry_run_would_place_order",
                ticker=bar.ticker,
                side=signal.type.value.lower(),
                qty=qty,
                price=bar.close,
            )
            # Still track positions in risk manager so counters are accurate
            if signal.type.value.lower() == "buy":
                risk.record_open(bar.ticker)
            else:
                risk.record_close(bar.ticker)
        else:
            await broker.place_market_order(
                ticker=bar.ticker,
                qty=qty,
                side=signal.type.value.lower(),
            )
            stats["orders_placed"] += 1
            if signal.type.value.lower() == "buy":
                risk.record_open(bar.ticker)
            else:
                risk.record_close(bar.ticker)

            # Refresh equity so the drawdown guard stays current
            acct = await broker.get_account()
            risk.update_equity(acct.equity)
