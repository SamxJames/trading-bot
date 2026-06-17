"""
Monthly GEM (Dual Momentum) rebalance entry point.

Runs once a month — on the last trading day — triggered by
.github/workflows/monthly_rebalance.yml. Completely separate from the daily
EMA job (bot/job.py): GEM manages its own paper-capital slice
(`gem_allocation_usd`) and holds exactly one of SPY / VEU / AGG at 100%.

Steps
-----
1. Load DualMomentumStrategy from settings.
2. Guard: only act on the last trading day of the month (strategy.should_rebalance);
   the cron fires on the 28th-31st and this guard isolates the real last weekday.
3. Fetch trailing-12m total returns for US (SPY), international (VEU), bonds (AGG)
   and the T-bill rate (^IRX) via yfinance.
4. Compute the target allocation.
5. Determine GEM's *current* holding from rebalance_log.jsonl (authoritative for
   GEM's slice — the daily EMA bot may independently hold SPY, so we must not
   infer GEM's position from the commingled paper account).
6. If the target differs: sell the current holding's GEM quantity, buy the target
   sized to `gem_allocation_usd` (paper account; skipped in --dry-run).
7. Append the decision to trade_journal/rebalance_log.jsonl.
8. Post a Discord notification.

PAPER-ONLY. The Alpaca base URL is the paper endpoint; no real capital is at risk.

Usage
-----
    python -m bot rebalance              # live paper rebalance (guarded by cadence)
    python -m bot rebalance --dry-run    # compute + log + notify, place no orders
    python -m bot rebalance --force      # ignore the last-trading-day guard
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional

from bot import notify
from bot.config import get_settings
from bot.execution.broker import BrokerClient
from bot.logging.logger import get_logger
from bot.strategies.dual_momentum import DualMomentumStrategy

log = get_logger(__name__)

REBALANCE_LOG_PATH = Path("bot/trade_journal/rebalance_log.jsonl")


# ---------------------------------------------------------------------------
# Data fetching (yfinance — same source as the backtest)
# ---------------------------------------------------------------------------

def _fetch_monthly_closes(symbol: str) -> List[float]:
    """Ascending list of month-end adjusted closes (~3 years) for `symbol`.

    Returns [] on any error — callers degrade to bonds via compute_target().
    """
    try:
        import yfinance as yf

        df = yf.Ticker(symbol).history(period="3y", interval="1mo", auto_adjust=True)
        if df is None or df.empty:
            return []
        return [float(x) for x in df["Close"].dropna().tolist()]
    except Exception as exc:
        log.warning("gem_price_fetch_failed", symbol=symbol, error=str(exc))
        return []


def _fetch_tbill_12m(lookback_months: int) -> Optional[float]:
    """Trailing-`lookback`-month T-bill total return, compounded from ^IRX
    (annualised 13-week T-bill discount yield, percent). None on data error."""
    try:
        import yfinance as yf

        df = yf.Ticker("^IRX").history(period="3y", interval="1mo", auto_adjust=True)
        if df is None or df.empty:
            return None
        ylds = [float(x) for x in df["Close"].dropna().tolist()]
        if len(ylds) < lookback_months:
            return None
        compounded = 1.0
        for y in ylds[-lookback_months:]:
            compounded *= 1.0 + (y / 100.0) / 12.0
        return compounded - 1.0
    except Exception as exc:
        log.warning("gem_tbill_fetch_failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Rebalance-log persistence
# ---------------------------------------------------------------------------

def _write_rebalance_log(record: dict) -> None:
    """Append one JSON line to the rebalance audit log. Never raises."""
    try:
        REBALANCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REBALANCE_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        log.warning("rebalance_log_write_failed", error=str(exc))


def _last_rebalance_state() -> Optional[dict]:
    """Most recent rebalance record that actually established a holding, or None.

    Used to know what GEM currently holds and at what quantity, independent of
    the commingled broker account.
    """
    if not REBALANCE_LOG_PATH.exists():
        return None
    last: Optional[dict] = None
    try:
        for line in REBALANCE_LOG_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("event") == "gem_rebalance" and entry.get("holding"):
                last = entry
    except Exception as exc:
        log.warning("rebalance_log_read_failed", error=str(exc))
    return last


def _pct(v: Optional[float]) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "n/a"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_rebalance(dry_run: bool = False, force: bool = False) -> None:
    """Run the monthly GEM rebalance. Re-raises on unhandled error so CI marks
    the run red (after sending a Discord alert)."""
    t_start = time.monotonic()
    try:
        await _run(dry_run=dry_run, force=force)
    except Exception as exc:
        log.exception("gem_rebalance_failed", error=str(exc))
        _write_rebalance_log({
            "event": "gem_rebalance_failed",
            "ts": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        })
        try:
            await notify.send("GEM Rebalance Failed", f"Error: {exc}", colour="red")
        except Exception:
            pass
        raise
    log.info("gem_rebalance_complete", duration_seconds=round(time.monotonic() - t_start, 2))


async def _run(dry_run: bool, force: bool) -> None:
    settings = get_settings()

    if not settings.gem_enabled:
        log.info("gem_disabled", note="gem_enabled is false — skipping rebalance")
        return

    strategy = DualMomentumStrategy(
        gem_lookback_months=settings.gem_lookback_months,
        gem_assets=settings.gem_assets,
        gem_enabled=settings.gem_enabled,
        gem_rebalance=settings.gem_rebalance,
    )

    today = date.today()
    if not force and not strategy.should_rebalance(today):
        log.info("gem_not_rebalance_day", date=str(today))
        return

    # ── Fetch trailing-12m momentum readings ──────────────────────────────────
    us_prices   = _fetch_monthly_closes(strategy.us_ticker)
    intl_prices = _fetch_monthly_closes(strategy.intl_ticker)
    bond_prices = _fetch_monthly_closes(strategy.bonds_ticker)

    us_12m   = strategy.trailing_return(us_prices)
    intl_12m = strategy.trailing_return(intl_prices)
    tbill_12m = _fetch_tbill_12m(strategy.gem_lookback_months)

    target = strategy.compute_target(us_12m, intl_12m, tbill_12m)

    # Reference price for sizing / logging (last month-end close of the target).
    price_by_ticker = {
        strategy.us_ticker:    us_prices[-1]   if us_prices   else None,
        strategy.intl_ticker:  intl_prices[-1] if intl_prices else None,
        strategy.bonds_ticker: bond_prices[-1] if bond_prices else None,
    }
    target_price = price_by_ticker.get(target)

    # ── Current GEM holding (from the log — authoritative for GEM's slice) ────
    prev = _last_rebalance_state()
    current_holding = prev.get("holding") if prev else None
    current_qty = float(prev.get("qty", 0) or 0) if prev else 0.0
    changed = target != current_holding

    # ── Broker context (connectivity + commingled positions, for logging) ────
    broker = BrokerClient(
        api_key=settings.apca_api_key_id,
        secret_key=settings.apca_api_secret_key,
        base_url=settings.apca_base_url,
    )
    account = await broker.get_account()
    log.info("gem_account_ok", equity=account.equity, status=account.status)

    target_qty = current_qty
    order_results: list[str] = []

    if not changed:
        log.info("gem_no_change", holding=target)
        order_results.append(f"No change — already holding {target}")
    else:
        # Size the new target position to GEM's capital slice.
        if target_price and target_price > 0:
            target_qty = float(max(1, int(settings.gem_allocation_usd / target_price)))
        else:
            target_qty = current_qty or 1.0

        if dry_run:
            if current_holding and current_qty > 0:
                order_results.append(f"[dry-run] SELL {current_qty:g} {current_holding}")
            order_results.append(f"[dry-run] BUY {target_qty:g} {target} @ ~${target_price:.2f}"
                                 if target_price else f"[dry-run] BUY {target_qty:g} {target}")
        else:
            # Sell the prior GEM holding (its tracked quantity), then buy target.
            if current_holding and current_qty > 0:
                try:
                    await broker.place_market_order(ticker=current_holding, qty=current_qty, side="sell")
                    order_results.append(f"SELL {current_qty:g} {current_holding}")
                except Exception as exc:
                    log.warning("gem_sell_failed", ticker=current_holding, error=str(exc))
                    order_results.append(f"SELL {current_holding} FAILED: {exc}")
            try:
                await broker.place_market_order(ticker=target, qty=target_qty, side="buy")
                order_results.append(f"BUY {target_qty:g} {target}")
            except Exception as exc:
                log.warning("gem_buy_failed", ticker=target, error=str(exc))
                order_results.append(f"BUY {target} FAILED: {exc}")

    # ── Audit log ─────────────────────────────────────────────────────────────
    record = {
        "event": "gem_rebalance",
        "ts": datetime.now(timezone.utc).isoformat(),
        "date": str(today),
        "holding": target,
        "previous_holding": current_holding,
        "changed": changed,
        "qty": target_qty,
        "price": round(target_price, 4) if target_price else None,
        "allocation_usd": settings.gem_allocation_usd,
        "us_ticker": strategy.us_ticker,
        "intl_ticker": strategy.intl_ticker,
        "bonds_ticker": strategy.bonds_ticker,
        "us_12m": _round(us_12m),
        "intl_12m": _round(intl_12m),
        "tbill_12m": _round(tbill_12m),
        "dry_run": dry_run,
        "orders": order_results,
    }
    _write_rebalance_log(record)

    # ── Discord notification ──────────────────────────────────────────────────
    change_line = (
        f"Rotated {current_holding or 'cash'} -> {target}" if changed
        else f"No change (still {target})"
    )
    await notify.send(
        title=f"GEM Monthly Rebalance: holding {target}",
        message=(
            f"{change_line}\n"
            f"{strategy.us_ticker} 12mo: {_pct(us_12m)}  |  "
            f"{strategy.intl_ticker} 12mo: {_pct(intl_12m)}  |  "
            f"T-bill: {_pct(tbill_12m)}\n"
            f"Allocation: ${settings.gem_allocation_usd:,.0f} (paper)\n"
            f"Orders: {', '.join(order_results) if order_results else 'none'}"
        ),
        colour="green" if changed else "grey",
    )


def _round(v: Optional[float]) -> Optional[float]:
    return round(v, 4) if isinstance(v, (int, float)) else None
