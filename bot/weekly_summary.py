"""
Weekly performance summary.

Reads trades_live.csv, computes rolling stats, and posts a structured
Discord summary every Monday morning before market open.

Designed to run as a standalone GitHub Actions job — no broker connection
needed, no credentials beyond DISCORD_WEBHOOK_URL.

Exit codes
----------
  0 — summary posted (or no trades yet — posts a "no trades" message)
  1 — unhandled exception

Usage
-----
    python -m bot weekly          # post summary to Discord
    python -m bot weekly --dry-run  # print summary to console only
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

from bot.logging.logger import get_logger

log = get_logger(__name__)

JOURNAL_PATH = Path("trades_live.csv")
ANALYTICS_PATH = Path("docs/analytics.json")
SIGNAL_LOG_PATH = Path("bot/trade_journal/signal_log.jsonl")

# Must match bot.job.FILTER_KEYS / scripts.analyse_trades.FILTER_BLOCK_KEYS.
FILTER_BLOCK_KEYS = [
    "trend_sma", "rsi_overbought", "volume", "vix",
    "spy_macro", "earnings", "correlation", "weekly_ema",
]

FILTER_BAR_WIDTH = 10
FILTER_OVERACTIVE_THRESHOLD = 0.60

# ── Data model ────────────────────────────────────────────────────────────────

class Trade(NamedTuple):
    ticker: str
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    entry_ts: datetime
    exit_ts: datetime
    qty: float


# ── CSV reader ────────────────────────────────────────────────────────────────

def load_trades(path: Path = JOURNAL_PATH) -> list[Trade]:
    """Load completed trades from the CSV journal. Returns [] if file missing."""
    if not path.exists():
        return []

    trades = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                trades.append(Trade(
                    ticker=row["ticker"],
                    side=row["side"],
                    entry_price=float(row["entry_price"]),
                    exit_price=float(row["exit_price"]),
                    pnl=float(row["pnl"]),
                    entry_ts=datetime.fromisoformat(row["entry_ts"]),
                    exit_ts=datetime.fromisoformat(row["exit_ts"]),
                    qty=float(row.get("qty", 0)),
                ))
            except (KeyError, ValueError):
                continue  # skip malformed rows

    return trades


# ── Stats engine ──────────────────────────────────────────────────────────────

def compute_stats(trades: list[Trade], since: date | None = None) -> dict:
    """
    Compute performance stats for a list of trades.

    Parameters
    ----------
    trades: all completed trades
    since:  if provided, also compute a rolling window slice

    Returns a dict with all_time and recent keys.
    """
    def _stats(subset: list[Trade]) -> dict:
        if not subset:
            return {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "best_trade": None,
                "worst_trade": None,
                "by_ticker": {},
            }

        wins   = [t for t in subset if t.pnl > 0]
        losses = [t for t in subset if t.pnl <= 0]

        by_ticker: dict[str, dict] = {}
        for t in subset:
            tk = by_ticker.setdefault(t.ticker, {"trades": 0, "pnl": 0.0, "wins": 0})
            tk["trades"] += 1
            tk["pnl"]    += t.pnl
            if t.pnl > 0:
                tk["wins"] += 1

        best  = max(subset, key=lambda t: t.pnl)
        worst = min(subset, key=lambda t: t.pnl)

        return {
            "trades":    len(subset),
            "wins":      len(wins),
            "losses":    len(losses),
            "win_rate":  len(wins) / len(subset) * 100 if subset else 0.0,
            "total_pnl": sum(t.pnl for t in subset),
            "avg_win":   sum(t.pnl for t in wins)   / len(wins)   if wins   else 0.0,
            "avg_loss":  sum(t.pnl for t in losses) / len(losses) if losses else 0.0,
            "best_trade":  best,
            "worst_trade": worst,
            "by_ticker": by_ticker,
        }

    recent_trades = (
        [t for t in trades if t.exit_ts.date() >= since]
        if since else trades
    )

    return {
        "all_time": _stats(trades),
        "recent":   _stats(recent_trades),
        "since":    since,
    }


# ── Filter activity (30d) ────────────────────────────────────────────────────

def _load_filter_blocks_30d(path: Path = ANALYTICS_PATH) -> dict[str, int]:
    """Read filter_blocks_30d from docs/analytics.json. Returns {} if missing/unreadable."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("filter_blocks_30d", {})


def _count_signal_evaluations_30d(path: Path = SIGNAL_LOG_PATH) -> int:
    """Count signal_evaluation records in signal_log.jsonl from the last 30 days."""
    if not path.exists():
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("event") != "signal_evaluation":
            continue
        ts_raw = entry.get("ts")
        if not ts_raw:
            continue
        ts = datetime.fromisoformat(ts_raw)
        if ts >= cutoff:
            count += 1
    return count


SIGNAL_FUNNEL_TOP_N = 3


def _load_signal_funnel(path: Path = ANALYTICS_PATH) -> dict:
    """Read signal_funnel from docs/analytics.json. Returns {} if missing/unreadable."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("signal_funnel", {})


def build_signal_funnel_lines() -> list[str]:
    """
    Build the "Signal Funnel (30d)" section: EMA crossovers detected vs.
    passed/blocked by filters, plus the top blocking filters.

    Skips gracefully (returns a placeholder line) if docs/analytics.json is
    missing or signal_funnel data is unavailable.
    """
    funnel = _load_signal_funnel()
    if not funnel:
        return ["No signal funnel data yet — check back after first live run"]

    crossovers = funnel.get("crossovers_30d", 0)
    fired      = funnel.get("signals_fired_30d", 0)
    blocked    = funnel.get("signals_blocked_30d", 0)

    if crossovers == 0:
        return ["No EMA crossovers detected in last 30 days — market may be trending sideways."]

    fired_pct   = round(fired / crossovers * 100)
    blocked_pct = round(blocked / crossovers * 100)

    lines = [
        f"EMA crossovers detected: {crossovers}",
        f"Passed all filters:      {fired}  ({fired_pct}%)",
        f"Blocked by filters:      {blocked}  ({blocked_pct}%)",
    ]

    if fired == 0:
        lines.append("")
        lines.append("⚠️ All crossovers blocked by filters — consider reviewing thresholds.")

    filter_funnel = funnel.get("filter_funnel", {})
    top_blockers = sorted(
        ((name, count) for name, count in filter_funnel.items() if count > 0),
        key=lambda x: x[1],
        reverse=True,
    )[:SIGNAL_FUNNEL_TOP_N]
    if top_blockers:
        lines.append("")
        lines.append("Top blockers:")
        for name, count in top_blockers:
            lines.append(f"  {name:<12} {count} blocks")

    return lines


def build_filter_activity_lines() -> list[str]:
    """
    Build the "Filter Activity (30d)" section: one Unicode block-bar per
    filter (scaled to the most-active filter), flagging any filter whose
    block rate exceeds FILTER_OVERACTIVE_THRESHOLD.

    Skips gracefully (returns a placeholder line) if docs/analytics.json is
    missing or filter_blocks_30d is all zeros.
    """
    filter_blocks = _load_filter_blocks_30d()
    if not filter_blocks or not any(filter_blocks.values()):
        return ["No filter data yet — check back after first live run"]

    total_evaluated = _count_signal_evaluations_30d()
    max_count = max(filter_blocks.values())

    lines: list[str] = []
    for name in FILTER_BLOCK_KEYS:
        count = filter_blocks.get(name, 0)
        filled = round((count / max_count) * FILTER_BAR_WIDTH) if max_count else 0
        bar = "█" * filled + "░" * (FILTER_BAR_WIDTH - filled)
        block_rate = count / total_evaluated if total_evaluated else 0.0
        flag = " ⚠️ potentially over-active" if block_rate > FILTER_OVERACTIVE_THRESHOLD else ""
        lines.append(f"`{bar}` {name}: {count}{flag}")
    return lines


# ── Message formatter ─────────────────────────────────────────────────────────

def _pnl_str(pnl: float) -> str:
    sign = "+" if pnl >= 0 else ""
    return f"{sign}${pnl:.2f}"


def _trade_line(t: Trade) -> str:
    return (
        f"{t.side.upper()} {t.ticker} "
        f"@ ${t.entry_price:.2f} → ${t.exit_price:.2f}  "
        f"({_pnl_str(t.pnl)})"
    )


def build_discord_message(stats: dict) -> tuple[str, str, str]:
    """
    Returns (title, message_body, colour) for the Discord embed.
    """
    today       = date.today()
    week_start  = stats["since"]
    all_time    = stats["all_time"]
    recent      = stats["recent"]

    # ── Colour logic ──────────────────────────────────────────────────────────
    if all_time["trades"] == 0:
        colour = "grey"
    elif all_time["total_pnl"] >= 0:
        colour = "green"
    else:
        colour = "red"

    title = f"📊 Weekly Performance Summary — {today.strftime('%d %b %Y')}"

    # ── No trades yet ─────────────────────────────────────────────────────────
    if all_time["trades"] == 0:
        message = (
            "No completed trades on record yet.\n\n"
            "The bot is live and running — signals will appear here once "
            "a full entry → exit cycle completes."
        )
        message += "\n\n**Filter Activity (30d)**\n" + "\n".join(build_filter_activity_lines())
        message += "\n\n**Signal Funnel (30d)**\n" + "\n".join(build_signal_funnel_lines())
        return title, message, colour

    # ── Build body ────────────────────────────────────────────────────────────
    lines: list[str] = []

    # This week
    lines.append(f"**This week** ({week_start} → {today})")
    if recent["trades"] == 0:
        lines.append("No completed trades this week.")
    else:
        lines.append(
            f"Trades: {recent['trades']}  |  "
            f"Win rate: {recent['win_rate']:.0f}%  |  "
            f"PnL: {_pnl_str(recent['total_pnl'])}"
        )

    lines.append("")

    # All time
    lines.append("**All time**")
    lines.append(
        f"Trades: {all_time['trades']}  |  "
        f"Win rate: {all_time['win_rate']:.0f}%  |  "
        f"Total PnL: {_pnl_str(all_time['total_pnl'])}"
    )
    lines.append(
        f"Avg win: {_pnl_str(all_time['avg_win'])}  |  "
        f"Avg loss: {_pnl_str(all_time['avg_loss'])}"
    )

    # Per-ticker breakdown
    if all_time["by_ticker"]:
        lines.append("")
        lines.append("**By ticker**")
        for ticker, tk in sorted(
            all_time["by_ticker"].items(),
            key=lambda x: x[1]["pnl"],
            reverse=True,
        ):
            win_rate = tk["wins"] / tk["trades"] * 100 if tk["trades"] else 0
            lines.append(
                f"`{ticker}` — {tk['trades']} trades  "
                f"{win_rate:.0f}% win  "
                f"{_pnl_str(tk['pnl'])}"
            )

    # Best / worst trade
    if all_time["best_trade"]:
        lines.append("")
        lines.append(f"**Best trade:** {_trade_line(all_time['best_trade'])}")
    if all_time["worst_trade"] and all_time["worst_trade"] != all_time["best_trade"]:
        lines.append(f"**Worst trade:** {_trade_line(all_time['worst_trade'])}")

    lines.append("")
    lines.append("**Filter Activity (30d)**")
    lines.extend(build_filter_activity_lines())

    lines.append("")
    lines.append("**Signal Funnel (30d)**")
    lines.extend(build_signal_funnel_lines())

    return title, "\n".join(lines), colour


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_weekly_summary(dry_run: bool = False) -> None:
    """Load trades, compute stats, post to Discord (or print if dry_run)."""
    trades = load_trades()
    log.info("weekly_summary_start", total_trades=len(trades))

    # Rolling window = last 7 days
    since = date.today() - timedelta(days=7)
    stats = compute_stats(trades, since=since)

    title, message, colour = build_discord_message(stats)

    if dry_run:
        print(f"\n{'='*60}")
        print(f"TITLE:  {title}")
        print(f"COLOUR: {colour}")
        print(f"\n{message}")
        print('='*60)
        log.info("weekly_summary_dry_run_complete")
        return

    from bot import notify
    await notify.send(title=title, message=message, colour=colour)
    log.info(
        "weekly_summary_sent",
        total_trades=stats["all_time"]["trades"],
        total_pnl=round(stats["all_time"]["total_pnl"], 2),
        recent_trades=stats["recent"]["trades"],
    )
