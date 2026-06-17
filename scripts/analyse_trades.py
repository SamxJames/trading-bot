"""
Trade analytics module.

Reads trades_live.csv and computes richer statistics than the weekly
Discord summary: Sharpe ratio, max consecutive losses, average hold time,
per-ticker win rate and R:R, equity curve, and drawdown.

Also reads signal_log.jsonl (written by the bot's structlog output,
captured by the daily job) to surface filter block reasons.

Usage:
    python scripts/analyse_trades.py                  # prints summary
    python scripts/analyse_trades.py --json           # outputs JSON (used by dashboard)
    python scripts/analyse_trades.py --out docs/      # writes JSON to docs/analytics.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

TRADES_PATH  = Path("bot/trade_journal/trades_live.csv")
LOG_PATH     = Path("bot/trade_journal/signal_log.jsonl")
SHADOW_TRADES_PATH = Path("bot/trade_journal/shadow_trades.csv")
GEM_LOG_PATH = Path("bot/trade_journal/rebalance_log.jsonl")
STARTING_EQUITY = 100_000.0
GEM_DEFAULT_ALLOCATION = 50_000.0
RISK_FREE_RATE  = 0.04

# Filter names shown on the dashboard's "filter blocks" bar chart, and the
# mapping from structlog event/reason -> one of those names.
FILTER_BLOCK_KEYS = [
    "trend_sma", "rsi_overbought", "volume", "vix",
    "spy_macro", "earnings", "correlation", "weekly_ema",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _sharpe(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    if len(returns) < 2:
        return 0.0
    daily_rf = rf / 252
    excess = returns - daily_rf
    std = excess.std()
    if std == 0:
        return 0.0
    trades_per_year = max(len(returns), 1)
    return float((excess.mean() / std) * math.sqrt(trades_per_year))


def _max_drawdown(equity_curve: list[float]) -> tuple[float, int | None]:
    """Return (max drawdown %, index of the trough) relative to the prior peak."""
    if not equity_curve:
        return 0.0, None
    peak = equity_curve[0]
    max_dd = 0.0
    max_dd_idx: int | None = None
    for i, v in enumerate(equity_curve):
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd
            max_dd_idx = i
    return round(max_dd, 2), max_dd_idx


def _flat_equity_curve() -> list[dict]:
    """A flat $100,000 line spanning the last 30 days, used when there are no trades yet."""
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=30)
    return [
        {"date": start.isoformat(), "equity": STARTING_EQUITY},
        {"date": today.isoformat(), "equity": STARTING_EQUITY},
    ]


# ── open positions ───────────────────────────────────────────────────────────

def _current_pnl_pct(ticker: str, entry_price: float | None) -> float | None:
    """Best-effort current unrealised PnL %, fetched via yfinance. Fails permissive (None)."""
    if not entry_price:
        return None
    try:
        import yfinance as yf

        hist = yf.Ticker(ticker).history(period="1d")
        if hist.empty:
            return None
        current = float(hist["Close"].iloc[-1])
        return round((current - entry_price) / entry_price * 100, 2)
    except Exception:
        return None


def _compute_open_positions(trades_path: Path) -> list[dict]:
    """
    FIFO-match BUY/SELL rows per ticker in trades_live.csv; any unmatched BUYs
    are still-open positions.
    """
    if not trades_path.exists():
        return []

    try:
        df = pd.read_csv(trades_path)
        df.columns = df.columns.str.strip()
    except Exception:
        return []

    if "side" not in df.columns or "ticker" not in df.columns:
        return []

    df["timestamp"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
    df = df.sort_values("timestamp")

    open_positions: list[dict] = []
    for ticker, group in df.groupby("ticker"):
        rows  = list(group.to_dict("records"))
        buys  = [r for r in rows if str(r.get("side", "")).upper() == "BUY"]
        sells = [r for r in rows if str(r.get("side", "")).upper() == "SELL"]
        for row in buys[len(sells):]:
            entry_price = row.get("entry_price", row.get("price"))
            entry_price = float(entry_price) if pd.notna(entry_price) else None
            entry_ts    = row.get("timestamp")
            open_positions.append({
                "ticker":          str(ticker),
                "entry_price":     entry_price,
                "entry_date":      entry_ts.date().isoformat() if pd.notna(entry_ts) else None,
                "current_pnl_pct": _current_pnl_pct(str(ticker), entry_price),
            })
    return open_positions


def _max_consecutive_losses(outcomes: list[bool]) -> int:
    max_streak = current = 0
    for win in outcomes:
        if not win:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _rr_ratio(wins: pd.Series, losses: pd.Series) -> float:
    if losses.empty or wins.empty:
        return 0.0
    avg_win  = wins.mean()
    avg_loss = abs(losses.mean())
    if avg_loss == 0:
        return 0.0
    return round(avg_win / avg_loss, 2)


# ── shadow trading ────────────────────────────────────────────────────────────

def _empty_shadow_stats() -> dict[str, Any]:
    return {
        "total_trades":     0,
        "win_rate":         0.0,
        "sharpe":           0.0,
        "total_return_pct": 0.0,
        "avg_rr":           0.0,
        "max_drawdown_pct": 0.0,
        "equity_curve":     _flat_equity_curve(),
    }


def _compute_shadow_stats(shadow_path: Path = SHADOW_TRADES_PATH) -> dict[str, Any]:
    """
    Read shadow_trades.csv (closed-trade rows: id, ticker, side, entry_price,
    exit_price, pnl_usd, pnl_pct, reason, entry_date, exit_date, bars_held)
    and compute the same headline stats as the live `overall` block, scoped
    to the shadow (simulated-fill) trade history.
    """
    if not shadow_path.exists():
        return _empty_shadow_stats()

    try:
        df = pd.read_csv(shadow_path)
        df.columns = df.columns.str.strip()
    except Exception:
        return _empty_shadow_stats()

    if df.empty:
        return _empty_shadow_stats()

    df["pnl_usd"] = pd.to_numeric(df["pnl_usd"], errors="coerce").fillna(0.0)
    df["entry_date"] = pd.to_datetime(df["entry_date"], utc=True, errors="coerce")
    df["exit_date"]  = pd.to_datetime(df["exit_date"], utc=True, errors="coerce")
    df = df.sort_values("exit_date").reset_index(drop=True)

    total_trades = len(df)
    wins   = df[df["pnl_usd"] > 0]["pnl_usd"]
    losses = df[df["pnl_usd"] <= 0]["pnl_usd"]
    win_rate = round(len(wins) / total_trades, 3) if total_trades else 0.0
    sharpe   = round(_sharpe(df["pnl_usd"]), 3)
    avg_rr   = _rr_ratio(wins, losses)

    first_entry = df["entry_date"].iloc[0]
    start_date = (first_entry - pd.Timedelta(days=1)).date().isoformat() if pd.notna(first_entry) else None

    equity = STARTING_EQUITY
    equity_curve: list[dict] = [{"date": start_date, "equity": round(equity, 2)}]
    for _, row in df.iterrows():
        equity += row["pnl_usd"]
        equity_curve.append({
            "date":     row["exit_date"].date().isoformat() if pd.notna(row["exit_date"]) else None,
            "equity":   round(equity, 2),
            "trade_id": str(row.get("id", "")),
            "ticker":   str(row.get("ticker", "")),
        })

    total_return_pct = round((equity - STARTING_EQUITY) / STARTING_EQUITY * 100, 2)
    max_dd, _ = _max_drawdown([e["equity"] for e in equity_curve])

    return {
        "total_trades":     total_trades,
        "win_rate":         win_rate,
        "sharpe":           sharpe,
        "total_return_pct": total_return_pct,
        "avg_rr":           avg_rr,
        "max_drawdown_pct": max_dd,
        "equity_curve":     equity_curve,
    }


# ── signal log ────────────────────────────────────────────────────────────────

def _empty_signal_funnel() -> dict[str, Any]:
    return {
        "crossovers_30d": 0,
        "signals_fired_30d": 0,
        "signals_blocked_30d": 0,
        "filter_funnel": {k: 0 for k in FILTER_BLOCK_KEYS},
    }


def _empty_signal_log() -> dict[str, Any]:
    return {
        "available": False,
        "last_run": None,
        "last_run_status": None,
        "signals_fired": 0,
        "stop_losses_triggered": 0,
        "blocked_reasons": {},
        "total_blocked": 0,
        "today_evaluated": [],
        "today_signals": [],
        "filter_blocks_30d": {k: 0 for k in FILTER_BLOCK_KEYS},
        "signal_funnel": _empty_signal_funnel(),
    }


def _read_signal_log(log_path: Path) -> dict[str, Any]:
    """
    Parse signal_log.jsonl (job_started/job_complete/job_failed/signal_evaluation
    records written by bot/job.py) for blocked signals, today's evaluated
    tickers and signals, 30-day filter block counts, and last-run metadata.
    """
    if not log_path.exists():
        return _empty_signal_log()

    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=30)
    today  = now.date().isoformat()

    blocked_reasons: dict[str, int] = {}
    filter_blocks_30d = {k: 0 for k in FILTER_BLOCK_KEYS}
    last_run = None
    last_run_status = None
    signals_fired = 0
    stop_losses_triggered = 0
    today_tickers: list[str] = []
    today_signal_events: dict[str, dict] = {}
    crossovers_30d = 0
    funnel_signals_fired_30d = 0
    signals_blocked_30d = 0

    try:
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            event  = entry.get("event", "")
            ts_raw = entry.get("ts")
            ts     = pd.to_datetime(ts_raw, utc=True, errors="coerce") if ts_raw else None
            if ts_raw:
                last_run = ts_raw

            if event == "job_complete":
                last_run_status = "ok"
                if ts is not None and pd.notna(ts) and ts >= cutoff:
                    crossovers_30d += entry.get("crossovers_detected", 0)
                    funnel_signals_fired_30d += entry.get("signals_fired", 0)
                    signals_blocked_30d += entry.get("signals_blocked", 0)
            elif event == "job_failed":
                last_run_status = "error"
            elif event == "job_started":
                if entry.get("date") == today:
                    for t in entry.get("tickers", []):
                        if t not in today_tickers:
                            today_tickers.append(t)
            elif event == "signal_evaluation":
                signal = entry.get("signal")
                ticker = entry.get("ticker")

                if signal in ("BUY", "SELL"):
                    signals_fired += 1
                    if signal == "SELL" and entry.get("signal_reason") == "stop_loss":
                        stop_losses_triggered += 1

                for key in entry.get("blocked_by") or []:
                    blocked_reasons[key] = blocked_reasons.get(key, 0) + 1
                    if ts is not None and pd.notna(ts) and ts >= cutoff and key in filter_blocks_30d:
                        filter_blocks_30d[key] += 1

                if entry.get("date") == today and ticker:
                    if ticker not in today_tickers:
                        today_tickers.append(ticker)
                    rec = today_signal_events.setdefault(ticker, {})
                    if signal in ("BUY", "SELL"):
                        rec["action"] = signal
                        rec["blocked"] = False
                    elif entry.get("blocked_by"):
                        rec.setdefault("action", "BUY")
                        rec["blocked"] = True
                        rec["block_reason"] = entry["blocked_by"][0]

    except Exception:
        return _empty_signal_log()

    today_signals = [
        {"ticker": t, **info}
        for t, info in today_signal_events.items()
        if info.get("action")
    ]

    return {
        "available": True,
        "last_run": last_run,
        "last_run_status": last_run_status,
        "signals_fired": signals_fired,
        "stop_losses_triggered": stop_losses_triggered,
        "blocked_reasons": blocked_reasons,
        "total_blocked": sum(blocked_reasons.values()),
        "today_evaluated": today_tickers,
        "today_signals": today_signals,
        "filter_blocks_30d": filter_blocks_30d,
        "signal_funnel": {
            "crossovers_30d": crossovers_30d,
            "signals_fired_30d": funnel_signals_fired_30d,
            "signals_blocked_30d": signals_blocked_30d,
            "filter_funnel": filter_blocks_30d,
        },
    }


# ── Dual Momentum (GEM) ─────────────────────────────────────────────────────

def _empty_gem() -> dict[str, Any]:
    return {
        "available":        False,
        "current_holding":  None,
        "allocation_usd":   GEM_DEFAULT_ALLOCATION,
        "last_rebalance":   None,
        "readings":         {},
        "history":          [],
        "equity_curve":     _flat_equity_curve(),
        "spy_curve":        _flat_equity_curve(),
        "cagr":             0.0,
        "spy_cagr":         0.0,
        "total_return_pct": 0.0,
    }


def _read_gem_records(log_path: Path) -> list[dict]:
    """Parsed `gem_rebalance` records (with a holding), sorted by date."""
    if not log_path.exists():
        return []
    records: list[dict] = []
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("event") == "gem_rebalance" and entry.get("holding"):
                records.append(entry)
    except Exception:
        return []
    records.sort(key=lambda r: r.get("date", ""))
    return records


def _monthly_closes_map(ticker: str, start: str) -> dict[str, float]:
    """{ 'YYYY-MM': month-end adjusted close } from yfinance. {} on any error."""
    try:
        import yfinance as yf

        df = yf.Ticker(ticker).history(start=start, interval="1mo", auto_adjust=True)
        if df is None or df.empty:
            return {}
        closes = df["Close"].dropna()
        return {ts.strftime("%Y-%m"): float(v) for ts, v in closes.items()}
    except Exception:
        return {}


def _cagr_from_curve(curve: list[dict]) -> float:
    if len(curve) < 2:
        return 0.0
    try:
        start_eq = curve[0]["equity"]
        end_eq   = curve[-1]["equity"]
        d0 = datetime.fromisoformat(curve[0]["date"]).date()
        d1 = datetime.fromisoformat(curve[-1]["date"]).date()
        years = (d1 - d0).days / 365.25
        if years <= 0 or start_eq <= 0 or end_eq <= 0:
            return 0.0
        return round(((end_eq / start_eq) ** (1.0 / years) - 1.0) * 100, 2)
    except Exception:
        return 0.0


def _gem_equity_curves(records: list[dict], allocation: float) -> tuple[list[dict], list[dict]]:
    """Reconstruct GEM and SPY-benchmark monthly equity curves from the rebalance
    log, replaying each held asset's monthly total return via yfinance.

    Returns (gem_curve, spy_curve); falls back to flat curves on data error.
    """
    if not records:
        return _flat_equity_curve(), _flat_equity_curve()

    first_month = records[0]["date"][:7]
    # Fetch a small buffer before inception so the first month has a prior close.
    start_dt = (datetime.fromisoformat(records[0]["date"]).date()
                - timedelta(days=40)).isoformat()

    held = {r["holding"] for r in records}
    needed = held | {"SPY"}
    maps = {t: _monthly_closes_map(t, start_dt) for t in needed}
    spy_map = maps.get("SPY", {})
    if not spy_map or any(not maps[t] for t in held):
        return _flat_equity_curve(), _flat_equity_curve()

    # Step function: holding effective from each rebalance month onward.
    schedule = [(r["date"][:7], r["holding"]) for r in records]

    def holding_for(month: str) -> str:
        current = schedule[0][1]
        for m, h in schedule:
            if m <= month:
                current = h
            else:
                break
        return current

    months = sorted(m for m in spy_map if m >= first_month)
    if len(months) < 2:
        return _flat_equity_curve(), _flat_equity_curve()

    equity = spy_equity = float(allocation)
    gem_curve = [{"date": f"{months[0]}-01", "equity": round(equity, 2), "holding": holding_for(months[0])}]
    spy_curve = [{"date": f"{months[0]}-01", "equity": round(spy_equity, 2)}]

    for prev, cur in zip(months[:-1], months[1:]):
        h = holding_for(prev)
        hmap = maps.get(h, {})
        if prev in hmap and cur in hmap and hmap[prev]:
            equity *= hmap[cur] / hmap[prev]
        if prev in spy_map and cur in spy_map and spy_map[prev]:
            spy_equity *= spy_map[cur] / spy_map[prev]
        gem_curve.append({"date": f"{cur}-01", "equity": round(equity, 2), "holding": holding_for(cur)})
        spy_curve.append({"date": f"{cur}-01", "equity": round(spy_equity, 2)})

    return gem_curve, spy_curve


def _compute_gem(log_path: Path = GEM_LOG_PATH) -> dict[str, Any]:
    """Build the `gem` analytics block: current allocation, momentum readings,
    rebalance history, and the GEM-vs-SPY equity curve since inception."""
    records = _read_gem_records(log_path)
    if not records:
        return _empty_gem()

    last = records[-1]
    allocation = float(last.get("allocation_usd") or GEM_DEFAULT_ALLOCATION)

    readings = {
        "us_ticker":    last.get("us_ticker", "SPY"),
        "intl_ticker":  last.get("intl_ticker", "VEU"),
        "bonds_ticker": last.get("bonds_ticker", "AGG"),
        "us_12m":       last.get("us_12m"),
        "intl_12m":     last.get("intl_12m"),
        "tbill_12m":    last.get("tbill_12m"),
    }
    history = [
        {
            "date":      r.get("date"),
            "holding":   r.get("holding"),
            "changed":   r.get("changed"),
            "us_12m":    r.get("us_12m"),
            "intl_12m":  r.get("intl_12m"),
            "tbill_12m": r.get("tbill_12m"),
        }
        for r in records
    ]

    gem_curve, spy_curve = _gem_equity_curves(records, allocation)
    gem_cagr = _cagr_from_curve(gem_curve)
    spy_cagr = _cagr_from_curve(spy_curve)
    total_return = round((gem_curve[-1]["equity"] - allocation) / allocation * 100, 2) if gem_curve else 0.0

    return {
        "available":        True,
        "current_holding":  last.get("holding"),
        "allocation_usd":   allocation,
        "last_rebalance":   last.get("date"),
        "readings":         readings,
        "history":          history,
        "equity_curve":     gem_curve,
        "spy_curve":        spy_curve,
        "cagr":             gem_cagr,
        "spy_cagr":         spy_cagr,
        "total_return_pct": total_return,
    }


# ── main analytics ────────────────────────────────────────────────────────────

def compute(trades_path: Path = TRADES_PATH,
            log_path: Path = LOG_PATH) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    signal_log   = _read_signal_log(log_path)

    # Last run metadata from env (set by GitHub Actions) or signal log
    last_run_info = {
        "timestamp":   signal_log.get("last_run") or os.environ.get("GITHUB_RUN_STARTED_AT"),
        "status":      signal_log.get("last_run_status"),
        "run_id":      os.environ.get("GITHUB_RUN_ID"),
        "run_number":  os.environ.get("GITHUB_RUN_NUMBER"),
        "actor":       os.environ.get("GITHUB_ACTOR", "cron"),
        "signals_fired":          signal_log.get("signals_fired", 0),
        "blocked_total":          signal_log.get("total_blocked", 0),
        "blocked_trend_filter":   signal_log.get("blocked_reasons", {}).get("trend_sma", 0),
        "blocked_rsi_overbought": signal_log.get("blocked_reasons", {}).get("rsi_overbought", 0),
    }

    # Fields populated regardless of trade history.
    base = {
        "generated_at":     generated_at,
        "last_run":         last_run_info,
        "last_run_status":  signal_log.get("last_run_status"),
        "signal_log":       signal_log,
        "open_positions":   _compute_open_positions(trades_path),
        "today_evaluated":  signal_log.get("today_evaluated", []),
        "today_signals":    signal_log.get("today_signals", []),
        "filter_blocks_30d": signal_log.get("filter_blocks_30d", {k: 0 for k in FILTER_BLOCK_KEYS}),
        "signal_funnel":    signal_log.get("signal_funnel", _empty_signal_funnel()),
        "shadow":           _compute_shadow_stats(),
        "gem":              _compute_gem(),
    }

    if not trades_path.exists():
        return {
            **base,
            "zero_trades": True,
            "reason": "trades_live.csv not found",
            "equity_curve": _flat_equity_curve(),
        }

    df = pd.read_csv(trades_path)
    df.columns = df.columns.str.strip()
    closed = df[df["side"].str.upper() == "SELL"].copy()

    if closed.empty:
        return {
            **base,
            "zero_trades": True,
            "reason": "no closed trades yet",
            "equity_curve": _flat_equity_curve(),
        }

    closed["pnl_usd"]    = pd.to_numeric(closed["pnl_usd"], errors="coerce").fillna(0.0)
    closed["timestamp"]  = pd.to_datetime(closed["timestamp"], utc=True, errors="coerce")
    closed = closed.sort_values("timestamp").reset_index(drop=True)

    # ── Equity curve ─────────────────────────────────────────────────────────
    # Start at $100,000 (paper account starting equity), one bar before the
    # first trade, then accumulate cumulative PnL after each closed trade.
    first_ts = closed["timestamp"].iloc[0]
    start_date = (first_ts - pd.Timedelta(days=1)).date().isoformat() if pd.notna(first_ts) else None
    equity = STARTING_EQUITY
    equity_curve: list[dict] = [{"date": start_date, "equity": round(equity, 2)}]
    for _, row in closed.iterrows():
        equity += row["pnl_usd"]
        equity_curve.append({
            "date":      row["timestamp"].date().isoformat() if pd.notna(row["timestamp"]) else None,
            "equity":    round(equity, 2),
            "trade_id":  str(row.get("id", "")),
            "ticker":    str(row.get("ticker", "")),
        })

    # ── Overall stats ─────────────────────────────────────────────────────────
    total_trades    = len(closed)
    wins            = closed[closed["pnl_usd"] > 0]["pnl_usd"]
    losses          = closed[closed["pnl_usd"] <= 0]["pnl_usd"]
    win_rate        = round(len(wins) / total_trades * 100, 1) if total_trades else 0.0
    total_pnl       = round(closed["pnl_usd"].sum(), 2)
    best_trade      = round(closed["pnl_usd"].max(), 2)
    worst_trade     = round(closed["pnl_usd"].min(), 2)
    avg_win         = round(wins.mean(), 2) if not wins.empty else 0.0
    avg_loss        = round(losses.mean(), 2) if not losses.empty else 0.0
    rr              = _rr_ratio(wins, losses)
    sharpe          = round(_sharpe(closed["pnl_usd"]), 3)
    max_dd, max_dd_idx = _max_drawdown([e["equity"] for e in equity_curve])
    outcomes        = (closed["pnl_usd"] > 0).tolist()
    max_consec      = _max_consecutive_losses(outcomes)

    stop_loss_exits = 0
    if "reason" in closed.columns:
        stop_loss_exits = int((closed["reason"].str.lower() == "stop_loss").sum())

    # ── Per-ticker breakdown ──────────────────────────────────────────────────
    per_ticker: dict[str, Any] = {}
    for ticker, group in closed.groupby("ticker"):
        t_wins   = group[group["pnl_usd"] > 0]["pnl_usd"]
        t_losses = group[group["pnl_usd"] <= 0]["pnl_usd"]
        t_total  = len(group)
        per_ticker[str(ticker)] = {
            "trades":      t_total,
            "win_rate":    round(len(t_wins) / t_total * 100, 1) if t_total else 0.0,
            "total_pnl":   round(group["pnl_usd"].sum(), 2),
            "avg_win":     round(t_wins.mean(), 2) if not t_wins.empty else 0.0,
            "avg_loss":    round(t_losses.mean(), 2) if not t_losses.empty else 0.0,
            "rr":          _rr_ratio(t_wins, t_losses),
            "best_trade":  round(group["pnl_usd"].max(), 2),
            "worst_trade": round(group["pnl_usd"].min(), 2),
        }

    # ── Rolling 7-day ─────────────────────────────────────────────────────────
    now      = pd.Timestamp.now(tz="UTC")
    week_ago = now - pd.Timedelta(days=7)
    recent   = closed[closed["timestamp"] >= week_ago]
    pnl_7d   = round(recent["pnl_usd"].sum(), 2)
    trades_7d = len(recent)

    return {
        **base,
        "zero_trades":  False,
        "overall": {
            "total_trades":           total_trades,
            "win_rate":               win_rate,
            "total_pnl":              total_pnl,
            "best_trade":             best_trade,
            "worst_trade":            worst_trade,
            "avg_win":                avg_win,
            "avg_loss":               avg_loss,
            "rr_ratio":               rr,
            "sharpe":                 sharpe,
            "max_drawdown_pct":       max_dd,
            "max_drawdown_idx":       max_dd_idx,
            "max_consecutive_losses": max_consec,
            "stop_loss_exits":        stop_loss_exits,
            "pnl_7d":                 pnl_7d,
            "trades_7d":              trades_7d,
            "current_equity":         round(STARTING_EQUITY + total_pnl, 2),
        },
        "per_ticker":   per_ticker,
        "equity_curve": equity_curve,
    }


def _print_summary(data: dict) -> None:
    if data.get("zero_trades"):
        print(f"No trades yet: {data.get('reason', '')}")
        lr = data.get("last_run", {})
        if lr.get("timestamp"):
            print(f"Last run: {lr['timestamp']}")
            print(f"Signals fired: {lr['signals_fired']}  Blocked: {lr['blocked_total']}")
        return

    o = data["overall"]
    print("\n── TRADING BOT ANALYTICS ──────────────────────────────")
    print(f"  Trades:           {o['total_trades']}  (last 7d: {o['trades_7d']})")
    print(f"  Win rate:         {o['win_rate']}%")
    print(f"  Total PnL:        ${o['total_pnl']:+,.2f}  (last 7d: ${o['pnl_7d']:+,.2f})")
    print(f"  Current equity:   ${o['current_equity']:,.2f}")
    print(f"  Avg win:          ${o['avg_win']:+,.2f}")
    print(f"  Avg loss:         ${o['avg_loss']:+,.2f}")
    print(f"  R:R ratio:        {o['rr_ratio']}x")
    print(f"  Sharpe:           {o['sharpe']}")
    print(f"  Max drawdown:     {o['max_drawdown_pct']}%")
    print(f"  Max consec. loss: {o['max_consecutive_losses']}")
    print(f"  Stop loss exits:  {o['stop_loss_exits']}")
    print(f"  Best trade:       ${o['best_trade']:+,.2f}")
    print(f"  Worst trade:      ${o['worst_trade']:+,.2f}")

    lr = data.get("last_run", {})
    if lr.get("timestamp"):
        print(f"\n── LAST RUN ────────────────────────────────────────────")
        print(f"  Timestamp:        {lr['timestamp']}")
        print(f"  Signals fired:    {lr['signals_fired']}")
        print(f"  Blocked (total):  {lr['blocked_total']}")
        print(f"    trend_filter:   {lr['blocked_trend_filter']}")
        print(f"    rsi_overbought: {lr['blocked_rsi_overbought']}")

    print("\n── PER TICKER ─────────────────────────────────────────")
    for ticker, t in data["per_ticker"].items():
        print(f"  {ticker:5s}  {t['trades']:2d} trades  "
              f"win={t['win_rate']}%  "
              f"PnL=${t['total_pnl']:+,.2f}  "
              f"R:R={t['rr']}x")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse trades_live.csv")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--out", type=str, default=None,
                        help="Write JSON to this directory (creates analytics.json)")
    args = parser.parse_args()

    data = compute()

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "analytics.json"
        out_path.write_text(json.dumps(data, indent=2))
        print(f"Written to {out_path}")

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        _print_summary(data)


if __name__ == "__main__":
    main()
