"""
Trade analytics module.

Reads trades_live.csv and computes richer statistics than the weekly
Discord summary: Sharpe ratio, max consecutive losses, average hold time,
per-ticker win rate and R:R, equity curve, and drawdown.

Usage:
    python scripts/analyse_trades.py                  # prints summary
    python scripts/analyse_trades.py --json           # outputs JSON (used by dashboard)
    python scripts/analyse_trades.py --out results/   # writes JSON to results/analytics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import pandas as pd

TRADES_PATH = Path("bot/trade_journal/trades_live.csv")
STARTING_EQUITY = 100_000.0   # paper account starting balance
RISK_FREE_RATE = 0.04          # annualised, for Sharpe calculation


# ── helpers ──────────────────────────────────────────────────────────────────

def _sharpe(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Annualised Sharpe ratio from a series of per-trade returns (as USD PnL)."""
    if len(returns) < 2:
        return 0.0
    daily_rf = rf / 252
    excess = returns - daily_rf
    std = excess.std()
    if std == 0:
        return 0.0
    # Annualise assuming ~252 trading days; approximate trades/year from sample
    trades_per_year = max(len(returns), 1)
    return float((excess.mean() / std) * math.sqrt(trades_per_year))


def _max_drawdown(equity_curve: list[float]) -> float:
    """Maximum percentage drawdown from an equity curve."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 2)


def _max_consecutive_losses(outcomes: list[bool]) -> int:
    """Count the longest streak of False (losing) trades."""
    max_streak = current = 0
    for win in outcomes:
        if not win:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _rr_ratio(wins: pd.Series, losses: pd.Series) -> float:
    """Average win / average absolute loss."""
    if losses.empty or wins.empty:
        return 0.0
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    if avg_loss == 0:
        return 0.0
    return round(avg_win / avg_loss, 2)


# ── main analytics ────────────────────────────────────────────────────────────

def compute(trades_path: Path = TRADES_PATH) -> dict[str, Any]:
    """
    Load trades_live.csv and return a dict of analytics.
    Returns a minimal structure with zero_trades=True if the file is
    missing or contains only the header row.
    """
    if not trades_path.exists():
        return {"zero_trades": True, "reason": "trades_live.csv not found"}

    df = pd.read_csv(trades_path)

    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Filter to SELL rows only — those are closed trades with a pnl_usd value
    closed = df[df["side"].str.upper() == "SELL"].copy()

    if closed.empty:
        return {"zero_trades": True, "reason": "no closed trades yet"}

    closed["pnl_usd"] = pd.to_numeric(closed["pnl_usd"], errors="coerce").fillna(0.0)
    closed["timestamp"] = pd.to_datetime(closed["timestamp"], utc=True, errors="coerce")
    closed = closed.sort_values("timestamp").reset_index(drop=True)

    # ── Equity curve ─────────────────────────────────────────────────────────
    equity = STARTING_EQUITY
    equity_curve: list[dict] = []
    for _, row in closed.iterrows():
        equity += row["pnl_usd"]
        equity_curve.append({
            "timestamp": row["timestamp"].isoformat() if pd.notna(row["timestamp"]) else None,
            "equity": round(equity, 2),
            "trade_id": str(row.get("id", "")),
            "ticker": str(row.get("ticker", "")),
        })

    # ── Overall stats ─────────────────────────────────────────────────────────
    total_trades = len(closed)
    wins = closed[closed["pnl_usd"] > 0]["pnl_usd"]
    losses = closed[closed["pnl_usd"] <= 0]["pnl_usd"]
    win_rate = round(len(wins) / total_trades * 100, 1) if total_trades else 0.0
    total_pnl = round(closed["pnl_usd"].sum(), 2)
    best_trade = round(closed["pnl_usd"].max(), 2)
    worst_trade = round(closed["pnl_usd"].min(), 2)
    avg_win = round(wins.mean(), 2) if not wins.empty else 0.0
    avg_loss = round(losses.mean(), 2) if not losses.empty else 0.0
    rr = _rr_ratio(wins, losses)
    sharpe = round(_sharpe(closed["pnl_usd"]), 3)
    max_dd = _max_drawdown([e["equity"] for e in equity_curve])
    outcomes = (closed["pnl_usd"] > 0).tolist()
    max_consec_losses = _max_consecutive_losses(outcomes)

    # ── Stop loss analysis ────────────────────────────────────────────────────
    stop_loss_exits = 0
    if "reason" in closed.columns:
        stop_loss_exits = int((closed["reason"].str.lower() == "stop_loss").sum())

    # ── Per-ticker breakdown ──────────────────────────────────────────────────
    per_ticker: dict[str, Any] = {}
    for ticker, group in closed.groupby("ticker"):
        t_wins = group[group["pnl_usd"] > 0]["pnl_usd"]
        t_losses = group[group["pnl_usd"] <= 0]["pnl_usd"]
        t_total = len(group)
        per_ticker[str(ticker)] = {
            "trades": t_total,
            "win_rate": round(len(t_wins) / t_total * 100, 1) if t_total else 0.0,
            "total_pnl": round(group["pnl_usd"].sum(), 2),
            "avg_win": round(t_wins.mean(), 2) if not t_wins.empty else 0.0,
            "avg_loss": round(t_losses.mean(), 2) if not t_losses.empty else 0.0,
            "rr": _rr_ratio(t_wins, t_losses),
            "best_trade": round(group["pnl_usd"].max(), 2),
            "worst_trade": round(group["pnl_usd"].min(), 2),
        }

    # ── Rolling 7-day PnL ─────────────────────────────────────────────────────
    now = pd.Timestamp.now(tz="UTC")
    week_ago = now - pd.Timedelta(days=7)
    recent = closed[closed["timestamp"] >= week_ago]
    pnl_7d = round(recent["pnl_usd"].sum(), 2)
    trades_7d = len(recent)

    return {
        "zero_trades": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "rr_ratio": rr,
            "sharpe": sharpe,
            "max_drawdown_pct": max_dd,
            "max_consecutive_losses": max_consec_losses,
            "stop_loss_exits": stop_loss_exits,
            "pnl_7d": pnl_7d,
            "trades_7d": trades_7d,
            "current_equity": round(STARTING_EQUITY + total_pnl, 2),
        },
        "per_ticker": per_ticker,
        "equity_curve": equity_curve,
    }


def _print_summary(data: dict) -> None:
    if data.get("zero_trades"):
        print(f"No trades yet: {data.get('reason', '')}")
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
