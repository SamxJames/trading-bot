"""
Generate synthetic trade_journal/ data for end-to-end dashboard testing.

Writes:
  bot/trade_journal/trades_live.csv   — ~20 closed round-trip trades + 1 open
                                         position, spread across the last
                                         6 months.
  bot/trade_journal/signal_log.jsonl  — 30 days of job_started /
                                         signal_evaluation / job_complete
                                         records, ending today.

Intended for verifying scripts/analyse_trades.py and the dashboard render a
realistic-looking docs/analytics.json. The generated files are gitignored
test fixtures — delete them after verification (see CLAUDE.md: never commit
trades_live.csv or signal_log.jsonl).

Baseline assumptions (from the 2005-2025 backtest, scripts/run_updated_backtest.py):
  - ~47.7% win rate
  - ~6.4 trades/year across the 5-ticker universe
  - avg R:R ~2.63x
  - stop_loss_pct = 2.5%, take_profit_rr = 3.0 (config.yaml)

Usage:
    python scripts/generate_synthetic_trades.py
"""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

TRADES_PATH = Path("bot/trade_journal/trades_live.csv")
LOG_PATH = Path("bot/trade_journal/signal_log.jsonl")

STRATEGY = "ema_cross_filtered"
MAX_NOTIONAL = 500.0
STOP_LOSS_PCT = 2.5
TAKE_PROFIT_RR = 3.0

# Approximate recent prices — just need to be "realistic" enough for qty/PnL math.
ENTRY_PRICES = {"GLD": 210.0, "AAPL": 190.0, "QQQ": 480.0, "NVDA": 140.0, "SPY": 550.0}

# Trade counts proportional to the backtest's relative activity per ticker
# (GLD most active, SPY fewest) — sums to 20.
TRADE_COUNTS = {"GLD": 6, "AAPL": 5, "QQQ": 4, "NVDA": 3, "SPY": 2}

FILTER_KEYS = [
    "trend_sma", "rsi_overbought", "volume", "vix",
    "spy_macro", "earnings", "correlation", "weekly_ema",
]
BLOCK_FILTERS = ["trend_sma", "rsi_overbought", "volume", "weekly_ema"]

RNG_SEED = 20240614


def _generate_trades(today: datetime) -> list[dict]:
    rng = random.Random(RNG_SEED)

    tickers: list[str] = []
    for ticker, count in TRADE_COUNTS.items():
        tickers.extend([ticker] * count)
    rng.shuffle(tickers)

    total = len(tickers)
    n_wins = round(total * 0.477)  # ~47.7% win rate
    wins = [True] * n_wins + [False] * (total - n_wins)
    rng.shuffle(wins)

    rows: list[dict] = []
    for i, (ticker, is_win) in enumerate(zip(tickers, wins)):
        base_price = ENTRY_PRICES[ticker]
        entry_price = round(base_price * (1 + rng.uniform(-0.05, 0.05)), 2)
        qty = max(1, int(MAX_NOTIONAL / entry_price))

        # Guarantee at least one trade closed within the last 7 days so
        # pnl_7d is non-zero.
        if i == total - 1:
            exit_offset = 2
        else:
            exit_offset = rng.randint(8, 170)
        hold_days = rng.randint(3, 15)
        entry_offset = exit_offset + hold_days

        entry_date = today - timedelta(days=entry_offset)
        exit_date = today - timedelta(days=exit_offset)

        if is_win:
            if rng.random() < 0.7:
                reason = "take_profit"
                pct = STOP_LOSS_PCT / 100.0 * TAKE_PROFIT_RR  # 7.5%
            else:
                reason = "ema_crossover"
                pct = rng.uniform(0.01, 0.04)
        else:
            if rng.random() < 0.8:
                reason = "stop_loss"
                pct = -STOP_LOSS_PCT / 100.0  # -2.5%
            else:
                reason = "ema_crossover"
                pct = rng.uniform(-0.02, -0.005)

        exit_price = round(entry_price * (1 + pct), 2)
        pnl_usd = round((exit_price - entry_price) * qty, 2)

        trade_id = f"T{i + 1:03d}"
        rows.append({
            "id": trade_id, "timestamp": entry_date.isoformat(),
            "ticker": ticker, "side": "BUY", "qty": qty,
            "entry_price": entry_price, "exit_price": "", "pnl_usd": "",
            "reason": "", "strategy": STRATEGY,
        })
        rows.append({
            "id": trade_id, "timestamp": exit_date.isoformat(),
            "ticker": ticker, "side": "SELL", "qty": qty,
            "entry_price": entry_price, "exit_price": exit_price,
            "pnl_usd": pnl_usd, "reason": reason, "strategy": STRATEGY,
        })

    # One still-open position (unmatched BUY), entered a few days ago.
    open_ticker = "AAPL"
    open_entry = round(ENTRY_PRICES[open_ticker] * (1 + rng.uniform(-0.02, 0.02)), 2)
    open_qty = max(1, int(MAX_NOTIONAL / open_entry))
    rows.append({
        "id": f"T{total + 1:03d}", "timestamp": (today - timedelta(days=3)).isoformat(),
        "ticker": open_ticker, "side": "BUY", "qty": open_qty,
        "entry_price": open_entry, "exit_price": "", "pnl_usd": "",
        "reason": "", "strategy": STRATEGY,
    })

    rows.sort(key=lambda r: r["timestamp"])
    return rows


def _write_trades(rows: list[dict]) -> None:
    fieldnames = [
        "id", "timestamp", "ticker", "side", "qty",
        "entry_price", "exit_price", "pnl_usd", "reason", "strategy",
    ]
    TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRADES_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _base_filters() -> dict[str, dict]:
    return {
        "trend_sma":       {"evaluated": True,  "passed": True,  "value": 102.5, "threshold": 98.0},
        "rsi_overbought":  {"evaluated": False, "passed": True,  "value": None,  "threshold": 75.0},
        "volume":          {"evaluated": True,  "passed": True,  "value": 1.15,  "threshold": 0.75},
        "vix":             {"evaluated": True,  "passed": True,  "value": 15.4,  "threshold": 25.0},
        "spy_macro":       {"evaluated": False, "passed": True,  "value": None,  "threshold": None},
        "earnings":        {"evaluated": True,  "passed": True,  "value": False, "threshold": 1},
        "correlation":     {"evaluated": False, "passed": True,  "value": None,  "threshold": 0.8},
        "weekly_ema":      {"evaluated": True,  "passed": True,  "value": True,  "threshold": None},
    }


def _blocked_filters(block: str) -> dict[str, dict]:
    filters = _base_filters()
    if block == "trend_sma":
        filters["trend_sma"] = {"evaluated": True, "passed": False, "value": 96.2, "threshold": 98.0}
    elif block == "rsi_overbought":
        filters["rsi_overbought"] = {"evaluated": True, "passed": False, "value": 81.3, "threshold": 75.0}
    elif block == "volume":
        filters["volume"] = {"evaluated": True, "passed": False, "value": 0.52, "threshold": 0.75}
    elif block == "weekly_ema":
        filters["weekly_ema"] = {"evaluated": True, "passed": False, "value": False, "threshold": None}
    return filters


def _generate_signal_log(today: datetime) -> list[dict]:
    tickers = list(ENTRY_PRICES.keys())
    entries: list[dict] = []

    for day_idx in range(29, -1, -1):  # 29 days ago .. today
        day = today - timedelta(days=day_idx)
        date_iso = day.date().isoformat()
        ts_start = day.replace(hour=20, minute=45, second=0, microsecond=0).isoformat()

        entries.append({
            "event": "job_started", "ts": ts_start,
            "date": date_iso, "tickers": tickers,
        })

        for t_idx, ticker in enumerate(tickers):
            slot = (day_idx + t_idx) % 6
            ts = day.replace(hour=20, minute=45, second=5 + t_idx, microsecond=0).isoformat()

            if slot == 0:
                # BUY signal fires cleanly
                signal, reason, blocked_by, filters = "BUY", "golden_cross", [], _base_filters()
            elif slot == 1:
                # SELL signal fires cleanly
                signal, reason, blocked_by, filters = "SELL", "stop_loss", [], _base_filters()
            elif slot in (2, 3, 4, 5):
                # Blocked entry — cycle through the 4 strategy-level filters
                block = BLOCK_FILTERS[slot - 2]
                signal, reason = None, None
                blocked_by = [block]
                filters = _blocked_filters(block)
            else:
                signal, reason, blocked_by, filters = None, None, [], _base_filters()

            entries.append({
                "event": "signal_evaluation", "ts": ts, "date": date_iso,
                "ticker": ticker, "signal": signal, "signal_reason": reason,
                "filters": filters, "blocked_by": blocked_by,
            })

        ts_complete = day.replace(hour=20, minute=46, second=0, microsecond=0).isoformat()
        entries.append({
            "event": "job_complete", "ts": ts_complete,
            "duration_seconds": round(random.Random(day_idx).uniform(1.5, 4.5), 2),
        })

    return entries


def _write_signal_log(entries: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def main() -> None:
    today = datetime.now(timezone.utc)

    trades = _generate_trades(today)
    _write_trades(trades)
    closed = [r for r in trades if r["side"] == "SELL"]
    print(f"Wrote {len(trades)} rows ({len(closed)} closed trades, "
          f"{len(trades) - 2 * len(closed)} open) to {TRADES_PATH}")

    signal_log = _generate_signal_log(today)
    _write_signal_log(signal_log)
    print(f"Wrote {len(signal_log)} signal_log entries (30 days) to {LOG_PATH}")


if __name__ == "__main__":
    main()
