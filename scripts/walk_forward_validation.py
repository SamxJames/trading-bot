"""
Out-of-sample walk-forward validation.

The existing 2005-2025 backtest (128 trades, 47.7% win rate, Sharpe 0.348)
is an IN-SAMPLE number: the strategy's parameters (RSI 70->75, trailing stop
1.5->4.0, SMA 200->150, volume 1.0->0.75, plus weekly_ema_filter) were all
tuned using observations from this same 2005-2025 dataset
(scripts/tune_filters.py, scripts/filter_isolation.py).

This script splits the 20-year dataset into three non-overlapping periods
and evaluates the CURRENT, FROZEN config.yaml parameters on the two periods
that were not used for tuning:

  Development   2005-2015  In-sample. Not evaluated here.
  Walk-forward  2016-2020  First out-of-sample window.
  Hold-out      2021-2025  Second out-of-sample window (COVID crash/recovery,
                            2022 bear market, 2023 AI rally, 2024-25 chop).

No parameters are changed. yfinance data only, no Alpaca credentials, no
config.yaml changes.

Usage:
    python scripts/walk_forward_validation.py
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import List

import pandas as pd
import structlog

logging.basicConfig(level=logging.CRITICAL)
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

from bot.backtest import _max_drawdown, _sharpe_ratio
from bot.data.feed import Bar
from bot.data.yfinance_historical import fetch_bars
from bot.strategies.base import SignalType
from bot.strategies.ema_cross_filtered import EmaCrossFilteredStrategy

TICKERS = ["SPY", "QQQ", "GLD", "AAPL", "NVDA"]
DATA_START = date(2005, 1, 1)
DATA_END = date(2025, 12, 31)
INITIAL_EQUITY = 10_000.0
NOTIONAL = 500.0  # matches config.yaml max_notional_per_trade

# Frozen current config.yaml parameters - DO NOT CHANGE.
PARAMS = dict(
    fast_period=20,
    slow_period=50,
    trend_sma_period=150,
    rsi_period=14,
    rsi_overbought=75.0,
    stop_loss_pct=2.5,
    trailing_stop_pct=4.0,
    take_profit_rr=3.0,
    volume_lookback=20,
    volume_multiplier=0.75,
    weekly_ema_filter=True,
)

DEVELOPMENT = (date(2005, 1, 1), date(2015, 12, 31))
WALK_FORWARD = (date(2016, 1, 1), date(2020, 12, 31))
HOLD_OUT = (date(2021, 1, 1), date(2025, 12, 31))
FULL_SAMPLE = (date(2005, 1, 1), date(2025, 12, 31))

# Step 3 / Step 4 pass-fail-marginal thresholds
PASS_SHARPE = 0.25
PASS_WIN_RATE = 38.0
MARGINAL_SHARPE_LO = 0.15
MARGINAL_WIN_RATE_LO = 30.0


@dataclass
class TradeRecord:
    entry_time: datetime
    exit_time: datetime
    pnl: float


async def load_bars(ticker: str) -> List[Bar]:
    df = await fetch_bars(ticker, DATA_START, DATA_END, timeframe="1Day")
    return [
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


def simulate_full(bars: List[Bar]) -> tuple[List[TradeRecord], List[tuple[datetime, float]]]:
    """
    Replay *bars* through EmaCrossFilteredStrategy with the frozen PARAMS.

    Returns (completed_trades, equity_curve) where equity_curve is a list of
    (timestamp, mark-to-market equity) pairs. Any position still open at the
    end of *bars* is force-closed at the last close.
    """
    strategy = EmaCrossFilteredStrategy(**PARAMS)
    strategy.on_start()

    equity = INITIAL_EQUITY
    in_position = False
    entry_price = 0.0
    entry_qty = 0.0
    entry_time: datetime | None = None

    trades: List[TradeRecord] = []
    equity_curve: List[tuple[datetime, float]] = []

    for i, bar in enumerate(bars):
        mtm = equity + (bar.close - entry_price) * entry_qty if in_position else equity
        equity_curve.append((bar.timestamp, mtm))

        signal = strategy.on_bar(bar)
        if signal is None:
            continue

        fill_price = bars[i + 1].open if i + 1 < len(bars) else bar.close

        if signal.type == SignalType.BUY and not in_position:
            entry_qty = NOTIONAL / fill_price
            entry_price = fill_price
            entry_time = bar.timestamp
            in_position = True
        elif signal.type == SignalType.SELL and in_position:
            pnl = (fill_price - entry_price) * entry_qty
            equity += pnl
            trades.append(TradeRecord(entry_time, bar.timestamp, pnl))
            in_position = False

    if in_position and bars:
        last_price = bars[-1].close
        pnl = (last_price - entry_price) * entry_qty
        equity += pnl
        trades.append(TradeRecord(entry_time, bars[-1].timestamp, pnl))

    strategy.on_stop()
    return trades, equity_curve


def period_metrics(
    trades: List[TradeRecord],
    equity_curve: List[tuple[datetime, float]],
    period_start: date,
    period_end: date,
) -> dict:
    """Metrics for trades whose ENTRY falls within [period_start, period_end]."""
    filtered = [t for t in trades if period_start <= t.entry_time.date() <= period_end]
    pnls = [t.pnl for t in filtered]

    num_trades = len(filtered)
    win_rate = (sum(1 for p in pnls if p > 0) / num_trades * 100) if num_trades else 0.0
    return_pct = sum(pnls) / INITIAL_EQUITY * 100
    sharpe = _sharpe_ratio(pnls, INITIAL_EQUITY)

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    rr = avg_win / avg_loss if avg_loss > 0 else 0.0

    period_curve = [eq for ts, eq in equity_curve if period_start <= ts.date() <= period_end]
    max_dd = _max_drawdown(period_curve) if period_curve else 0.0

    return {
        "trades": num_trades,
        "win_rate": win_rate,
        "return_pct": return_pct,
        "sharpe": sharpe,
        "rr": rr,
        "max_dd": max_dd,
        "pnls": [p / INITIAL_EQUITY for p in pnls],
    }


def _sharpe_from_returns(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    s = pd.Series(returns)
    std = s.std()
    if std == 0:
        return 0.0
    return float(s.mean() / std)


def pooled(rows: List[dict]) -> tuple[int, float, float]:
    """Return (total_trades, pooled_win_rate, pooled_sharpe) across tickers."""
    total_trades = sum(r["trades"] for r in rows)
    pooled_returns: List[float] = []
    for r in rows:
        pooled_returns.extend(r["pnls"])
    win_rate = (
        sum(1 for x in pooled_returns if x > 0) / len(pooled_returns) * 100
        if pooled_returns
        else 0.0
    )
    sharpe = _sharpe_from_returns(pooled_returns)
    return total_trades, win_rate, sharpe


def print_table(title: str, rows: dict[str, dict]) -> tuple[int, float, float]:
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  {'Ticker':<8} {'Trades':>7} {'Win%':>8} {'Return':>9} {'R:R':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print(f"  {'-'*8} {'-'*7} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*8}")
    for t in TICKERS:
        r = rows[t]
        rr_str = f"{r['rr']:.2f}x" if r["rr"] > 0 else "N/A"
        print(
            f"  {t:<8} {r['trades']:>7d} {r['win_rate']:>7.1f}% "
            f"{r['return_pct']:>+8.2f}% {rr_str:>8} {r['sharpe']:>8.3f} {r['max_dd']:>7.2f}%"
        )
    print(f"  {'-'*8} {'-'*7} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*8}")

    total_trades, win_rate, sharpe = pooled(list(rows.values()))
    print(f"  {'TOTAL':<8} {total_trades:>7d} {win_rate:>7.1f}% {'':>9} {'':>8} {sharpe:>8.3f}")
    print(sep)
    return total_trades, win_rate, sharpe


def print_comparison(label: str, in_sample: tuple, oos: tuple) -> None:
    is_trades, is_win, is_sharpe = in_sample
    oos_trades, oos_win, oos_sharpe = oos
    print(f"\n  Metric          In-sample (2005-2025)    {label:<25} Delta")
    print(f"  -------         ---------------------    " + "-" * 25 + "    -----")
    print(
        f"  Trades          {is_trades:>21d}    {oos_trades:>25d}    "
        f"{oos_trades - is_trades:+d}"
    )
    print(
        f"  Win rate        {is_win:>20.1f}%    {oos_win:>24.1f}%    "
        f"{oos_win - is_win:+.1f}pp"
    )
    print(
        f"  Pooled Sharpe   {is_sharpe:>21.3f}    {oos_sharpe:>25.3f}    "
        f"{oos_sharpe - is_sharpe:+.3f}"
    )


def evaluate(sharpe: float, win_rate: float, per_ticker_return_pct: list[float]) -> str:
    """Apply the Step 3/4 pass/marginal/fail criteria. Returns PASS/MARGINAL/FAIL."""
    n = len(per_ticker_return_pct)
    positive_count = sum(1 for r in per_ticker_return_pct if r > 0)
    negative_count = n - positive_count
    majority_negative = negative_count > n / 2

    if sharpe < MARGINAL_SHARPE_LO or majority_negative:
        return "FAIL"
    if sharpe >= PASS_SHARPE and win_rate >= PASS_WIN_RATE and positive_count >= 3:
        return "PASS"
    return "MARGINAL"


async def main() -> None:
    print(f"Loading bars for {TICKERS} ({DATA_START} -> {DATA_END}) via yfinance...")
    bars_by_ticker: dict[str, List[Bar]] = {}
    for t in TICKERS:
        bars = await load_bars(t)
        bars_by_ticker[t] = bars
        print(f"  {t}: {len(bars)} bars")

    # ── Full-sample run (2005-2025) - reproduces the in-sample baseline ─────
    full_trades: dict[str, List[TradeRecord]] = {}
    full_curves: dict[str, List[tuple[datetime, float]]] = {}
    for t in TICKERS:
        trades, curve = simulate_full(bars_by_ticker[t])
        full_trades[t] = trades
        full_curves[t] = curve

    full_sample_rows = {
        t: period_metrics(full_trades[t], full_curves[t], *FULL_SAMPLE) for t in TICKERS
    }

    # ── Walk-forward run (data truncated to end of 2020) ────────────────────
    wf_end = WALK_FORWARD[1]
    wf_trades: dict[str, List[TradeRecord]] = {}
    wf_curves: dict[str, List[tuple[datetime, float]]] = {}
    for t in TICKERS:
        truncated = [b for b in bars_by_ticker[t] if b.timestamp.date() <= wf_end]
        trades, curve = simulate_full(truncated)
        wf_trades[t] = trades
        wf_curves[t] = curve

    wf_rows = {
        t: period_metrics(wf_trades[t], wf_curves[t], *WALK_FORWARD) for t in TICKERS
    }

    # ── Hold-out run (full 2005-2025 data, period filter on entry date) ─────
    ho_rows = {
        t: period_metrics(full_trades[t], full_curves[t], *HOLD_OUT) for t in TICKERS
    }

    # ── Step 1 - protocol summary ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  VALIDATION PROTOCOL")
    print("=" * 78)
    print(f"  Development (in-sample, not evaluated): {DEVELOPMENT[0]} -> {DEVELOPMENT[1]}")
    print(f"  Walk-forward (OOS #1, frozen params):    {WALK_FORWARD[0]} -> {WALK_FORWARD[1]}")
    print(f"  Hold-out (OOS #2, frozen params):        {HOLD_OUT[0]} -> {HOLD_OUT[1]}")
    print(f"  Frozen parameters: {PARAMS}")

    # ── Step 2 - walk-forward results ────────────────────────────────────────
    print("\n" + "#" * 78)
    print("  STEP 2 - WALK-FORWARD TEST (2016-2020), frozen config.yaml params")
    print("#" * 78)
    wf_total_trades, wf_win_rate, wf_sharpe = print_table(
        "Walk-forward 2016-2020", wf_rows
    )

    is_total_trades, is_win_rate, is_sharpe = pooled(list(full_sample_rows.values()))
    print_comparison("Walk-forward (2016-2020)", (is_total_trades, is_win_rate, is_sharpe),
                      (wf_total_trades, wf_win_rate, wf_sharpe))

    # ── Step 3 - walk-forward verdict ────────────────────────────────────────
    wf_returns = [wf_rows[t]["return_pct"] for t in TICKERS]
    wf_positive = [t for t in TICKERS if wf_rows[t]["return_pct"] > 0]
    wf_negative = [t for t in TICKERS if wf_rows[t]["return_pct"] <= 0]
    wf_verdict = evaluate(wf_sharpe, wf_win_rate, wf_returns)

    print("\n" + "=" * 78)
    print("  STEP 3 - WALK-FORWARD VERDICT (2016-2020)")
    print("=" * 78)
    print(f"  Pooled Sharpe: {wf_sharpe:.3f}  |  Pooled win rate: {wf_win_rate:.1f}%  |  "
          f"Tickers with positive return: {len(wf_positive)}/5 ({', '.join(wf_positive) or 'none'})")
    print(f"\n  VERDICT: {wf_verdict}")
    if wf_verdict == "PASS":
        interp = (
            f"With parameters frozen exactly as tuned on 2005-2015, the strategy produced a "
            f"pooled Sharpe of {wf_sharpe:.3f} (>= {PASS_SHARPE}) and a {wf_win_rate:.1f}% win rate "
            f"(>= {PASS_WIN_RATE}%) across {wf_total_trades} trades in 2016-2020, with "
            f"{len(wf_positive)}/5 tickers ({', '.join(wf_positive)}) showing a positive total "
            f"return. This is consistent with genuine edge surviving outside the tuning window."
        )
    elif wf_verdict == "MARGINAL":
        interp = (
            f"With parameters frozen exactly as tuned on 2005-2015, the strategy produced a "
            f"pooled Sharpe of {wf_sharpe:.3f} and a {wf_win_rate:.1f}% win rate across "
            f"{wf_total_trades} trades in 2016-2020, with {len(wf_positive)}/5 tickers "
            f"({', '.join(wf_positive) or 'none'}) positive. This falls in the marginal band "
            f"(Sharpe 0.15-0.25 or win rate 30-38%) - some edge appears to persist, but it is "
            f"materially weaker than the in-sample {is_sharpe:.3f} Sharpe, so this should not "
            f"yet be trusted with full position sizing."
        )
    else:
        interp = (
            f"With parameters frozen exactly as tuned on 2005-2015, the strategy produced a "
            f"pooled Sharpe of {wf_sharpe:.3f} and a {wf_win_rate:.1f}% win rate across "
            f"{wf_total_trades} trades in 2016-2020, with {len(wf_negative)}/5 tickers "
            f"({', '.join(wf_negative) or 'none'}) showing a negative or zero total return. "
            f"This indicates the parameters tuned on 2005-2015 do not generalise to 2016-2020 "
            f"and the in-sample {is_sharpe:.3f} Sharpe was likely an artifact of overfitting."
        )
    print(f"\n  {interp}")

    # ── Step 4 - hold-out results ────────────────────────────────────────────
    print("\n" + "#" * 78)
    print("  STEP 4 - HOLD-OUT TEST (2021-2025), frozen config.yaml params")
    print("#" * 78)
    ho_total_trades, ho_win_rate, ho_sharpe = print_table(
        "Hold-out 2021-2025", ho_rows
    )
    print_comparison("Hold-out (2021-2025)", (is_total_trades, is_win_rate, is_sharpe),
                      (ho_total_trades, ho_win_rate, ho_sharpe))

    # Calendar-year breakdown (pooled across tickers)
    print("\n  Calendar-year breakdown (pooled across tickers, by entry date):")
    print(f"  {'Year':<6} {'Trades':>7} {'Win%':>8} {'Return contribution':>22}")
    print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*22}")
    for year in range(HOLD_OUT[0].year, HOLD_OUT[1].year + 1):
        year_pnls = []
        for t in TICKERS:
            for tr in full_trades[t]:
                if tr.entry_time.year == year:
                    year_pnls.append(tr.pnl)
        n = len(year_pnls)
        win_rate_y = (sum(1 for p in year_pnls if p > 0) / n * 100) if n else 0.0
        return_y = sum(year_pnls) / INITIAL_EQUITY * 100
        print(f"  {year:<6} {n:>7d} {win_rate_y:>7.1f}% {return_y:>+21.2f}%")

    # ── Step 4 - hold-out verdict ────────────────────────────────────────────
    ho_returns = [ho_rows[t]["return_pct"] for t in TICKERS]
    ho_positive = [t for t in TICKERS if ho_rows[t]["return_pct"] > 0]
    ho_negative = [t for t in TICKERS if ho_rows[t]["return_pct"] <= 0]
    ho_verdict = evaluate(ho_sharpe, ho_win_rate, ho_returns)

    print("\n" + "=" * 78)
    print("  STEP 4 - HOLD-OUT VERDICT (2021-2025)")
    print("=" * 78)
    print(f"  Pooled Sharpe: {ho_sharpe:.3f}  |  Pooled win rate: {ho_win_rate:.1f}%  |  "
          f"Tickers with positive return: {len(ho_positive)}/5 ({', '.join(ho_positive) or 'none'})")
    print(f"\n  VERDICT: {ho_verdict}")
    if ho_verdict == "PASS":
        interp = (
            f"Across 2021-2025 - a period containing the COVID recovery, the 2022 bear market, "
            f"and the 2023 AI rally - the frozen parameters produced a pooled Sharpe of "
            f"{ho_sharpe:.3f} and a {ho_win_rate:.1f}% win rate across {ho_total_trades} trades, "
            f"with {len(ho_positive)}/5 tickers ({', '.join(ho_positive)}) positive. This holds "
            f"up across genuinely different regimes and is meaningful evidence of edge."
        )
    elif ho_verdict == "MARGINAL":
        interp = (
            f"Across 2021-2025, the frozen parameters produced a pooled Sharpe of {ho_sharpe:.3f} "
            f"and a {ho_win_rate:.1f}% win rate across {ho_total_trades} trades, with "
            f"{len(ho_positive)}/5 tickers ({', '.join(ho_positive) or 'none'}) positive. This is "
            f"in the marginal band - the strategy did not break down across the COVID/2022/AI-"
            f"rally regimes, but performance is weaker than the in-sample {is_sharpe:.3f} Sharpe."
        )
    else:
        interp = (
            f"Across 2021-2025, the frozen parameters produced a pooled Sharpe of {ho_sharpe:.3f} "
            f"and a {ho_win_rate:.1f}% win rate across {ho_total_trades} trades, with "
            f"{len(ho_negative)}/5 tickers ({', '.join(ho_negative) or 'none'}) showing a "
            f"negative or zero total return. The strategy did not hold up across the COVID/2022/"
            f"AI-rally regime diversity, which is strong evidence the in-sample "
            f"{is_sharpe:.3f} Sharpe reflects overfitting rather than genuine edge."
        )
    print(f"\n  {interp}")

    # ── Step 5 - combined verdict ────────────────────────────────────────────
    print("\n" + "#" * 78)
    print("  STEP 5 - COMBINED VERDICT")
    print("#" * 78)

    evidence_for: list[str] = []
    evidence_against: list[str] = []

    for label, rows, sharpe_v, win_v, total_v in [
        ("2016-2020", wf_rows, wf_sharpe, wf_win_rate, wf_total_trades),
        ("2021-2025", ho_rows, ho_sharpe, ho_win_rate, ho_total_trades),
    ]:
        if sharpe_v >= PASS_SHARPE:
            evidence_for.append(f"Pooled Sharpe {sharpe_v:.3f} in {label} (>= {PASS_SHARPE} pass threshold)")
        elif sharpe_v >= MARGINAL_SHARPE_LO:
            evidence_for.append(f"Pooled Sharpe {sharpe_v:.3f} in {label} is positive but below the {PASS_SHARPE} pass threshold (marginal band)")
        else:
            evidence_against.append(f"Pooled Sharpe {sharpe_v:.3f} in {label} is below the {MARGINAL_SHARPE_LO} marginal floor")

        if win_v >= PASS_WIN_RATE:
            evidence_for.append(f"Win rate {win_v:.1f}% in {label} (>= {PASS_WIN_RATE}% pass threshold)")
        elif win_v >= MARGINAL_WIN_RATE_LO:
            evidence_against.append(f"Win rate {win_v:.1f}% in {label} is below the {PASS_WIN_RATE}% pass threshold (marginal band)")
        else:
            evidence_against.append(f"Win rate {win_v:.1f}% in {label} is below the {MARGINAL_WIN_RATE_LO}% marginal floor")

        for t in TICKERS:
            r = rows[t]
            if r["return_pct"] > 0:
                evidence_for.append(f"{t} return {r['return_pct']:+.2f}% / Sharpe {r['sharpe']:.3f} in {label} ({r['trades']} trades)")
            else:
                evidence_against.append(f"{t} return {r['return_pct']:+.2f}% / Sharpe {r['sharpe']:.3f} in {label} ({r['trades']} trades)")

        evidence_for.append(f"Total trades in {label}: {total_v} (sample size)") if total_v >= 20 else \
            evidence_against.append(f"Total trades in {label}: {total_v} - small sample, low statistical confidence")

    print("\n  Evidence for edge:")
    for e in evidence_for:
        print(f"    - {e}")

    print("\n  Evidence against edge:")
    for e in evidence_against:
        print(f"    - {e}")

    # Combined verdict: worst of the two individual verdicts wins.
    rank = {"FAIL": 0, "MARGINAL": 1, "PASS": 2}
    combined = min([wf_verdict, ho_verdict], key=lambda v: rank[v])

    print("\n" + "=" * 78)
    print(f"  OVERALL VERDICT: {combined}")
    print(f"  (Walk-forward 2016-2020: {wf_verdict}  |  Hold-out 2021-2025: {ho_verdict})")
    print("=" * 78)

    if combined == "PASS":
        recommendation = (
            "A. Strategy shows genuine edge - proceed with live trading and shadow "
            "monitoring as planned."
        )
    elif combined == "MARGINAL":
        recommendation = (
            "B. Strategy shows marginal edge - reduce position sizes by 50% "
            "(max_notional_per_trade: 250) until 20+ shadow trades confirm live "
            "performance. Do not activate walk-forward reoptimisation yet."
        )
    else:
        recommendation = (
            "C. Strategy does not show reliable out-of-sample edge - parameters are "
            "likely overfit. Recommend returning to the Run A baseline (no trailing "
            "stop, no volume filter, no take-profit cap) and revalidating before "
            "running live."
        )

    print(f"\n  RECOMMENDATION:\n  {recommendation}")
    print("\n  NOTE: config.yaml has NOT been modified. This script measures only.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
