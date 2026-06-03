"""
Backtesting runner.

Replays historical OHLCV bars through a strategy in a simulated environment
with no live broker connection.  After the replay, prints a performance
summary and writes trades.csv to the working directory.

Simulation rules:
  - Long-only (BUY opens, SELL closes)
  - Fills at the next bar's open price (avoids look-ahead bias)
  - Flat notional sizing: qty = max_notional_per_trade / fill_price
  - Any position still open at the end of data is closed at the last close

Metrics reported:
  - Total return (%)
  - Number of completed round-trip trades
  - Win rate (%)
  - Max drawdown (% peak-to-trough on the mark-to-market equity curve)
  - Sharpe ratio (simplified: mean / std of per-trade returns, not annualised)

CLI usage (via bot.py):
    python bot.py backtest --strategy ema_cross --ticker AAPL \
        --from 2024-01-01 --to 2024-06-01
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")   # must be set before any pyplot import; Agg = file-only, no display

import pandas as pd

from bot.config import get_settings
from bot.data.feed import Bar
from bot.data.historical import fetch_bars
from bot.logging.logger import TradeJournal, TradeRecord, get_logger
from bot.strategies.base import SignalType
from bot.strategies.registry import get_strategy

log = get_logger(__name__)


@dataclass
class BacktestResult:
    """Aggregated performance metrics from a completed backtest run."""

    ticker: str
    strategy: str
    start_date: date
    end_date: date
    total_return_pct: float = 0.0
    num_trades: int = 0
    win_rate_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    reward_risk_ratio: float = 0.0
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)


async def run_backtest(
    strategy_name: str,
    ticker: str,
    start: date,
    end: date,
) -> BacktestResult:
    """
    Load historical bars, run the strategy bar-by-bar, and return results.

    Fills are simulated at the next bar's open price (no look-ahead).
    Slippage and commissions are not modelled in v1.
    """
    settings = get_settings()
    log = get_logger(__name__)

    log.info(
        "backtest_start",
        strategy=strategy_name,
        ticker=ticker,
        start=str(start),
        end=str(end),
    )

    df = await fetch_bars(ticker, start, end)
    bars: List[Bar] = [
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

    log.info("bars_loaded", count=len(bars))

    strategy = get_strategy(
        strategy_name,
        fast_period=settings.fast_period,
        slow_period=settings.slow_period,
        trend_sma_period=settings.trend_sma_period,
        rsi_period=settings.rsi_period,
        rsi_oversold=settings.rsi_oversold,
        rsi_overbought=settings.rsi_overbought,
        stop_loss_pct=settings.stop_loss_pct,
    )
    strategy.on_start()

    initial_equity = 10_000.0
    equity = initial_equity
    notional = settings.max_notional_per_trade

    # Position state
    in_position = False
    entry_price = 0.0
    entry_qty = 0.0
    entry_time = None

    completed_trades: List[TradeRecord] = []
    equity_curve: List[float] = [initial_equity]
    equity_series: List[Tuple[datetime, float]] = []   # (timestamp, mtm_equity)

    for i, bar in enumerate(bars):
        # Mark-to-market equity for drawdown tracking
        if in_position:
            mtm = equity + (bar.close - entry_price) * entry_qty
        else:
            mtm = equity
        equity_curve.append(mtm)
        equity_series.append((bar.timestamp, mtm))

        signal = strategy.on_bar(bar)
        if signal is None:
            continue

        # Fill at the next bar's open; fall back to current close at end of data
        fill_price = bars[i + 1].open if i + 1 < len(bars) else bar.close

        if signal.type == SignalType.BUY and not in_position:
            entry_qty = notional / fill_price
            entry_price = fill_price
            entry_time = bar.timestamp
            in_position = True
            log.info(
                "sim_open",
                ticker=ticker,
                price=round(fill_price, 4),
                qty=round(entry_qty, 4),
            )

        elif signal.type == SignalType.SELL and in_position:
            pnl = (fill_price - entry_price) * entry_qty
            equity += pnl
            completed_trades.append(
                TradeRecord(
                    timestamp=entry_time,
                    ticker=ticker,
                    side="buy",
                    qty=entry_qty,
                    entry_price=entry_price,
                    exit_price=fill_price,
                    exit_timestamp=bar.timestamp,
                    pnl=pnl,
                    strategy=strategy_name,
                )
            )
            log.info(
                "sim_close",
                ticker=ticker,
                price=round(fill_price, 4),
                pnl=round(pnl, 2),
                equity=round(equity, 2),
            )
            in_position = False

    # Force-close any open position at the last bar's close
    if in_position and bars:
        last_price = bars[-1].close
        pnl = (last_price - entry_price) * entry_qty
        equity += pnl
        completed_trades.append(
            TradeRecord(
                timestamp=entry_time,
                ticker=ticker,
                side="buy",
                qty=entry_qty,
                entry_price=entry_price,
                exit_price=last_price,
                exit_timestamp=bars[-1].timestamp,
                pnl=pnl,
                strategy=strategy_name,
            )
        )
        log.info("sim_force_close", ticker=ticker, price=round(last_price, 4), pnl=round(pnl, 2))

    strategy.on_stop()

    # Write trade journal — one CSV per ticker
    journal = TradeJournal()
    for t in completed_trades:
        journal.record_trade(t)
    journal.write(f"trades_{ticker}.csv")

    # Metrics
    num_trades = len(completed_trades)
    total_return_pct = (equity - initial_equity) / initial_equity * 100
    win_rate_pct = (
        sum(1 for t in completed_trades if t.pnl > 0) / num_trades * 100
        if num_trades > 0
        else 0.0
    )
    max_drawdown_pct = _max_drawdown(equity_curve)
    sharpe = _sharpe_ratio([t.pnl for t in completed_trades], initial_equity)

    # Reward : risk ratio — avg winning trade / avg losing trade (absolute)
    wins   = [t.pnl for t in completed_trades if t.pnl is not None and t.pnl > 0]
    losses = [t.pnl for t in completed_trades if t.pnl is not None and t.pnl <= 0]
    avg_win  = sum(wins)   / len(wins)   if wins   else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    reward_risk_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    log.info(
        "backtest_complete",
        trades=num_trades,
        total_return_pct=round(total_return_pct, 2),
        win_rate_pct=round(win_rate_pct, 1),
        max_drawdown_pct=round(max_drawdown_pct, 2),
        sharpe=round(sharpe, 3),
        reward_risk_ratio=round(reward_risk_ratio, 2),
    )

    result = BacktestResult(
        ticker=ticker,
        strategy=strategy_name,
        start_date=start,
        end_date=end,
        total_return_pct=total_return_pct,
        num_trades=num_trades,
        win_rate_pct=win_rate_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe,
        reward_risk_ratio=reward_risk_ratio,
        trades=completed_trades,
        equity_curve=equity_series,
    )
    save_equity_chart(result)
    return result


def print_summary(result: BacktestResult) -> None:
    """Print a human-readable performance summary to stdout."""
    sep = "=" * 52
    rr_str = f"{result.reward_risk_ratio:.2f}x" if result.reward_risk_ratio > 0 else "N/A"
    print(f"\n{sep}")
    print(f"  Backtest: {result.strategy.upper()}  |  {result.ticker}")
    print(f"  Period:   {result.start_date}  ->  {result.end_date}")
    print(sep)
    print(f"  Total return   : {result.total_return_pct:+.2f}%")
    print(f"  Trades         : {result.num_trades}")
    print(f"  Win rate       : {result.win_rate_pct:.1f}%")
    print(f"  Reward:risk    : {rr_str}")
    print(f"  Max drawdown   : {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe ratio   : {result.sharpe_ratio:.3f}  (simplified)")
    print(sep)
    if result.trades:
        print(f"  Trade journal  : trades_{result.ticker}.csv  ({result.num_trades} rows)\n")
    else:
        print(f"  No trades executed in this period.\n")


def print_comparison(results: list) -> None:
    """Print N strategies side by side for the same ticker and date range."""
    if not results:
        return
    first = results[0]
    col_w = max(12, max(len(r.strategy) for r in results) + 2)
    label_w = 20
    total_w = label_w + col_w * len(results) + 4
    sep = "=" * total_w

    def _val(v: str) -> str:
        return f"{v:>{col_w}s}"

    print(f"\n{sep}")
    print(f"  Comparison: {first.ticker}  |  {first.start_date} -> {first.end_date}")
    print(sep)
    print(f"  {'':20s}" + "".join(_val(r.strategy) for r in results))
    print(f"  {'Total return':20s}" + "".join(_val(f"{r.total_return_pct:+.2f}%") for r in results))
    print(f"  {'Trades':20s}" + "".join(_val(str(r.num_trades)) for r in results))
    print(f"  {'Win rate':20s}" + "".join(_val(f"{r.win_rate_pct:.1f}%") for r in results))
    print(f"  {'Reward:risk':20s}" + "".join(_val(f"{r.reward_risk_ratio:.2f}x" if r.reward_risk_ratio > 0 else "N/A") for r in results))
    print(f"  {'Max drawdown':20s}" + "".join(_val(f"{r.max_drawdown_pct:.2f}%") for r in results))
    print(f"  {'Sharpe ratio':20s}" + "".join(_val(f"{r.sharpe_ratio:.3f}") for r in results))
    print(f"{sep}\n")


async def run_multi_ticker_backtest(
    strategy_name: str,
    tickers: list,
    start: date,
    end: date,
) -> list:
    """Run backtest for each ticker sequentially; return list of BacktestResult."""
    results = []
    for ticker in tickers:
        result = await run_backtest(strategy_name, ticker, start, end)
        results.append(result)
    return results


def print_multi_ticker_summary(results: list) -> None:
    """Print a compact side-by-side summary table for multiple tickers."""
    if not results:
        return
    first = results[0]
    sep = "=" * 59
    print(f"\n{sep}")
    print(f"  Multi-ticker: {first.strategy.upper()}  |  {first.start_date.year} -> {first.end_date.year}")
    print(sep)
    print(f"  {'Ticker':<8} {'Trades':>7} {'Return':>9} {'Win%':>7} {'R:R':>7} {'MaxDD':>7}")
    print(f"  {'-'*8} {'-'*7} {'-'*9} {'-'*7} {'-'*7} {'-'*7}")
    for r in results:
        rr = f"{r.reward_risk_ratio:.2f}x" if r.reward_risk_ratio > 0 else "  N/A"
        print(
            f"  {r.ticker:<8} {r.num_trades:>7d}"
            f" {r.total_return_pct:>+8.2f}%"
            f" {r.win_rate_pct:>6.1f}%"
            f" {rr:>7}"
            f" {r.max_drawdown_pct:>6.2f}%"
        )
    print(f"{sep}\n")


def save_equity_chart(result: "BacktestResult", charts_dir: str = "charts") -> None:
    """
    Save an equity-curve PNG chart to *charts_dir*.

    Chart shows:
      - Portfolio value as a steelblue line
      - Dashed horizontal line at starting equity ($10,000)
      - Green up-triangles at trade entry bars
      - Red down-triangles at trade exit bars

    File name: equity_{ticker}_{strategy}_{start}_{end}.png
    Silently skipped if matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        log.warning("chart_skipped", reason="matplotlib not installed")
        return

    if not result.equity_curve:
        return

    Path(charts_dir).mkdir(exist_ok=True)

    timestamps = [ts for ts, _ in result.equity_curve]
    equities   = [eq for _, eq in result.equity_curve]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Equity curve
    ax.plot(timestamps, equities, linewidth=1.5, color="steelblue",
            label="Portfolio value")

    # Starting equity reference line
    ax.axhline(y=10_000.0, linestyle="--", color="gray", alpha=0.6,
               linewidth=1.0, label="Starting equity ($10,000)")

    # Trade markers — look up equity at signal bar timestamps
    eq_by_ts = {ts: eq for ts, eq in result.equity_curve}

    entry_ts_list, entry_eq_list = [], []
    exit_ts_list,  exit_eq_list  = [], []

    for trade in result.trades:
        if trade.timestamp in eq_by_ts:
            entry_ts_list.append(trade.timestamp)
            entry_eq_list.append(eq_by_ts[trade.timestamp])
        if trade.exit_timestamp is not None and trade.exit_timestamp in eq_by_ts:
            exit_ts_list.append(trade.exit_timestamp)
            exit_eq_list.append(eq_by_ts[trade.exit_timestamp])

    if entry_ts_list:
        ax.scatter(entry_ts_list, entry_eq_list, marker="^", color="lime",
                   s=100, zorder=5, label="Entry",
                   edgecolors="darkgreen", linewidths=0.8)
    if exit_ts_list:
        ax.scatter(exit_ts_list, exit_eq_list, marker="v", color="crimson",
                   s=100, zorder=5, label="Exit",
                   edgecolors="darkred", linewidths=0.8)

    # Labels and formatting
    ax.set_title(
        f"{result.strategy} | {result.ticker} | {result.start_date} → {result.end_date}",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    fname = (
        f"equity_{result.ticker}_{result.strategy}_"
        f"{result.start_date}_{result.end_date}.png"
    )
    fpath = Path(charts_dir) / fname
    plt.savefig(fpath, dpi=100)
    plt.close(fig)

    log.info("chart_saved", path=str(fpath))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the backtest sub-command (standalone use)."""
    p = argparse.ArgumentParser(description="Run a strategy backtest")
    p.add_argument("--strategy", required=True)
    p.add_argument("--ticker", required=True)
    p.add_argument("--from", dest="from_date", required=True)
    p.add_argument("--to",   dest="to_date",   required=True)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _max_drawdown(equity_curve: List[float]) -> float:
    """Return the maximum peak-to-trough drawdown as a percentage."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _sharpe_ratio(trade_pnls: List[float], initial_equity: float) -> float:
    """
    Simplified Sharpe: mean per-trade return / std per-trade return.

    Not annualised — intended as a relative quality signal between strategy
    runs, not an absolute risk-adjusted return measure.
    """
    if len(trade_pnls) < 2 or initial_equity == 0:
        return 0.0
    returns = pd.Series([p / initial_equity for p in trade_pnls])
    std = returns.std()
    if std == 0:
        return 0.0
    return float(returns.mean() / std)
