"""
Walk-forward parameter optimisation — Level 2.

Validates that strategy parameters generalise beyond their training window
before promoting them. Prevents overfitting.

HOW IT WORKS:
  1. Divide historical data into overlapping windows:
       Train: 5 years  →  Validate: 1 year  →  Slide forward 1 year  →  Repeat
  2. For each window, grid-search all param combos on the TRAIN set
  3. Take the best combo (by Sharpe) and test it on the VALIDATE set
  4. Only promote params that ALSO win on validation (win rate ≥ 45%, Sharpe ≥ 0)
  5. Aggregate results across all windows → consensus best params

USAGE:
  python scripts/walk_forward.py                    # run with defaults, print results
  python scripts/walk_forward.py --out results/     # save JSON to results/wf_results.json
  python scripts/walk_forward.py --promote          # overwrite config.yaml if validated

REQUIRED:
  pip install yfinance pyyaml  (already in requirements.txt)
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pandas_ta as ta
import yaml

# ── Config ────────────────────────────────────────────────────────────────────

TICKERS        = ["SPY", "QQQ", "GLD", "AAPL", "NVDA"]
TRAIN_YEARS    = 5
VALIDATE_YEARS = 1
SLIDE_YEARS    = 1
DATA_START     = "2003-01-01"   # fetch extra for warm-up
DATA_END       = date.today().isoformat()
CONFIG_PATH    = Path("config.yaml")

# Grid search space — keep it coarse to avoid overfitting
PARAM_GRID = {
    "fast_period":      [10, 15, 20, 25],
    "slow_period":      [40, 50, 60],
    "trend_sma_period": [100, 150, 200],
    "rsi_overbought":   [70, 75, 80],
    "stop_loss_pct":    [1.5, 2.0, 2.5, 3.0],
}

# Promotion thresholds — params only promoted if validation passes both
MIN_WIN_RATE   = 45.0   # %
MIN_SHARPE     = 0.0


# ── Backtester (self-contained, no broker) ────────────────────────────────────

@dataclass
class TradeResult:
    ticker:      str
    entry_price: float
    exit_price:  float
    pnl_pct:     float
    win:         bool
    reason:      str


def _compute_rsi(prices: pd.Series, period: int) -> pd.Series:
    return ta.rsi(prices, length=period)


def backtest_ticker(
    prices: pd.DataFrame,
    fast_period: int,
    slow_period: int,
    trend_sma_period: int,
    rsi_overbought: float,
    stop_loss_pct: float,
) -> list[TradeResult]:
    """
    Minimal backtester. prices must have columns: open, close (lowercase).
    Returns list of closed TradeResult objects.
    """
    close = prices["close"]
    open_ = prices["open"]

    ema_fast  = close.ewm(span=fast_period,  adjust=False).mean()
    ema_slow  = close.ewm(span=slow_period,  adjust=False).mean()
    trend_sma = close.rolling(trend_sma_period).mean()
    rsi       = _compute_rsi(close, 14)

    trades: list[TradeResult] = []
    in_position   = False
    entry_price   = 0.0
    pending_entry = False

    for i in range(1, len(close)):
        c    = close.iloc[i]
        o    = open_.iloc[i]
        ef   = ema_fast.iloc[i]
        ef_p = ema_fast.iloc[i - 1]
        es   = ema_slow.iloc[i]
        es_p = ema_slow.iloc[i - 1]
        sma  = trend_sma.iloc[i]
        rsi_v = rsi.iloc[i] if rsi is not None and not pd.isna(rsi.iloc[i]) else None

        # Resolve pending entry fill at open
        if pending_entry:
            entry_price   = o
            pending_entry = False

        # Stop loss check
        if in_position:
            stop = entry_price * (1.0 - stop_loss_pct / 100.0)
            if c < stop:
                pnl = (c - entry_price) / entry_price * 100.0
                trades.append(TradeResult(
                    ticker=prices.index.name or "?",
                    entry_price=entry_price, exit_price=c,
                    pnl_pct=pnl, win=False, reason="stop_loss"
                ))
                in_position = False
                continue

        # EMA crossover signals
        cross_up   = ef_p <= es_p and ef > es
        cross_down = ef_p >= es_p and ef < es

        if cross_down and in_position:
            pnl = (c - entry_price) / entry_price * 100.0
            trades.append(TradeResult(
                ticker=prices.index.name or "?",
                entry_price=entry_price, exit_price=c,
                pnl_pct=pnl, win=pnl > 0, reason="signal"
            ))
            in_position = False

        if cross_up and not in_position:
            # Trend filter
            if not pd.isna(sma) and c <= sma:
                continue
            # RSI filter
            if rsi_v is not None and rsi_v >= rsi_overbought:
                continue
            in_position   = True
            entry_price   = c    # provisional
            pending_entry = True

    return trades


def _sharpe(trades: list[TradeResult]) -> float:
    if len(trades) < 2:
        return 0.0
    pnls = pd.Series([t.pnl_pct for t in trades])
    std  = pnls.std()
    if std == 0:
        return 0.0
    return float(pnls.mean() / std * math.sqrt(max(len(pnls), 1)))


def _win_rate(trades: list[TradeResult]) -> float:
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.win) / len(trades) * 100.0


def run_params(data: dict[str, pd.DataFrame], params: dict) -> dict[str, float]:
    """Run a param combo across all tickers; return aggregate metrics."""
    all_trades: list[TradeResult] = []
    for ticker, df in data.items():
        df.index.name = ticker
        all_trades.extend(backtest_ticker(df, **params))
    return {
        "win_rate": _win_rate(all_trades),
        "sharpe":   _sharpe(all_trades),
        "trades":   len(all_trades),
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> dict[str, pd.DataFrame]:
    """Download OHLCV data for all tickers via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("yfinance not installed — run: pip install yfinance")

    print(f"Downloading data for {TICKERS} ({DATA_START} → {DATA_END})…")
    data = {}
    for ticker in TICKERS:
        raw = yf.download(ticker, start=DATA_START, end=DATA_END,
                          auto_adjust=True, progress=False)
        if raw.empty:
            print(f"  WARNING: no data for {ticker}, skipping")
            continue
        df = raw[["Open", "Close"]].copy()
        df.columns = ["open", "close"]
        df.index = pd.to_datetime(df.index)
        data[ticker] = df
        print(f"  {ticker}: {len(df)} bars")
    return data


# ── Walk-forward engine ───────────────────────────────────────────────────────

@dataclass
class WindowResult:
    train_start:  str
    train_end:    str
    validate_end: str
    best_params:  dict
    train_metrics:    dict
    validate_metrics: dict
    promoted:     bool


def walk_forward(data: dict[str, pd.DataFrame]) -> list[WindowResult]:
    results: list[WindowResult] = []

    # Find the common date range
    start = max(df.index.min() for df in data.values())
    end   = min(df.index.max() for df in data.values())

    train_delta    = pd.DateOffset(years=TRAIN_YEARS)
    validate_delta = pd.DateOffset(years=VALIDATE_YEARS)
    slide_delta    = pd.DateOffset(years=SLIDE_YEARS)

    cursor = start
    window_num = 0

    while True:
        train_start  = cursor
        train_end    = cursor + train_delta
        validate_end = train_end + validate_delta

        if validate_end > end:
            break

        window_num += 1
        print(f"\nWindow {window_num}: train {train_start.date()} → {train_end.date()}"
              f"  validate → {validate_end.date()}")

        # Slice data for this window
        train_data = {
            t: df[(df.index >= train_start) & (df.index < train_end)].copy()
            for t, df in data.items()
        }
        val_data = {
            t: df[(df.index >= train_end) & (df.index < validate_end)].copy()
            for t, df in data.items()
        }

        # Grid search on training set
        param_keys   = list(PARAM_GRID.keys())
        param_values = list(PARAM_GRID.values())
        best_sharpe  = -999.0
        best_params  = {}
        best_train_m = {}

        combos = list(itertools.product(*param_values))
        print(f"  Grid search: {len(combos)} combos…", end="", flush=True)

        for combo in combos:
            params = dict(zip(param_keys, combo))
            # Skip invalid combos (fast must be < slow)
            if params["fast_period"] >= params["slow_period"]:
                continue
            metrics = run_params(train_data, params)
            if metrics["sharpe"] > best_sharpe:
                best_sharpe  = metrics["sharpe"]
                best_params  = params
                best_train_m = metrics

        print(f" done. Best train Sharpe: {best_sharpe:.3f}")

        # Validate best params on out-of-sample data
        val_metrics = run_params(val_data, best_params)
        promoted    = (val_metrics["win_rate"] >= MIN_WIN_RATE
                       and val_metrics["sharpe"] >= MIN_SHARPE
                       and val_metrics["trades"] >= 2)

        status = "✅ PROMOTED" if promoted else "❌ REJECTED"
        print(f"  Validate → win={val_metrics['win_rate']:.1f}%  "
              f"sharpe={val_metrics['sharpe']:.3f}  "
              f"trades={val_metrics['trades']}  {status}")

        results.append(WindowResult(
            train_start=str(train_start.date()),
            train_end=str(train_end.date()),
            validate_end=str(validate_end.date()),
            best_params=best_params,
            train_metrics=best_train_m,
            validate_metrics=val_metrics,
            promoted=promoted,
        ))

        cursor += slide_delta

    return results


# ── Consensus params ──────────────────────────────────────────────────────────

def consensus_params(results: list[WindowResult]) -> dict | None:
    """
    From all promoted windows, pick the param combo that appeared most
    often as best. Ties broken by average validation Sharpe.
    """
    promoted = [r for r in results if r.promoted]
    if not promoted:
        return None

    counts: dict[str, int]    = defaultdict(int)
    sharpes: dict[str, list]  = defaultdict(list)

    for r in promoted:
        key = json.dumps(r.best_params, sort_keys=True)
        counts[key]  += 1
        sharpes[key].append(r.validate_metrics["sharpe"])

    best_key = max(counts, key=lambda k: (counts[k], sum(sharpes[k]) / len(sharpes[k])))
    return json.loads(best_key)


# ── Config promotion ──────────────────────────────────────────────────────────

def promote_to_config(params: dict, config_path: Path = CONFIG_PATH) -> None:
    """Overwrite strategy params in config.yaml with validated params."""
    if not config_path.exists():
        print(f"WARNING: {config_path} not found — cannot promote")
        return

    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    cfg["fast_period"]       = params["fast_period"]
    cfg["slow_period"]       = params["slow_period"]
    cfg["trend_sma_period"]  = params["trend_sma_period"]
    cfg["rsi_overbought"]    = params["rsi_overbought"]
    cfg["stop_loss_pct"]     = params["stop_loss_pct"]

    with config_path.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"Promoted params written to {config_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward parameter optimisation")
    parser.add_argument("--out",     type=str, default=None,
                        help="Directory to write wf_results.json")
    parser.add_argument("--promote", action="store_true",
                        help="Write consensus params to config.yaml if validated")
    args = parser.parse_args()

    data    = load_data()
    results = walk_forward(data)
    best    = consensus_params(results)

    print("\n── WALK-FORWARD SUMMARY ────────────────────────────────")
    promoted_count = sum(1 for r in results if r.promoted)
    print(f"  Windows run:     {len(results)}")
    print(f"  Windows promoted:{promoted_count} / {len(results)}")

    if best:
        print(f"\n  Consensus params (from {promoted_count} promoted windows):")
        for k, v in best.items():
            print(f"    {k}: {v}")
    else:
        print("\n  No windows promoted — current params retained.")

    output = {
        "generated_at":     date.today().isoformat(),
        "windows":          [asdict(r) for r in results],
        "consensus_params": best,
        "promoted_count":   promoted_count,
        "total_windows":    len(results),
    }

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "wf_results.json"
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\nResults written to {out_path}")

    if args.promote and best:
        promote_to_config(best)
    elif args.promote and not best:
        print("Nothing to promote — no windows passed validation.")


if __name__ == "__main__":
    main()
