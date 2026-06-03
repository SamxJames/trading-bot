"""
Package entry point — enables `python -m bot <command>`.

Commands:
    python -m bot backtest --strategy ema_cross_filtered --tickers AAPL --from 2024-01-01 --to 2024-06-01
    python -m bot backtest --strategy ema_cross_filtered --tickers AAPL,MSFT,SPY --from 2020-01-01 --to 2024-06-01
    python -m bot compare  --tickers AAPL --from 2024-01-01 --to 2024-06-01
    python -m bot compare  --strategies ema_cross_filtered,rsi --tickers AAPL,MSFT --from 2020-01-01 --to 2024-06-01
    python -m bot live
    python -m bot live --dry-run
    python -m bot job
    python -m bot job --dry-run

NOTE: The backtest makes ZERO live broker API calls.
Alpaca is used only to fetch historical price data.
Your paper account will show no positions after running a backtest — this is correct.
"""

import sys
import asyncio
import argparse
from datetime import date


def _parse_tickers(raw: str) -> list:
    """Split 'AAPL,MSFT, SPY' → ['AAPL', 'MSFT', 'SPY']."""
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def main() -> None:
    from bot.logging.logger import configure_logging
    configure_logging()

    parser = argparse.ArgumentParser(prog="bot")
    subparsers = parser.add_subparsers(dest="command")

    # ── backtest ──────────────────────────────────────────────────────────────
    bt = subparsers.add_parser("backtest", help="Run a strategy backtest on historical data")
    bt.add_argument("--strategy", required=True, help="Strategy name (e.g. ema_cross_filtered)")
    bt.add_argument(
        "--tickers", required=True,
        help="Comma-separated ticker symbols (e.g. AAPL  or  AAPL,MSFT,SPY)",
    )
    bt.add_argument("--from", dest="from_date", required=True, help="Start date YYYY-MM-DD")
    bt.add_argument("--to",   dest="to_date",   required=True, help="End date YYYY-MM-DD")

    # ── compare ───────────────────────────────────────────────────────────────
    cmp = subparsers.add_parser(
        "compare", help="Run multiple strategies side by side"
    )
    cmp.add_argument(
        "--tickers", required=True,
        help="Comma-separated tickers (e.g. AAPL  or  AAPL,MSFT)",
    )
    cmp.add_argument(
        "--strategies", default=None,
        help="Comma-separated strategy names to compare (default: all registered)",
    )
    cmp.add_argument("--from", dest="from_date", required=True)
    cmp.add_argument("--to",   dest="to_date",   required=True)

    # ── live ──────────────────────────────────────────────────────────────────
    live = subparsers.add_parser("live", help="Start the live paper-trading session")
    live.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=False,
        help="Simulate the live loop without placing real orders",
    )

    # ── job ───────────────────────────────────────────────────────────────────
    job = subparsers.add_parser(
        "job",
        help="Run the daily scheduled job (fetch today's bar, signal, order, notify)",
    )
    job.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=False,
        help="Simulate the job without placing real orders",
    )

    args = parser.parse_args()

    # ── dispatch ──────────────────────────────────────────────────────────────
    if args.command == "backtest":
        from bot.backtest import (
            run_backtest,
            run_multi_ticker_backtest,
            print_summary,
            print_multi_ticker_summary,
        )
        tickers = _parse_tickers(args.tickers)
        start   = date.fromisoformat(args.from_date)
        end     = date.fromisoformat(args.to_date)

        if len(tickers) == 1:
            result = asyncio.run(run_backtest(args.strategy, tickers[0], start, end))
            print_summary(result)
        else:
            results = asyncio.run(
                run_multi_ticker_backtest(args.strategy, tickers, start, end)
            )
            print_multi_ticker_summary(results)

    elif args.command == "compare":
        from bot.backtest import run_backtest, print_comparison
        from bot.strategies.registry import REGISTRY

        tickers = _parse_tickers(args.tickers)
        start   = date.fromisoformat(args.from_date)
        end     = date.fromisoformat(args.to_date)

        if args.strategies:
            strategy_names = [s.strip() for s in args.strategies.split(",") if s.strip()]
        else:
            strategy_names = sorted(REGISTRY.keys())

        for ticker in tickers:
            results = []
            for name in strategy_names:
                results.append(asyncio.run(run_backtest(name, ticker, start, end)))
            print_comparison(results)

    elif args.command == "live":
        from bot.main import run
        try:
            asyncio.run(run(dry_run=args.dry_run))
        except KeyboardInterrupt:
            pass

    elif args.command == "job":
        from bot.job import run_job
        try:
            asyncio.run(run_job(dry_run=args.dry_run))
        except KeyboardInterrupt:
            pass
        except Exception:
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
