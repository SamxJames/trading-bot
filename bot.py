"""
Trading bot CLI entry point.

Usage:
    python bot.py live
    python bot.py backtest --strategy ema_cross --ticker AAPL \
        --from 2024-01-01 --to 2024-06-01
"""

from bot.main import main

if __name__ == "__main__":
    main()
