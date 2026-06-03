"""
Logging and observability.

Configures structlog to emit:
  - Human-readable, colourised output to stdout during a live session
  - Machine-readable JSON to a rotating log file (logs/bot.log)

Every significant event in the pipeline should be logged with structured
key=value context so sessions are fully reconstructable after the fact:
  bar_received, signal_generated, risk_approved, risk_rejected,
  order_placed, order_filled, order_rejected, error, session_start,
  session_end

Trade journal:
  At session end, write_trade_journal() serialises the in-memory trade
  list to trades.csv with columns:
    timestamp, ticker, side, qty, entry_price, exit_price, pnl, strategy

Usage:
    from bot.logging.logger import get_logger, TradeJournal
    log = get_logger(__name__)
    log.info("signal_generated", ticker="AAPL", signal="BUY", reason="ema_cross")
"""

from __future__ import annotations

import csv
import logging
import logging.handlers
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import structlog

_shared_processors: list = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.StackInfoRenderer(),
]


def configure_logging(log_dir: str = "logs") -> None:
    """
    Set up dual-sink logging: JSON to a rotating file, coloured output to stdout.
    Call exactly once at startup before any other logging occurs.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        Path(log_dir) / "bot.log",
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=_shared_processors,
        )
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            foreign_pre_chain=_shared_processors,
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stdout_handler)
    root.setLevel(logging.INFO)

    structlog.configure(
        processors=_shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger namespaced to *name*."""
    return structlog.get_logger(name)


@dataclass
class TradeRecord:
    """One round-trip trade (entry + exit) for the journal CSV."""

    timestamp: datetime          # signal bar that opened the trade
    ticker: str
    side: str
    qty: float
    entry_price: float
    exit_price: float | None
    pnl: float | None
    strategy: str
    exit_timestamp: datetime | None = None   # signal bar that closed the trade


class TradeJournal:
    """
    In-memory list of trades that is flushed to trades.csv on session end.

    Accumulate records with record_trade(); flush with write().
    """

    _COLUMNS = [
        "timestamp", "ticker", "side", "qty",
        "entry_price", "exit_price", "pnl", "strategy",
    ]

    def __init__(self) -> None:
        self._trades: List[TradeRecord] = []

    def record_trade(self, trade: TradeRecord) -> None:
        self._trades.append(trade)

    def write(self, path: str = "trades.csv") -> None:
        """Write all recorded trades to *path* as a CSV file.  No-op if no trades."""
        if not self._trades:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._COLUMNS)
            writer.writeheader()
            for t in self._trades:
                writer.writerow(
                    {
                        "timestamp": t.timestamp.isoformat(),
                        "ticker": t.ticker,
                        "side": t.side,
                        "qty": t.qty,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "pnl": t.pnl,
                        "strategy": t.strategy,
                    }
                )
