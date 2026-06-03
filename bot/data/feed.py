"""
Live data feed.

Connects to Alpaca's real-time WebSocket stream and delivers OHLCV bars
to the strategy engine as they arrive.

Responsibilities:
  - Open and maintain the WebSocket connection (alpaca-py handles reconnects)
  - Parse incoming bar messages into our internal Bar dataclass
  - Put each Bar onto the shared asyncio.Queue consumed by the trading loop
  - Expose an async context manager so main.py can do:
        async with LiveFeed(tickers, queue) as feed:
            await feed.start()

Notes:
  - The free Alpaca tier uses the IEX feed (15-min delayed intraday).
    This is fine for paper trading.
  - StockDataStream.run() is a long-running coroutine; run it as an
    asyncio Task so the trading loop can run concurrently.
  - Bars only arrive during market hours.  Outside hours the queue will be
    empty and the trading loop will time out on every iteration — this is
    expected behaviour.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import List

from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream

from bot.logging.logger import get_logger

log = get_logger(__name__)


@dataclass
class Bar:
    """A single OHLCV bar for one ticker."""

    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class LiveFeed:
    """
    Async WebSocket feed for one or more tickers.

    Subscribes to 1-minute bars and puts each arriving Bar onto *queue*.
    Use as an async context manager to ensure the stream is closed on exit.
    """

    def __init__(self, tickers: List[str], queue: asyncio.Queue) -> None:
        from bot.config import get_settings
        s = get_settings()
        self._tickers = tickers
        self._queue = queue
        self._stream = None
        self._stream = StockDataStream(
            api_key=s.apca_api_key_id,
            secret_key=s.apca_api_secret_key,
            feed=DataFeed.IEX,
        )

    async def start(self) -> None:
        """Subscribe to bars and start the WebSocket stream (blocks until stopped)."""
        async def _on_bar(alpaca_bar: object) -> None:
            bar = Bar(
                ticker=alpaca_bar.symbol,
                timestamp=alpaca_bar.timestamp,
                open=float(alpaca_bar.open),
                high=float(alpaca_bar.high),
                low=float(alpaca_bar.low),
                close=float(alpaca_bar.close),
                volume=int(alpaca_bar.volume),
            )
            log.debug("bar_received", ticker=bar.ticker, close=bar.close)
            await self._queue.put(bar)

        self._stream.subscribe_bars(_on_bar, *self._tickers)
        log.info("feed_starting", tickers=self._tickers)
        await self._stream.run()

    async def stop(self) -> None:
        """Gracefully close the WebSocket connection."""
        if self._stream is not None:
            try:
                await self._stream.stop()
            except AttributeError:
                pass
        log.info("feed_stopped")

    async def __aenter__(self) -> "LiveFeed":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.stop()
