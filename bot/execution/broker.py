"""
Broker client (Alpaca).

Thin async wrapper around the Alpaca REST API via alpaca-py.
All order placement, position queries, and account information go through
this class.  The rest of the codebase never imports alpaca-py directly —
only this module does, so swapping brokers in the future means changing
one file.

Supported operations (v1):
  - get_account()            → account equity, buying power, status
  - get_positions()          → list of currently open positions
  - get_open_orders()        → list of pending orders
  - place_market_order(...)
  - place_limit_order(...)
  - cancel_all_orders()
  - close_all_positions()    ← used by the kill switch

The alpaca-py TradingClient is synchronous, so every method wraps calls in
asyncio.to_thread() to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
from typing import List

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetCalendarRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)

from bot.logging.logger import get_logger

log = get_logger(__name__)


@dataclass
class AccountInfo:
    equity: float
    buying_power: float
    status: str


@dataclass
class Position:
    ticker: str
    qty: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float


@dataclass
class Order:
    order_id: str
    ticker: str
    side: str
    qty: float
    order_type: str
    status: str
    filled_avg_price: float | None = None


class BrokerClient:
    """Async Alpaca REST client for paper trading."""

    def __init__(self, api_key: str, secret_key: str, base_url: str) -> None:
        paper = "paper-api" in base_url
        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )
        log.info("broker_init", paper=paper, base_url=base_url)

    async def get_account(self) -> AccountInfo:
        account = await asyncio.to_thread(self._client.get_account)
        return AccountInfo(
            equity=float(account.equity),
            buying_power=float(account.buying_power),
            status=str(account.status.value),
        )

    async def get_positions(self) -> List[Position]:
        positions = await asyncio.to_thread(self._client.get_all_positions)
        return [
            Position(
                ticker=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price or 0),
                unrealized_pnl=float(p.unrealized_pl or 0),
            )
            for p in positions
        ]

    async def get_open_orders(self) -> List[Order]:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = await asyncio.to_thread(self._client.get_orders, filter=request)
        return [_to_order(o) for o in orders]

    async def place_market_order(
        self, ticker: str, qty: float, side: str
    ) -> Order:
        request = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide(side.lower()),
            time_in_force=TimeInForce.DAY,
        )
        order = await asyncio.to_thread(self._client.submit_order, order_data=request)
        log.info(
            "order_placed",
            ticker=ticker,
            side=side,
            qty=qty,
            order_type="market",
            order_id=str(order.id),
        )
        return _to_order(order)

    async def place_limit_order(
        self, ticker: str, qty: float, side: str, limit_price: float
    ) -> Order:
        request = LimitOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide(side.lower()),
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
        )
        order = await asyncio.to_thread(self._client.submit_order, order_data=request)
        log.info(
            "order_placed",
            ticker=ticker,
            side=side,
            qty=qty,
            order_type="limit",
            limit_price=limit_price,
            order_id=str(order.id),
        )
        return _to_order(order)

    async def is_trading_day(self, d: date) -> bool:
        """Return True if *d* is a NYSE trading day (Alpaca calendar is non-empty)."""
        calendar = await asyncio.to_thread(
            self._client.get_calendar,
            filters=GetCalendarRequest(start=d, end=d),
        )
        return len(calendar) > 0

    async def cancel_all_orders(self) -> None:
        await asyncio.to_thread(self._client.cancel_orders)
        log.info("orders_cancelled_all")

    async def close_all_positions(self) -> None:
        await asyncio.to_thread(self._client.close_all_positions, cancel_orders=True)
        log.info("positions_closed_all")


def _to_order(o: object) -> Order:
    """Convert an alpaca-py Order object to our internal Order dataclass."""
    return Order(
        order_id=str(o.id),
        ticker=o.symbol,
        side=str(o.side.value),
        qty=float(o.qty or 0),
        order_type=str(o.order_type.value),
        status=str(o.status.value),
        filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
    )
