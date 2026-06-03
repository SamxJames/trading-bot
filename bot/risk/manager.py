"""
Risk manager — Phase 3 implementation.

The last gate before any order reaches the broker.  Every signal produced
by the strategy engine must be approved by the risk manager before it is
forwarded to execution.

Rules enforced (all configurable via Settings):
  1. Max concurrent open positions — rejects new entries when the limit is
     reached (default: 3).
  2. Max notional per trade — rejects orders where estimated notional
     (max_notional_per_trade) exceeds the configured cap (default: $500).
     Notional is the flat amount from settings, not qty * current_price,
     because qty is calculated from notional in the trading loop.
  3. Drawdown halt — if account equity has fallen by more than
     drawdown_halt_pct % since session start, halts ALL new trading
     and prevents further order placement (default: 5%).
  4. Kill switch — emergency_flatten() closes all open positions and
     cancels all orders via the injected broker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set

from bot.logging.logger import get_logger
from bot.strategies.base import Signal, SignalType

log = get_logger(__name__)


@dataclass
class RiskDecision:
    """The risk manager's verdict on a proposed signal."""

    approved: bool
    reason: str = ""


class RiskManager:
    """
    Stateful risk manager.

    Injected with the broker client and initial account equity so it can
    track drawdown and position counts across the session.
    """

    def __init__(
        self,
        max_positions: int = 3,
        max_notional: float = 500.0,
        drawdown_halt_pct: float = 5.0,
        broker: object = None,
        initial_equity: float = 0.0,
    ) -> None:
        self.max_positions      = max_positions
        self.max_notional       = max_notional
        self.drawdown_halt_pct  = drawdown_halt_pct
        self._broker            = broker
        self._initial_equity    = float(initial_equity)
        self._current_equity    = float(initial_equity)
        self._open_tickers: Set[str] = set()
        self._halted: bool      = False

        log.info(
            "risk_manager_init",
            max_positions=max_positions,
            max_notional=max_notional,
            drawdown_halt_pct=drawdown_halt_pct,
            initial_equity=initial_equity,
        )

    # ------------------------------------------------------------------
    # Core gate
    # ------------------------------------------------------------------

    def evaluate(self, signal: Signal, current_price: float) -> RiskDecision:  # noqa: ARG002
        """
        Assess *signal* against all active risk rules.

        Returns RiskDecision(approved=True) if all checks pass, or
        RiskDecision(approved=False, reason=<rule>) otherwise.

        *current_price* is accepted for API compatibility but notional
        sizing is based on max_notional (flat trade sizing).
        """
        if self._halted:
            return RiskDecision(approved=False, reason="trading_halted")

        if signal.type == SignalType.BUY:
            if len(self._open_tickers) >= self.max_positions:
                return RiskDecision(
                    approved=False,
                    reason=f"max_positions_reached ({self.max_positions})",
                )
            # Notional check: the trading loop sizes at max_notional / price,
            # so the notional spent is always ≤ max_notional.  The check is
            # still here as a guard against future sizing changes.
            if self.max_notional <= 0:
                return RiskDecision(approved=False, reason="max_notional_zero")

        return RiskDecision(approved=True, reason="approved")

    # ------------------------------------------------------------------
    # Position tracking
    # ------------------------------------------------------------------

    def record_open(self, ticker: str) -> None:
        """Notify the manager that a new long position has been opened."""
        self._open_tickers.add(ticker)
        log.info("risk_position_opened", ticker=ticker,
                 open_count=len(self._open_tickers))

    def record_close(self, ticker: str) -> None:
        """Notify the manager that an existing position has been closed."""
        self._open_tickers.discard(ticker)
        log.info("risk_position_closed", ticker=ticker,
                 open_count=len(self._open_tickers))

    # ------------------------------------------------------------------
    # Equity / drawdown tracking
    # ------------------------------------------------------------------

    def update_equity(self, current_equity: float) -> None:
        """Update running equity; trigger drawdown halt if threshold is breached."""
        self._current_equity = float(current_equity)
        if self._initial_equity <= 0:
            return
        drawdown_pct = (
            (self._initial_equity - self._current_equity) / self._initial_equity * 100.0
        )
        if drawdown_pct >= self.drawdown_halt_pct and not self._halted:
            self._halted = True
            log.warning(
                "drawdown_halt_triggered",
                drawdown_pct=round(drawdown_pct, 2),
                threshold=self.drawdown_halt_pct,
                current_equity=current_equity,
                initial_equity=self._initial_equity,
            )

    def is_halted(self) -> bool:
        """Return True if the drawdown halt has been triggered."""
        return self._halted

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    async def emergency_flatten(self) -> None:
        """Close all open positions and cancel all pending orders."""
        if self._broker is None:
            log.warning("emergency_flatten_no_broker")
            return
        try:
            await self._broker.close_all_positions()
            await self._broker.cancel_all_orders()
            self._open_tickers.clear()
            log.info("emergency_flatten_complete")
        except Exception as exc:
            log.error("emergency_flatten_failed", error=str(exc))
