"""Main polling loop tying together data, signals, risk, and execution."""

import logging
import time
from decimal import Decimal

from execution.broker_adapter import BrokerAdapter
from risk.risk_manager import RiskManager
from signals.sma_crossover import SMACrossoverSignal
from storage.db import Database

logger = logging.getLogger(__name__)

_PRICE_BUFFER_MAX_LEN = 20


class TradingLoop:
    """Polls a broker for quotes, computes a signal, and places risk-checked orders."""

    def __init__(
        self,
        adapter: BrokerAdapter,
        risk_manager: RiskManager,
        signal: SMACrossoverSignal,
        db: Database,
        poll_interval_seconds: int = 60,
        order_size_usd: float = 10.0,
        symbol: str = "BTC-USDC",
    ) -> None:
        """Store collaborators and configuration for the trading loop.

        Args:
            adapter: Broker/exchange adapter used for quotes, positions, and orders.
            risk_manager: Pre-trade risk gate that all orders must pass through.
            signal: Strategy used to turn price history into a trade signal.
            db: Storage layer for logging quotes, signals, and orders.
            poll_interval_seconds: Seconds to sleep between polling iterations.
            order_size_usd: USD size to use for each order attempt.
            symbol: Instrument to trade, e.g. "BTC-USDC".
        """
        self.adapter = adapter
        self.risk_manager = risk_manager
        self.signal = signal
        self.db = db
        self.poll_interval_seconds = poll_interval_seconds
        self.order_size_usd = order_size_usd
        self.symbol = symbol
        self._price_buffer: list[float] = []
        self._running: bool = False

    def run(self) -> None:
        """Run the main polling loop until stopped or interrupted.

        Each iteration fetches a quote, updates the signal, runs the risk
        check, and places an order if approved. Errors from a single
        iteration are logged and swallowed so a bad quote or API hiccup
        never crashes the loop; only a KeyboardInterrupt (Ctrl+C) stops it,
        triggering a clean shutdown.
        """
        self._running = True
        logger.info(
            "Starting trading loop: symbol=%s poll_interval_seconds=%d",
            self.symbol,
            self.poll_interval_seconds,
        )

        try:
            while self._running:
                try:
                    self._run_iteration()
                except Exception:
                    logger.exception("Error during trading loop iteration; continuing")

                time.sleep(self.poll_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Shutting down cleanly")
            self._running = False

    def _run_iteration(self) -> None:
        """Run a single poll/signal/risk/execution cycle."""
        quote = self.adapter.get_quote(self.symbol)
        logger.info("Quote: %s", quote)
        self.db.log_quote(quote)

        self._price_buffer.append(float(quote.last))
        self._price_buffer = self._price_buffer[-_PRICE_BUFFER_MAX_LEN:]

        result = self.signal.compute_with_metadata(self.symbol, self._price_buffer)
        self.db.log_signal(self.symbol, result.signal_name, result.value)

        if result.value == 1:
            logger.info("Signal: 1 (bullish)")
        elif result.value == -1:
            logger.info("Signal: -1 (bearish)")
        else:
            logger.info("Signal: 0 (neutral/insufficient data)")

        position = self.adapter.get_position(self.symbol)
        current_exposure = (
            position.quantity * float(quote.last) if position is not None else 0.0
        )

        approved, reason = self.risk_manager.check(
            signal_value=result.value,
            order_size_usd=self.order_size_usd,
            current_exposure_usd=current_exposure,
            paper_mode=self.adapter.paper_mode,
        )

        if approved:
            side = "buy" if result.value == 1 else "sell"
            order_result = self.adapter.place_order(
                self.symbol, side, Decimal(str(self.order_size_usd))
            )
            self.db.log_order(
                order_result,
                self.symbol,
                side,
                self.order_size_usd,
                paper_mode=True,
            )
            logger.info("Order result: %s", order_result)
        else:
            logger.info("Order not placed: %s", reason)

    def stop(self) -> None:
        """Signal the loop to stop after its current iteration."""
        self._running = False
