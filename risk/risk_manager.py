"""Order-level risk checks.

All orders must pass through RiskManager.check() before reaching execution/.
"""

import os
from datetime import datetime, timedelta, UTC


class RiskManager:
    """Runs a sequence of pre-trade risk checks on a candidate order."""

    def __init__(
        self,
        max_order_size_usd: float = 10.0,
        max_exposure_usd: float = 50.0,
        min_order_size_usd: float = 1.0,
        max_orders_per_hour: int = 5,
        kill_switch_path: str = "KILL_SWITCH",
    ) -> None:
        """Store risk limits and rate-limit configuration.

        Args:
            max_order_size_usd: Largest single order allowed, in USD.
            max_exposure_usd: Largest total exposure allowed, in USD.
            min_order_size_usd: Smallest single order allowed, in USD.
            max_orders_per_hour: Max number of approved orders in a
                rolling one-hour window.
            kill_switch_path: Path to a file whose mere presence halts
                all order approval.
        """
        self.max_order_size_usd = max_order_size_usd
        self.max_exposure_usd = max_exposure_usd
        self.min_order_size_usd = min_order_size_usd
        self.max_orders_per_hour = max_orders_per_hour
        self.kill_switch_path = kill_switch_path
        self._order_timestamps: list[datetime] = []
        self._peak_value: float | None = None
        self._drawdown_circuit_tripped = False

    def check(
        self,
        signal_value: int,
        order_size_usd: float,
        current_exposure_usd: float,
        paper_mode: bool,
    ) -> tuple[bool, str]:
        """Run all pre-trade risk checks in sequence.

        Args:
            signal_value: Signal output; 0 means no trade.
            order_size_usd: Proposed order size, in USD.
            current_exposure_usd: Existing exposure before this order, in USD.
            paper_mode: Whether the system is running in paper (simulated)
                mode. Must be True for any order to be approved.

        Returns:
            (True, 'approved') if the order passes every check, otherwise
            (False, reason) where reason describes the first failed check.
        """
        if self.is_kill_switch_active():
            return False, "kill switch active"

        if not paper_mode:
            return False, "paper mode required"

        if signal_value == 0:
            return False, "no signal"

        if order_size_usd < self.min_order_size_usd:
            return False, "order below minimum size"

        if order_size_usd > self.max_order_size_usd:
            return False, "order exceeds max size"

        if current_exposure_usd + order_size_usd > self.max_exposure_usd:
            return False, "exceeds max exposure"

        cutoff = datetime.now(UTC) - timedelta(hours=1)
        self._order_timestamps = [
            ts for ts in self._order_timestamps if ts > cutoff
        ]
        if len(self._order_timestamps) >= self.max_orders_per_hour:
            return False, "order rate limit reached"

        self._order_timestamps.append(datetime.now(UTC))
        return True, "approved"

    def reset_order_history(self) -> None:
        """Clear tracked order timestamps. Useful for testing."""
        self._order_timestamps = []

    def check_portfolio_drawdown(
        self,
        current_value: float,
        peak_value: float,
        max_drawdown_threshold: float = 0.15,
    ) -> tuple[bool, str]:
        """Check current drawdown from peak portfolio value.

        Tracks the highest peak value seen so far internally. Once the
        drawdown limit is breached, the circuit stays tripped on every
        subsequent call — even if current_value recovers — until
        reset_drawdown_circuit() is called.

        Args:
            current_value: Current portfolio value, in USD.
            peak_value: Peak portfolio value observed by the caller, in USD.
            max_drawdown_threshold: Max allowed fractional drawdown from peak.

        Returns:
            (True, 'approved') if within the drawdown limit, otherwise
            (False, 'portfolio drawdown limit reached — manual reset required').
        """
        if self._peak_value is None or peak_value > self._peak_value:
            self._peak_value = peak_value
        if current_value > self._peak_value:
            self._peak_value = current_value

        if self._drawdown_circuit_tripped:
            return False, "portfolio drawdown limit reached — manual reset required"

        drawdown = (self._peak_value - current_value) / self._peak_value
        if drawdown > max_drawdown_threshold:
            self._drawdown_circuit_tripped = True
            return False, "portfolio drawdown limit reached — manual reset required"

        return True, "approved"

    def reset_drawdown_circuit(self) -> None:
        """Re-enable trading after manual review of a tripped drawdown circuit."""
        self._drawdown_circuit_tripped = False

    def is_kill_switch_active(self) -> bool:
        """Check whether the kill switch file currently exists.

        Returns:
            True if a file exists at kill_switch_path, False otherwise.
        """
        return os.path.exists(self.kill_switch_path)
