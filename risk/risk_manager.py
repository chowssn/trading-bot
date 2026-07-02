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

    def is_kill_switch_active(self) -> bool:
        """Check whether the kill switch file currently exists.

        Returns:
            True if a file exists at kill_switch_path, False otherwise.
        """
        return os.path.exists(self.kill_switch_path)
