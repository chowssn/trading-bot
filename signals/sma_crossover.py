"""Simple moving average crossover signal.

Pure strategy logic — must never import from execution/ or storage/.
"""

from dataclasses import dataclass
from datetime import datetime, UTC


@dataclass
class SignalResult:
    """Container for a computed signal value with metadata for logging."""

    symbol: str
    signal_name: str
    value: int
    timestamp: str


class SMACrossoverSignal:
    """Compares the latest price to a simple moving average.

    Emits 1 (bullish) when price is above the SMA, -1 (bearish) when
    price is below the SMA, and 0 when equal or when there is not
    enough data to compute the SMA.
    """

    def __init__(self, period: int = 20) -> None:
        """Store the lookback period used to compute the SMA.

        Args:
            period: Number of most recent prices to average.
        """
        self.period = period

    def compute(self, prices: list[float]) -> int:
        """Compute the crossover signal from a list of prices.

        Args:
            prices: Historical prices, ordered oldest to newest.

        Returns:
            0 if fewer than `period` prices are available or the last
            price equals the SMA, 1 if the last price is above the
            SMA, -1 if it is below.
        """
        if len(prices) < self.period:
            return 0

        window = prices[-self.period :]
        sma = sum(window) / self.period
        last_price = prices[-1]

        if last_price > sma:
            return 1
        if last_price < sma:
            return -1
        return 0

    def compute_with_metadata(self, symbol: str, prices: list[float]) -> SignalResult:
        """Compute the signal and wrap it with metadata for logging.

        Args:
            symbol: Instrument the signal was computed for, e.g. "BTC-USDC".
            prices: Historical prices, ordered oldest to newest.

        Returns:
            A SignalResult carrying the symbol, signal name, computed
            value, and UTC timestamp of computation.
        """
        value = self.compute(prices)
        return SignalResult(
            symbol=symbol,
            signal_name="sma_crossover",
            value=value,
            timestamp=datetime.now(UTC).isoformat(),
        )
