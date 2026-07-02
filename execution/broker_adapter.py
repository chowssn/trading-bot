"""Abstract base class for broker adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional


@dataclass
class Quote:
    bid: Decimal
    ask: Decimal
    last: Decimal
    timestamp: datetime
    symbol: str


@dataclass
class Order:
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: Decimal
    price: Optional[Decimal]
    status: str


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_cost: float


@dataclass
class OrderResult:
    order_id: str
    status: str
    filled_qty: Decimal
    avg_fill_price: Optional[Decimal]


class BrokerAdapter(ABC):
    """Interface for all broker integrations."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection / authenticate with the broker."""
        ...

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Return the current best bid/ask/last for *symbol*."""
        ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str = "market",
        limit_price: Optional[Decimal] = None,
    ) -> Order:
        """Submit an order and return an Order with broker-assigned ID."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if successfully cancelled."""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """Fetch the current status of a previously placed order."""
        ...

    @abstractmethod
    def get_balances(self) -> dict[str, Decimal]:
        """Return a mapping of asset symbol → available balance."""
        ...

    @abstractmethod
    def get_balance(self) -> float:
        """Return the available balance of the account's quote currency (e.g. USDC)."""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Return the current Position for *symbol*, or None if there is none."""
        ...
