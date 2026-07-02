"""Coinbase Advanced Trade adapter using the coinbase-advanced-py SDK."""

import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import Optional

from coinbase.rest import RESTClient
from requests.exceptions import RequestException

from execution.broker_adapter import BrokerAdapter, Order, OrderResult, Position, Quote

_CDP_KEY_FILE = Path(__file__).resolve().parents[1] / "cdp_api_key.json"

logger = logging.getLogger(__name__)


class CoinbaseAdapter(BrokerAdapter):
    """BrokerAdapter implementation backed by the Coinbase Advanced Trade REST API.

    Credentials are read from cdp_api_key.json at the repo root:
        id         — CDP API key ID (used to build the full apiKeys/ path)
        privateKey — EC private key PEM string
    """

    def __init__(self, paper_mode: bool = True) -> None:
        """Create the adapter.

        Args:
            paper_mode: When True (the default), place_order() logs the
                intended order but never calls the live Coinbase API — it
                returns a mock OrderResult with status "paper" instead.
                Must be explicitly set to False to submit real orders.
        """
        self.client: Optional[RESTClient] = None
        self.paper_mode = paper_mode

    def connect(self) -> None:
        """Instantiate and store a RESTClient using credentials from cdp_api_key.json."""
        key_data = json.loads(_CDP_KEY_FILE.read_text())
        full_name = f"organizations/cdfa7c9a-4800-53ca-992e-a39d9cbc394d/apiKeys/{key_data['id']}"
        self.client = RESTClient(
            key_file=StringIO(
                json.dumps({"name": full_name, "privateKey": key_data["privateKey"]})
            )
        )

    def _require_client(self) -> RESTClient:
        if self.client is None:
            raise RuntimeError("Not connected — call connect() first.")
        return self.client

    def get_quote(self, symbol: str) -> Quote:
        """Return the best bid, ask, and last price for *symbol* (e.g. 'BTC-USDC').

        Makes two calls:
          - get_best_bid_ask for live bid/ask from the order book
          - get_product for the most-recent trade price
        """
        client = self._require_client()

        bid_ask_resp = client.get_best_bid_ask(product_ids=[symbol])
        pricebooks = bid_ask_resp.pricebooks
        if not pricebooks:
            raise ValueError(f"No order book data returned for {symbol!r}")

        book = pricebooks[0]
        bid = Decimal(book.bids[0].price) if book.bids else Decimal("0")
        ask = Decimal(book.asks[0].price) if book.asks else Decimal("0")

        # book.time is a dict with an "seconds" key from the Timestamp proto
        raw_time = book.time
        if raw_time and "seconds" in raw_time:
            timestamp = datetime.fromtimestamp(
                float(raw_time["seconds"]), tz=timezone.utc
            )
        else:
            timestamp = datetime.now(tz=timezone.utc)

        product_resp = client.get_product(symbol)
        last = Decimal(product_resp.price)

        return Quote(
            bid=bid,
            ask=ask,
            last=last,
            timestamp=timestamp,
            symbol=symbol,
        )

    # --- unimplemented abstract methods ---

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str = "market",
        limit_price: Optional[Decimal] = None,
    ) -> OrderResult:
        """Submit a market order and return the resulting OrderResult.

        Args:
            symbol: Product ID, e.g. "BTC-USDC".
            side: "buy" or "sell".
            quantity: For "buy", the USDC amount to spend (quote_size).
                For "sell", the BTC amount to sell (base_size).
            order_type: Unused for now — only market orders are supported.
            limit_price: Unused for now — only market orders are supported.

        Returns:
            An OrderResult with the broker-assigned order_id (or the
            generated client_order_id in paper mode / on failure), status,
            filled_qty, and avg_fill_price.

        Note:
            The Coinbase order-creation response does not include fill
            details for market orders — filled_qty and avg_fill_price are
            only populated once available; otherwise callers should follow
            up with get_order_status().

        Raises:
            ValueError: If side is not "buy" or "sell".
            RuntimeError: If not paper mode and not connected, or if the
                order request itself fails (network/API error).
        """
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

        client_order_id = str(uuid.uuid4())

        if self.paper_mode:
            logger.info(
                "[PAPER] place_order symbol=%s side=%s quantity=%s "
                "client_order_id=%s",
                symbol,
                side,
                quantity,
                client_order_id,
            )
            return OrderResult(
                order_id=client_order_id,
                status="paper",
                filled_qty=Decimal("0"),
                avg_fill_price=None,
            )

        client = self._require_client()

        try:
            if side == "buy":
                response = client.market_order_buy(
                    client_order_id=client_order_id,
                    product_id=symbol,
                    quote_size=str(quantity),
                )
            else:
                response = client.market_order_sell(
                    client_order_id=client_order_id,
                    product_id=symbol,
                    base_size=str(quantity),
                )
        except RequestException as exc:
            raise RuntimeError(f"Failed to place order on Coinbase: {exc}") from exc

        success = bool(getattr(response, "success", False))
        order_id = getattr(response, "order_id", None) or client_order_id
        status = "submitted" if success else "rejected"

        return OrderResult(
            order_id=order_id,
            status=status,
            filled_qty=Decimal("0"),
            avg_fill_price=None,
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by ID.

        Args:
            order_id: The broker-assigned order ID to cancel.

        Returns:
            True if Coinbase confirmed the cancellation succeeded, False
            otherwise (including on request failure).
        """
        client = self._require_client()

        try:
            response = client.cancel_orders(order_ids=[order_id])
        except RequestException as exc:
            logger.warning("cancel_order request failed for %s: %s", order_id, exc)
            return False

        results = getattr(response, "results", None) or []
        if not results:
            return False

        return bool(getattr(results[0], "success", False))

    def get_order_status(self, order_id: str) -> Order:
        raise NotImplementedError

    def get_balances(self) -> dict[str, Decimal]:
        """Return a mapping of currency -> available balance across all accounts."""
        client = self._require_client()

        balances: dict[str, Decimal] = {}
        cursor: Optional[str] = None
        try:
            while True:
                resp = client.get_accounts(limit=250, cursor=cursor)
                for account in resp.accounts or []:
                    if not account.currency or not account.available_balance:
                        continue
                    balances[account.currency] = Decimal(account.available_balance["value"])
                if not getattr(resp, "has_next", False):
                    break
                cursor = resp.cursor
        except RequestException as exc:
            raise RuntimeError(f"Failed to fetch account balances from Coinbase: {exc}") from exc

        return balances

    def get_balance(self) -> float:
        """Return the available balance of the account's quote currency (USDC)."""
        return self.get_usdc_balance()

    def get_usdc_balance(self) -> float:
        """Coinbase-specific convenience: available USDC balance as a float."""
        return float(self.get_balances().get("USDC", Decimal("0")))

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return the Position for *symbol* (e.g. 'BTC-USDC'), or None if flat.

        avg_cost is always 0.0 — Coinbase's account balances don't track cost basis.
        """
        base_asset = symbol.split("-")[0]
        balance = self.get_balances().get(base_asset, Decimal("0"))
        if balance == 0:
            return None
        return Position(symbol=symbol, quantity=float(balance), avg_cost=0.0)
