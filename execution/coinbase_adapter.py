"""Coinbase Advanced Trade adapter using the coinbase-advanced-py SDK."""

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

from coinbase.rest import RESTClient

from execution.broker_adapter import BrokerAdapter, Order, Quote

_CDP_KEY_FILE = Path(__file__).resolve().parents[1] / "cdp_api_key.json"


class CoinbaseAdapter(BrokerAdapter):
    """BrokerAdapter implementation backed by the Coinbase Advanced Trade REST API.

    Credentials are read from cdp_api_key.json at the repo root:
        id         — CDP API key name (e.g. "organizations/.../apiKeys/...")
        privateKey — EC private key PEM string
    """

    def __init__(self) -> None:
        self.client: Optional[RESTClient] = None

    def connect(self) -> None:
        """Instantiate and store a RESTClient using credentials from cdp_api_key.json."""
        creds = json.loads(_CDP_KEY_FILE.read_text())
        self.client = RESTClient(api_key=creds["id"], api_secret=creds["privateKey"])

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
    ) -> Order:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def get_order_status(self, order_id: str) -> Order:
        raise NotImplementedError

    def get_account_balance(self) -> dict[str, Decimal]:
        raise NotImplementedError
