# scripts/test_quote.py
from dotenv import load_dotenv
from execution.coinbase_adapter import CoinbaseAdapter

load_dotenv()

adapter = CoinbaseAdapter()
adapter.connect()
quote = adapter.get_quote("BTC-USDC")
print(quote)
