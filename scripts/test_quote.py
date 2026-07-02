# scripts/test_quote.py
from dotenv import load_dotenv
from execution.coinbase_adapter import CoinbaseAdapter

load_dotenv()

adapter = CoinbaseAdapter()
adapter.connect()
quote = adapter.get_quote("AAVE-USDC")
print(quote)

balance = adapter.get_balance()
print(f"USDC Balance: {balance}")

position = adapter.get_position("AAVE-USDC")
print(f"AAVE Position: {position}")

# Add to scripts/test_quote.py temporarily
result = adapter.place_order("AAVE-USDC", "buy", 5.0)  # $10 USDC
print(f"Paper order result: {result}")
