# scripts/test_adapter.py
"""Runs CoinbaseAdapter's currently implemented methods in sequence and
prints a PASS/FAIL report. Adapter runs in paper mode, so place_order()
never touches the live API.
"""
from decimal import Decimal

from dotenv import load_dotenv

from execution.coinbase_adapter import CoinbaseAdapter

load_dotenv()

SYMBOL = "BTC-USDC"

adapter = CoinbaseAdapter(paper_mode=True)

passed = 0
failed = 0


def run_step(label, fn):
    global passed, failed
    print(f"\n[{label}]")
    try:
        result = fn()
        print(f"  result: {result}")
        print(f"  PASS: {label}")
        passed += 1
        return result
    except Exception as exc:
        print(f"  error: {exc!r}")
        print(f"  FAIL: {label}")
        failed += 1
        return None


run_step("connect()", adapter.connect)
run_step(f"get_quote({SYMBOL!r})", lambda: adapter.get_quote(SYMBOL))
run_step("get_balances()", adapter.get_balances)
run_step("get_usdc_balance()", adapter.get_usdc_balance)
run_step(f"get_position({SYMBOL!r})", lambda: adapter.get_position(SYMBOL))
run_step(
    f"place_order({SYMBOL!r}, 'buy', 10.0) [paper]",
    lambda: adapter.place_order(SYMBOL, "buy", Decimal("10.0")),
)

print(f"\n{'=' * 40}")
print(f"Results: {passed} passed, {failed} failed")
print("=" * 40)
