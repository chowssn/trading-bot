# Trading Bot — Project Context

## Purpose
Personal algorithmic trading system. Starts with one venue/instrument,
designed to scale to multiple brokers/exchanges across TradFi and DeFi,
and potentially to enterprise/institutional use later.

## Architecture (layers, in order of data flow)
1. `data/` — market data ingestion, normalized into our own schema
2. `signals/` — strategy logic. Pure functions. MUST NOT import anything
   from `execution/` — signals know nothing about brokers.
3. `risk/` — position sizing, exposure limits, kill switches. All orders
   must pass through this layer before reaching execution.
4. `execution/` — broker/exchange adapters. All adapters implement the
   `BrokerAdapter` interface in `execution/broker_adapter.py`. New venues
   = new adapter class, nothing else changes.
5. `orchestrator/` — scheduling/event loop that ties layers together.
6. `storage/` — DB models and trade/signal logging.
7. `backtest/` — backtesting harness, separate from live orchestration
   but reuses the same `signals/` code.

## House rules
- NEVER commit secrets. All credentials go in `.env` (gitignored), loaded
  via `config/settings.py` using python-dotenv.
  explicitly say to use live trading.
- Every order attempt, fill, and signal must be logged with a timestamp
  via `storage/`.
- `signals/` code must never import from `execution/` — keep strategy
  logic broker-agnostic.
- Favor small, testable functions. Write a test in `tests/` alongside
  new logic in `risk/` and `signals/` especially.
- Ask before adding new dependencies to requirements.txt for anything
  beyond standard data/finance libraries.

## Current status
- Repo scaffolded, folder structure in place.
- `BrokerAdapter` interface defined in execution/broker_adapter.py.
- Venue not yet chosen — deciding between IBKR+SPY and Coinbase+BTC/USDC.
- `CoinbaseAdapter.connect()` and `get_quote()` are working against the live
  Coinbase Advanced Trade API, using a CDP production key loaded from
  `cdp_api_key.json`.

## Tech stack
- Python 3.12, venv
- pandas for data handling
- SQLite (storage/) for now, may move to Postgres later
- python-dotenv for config
- pytest for testing

Storage uses SQLite at DB_PATH (from .env, default storage/trading.db). Indexes exist on (symbol, timestamp) for all tables. Migration path to Postgres is via storage/db.py only — no other layer touches DB internals directly.
