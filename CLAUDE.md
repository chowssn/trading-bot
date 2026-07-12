# Trading Bot — Project Context

## Purpose
Personal algorithmic trading system. Starts with one venue/instrument,
designed to scale to multiple brokers/exchanges across TradFi and DeFi,
and potentially to enterprise/institutional use later.

The objective is not to predict market moves but to deploy capital where
multiple independent evidence sources align, while preserving capital and
optionality for high-conviction opportunities. The system follows a
barbell architecture: a regime engine classifies the macro environment and
dynamically adjusts exposure; a convex sleeve deploys into liquid momentum
opportunities (BTC-USDC initially) only when technical, macro, and regime
layers independently agree. Decision quality over prediction frequency.
Position size scales with conviction. When evidence is ambiguous, the
correct output is flat.

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

## Data Maintenance
ETF flow data is sourced from a manually downloaded SoSoValue CSV at
`backtest/data/btc_etf_flows_sosovalue.csv`. This file does not auto-update.
Re-download periodically (monthly or when data appears stale) from
sosovalue.com/assets/etf/us-btc-spot — export the total net flow history,
replace the existing file, and delete `backtest/data/btc_etf_flows.csv` to
force cache refresh on next run.

yfinance shares outstanding for BTC ETFs (IBIT, FBTC, ARKB, BITB, GBTC)
currently returns None from Yahoo Finance — the yfinance extension path is
implemented but contributes zero rows until Yahoo fixes this. The SoSoValue
seed is the sole data source until then.

## Tech stack
- Python 3.12, venv
- pandas for data handling
- SQLite (storage/) for now, may move to Postgres later
- python-dotenv for config
- pytest for testing

Storage uses SQLite at DB_PATH (from .env, default storage/trading.db). Indexes exist on (symbol, timestamp) for all tables. Migration path to Postgres is via storage/db.py only — no other layer touches DB internals directly.
