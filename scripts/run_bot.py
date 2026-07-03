"""Main entry point for running the trading bot's live polling loop."""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from execution.coinbase_adapter import CoinbaseAdapter
from orchestrator.loop import TradingLoop
from risk.risk_manager import RiskManager
from signals.sma_crossover import SMACrossoverSignal
from storage.db import Database

load_dotenv()

LOGS_DIR = Path(__file__).resolve().parents[1] / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "trading.log"),
    ],
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Wire up the bot's components and run the trading loop."""
    db = Database()
    adapter = CoinbaseAdapter(paper_mode=True)
    adapter.connect()
    risk_manager = RiskManager()
    signal = SMACrossoverSignal(period=20)

    loop = TradingLoop(
        adapter,
        risk_manager,
        signal,
        db,
        poll_interval_seconds=60,
        order_size_usd=10.0,
    )

    logger.info("Bot starting up — paper mode enabled")
    loop.run()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error during bot startup")
        sys.exit(1)
