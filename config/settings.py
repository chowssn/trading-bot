"""Centralized app settings, loaded from .env via python-dotenv."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "storage" / "trading.db"))
