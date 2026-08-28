"""Centralized settings for the equity screener, loaded from .env via python-dotenv."""

import os

from dotenv import load_dotenv

load_dotenv()

# Data sources
FMP_API_KEY: str = os.getenv("FMP_API_KEY", "")
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_USER_ID: int = int(os.getenv("TELEGRAM_USER_ID", "0"))

# Universe
#
# BlackRock's product-data API (reverse-engineered from the "Download
# holdings" button network call on the IWB product page) replaces the old
# ishares.com .ajax CSV endpoint, which now serves an HTML page instead of
# CSV from this environment.
#
# Deliberately no `asOfDate` param: holdings publish with a ~2 business-day
# lag, and passing today's (or even yesterday's) date returns a
# metadata-only stub — "Fund Holdings as of" blank and zero data rows,
# still HTTP 200 — rather than an error, so it fails silently. Confirmed
# 2026-08-28: asOfDate=20260828 and 20260827 both came back empty;
# asOfDate=20260826 and omitting the param entirely both returned the same
# real holdings. Omitting it lets BlackRock resolve "latest available"
# itself instead of us guessing the lag.
def universe_url() -> str:
    return (
        "https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document"
        "?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares&locale=en_US"
        "&portfolioId=239707&userType=individual&component=holdings"
    )


UNIVERSE_MIN_MARKET_CAP_B = 5.0  # $5B minimum
UNIVERSE_MIN_AVG_VOLUME = 500_000  # 500K shares/day

# Screener thresholds
ROIC_TIER1 = 25.0  # >25% exceptional
ROIC_TIER2 = 20.0  # 20-25% excellent
ROIC_TIER3 = 15.0  # 15-20% good
ROIC_MIN = 10.0  # <10% generally excluded

DISLOCATION_ZONE_MIN = -10.0  # 1Y return floor for interesting zone
DISLOCATION_ZONE_MAX = -50.0  # 1Y return ceiling (beyond this investigate hard)
RSI_30D_MAX = 40.0  # RSI 30D threshold for sustained weakness

NET_DEBT_EBITDA_EXCELLENT = 1.0
NET_DEBT_EBITDA_GOOD = 2.0
NET_DEBT_EBITDA_FLAG = 3.0
NET_DEBT_EBITDA_EXCLUDE = 4.0

MIN_QUALITY_SCORE = 50  # minimum score to appear in output
MAX_SCREENER_OUTPUT = 20  # top N names in Telegram output

# Cache
CACHE_DIR = "equity/data/cache"
UNIVERSE_CACHE_HOURS = 24
PRICE_CACHE_HOURS = 1  # refresh intraday
FUNDAMENTAL_CACHE_HOURS = 24
