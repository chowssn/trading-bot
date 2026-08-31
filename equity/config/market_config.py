# ============================================================
# MARKET BRIEF CONFIGURATION
# Last reviewed: 2026-08
# All thresholds documented with rationale.
# Update via config_manager.update_market_config() to ensure
# changes are tracked in git and changelog.md.
# ============================================================

# EQUITY BENCHMARKS
EQUITY_BENCHMARKS = {
    'SPY': 'S&P 500',
    'QQQ': 'Nasdaq 100',
    'IWM': 'Russell 2000',  # small cap — useful divergence signal vs large cap
}

# VOLATILITY THRESHOLDS
# Based on historical VIX distribution — 20 is roughly 60th percentile (elevated but not panic)
VIX_ELEVATED = 20
VIX_HIGH = 30       # roughly 85th percentile historically
VIX_EXTREME = 40    # crisis level — 2020 COVID peak was ~82, 2008 peak was ~80

# RATES
RATE_TICKERS = {
    '^TNX': '10Y Treasury',
    '^IRX': '3M Treasury',
}
YIELD_CURVE_INVERTED_THRESHOLD = 0.0  # 2s10s below 0 = inverted

# COMMODITIES AND FX
COMMODITY_TICKERS = {
    'DX-Y.NYB': 'DXY Dollar Index',
    'GC=F': 'Gold',
    'CL=F': 'Crude Oil',
    'HG=F': 'Copper',  # global growth proxy
}

# REGIME DETECTION
# Thresholds for daily regime flag classification in morning brief
# Review after major macro regime shifts
REGIME_RULES = {
    'RISK_OFF_DAY':       {'spy_change_1d_lt': -1.5},
    'RISK_ON_DAY':        {'spy_change_1d_gt': 1.5},
    'ELEVATED_VOL':       {'vix_gt': VIX_ELEVATED},
    'HIGH_VOL':           {'vix_gt': VIX_HIGH},
    'TECH_UNDERPERFORM':  {'qqq_vs_spy_lt': -0.5},  # QQQ lagging SPY by 0.5%+ = value rotation
    'TECH_OUTPERFORM':    {'qqq_vs_spy_gt': 0.5},
    'DOLLAR_STRENGTH':    {'dxy_change_gt': 0.5},   # DXY up 0.5%+ = headwind for risk assets
    'DOLLAR_WEAKNESS':    {'dxy_change_lt': -0.5},
    'RATES_RISING':       {'yield_10y_change_bps_gt': 5},
    'RATES_FALLING':      {'yield_10y_change_bps_lt': -5},
}

# ECONOMIC CALENDAR
IMPORTANT_RELEASES = {
    'Consumer Price Index':                              ('CPI', 'HIGH'),
    'Employment Situation':                              ('NFP/Jobs', 'HIGH'),
    'Gross Domestic Product':                           ('GDP', 'HIGH'),
    'Personal Income and Outlays':                      ('PCE/Income', 'HIGH'),
    'Producer Price Index':                             ('PPI', 'MEDIUM'),
    'Retail Sales':                                     ('Retail Sales', 'MEDIUM'),
    'Industrial Production and Capacity Utilization':   ('Industrial Production', 'MEDIUM'),
    'Housing Starts':                                   ('Housing Starts', 'LOW'),
    'Consumer Sentiment':                               ('UMich Sentiment', 'MEDIUM'),
    'Job Openings and Labor Turnover Survey':           ('JOLTS', 'MEDIUM'),
    'ISM Manufacturing PMI':                            ('ISM Manufacturing', 'MEDIUM'),
    'ISM Services PMI':                                 ('ISM Services', 'MEDIUM'),
}

# FOMC CALENDAR
# Auto-fetched from Fed website at runtime (see eco_calendar.py).
# Hardcoded list below is the fallback if fetch fails.
# FOMC_DATES_VALID_THROUGH: when hardcoded list expires.
# System warns 60 days before expiry if no auto-fetch available.
FOMC_DATES_VALID_THROUGH = '2027-12-31'
FOMC_DATES_FALLBACK = ['2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17', '2026-07-29', '2026-09-16', '2026-10-28', '2026-12-09']
FOMC_PROXIMITY_DAYS = 2   # flag if FOMC within this many days
FOMC_FETCH_URL = 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'
FOMC_CACHE_DAYS = 30      # re-fetch from Fed website every 30 days

# SCREENER REGIME ADJUSTMENTS
REGIME_SCREENER_ADJUSTMENTS = {
    'HIGH_VOL':     {'min_quality_score_override': 60, 'note': 'High vol — higher conviction bar'},
    'ELEVATED_VOL': {'min_quality_score_override': 55, 'note': 'Elevated vol — slightly higher bar'},
    'RISK_OFF_DAY': {'note': 'Risk-off day — review all WATCH signals before acting'},
}

# PORTFOLIO MONITOR THRESHOLDS
LARGE_MOVE_THRESHOLD_PCT = 3.0    # flag LARGE_UP / LARGE_DOWN
MEDIUM_MOVE_THRESHOLD_PCT = 1.0   # flag UP / DOWN

# NEWS TRIAGE
MAX_HEADLINES_PER_TICKER = 3
NEWS_KEYWORD_MIN_MATCHES = 2      # minimum keyword matches to flag thesis_breaker
NEWS_IGNORE_WORDS = {             # common words excluded from keyword matching
    'the', 'a', 'an', 'is', 'are', 'was', 'has', 'have',
    'for', 'in', 'of', 'to', 'and', 'or', 'with', 'on',
    'at', 'by', 'from', 'as', 'its', 'this', 'that', 'it',
    'be', 'been', 'will', 'would', 'could', 'may', 'might',
}

# ============================================================
# BENCHMARKS AND SECTOR ETFS
# ============================================================
BENCHMARK_TICKERS = {
    # Broad market
    'SPY':  'S&P 500',
    'QQQ':  'Nasdaq 100',
    'IWM':  'Russell 2000',
    'EEM':  'Emerging Markets',
    'EFA':  'Developed International (EAFE)',
    # Sectors
    'XLK':  'Technology',
    'XLB':  'Basic Materials',
    'XLF':  'Financials',
    'XLE':  'Energy',
    'XLI':  'Industrials',
    'TLT':  'Long Treasury / Government',
    'XLY':  'Consumer Cyclical',
    'XLC':  'Communication Services',
    'XLRE': 'Real Estate',
    'XLV':  'Healthcare',
    'XLU':  'Utilities',
    'BIL':  'Cash (3M T-Bill proxy)',
    # Specialist
    'URA':  'Uranium / Nuclear',
    # Factors and commodities
    'GLD':  'Gold',
    'SLV':  'Silver',
    'CPER': 'Copper (COMEX — LME proxy)',
    'SHY':  'Short Treasury (1-3Y)',
    'IEF':  'Intermediate Treasury (7-10Y)',
}

# Position-to-sector ETF mapping for relative performance
POSITION_SECTOR_MAP = {
    'CCJ':  'URA',    # uranium ETF
    'CEG':  'XLU',    # utilities ETF
    'MSFT': 'XLK',    # tech ETF
    'PGR':  'XLF',    # financials ETF
    'UMAC': 'XLI',    # industrials ETF
    'APP':  'XLC',    # communication services ETF
}

# SLV/GLD ratio — risk-on indicator
# Rising ratio = silver outperforming gold = risk-on sentiment
# (silver has more industrial demand than gold)
SILVER_GOLD_RATIO_RISK_ON_THRESHOLD = 0.0  # positive change = risk-on signal

# PERFORMANCE / EARNINGS / SECTOR MONITOR THRESHOLDS
POSITION_UNDERPERFORM_ALERT_PCT = 2.0  # flag if position lags sector by this much
CORRELATION_CONCENTRATION_THRESHOLD = 0.70  # flag correlated positions
CORRELATION_HEDGE_THRESHOLD = -0.30  # flag natural hedges
SECTOR_CONCENTRATION_MAX_PCT = 40    # flag if one sector > this % of portfolio
EARNINGS_LOOKAHEAD_DAYS = 7          # days ahead to show earnings
EARNINGS_ALERT_DAYS = 2              # days before earnings to send alert

# Precious metals futures tickers (yfinance front-month)
GOLD_FUTURES_TICKER = 'GC=F'    # COMEX gold front-month, USD/troy oz
SILVER_FUTURES_TICKER = 'SI=F'  # COMEX silver front-month, USD/troy oz

# SLV holds ~1 troy oz of silver per share, so unlike GLD (~0.1oz/share,
# intentionally ~10x below GC=F) its ETF price should track SI=F closely.
# In practice a several-% gap is routine (SLV's cash-market close vs.
# SI=F's front-month futures basis, plus trust expense drag) — e.g. SLV
# $60.0 vs. SI=F $66.7 (~10%) is not itself a problem. Flag the SLV
# commodities line only past this, wider, threshold — signals a data
# problem (wrong ticker, stale price, decimal error), not a level worth
# reading into.
SLV_SI_DIVERGENCE_ALERT_PCT = 15.0
# Note: FRED does not carry gold/silver spot series (removed ~2015)
# Front-month futures are the standard market reference price
# Same approach as copper: HG=F (COMEX front month) via yfinance
# LME copper monthly available via FRED PCOPPUSDM but too stale for daily brief
