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

# FRED series carrying the headline data point for each IMPORTANT_RELEASES
# release — used by eco_calendar.py to show the prior reading alongside an
# upcoming release date. Distinct from the release-*dates* calendar (which
# uses FRED's /fred/releases/dates endpoint and release *names*, not data
# series ids). ISM Manufacturing/Services have no entry — ISM is
# proprietary data FRED does not carry at all (same limitation documented
# on IMPORTANT_RELEASES' release-dates side in eco_calendar.py).
RELEASE_DATA_SERIES = {
    'Consumer Price Index':                            'CPIAUCSL',
    'Employment Situation':                            'PAYEMS',
    'Gross Domestic Product':                          'GDP',
    'Personal Income and Outlays':                     'PCEPI',
    'Producer Price Index':                            'PPIACO',
    'Retail Sales':                                    'RSAFS',
    'Industrial Production and Capacity Utilization':  'INDPRO',
    'Housing Starts':                                  'HOUST',
    'Consumer Sentiment':                               'UMCSENT',
    'Job Openings and Labor Turnover Survey':          'JTSJOL',
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
    'ABT': 'XLV',    # healthcare ETF
    'ADP': 'XLK',    # tech ETF
    'AMD': 'XLK',    # tech ETF
    'AMZN': 'XLY',    # consumer discretionary ETF
    'ANET': 'XLK',    # tech ETF
    'BSX': 'XLV',    # healthcare ETF
    'BYDDY': 'XLY',    # consumer discretionary ETF
    'CAT': 'XLI',    # industrials ETF
    'CTAS': 'XLI',    # industrials ETF
    'FCX': 'XLB',    # materials ETF
    'GOOGL': 'XLC',    # communication services ETF
    'HLI': 'XLF',    # financials ETF
    'JKHY': 'XLK',    # tech ETF
    'META': 'XLC',    # communication services ETF
    'MRK': 'XLV',    # healthcare ETF
    'NUE': 'XLB',    # materials ETF
    'NVDA': 'XLK',    # tech ETF
    'PLTR': 'XLK',    # tech ETF
    'PWR': 'XLI',    # industrials ETF
    'QBTS': 'XLK',    # tech ETF
    'RDDT': 'XLC',    # communication services ETF
    'TSLA': 'XLY',    # consumer discretionary ETF
    'TSM': 'XLK',    # tech ETF
    'VRT': 'XLI',    # industrials ETF
    'ZTS': 'XLV',    # healthcare ETF
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

# ============================================================
# TREASURY CURVE
# ============================================================
TREASURY_TICKERS = {
    '^IRX':  '3M',
    '^FVX':  '5Y',
    '^TNX':  '10Y',
    '^TYX':  '30Y',
}
# 2Y and 20Y from FRED (not available on yfinance)
TREASURY_FRED_SERIES = {
    'DGS2':  '2Y',
    'DGS20': '20Y',
}
# JGB 10Y from FRED — monthly, forward-filled
JGB_10Y_FRED_SERIES = 'IRLTLT01JPM156N'

TREASURY_SPREADS = {
    '3M10Y': ('3M', '10Y'),
    '2Y10Y': ('2Y', '10Y'),
    '2Y30Y': ('2Y', '30Y'),
}

# ============================================================
# FX PAIRS
# ============================================================
FX_TICKERS = {
    'EURUSD=X': 'EUR/USD',
    'USDJPY=X': 'USD/JPY',
    'GBPUSD=X': 'GBP/USD',
    'USDCHF=X': 'USD/CHF',
    'AUDUSD=X': 'AUD/USD',
    'USDCAD=X': 'USD/CAD',
    'USDCNH=X': 'USD/CNH',
    'EURJPY=X': 'EUR/JPY',
    'EURGBP=X': 'EUR/GBP',
}

# FX forwards via CIP derivation
FX_FORWARD_FOREIGN_RATES = {
    'EURUSD=X': {'1M': 'ECBDFR', '3M': 'IRLTLT01DEM156N', '1Y': 'IRLTLT01DEM156N'},
    'USDJPY=X': {'1M': 'IRSTCI01JPM156N', '3M': 'IRSTCI01JPM156N', '1Y': 'IRLTLT01JPM156N'},
    'GBPUSD=X': {'1M': 'IUDSOIA', '3M': 'IUDSOIA', '1Y': 'IRLTLT01GBM156N'},
    'USDCAD=X': {'1M': 'IRSTCI01CAM156N', '3M': 'IRSTCI01CAM156N', '1Y': 'IRSTCI01CAM156N'},
    'AUDUSD=X': {'1M': 'IRSTCI01AUM156N', '3M': 'IRSTCI01AUM156N', '1Y': 'IRSTCI01AUM156N'},
    'USDCHF=X': {'1M': 'IRSTCI01CHM156N', '3M': 'IRSTCI01CHM156N', '1Y': 'IRSTCI01CHM156N'},
}
FX_FORWARD_TENORS = ['1M', '3M', '6M', '1Y']
FX_TENOR_DAYS = {'1W': 7, '1M': 30, '3M': 90, '6M': 180, '1Y': 365}

# ============================================================
# EXTENDED COMMODITIES
# ============================================================
COMMODITY_TICKERS_EXTENDED = {
    'BZ=F':  'Brent Crude',
    'CL=F':  'WTI Crude',
    'NG=F':  'Natural Gas',
    'GC=F':  'Gold',
    'SI=F':  'Silver',
    'PL=F':  'Platinum',
    'HG=F':  'Copper (COMEX)',
    'URA':   'Uranium (URA ETF)',
}

# ============================================================
# MOVING AVERAGE AND EXTREMES HIGHLIGHTING
# Applied to: rates, spreads, FX, commodities, positions
# ============================================================
HIGHLIGHT_MA_PERIODS = [20, 50, 200]
HIGHLIGHT_MA_PROXIMITY_PCT = 1.0      # flag if within 1% of any SMA
HIGHLIGHT_EXTREMES_LOOKBACK = '5y'
HIGHLIGHT_EXTREMES_PCT = 2.0          # flag if within 2% of 5Y high or low

# ============================================================
# PERFORMANCE PERIODS
# ============================================================
PERFORMANCE_PERIODS = ['1D', '1W', '1M', '1Y', '3Y', '5Y']
# 3Y and 5Y displayed as annualized CAGR

# ============================================================
# KNOWN ECO RELEASE TIMES (ET)
# FRED does not provide release times — hardcoded for common releases
# ============================================================
KNOWN_RELEASE_TIMES = {
    'Employment Situation':                              '8:30 AM ET',
    'Consumer Price Index':                              '8:30 AM ET',
    'Personal Income and Outlays':                       '8:30 AM ET',
    'Producer Price Index':                              '8:30 AM ET',
    'Retail Sales':                                      '8:30 AM ET',
    'Gross Domestic Product':                            '8:30 AM ET',
    'Consumer Sentiment':                                '10:00 AM ET',
    'Job Openings and Labor Turnover Survey':            '10:00 AM ET',
    'Advance Monthly Sales for Retail and Food Services':'8:30 AM ET',
    'New Residential Construction':                      '8:30 AM ET',
    'Surveys of Consumers':                              '10:00 AM ET',
}

# ============================================================
# NEWS SOURCE CREDIBILITY TIERS
# ============================================================
NEWS_SOURCE_TIER1 = {
    'reuters', 'bloomberg', 'associated press', 'ap news',
    'wall street journal', 'wsj', 'financial times', 'ft',
    'new york times', 'nyt', 'cnbc', 'marketwatch',
    'barrons', "barron's", 'the economist', 'washington post',
}
NEWS_SOURCE_TIER2 = {
    'seeking alpha', 'benzinga', 'the fly', 'streetinsider',
    'globe newswire', 'pr newswire', 'business wire',
    'motley fool', "investor's business daily", 'ibd',
    'nuclear engineering international', 'defense news',
    'breaking defense', 'aviation week', 'the war zone',
    'uranium insider', 'world nuclear news',
}
NEWS_MAX_TIER3_PER_TICKER = 1
NEWS_HEADLINE_PAGE_SIZE = 10

# ============================================================
# REGIME-ADJUSTED RSI THRESHOLDS
# Applied in price_filter.py — RSI 30D threshold varies by regime.
# RSI_30D_MAX (settings.py) remains the default/fallback value used
# here when regime is unknown — not redeclared in this module to avoid
# a second source of truth for the same constant.
# ============================================================
RSI_30D_THRESHOLD_BY_REGIME = {
    'RISK_ON':      45,   # bull market — mild weakness is enough
    'NEUTRAL':      40,   # base case (default)
    'ELEVATED_VOL': 35,   # stress — only deeply oversold
    'HIGH_VOL':     30,   # crisis — maximum selectivity
}

# ============================================================
# CORRELATION GATE — entry check against existing positions
# ============================================================
CORRELATION_ENTRY_THRESHOLD = 0.70   # flag new candidate if correlated > this with any position
CORRELATION_LOOKBACK_DAYS = 60

# Extended correlation categories beyond price correlation
# Used to surface relationship context in screener output
CORRELATION_CATEGORIES = {
    # Style
    'value_etf':   'IVE',    # iShares S&P 500 Value ETF
    'growth_etf':  'IVW',    # iShares S&P 500 Growth ETF
    'em_etf':      'EEM',    # Emerging Markets
    # Sectors (same as POSITION_SECTOR_MAP sources)
    # Sub-sector relationships defined per-position in positions.py
    # under optional 'peer_tickers' field
}

# ============================================================
# VOLUME CONFIRMATION FOR ENTRY SIGNAL
# ============================================================
VOLUME_CONFIRMATION_DAYS = 5         # look back N days for volume analysis
VOLUME_UP_DOWN_RATIO_MIN = 1.1       # buying pressure: up-day volume / down-day volume

# ============================================================
# RETURN ZONE — 3M window additions
# DISLOCATION_ZONE_MIN/MAX (1Y return floor/ceiling) and RSI_30D_MAX
# already live in equity.config.settings — not redeclared here.
# ============================================================
RETURN_3M_SHARP_DROP = -15.0         # flag as recent sharp drop if 3M return < this
RETURN_3M_GRADUAL_GRIND = -5.0       # flag as slow grind if 1Y bad but 3M only slightly negative

# ============================================================
# POSITION TIER FRAMEWORK
# Developed: 2026-09
# Reformulation rule: thesis development may suggest reclassification — no position is locked
# ============================================================

POSITION_TIERS = {
    'core_compounder': {
        'label': 'Core — Compounder',
        'behavior': 'Buy dips, thesis-based exits, let run',
        'max_size_pct': 6.0,
        'min_size_pct': 4.0,
        'description': 'High-quality compounding businesses held for multi-year thesis delivery',
        'exit_rule': 'Thesis broken only — not price target',
    },
    'core_macro': {
        'label': 'Core — Macro/Structural',
        'behavior': 'Hold through volatility, defined thesis-breakers',
        'max_size_pct': 5.0,
        'min_size_pct': 3.0,
        'description': 'Positions driven by macro or structural thesis — regime-dependent',
        'exit_rule': 'Thesis-breaker triggered or regime shift',
    },
    'tactical': {
        'label': 'Tactical — Range Trade',
        'behavior': 'Active management, defined levels',
        'max_size_pct': 5.0,
        'min_size_pct': 2.0,
        'description': 'Range-bound or mean-reversion trades with defined entry/exit levels',
        'exit_rule': 'Price target or defined level breach',
    },
    'speculative_high': {
        'label': 'Speculative — High Conviction',
        'behavior': 'Sized for asymmetric upside',
        'max_size_pct': 3.0,
        'min_size_pct': 1.0,
        'description': 'High-conviction speculative positions with asymmetric payoff potential',
        'exit_rule': 'Thesis broken or target reached',
    },
    'speculative_exploratory': {
        'label': 'Speculative — Exploratory',
        'behavior': 'Small, documented, learning position',
        'max_size_pct': 1.5,
        'min_size_pct': 0.5,
        'description': 'Small exploratory positions to build knowledge before sizing up',
        'exit_rule': 'Discretionary — thesis development or stop-loss',
    },
}

# Secondary classification dimension
POSITION_STYLES = {
    'growth':    'Growth — earnings/revenue expansion primary driver',
    'value':     'Value — multiple expansion or mean reversion primary driver',
    'defensive': 'Defensive — capital preservation, low correlation to risk assets',
    'macro':     'Macro — regime/thematic driver dominates fundamentals',
    'commodity': 'Commodity — physical asset price is primary driver',
}

# Classification status for tracking thesis completion
CLASSIFICATION_STATUS = {
    'complete':   '✅ Classified + thesis documented',
    'needs_thesis': '✅ Classified, thesis needed',
    'tentative':  '⚠️ Classification tentative, thesis needed',
    'unclassified': '🔴 Unclassified, no thesis',
}

# Position sizing alerts — flag when position is outside tier bounds
POSITION_SIZE_ALERT_ENABLED = True
