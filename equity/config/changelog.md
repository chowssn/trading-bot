# Config Changelog

## 2026-08-28 — Initial Setup
- positions.py: Initial positions loaded (CCJ, CEG, MSFT, UMAC, PGR)
- positions.py: APP added to watchlist (limit orders pending)
- market_config.py: Initial regime thresholds established

## 2026-08-30
- market_config.py: FOMC_DATES_FALLBACK updated to ['2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17', '2026-07-29', '2026-09-16', '2026-10-28', '2026-12-09'] — Fix 2026-05-06 → 2026-04-29 per live Fed website scrape
- market_config.py: FOMC_DATES_VALID_THROUGH updated to '2027-12-31' — Extended through 2027 — live Fed scrape confirmed 2027 dates available

## 2026-09-01
- positions.py: ONON added to watchlist — Added via Telegram advisor

## 2026-09-04
- positions.py: ONON removed from watchlist — Removed via Telegram
- positions.py: SMFG added to watchlist — Added via Telegram advisor
- positions.py: ONON added to watchlist — Added via Telegram advisor
- positions.py: ONON removed from watchlist — Removed via Telegram

## 2026-09-05
- positions.py: CDNS added to watchlist — Added via Telegram advisor
- positions.py: Updated MSFT thesis — Updated from Telegram discussion 2026-09-05
- positions.py: Updated MSFT thesis — Classified from Telegram discussion 2026-09-05
