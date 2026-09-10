"""Tests for equity/telegram/bot.py's intraday macro alert logic (Treasury
yields, FX, commodities, volatility, crypto, international indices, and
cross-asset ratios) — the alert-side half of the macro monitoring
extension. The advisor-side half (get_live_macro_snapshot) is covered in
test_advisor_macro.py.

`_check_macro_alerts()` reads the shared `equity.data.price_cache`
singleton — these tests patch its `get`/`get_yield` methods rather than
hitting real yfinance/FRED, and reset `bot._alerted_today` between tests
since it's shared, dedup-tracking module state.
"""

import asyncio
import time
import unittest
from unittest.mock import patch

from equity.telegram import bot


class TestLevelCrossed(unittest.TestCase):
    def test_crossed_upward(self):
        self.assertTrue(bot._level_crossed(prev=4.45, curr=4.55, level=4.5))

    def test_crossed_downward(self):
        self.assertTrue(bot._level_crossed(prev=4.55, curr=4.45, level=4.5))

    def test_not_crossed(self):
        self.assertFalse(bot._level_crossed(prev=4.40, curr=4.48, level=4.5))

    def test_landing_exactly_on_level_counts_as_crossed(self):
        self.assertTrue(bot._level_crossed(prev=4.45, curr=4.50, level=4.5))


class TestFxContextNote(unittest.TestCase):
    def test_known_pair_positive(self):
        self.assertIn("EUR strengthening", bot._fx_context_note("EURUSD=X", 0.6))

    def test_known_pair_negative(self):
        self.assertIn("JPY strengthening", bot._fx_context_note("USDJPY=X", -0.6))

    def test_unknown_pair_falls_back(self):
        self.assertIn("check commodity and EM exposure", bot._fx_context_note("AUDUSD=X", 0.6))


class TestCommodityContextNote(unittest.TestCase):
    def test_known_commodity(self):
        self.assertIn("Gold surging", bot._commodity_context_note("GC=F", 2.0))

    def test_unknown_commodity_falls_back(self):
        self.assertIn("check portfolio exposure", bot._commodity_context_note("PL=F", 2.0))


# Keyed by whatever price_cache.get()/get_yield() would be called with:
# tenor labels for yields (get_yield resolves the ^ticker/FRED-series
# mapping itself, tested separately on PriceCache), raw tickers for
# everything else. Crafted to trip: a >8bp 10Y move + a 4.5% level
# breach, a >0.5% EUR/USD move, a sub-threshold DXY move (should NOT
# alert), a >1.5% gold move, a VIX point move AND a VIX/VVIX ratio above
# 0.30, a >4% BTC move, and a >1.5% Nikkei move.
_FAKE_CACHE = {
    "10Y": {"price": 4.58, "prev_close": 4.49, "change_1d_bps": 9.0, "is_yield": True},
    "EURUSD=X": {"price": 1.09, "prev_close": 1.0827, "change_1d_pct": 0.7, "is_yield": False},
    "DX-Y.NYB": {"price": 104.0, "prev_close": 103.9, "change_1d_pct": 0.1, "is_yield": False},
    "GC=F": {"price": 2700.0, "prev_close": 2647.0, "change_1d_pct": 2.0, "is_yield": False},
    "^VIX": {"price": 30.0, "prev_close": 25.0, "change_1d_pct": 20.0, "is_yield": False},
    "^VVIX": {"price": 95.0, "prev_close": 90.0, "change_1d_pct": 5.5, "is_yield": False},
    "BTC-USD": {"price": 90000.0, "prev_close": 83000.0, "change_1d_pct": 8.4, "is_yield": False},
    "^N225": {"price": 40000.0, "prev_close": 39000.0, "change_1d_pct": 2.5, "is_yield": False},
}


def _fake_get(ticker):
    return _FAKE_CACHE.get(ticker)


def _fake_get_yield(tenor):
    return _FAKE_CACHE.get(tenor)


def _patched_cache():
    return (
        patch.object(bot.price_cache, "get", side_effect=_fake_get),
        patch.object(bot.price_cache, "get_yield", side_effect=_fake_get_yield),
    )


class TestCheckMacroAlerts(unittest.TestCase):
    def setUp(self):
        bot._alerted_today.clear()
        self._patches = _patched_cache()
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_detects_yield_move_and_level_breach(self):
        types = {a["type"] for a in bot._check_macro_alerts("2026-09-06")}
        self.assertIn("macro_yield", types)
        self.assertIn("macro_yield_level", types)

    def test_detects_fx_move_above_threshold(self):
        fx_alerts = [a for a in bot._check_macro_alerts("2026-09-06") if a["type"] == "macro_fx"]
        self.assertEqual(len(fx_alerts), 1)
        self.assertEqual(fx_alerts[0]["ticker"], "EUR/USD")

    def test_dxy_below_threshold_not_alerted(self):
        alerts = bot._check_macro_alerts("2026-09-06")
        self.assertFalse(any(a["ticker"] == "DXY Dollar Index" for a in alerts))

    def test_detects_commodity_move(self):
        comm_alerts = [a for a in bot._check_macro_alerts("2026-09-06") if a["type"] == "macro_commodity"]
        self.assertEqual(len(comm_alerts), 1)
        self.assertEqual(comm_alerts[0]["ticker"], "Gold")

    def test_detects_volatility_move(self):
        # Both ^VIX (+5pts) and ^VVIX (+5pts) clear their respective
        # VOLATILITY_ALERT thresholds (3.0 / 5.0) in the fixture above.
        vol_alerts = [a for a in bot._check_macro_alerts("2026-09-06") if a["type"] == "volatility"]
        self.assertEqual({a["ticker"] for a in vol_alerts}, {"VIX (S&P 500 vol)", "VVIX (Vol of Vol)"})

    def test_detects_crypto_move(self):
        crypto_alerts = [a for a in bot._check_macro_alerts("2026-09-06") if a["type"] == "crypto"]
        self.assertEqual(len(crypto_alerts), 1)
        self.assertEqual(crypto_alerts[0]["ticker"], "Bitcoin (XBT)")

    def test_detects_international_move(self):
        intl_alerts = [a for a in bot._check_macro_alerts("2026-09-06") if a["type"] == "international"]
        self.assertEqual(len(intl_alerts), 1)
        self.assertEqual(intl_alerts[0]["ticker"], "Nikkei 225")

    def test_detects_vix_vvix_ratio_alert(self):
        ratio_alerts = [a for a in bot._check_macro_alerts("2026-09-06") if a["type"] == "ratio"]
        self.assertTrue(any("VIX/VVIX" in a["ticker"] for a in ratio_alerts))

    def test_every_alert_has_a_magnitude(self):
        for alert in bot._check_macro_alerts("2026-09-06"):
            self.assertIn("magnitude", alert)

    def test_dedup_skips_already_alerted_key(self):
        today = "2026-09-06"
        bot._alerted_today[f"yield_10Y_{today}"] = True
        alerts = bot._check_macro_alerts(today)
        self.assertFalse(any(a["type"] == "macro_yield" for a in alerts))


class TestShouldEnrich(unittest.TestCase):
    def test_zero_threshold_always_enriches(self):
        self.assertTrue(bot._should_enrich({"type": "news", "magnitude": 0}))

    def test_below_threshold_not_enriched(self):
        self.assertFalse(bot._should_enrich({"type": "price", "magnitude": 2.0}))

    def test_above_threshold_enriched(self):
        self.assertTrue(bot._should_enrich({"type": "price", "magnitude": 6.0}))

    def test_unmapped_type_never_enriched(self):
        self.assertFalse(bot._should_enrich({"type": "unknown_type", "magnitude": 999}))


class TestBuildEnrichmentPrompt(unittest.TestCase):
    def test_known_types_produce_a_prompt(self):
        for alert_type in (
            "price", "news", "macro_yield", "macro_yield_level", "macro_fx",
            "macro_commodity", "volatility", "crypto", "international",
        ):
            prompt = bot._build_enrichment_prompt({"type": alert_type, "ticker": "SPY", "message": "moved"})
            self.assertIsNotNone(prompt, f"{alert_type} should produce a prompt")

    def test_ratio_type_has_no_prompt(self):
        self.assertIsNone(bot._build_enrichment_prompt({"type": "ratio", "ticker": "x", "message": "y"}))


class TestIsAlertDataFresh(unittest.TestCase):
    def test_no_raw_ticker_passes_through(self):
        # News/ratio alerts don't carry raw_ticker — staleness of some
        # price reading isn't relevant to whether they should fire.
        self.assertTrue(bot._is_alert_data_fresh({"type": "news", "message": "x"}))

    def test_missing_from_cache_is_not_fresh(self):
        with patch.object(bot.price_cache, "get", return_value=None):
            self.assertFalse(bot._is_alert_data_fresh({"raw_ticker": "NOPE"}))

    def test_stale_entry_is_not_fresh(self):
        with patch.object(bot.price_cache, "get", return_value={"is_stale": True, "session_label": "3d ago"}):
            self.assertFalse(bot._is_alert_data_fresh({"raw_ticker": "SPY"}))

    def test_fresh_entry_passes(self):
        with patch.object(bot.price_cache, "get", return_value={"is_stale": False}):
            self.assertTrue(bot._is_alert_data_fresh({"raw_ticker": "SPY"}))

    def test_intraday_lag_within_alert_threshold_passes(self):
        # equity max is 4h — 10min lag is well within it.
        data = {"instrument_type": "equity", "data_lag_minutes": 10.0, "is_stale": False}
        with patch.object(bot.price_cache, "get", return_value=data):
            self.assertTrue(bot._is_alert_data_fresh({"raw_ticker": "MSFT"}))

    def test_intraday_lag_beyond_alert_threshold_fails(self):
        # crypto max is 1h — 90min lag exceeds it, even though this would
        # be well within price_cache's own (much wider) display threshold.
        data = {"instrument_type": "crypto", "data_lag_minutes": 90.0, "is_stale": False}
        with patch.object(bot.price_cache, "get", return_value=data):
            self.assertFalse(bot._is_alert_data_fresh({"raw_ticker": "BTC-USD"}))

    def test_daily_bar_within_alert_threshold_passes(self):
        # yield max is 6h.
        data = {"instrument_type": "yield", "data_lag_minutes": None, "data_age_hours": 2.0, "is_stale": False}
        with patch.object(bot.price_cache, "get", return_value=data):
            self.assertTrue(bot._is_alert_data_fresh({"raw_ticker": "^TNX"}))

    def test_daily_bar_beyond_alert_threshold_fails_even_when_not_display_stale(self):
        # A weekend-gap equity bar can be well within price_cache's 75h
        # display-stale threshold (is_stale=False) but still far too old
        # for an alert — this is the whole point of ALERT_MAX_AGE_HOURS
        # being tighter than is_stale.
        data = {"instrument_type": "equity", "data_lag_minutes": None, "data_age_hours": 20.0, "is_stale": False}
        with patch.object(bot.price_cache, "get", return_value=data):
            self.assertFalse(bot._is_alert_data_fresh({"raw_ticker": "MSFT"}))

    def test_unknown_instrument_type_uses_default_threshold(self):
        data = {"instrument_type": "something_new", "data_lag_minutes": 300.0, "is_stale": False}
        with patch.object(bot.price_cache, "get", return_value=data):
            self.assertFalse(bot._is_alert_data_fresh({"raw_ticker": "X"}))  # 5h > default 4h


class TestDataDateKey(unittest.TestCase):
    def test_uses_last_update_date_portion(self):
        entry = {"last_update": "2026-09-05 16:00 UTC"}
        self.assertEqual(bot._data_date_key(entry, fallback="2026-09-08"), "2026-09-05")

    def test_none_entry_falls_back(self):
        self.assertEqual(bot._data_date_key(None, fallback="2026-09-08"), "2026-09-08")

    def test_missing_last_update_falls_back(self):
        self.assertEqual(bot._data_date_key({}, fallback="2026-09-08"), "2026-09-08")

    def test_does_not_use_exact_timestamp_daily_bar_form(self):
        # exact_timestamp for a daily bar is "Weekday YYYY-MM-DD close" —
        # slicing that instead of last_update would silently collide
        # same-weekday bars a week apart. Confirm last_update wins even
        # when exact_timestamp is also present in the entry.
        entry = {"last_update": "2026-09-09 00:00 UTC", "exact_timestamp": "Wed 2026-09-09 close"}
        self.assertEqual(bot._data_date_key(entry, fallback="x"), "2026-09-09")


class TestAlertStartupGracePeriod(unittest.TestCase):
    def test_job_skips_during_grace_period(self):
        with patch.object(bot, "_BOT_START_TIME", time.time()):
            with patch.object(bot, "_get_market_hours_now", return_value=True) as mock_hours:
                asyncio.run(bot.intraday_alert_job(context=None))
        mock_hours.assert_not_called()  # returned before even checking market hours

    def test_job_proceeds_past_grace_period(self):
        with patch.object(bot, "_BOT_START_TIME", time.time() - bot.ALERT_STARTUP_GRACE_SECONDS - 1):
            with patch.object(bot, "_get_market_hours_now", return_value=False) as mock_hours:
                asyncio.run(bot.intraday_alert_job(context=None))
        mock_hours.assert_called_once()


class TestFormatPriceLine(unittest.TestCase):
    def test_basic_line_shape(self):
        line = bot._format_price_line("Gold", {"price": 4476.60, "change_1d_pct": 1.06, "is_stale": False})
        self.assertIn("Gold", line)
        self.assertIn("+1.06%", line)
        self.assertIn("🟢", line)

    def test_stale_flag_shown(self):
        line = bot._format_price_line("ES Futures", {"price": 5720.0, "change_1d_pct": 0.0, "is_stale": True})
        self.assertIn("⚠️ STALE", line)

    def test_session_suffix_included_by_default(self):
        data = {"price": 100.0, "change_1d_pct": 0.5, "session_label": "14h ago", "condition_note": "Fri 16:00 ET"}
        line = bot._format_price_line("SPY", data)
        self.assertIn("14h ago", line)
        self.assertIn("Fri 16:00 ET", line)

    def test_session_suffix_omitted_when_disabled(self):
        data = {"price": 100.0, "change_1d_pct": 0.5, "session_label": "14h ago"}
        line = bot._format_price_line("SPY", data, show_session=False)
        self.assertNotIn("14h ago", line)

    def test_negative_change_is_red(self):
        line = bot._format_price_line("QQQ", {"price": 500.0, "change_1d_pct": -1.2})
        self.assertIn("🔴", line)

    def test_intraday_lag_shown_with_exact_timestamp(self):
        data = {
            "price": 420.0, "change_1d_pct": 0.3,
            "data_lag_minutes": 3.0, "exact_timestamp": "2026-09-07 14:35 ET",
        }
        line = bot._format_price_line("MSFT", data)
        self.assertIn("3min ago", line)
        self.assertIn("2026-09-07 14:35 ET", line)

    def test_daily_bar_shows_exact_timestamp_alone(self):
        data = {"price": 4476.60, "change_1d_pct": 1.06, "exact_timestamp": "Fri 2026-09-04 close"}
        line = bot._format_price_line("Gold", data)
        self.assertIn("Fri 2026-09-04 close", line)

    def test_prefers_official_close_over_raw_price(self):
        # After hours, `price` becomes the AH print — the headline number
        # must stay the regular-session close, not that AH print.
        data = {"price": 366.42, "official_close": 368.16, "change_1d_pct": -0.05}
        line = bot._format_price_line("TSLA", data)
        self.assertIn("368.16", line)
        self.assertNotIn("366.42 (", line)  # not as the headline price

    def test_afterhours_price_appended_when_present(self):
        data = {
            "price": 366.42, "official_close": 368.16, "afterhours_price": 366.42,
            "session_type": "afterhours", "change_1d_pct": -0.05,
        }
        line = bot._format_price_line("TSLA", data)
        self.assertIn("AH: 366.42", line)
        self.assertIn("-0.47%", line)  # 366.42/368.16 - 1

    def test_premarket_price_labeled_pm_not_ah(self):
        data = {
            "price": 370.00, "official_close": 368.16, "afterhours_price": 370.00,
            "session_type": "premarket", "change_1d_pct": -0.05,
        }
        line = bot._format_price_line("TSLA", data)
        self.assertIn("PM: 370", line)
        self.assertNotIn("AH:", line)

    def test_no_afterhours_line_without_extended_price(self):
        data = {"price": 368.16, "official_close": 368.16, "change_1d_pct": -0.05}
        line = bot._format_price_line("TSLA", data)
        self.assertNotIn("AH:", line)
        self.assertNotIn("PM:", line)

    def test_afterhours_line_omitted_when_flat(self):
        # AH print within a cent of the close isn't worth its own line.
        data = {
            "price": 368.17, "official_close": 368.16, "afterhours_price": 368.17,
            "session_type": "afterhours", "change_1d_pct": 0.0,
        }
        line = bot._format_price_line("TSLA", data)
        self.assertNotIn("AH:", line)

    def test_no_official_close_falls_back_to_price(self):
        # Crypto/futures never get official_close — must still render.
        data = {"price": 90123.45, "change_1d_pct": 1.5}
        line = bot._format_price_line("BTC-USD", data)
        self.assertIn("90,123", line)


class TestFormatAlertMessage(unittest.TestCase):
    """_format_alert_message() — the no-web-search, price_cache-only alert
    formatter that replaced _enrich_alert_with_context() in the alert send
    loop (see intraday_alert_job())."""

    def test_includes_base_message(self):
        alert = {"ticker": "TSLA", "raw_ticker": "TSLA", "message": "🚀 *TSLA* +5.0%"}
        with patch.object(bot.price_cache, "get", return_value=None):
            msg = bot._format_alert_message(alert)
        self.assertIn("🚀 *TSLA* +5.0%", msg)

    def test_recent_intraday_tick_shows_minute_lag(self):
        alert = {"ticker": "TSLA", "raw_ticker": "TSLA", "message": "move"}
        data = {"exact_timestamp": "2026-09-07 14:35 ET", "data_lag_minutes": 3.0, "instrument_type": "equity"}
        with patch.object(bot.price_cache, "get", return_value=data):
            msg = bot._format_alert_message(alert)
        self.assertIn("2026-09-07 14:35 ET", msg)
        self.assertIn("equity", msg)

    def test_daily_bar_shows_timestamp_without_lag(self):
        alert = {"ticker": "GLD", "raw_ticker": "GC=F", "message": "move"}
        data = {"exact_timestamp": "Fri 2026-09-04 close", "data_lag_minutes": None, "instrument_type": "futures"}
        with patch.object(bot.price_cache, "get", return_value=data):
            msg = bot._format_alert_message(alert)
        self.assertIn("Fri 2026-09-04 close", msg)
        self.assertNotIn("min ago)", msg)

    def test_missing_price_cache_entry_falls_back_to_bare_message(self):
        alert = {"ticker": "DXY", "raw_ticker": "DX-Y.NYB", "message": "📈 *DXY* +0.5%"}
        with patch.object(bot.price_cache, "get", return_value=None):
            msg = bot._format_alert_message(alert)
        self.assertEqual(msg, "📈 *DXY* +0.5%")

    def test_no_raw_ticker_uses_display_ticker_or_skips_lookup(self):
        # ratio/news alerts carry no raw_ticker — must not blow up.
        alert = {"ticker": "Copper/Gold ratio", "message": "ratio move"}
        with patch.object(bot.price_cache, "get", return_value=None):
            msg = bot._format_alert_message(alert)
        self.assertIn("ratio move", msg)


class TestRecentAlertHistory(unittest.TestCase):
    def setUp(self):
        bot._recent_alerts.clear()

    def tearDown(self):
        bot._recent_alerts.clear()

    def test_record_alert_appends_and_context_reflects_it(self):
        bot._record_alert({"ticker": "TSLA", "type": "price", "message": "🚀 *TSLA* +5.0%"})
        ctx = bot._get_recent_alert_context()
        self.assertIn("TSLA", ctx)

    def test_no_alerts_yields_empty_context(self):
        self.assertEqual(bot._get_recent_alert_context(), "")

    def test_history_caps_at_max_recent_alerts(self):
        for i in range(bot.MAX_RECENT_ALERTS + 5):
            bot._record_alert({"ticker": f"T{i}", "type": "price", "message": "x"})
        self.assertEqual(len(bot._recent_alerts), bot.MAX_RECENT_ALERTS)
        # Oldest entries evicted first.
        self.assertEqual(bot._recent_alerts[0]["ticker"], "T5")


class TestRestartKillHandlersRegistered(unittest.TestCase):
    """send_restart/send_kill must go through the same
    require_email_auth -> authorized_only -> register_command stack as the
    other write handlers (send_add etc.) — see the comment above that
    block in bot.py. Missing @register_command means the post-auth
    re-dispatch in handle_message() silently no-ops after the user enters
    the 2FA code.
    """

    def test_both_registered_for_post_auth_dispatch(self):
        self.assertIn("send_restart", bot.COMMAND_REGISTRY)
        self.assertIn("send_kill", bot.COMMAND_REGISTRY)

    def test_both_in_known_commands(self):
        self.assertIn("restart", bot.KNOWN_COMMANDS)
        self.assertIn("kill", bot.KNOWN_COMMANDS)


if __name__ == "__main__":
    unittest.main()
