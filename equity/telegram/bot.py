"""Telegram portfolio advisor bot — main entry point.

Run with: python -m equity.telegram.bot
Before the first run, always: python equity/telegram/security_check.py
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from functools import partial

import yfinance as yf
from dotenv import load_dotenv
from telegram import Update
from telegram.error import BadRequest

load_dotenv()

# ---------------------------------------------------------------------------
# Startup security check — fail loud if any required var missing
# ---------------------------------------------------------------------------

required_vars = [
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_USER_ID",
    "ANTHROPIC_API_KEY", "FMP_API_KEY", "FRED_API_KEY",
    "BOT_EMAIL", "BOT_EMAIL_PASSWORD", "YOUR_EMAIL",
]
missing = [v for v in required_vars if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Missing required environment variables: {missing}")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_USER_ID = int(os.getenv("TELEGRAM_USER_ID"))
READONLY_MODE = os.getenv("BOT_READONLY", "false").lower() == "true"

_BOT_START_TIME = time.time()  # for /status uptime

logger = logging.getLogger(__name__)

from equity.brief.brief_builder import (
    _format_section2_eco_calendar,
    build_morning_brief,
    get_last_brief_synthesis,
    save_brief_to_thread,
)
from equity.brief.earnings_monitor import fetch_earnings_calendar
from equity.brief.eco_calendar import fetch_eco_calendar
from equity.brief.market_snapshot import _get_market_session_status, fetch_market_snapshot
from equity.brief.performance_tracker import (
    fetch_benchmark_performance,
    fetch_portfolio_performance,
    fetch_position_relative_performance,
    fetch_spot_prices,
    format_performance_section,
)
from equity.brief.sector_monitor import fetch_sector_data, format_sector_section
from equity.config import config_manager
from equity.config import positions as positions_module
from equity.config.market_config import (
    COMMODITY_ALERT_PCT,
    COMMODITY_TICKERS_EXTENDED,
    CREDIT_TICKERS,
    CROSS_ASSET_RATIOS,
    CRYPTO_ALERT_PCT,
    CRYPTO_TICKERS,
    EQUITY_FUTURES,
    FX_ALERT_PCT,
    FX_TICKERS,
    INTERNATIONAL_INDICES,
    INTL_ALERT_PCT,
    LARGE_MOVE_THRESHOLD_PCT,
    NEWS_HEADLINE_PAGE_SIZE,
    VOLATILITY_ALERT,
    VOLATILITY_TICKERS,
    YIELD_ALERT_BP,
    YIELD_LEVEL_ALERTS,
)
from equity.data.price_cache import TENOR_TO_CACHE_KEY, price_cache
from equity.portfolio.monitor import format_portfolio_monitor, run_portfolio_monitor
from equity.portfolio.news_triage import format_news_triage, run_news_triage
from equity.screener.quality_scorer import score_ticker
from equity.screener.screener import format_screener_output, run_screener
from equity.telegram import config_commands
from equity.telegram.advisor import MODEL as ADVISOR_MODEL
from equity.telegram.advisor import Advisor
from equity.telegram.advisor import _is_valid_ticker
from equity.telegram.auth import AuthManager
from equity.telegram.formatters import (
    format_headline_page,
    format_thread_list,
    make_confirm_cancel,
    make_discuss_menu,
    make_headline_page_keyboard,
    make_main_menu,
    make_news_actions,
    make_portfolio_actions,
    make_screener_actions,
    make_suggestions_keyboard,
    make_thread_list_keyboard,
    make_ticker_actions,
    send_in_parts,
    send_safe,
)
from equity.telegram.threads import ThreadManager


async def reply(update: Update, context, text: str,
                 reply_markup=None, parse_mode: str = 'Markdown') -> None:
    '''
    Sends a reply regardless of whether the update came from a command or callback query.
    Uses update.effective_message which works for both.
    Falls back to send_safe for Markdown error handling.
    '''
    chat_id = update.effective_chat.id
    await send_safe(context.bot, chat_id, text, reply_markup=reply_markup)


POSITIONS = positions_module.POSITIONS
WATCHLIST = positions_module.WATCHLIST

thread_manager = ThreadManager()
auth_manager = AuthManager()
advisor = Advisor(api_key=os.getenv("ANTHROPIC_API_KEY"), thread_manager=thread_manager)

# ---------------------------------------------------------------------------
# Security logging
# ---------------------------------------------------------------------------

# Handler is attached by setup_logging() (equity/config/logging_config.py),
# which routes the "security" logger to equity/data/logs/security.log.
security_logger = logging.getLogger("security")
security_logger.setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

_user_message_times: dict[int, list[float]] = defaultdict(list)
_claude_call_times: list[float] = []
MAX_MESSAGES_PER_MINUTE = 20
MAX_CLAUDE_CALLS_PER_HOUR = 30


def check_rate_limit(user_id: int) -> bool:
    now = time.time()
    times = _user_message_times[user_id]
    times[:] = [t for t in times if now - t < 60]
    if len(times) >= MAX_MESSAGES_PER_MINUTE:
        return False
    times.append(now)
    return True


def check_claude_rate_limit() -> bool:
    now = time.time()
    _claude_call_times[:] = [t for t in _claude_call_times if now - t < 3600]
    if len(_claude_call_times) >= MAX_CLAUDE_CALLS_PER_HOUR:
        return False
    _claude_call_times.append(now)
    return True


# ---------------------------------------------------------------------------
# Helper: run blocking calls in thread pool executor
# ---------------------------------------------------------------------------

async def run_in_executor(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(func, *args))


# ---------------------------------------------------------------------------
# Command registry for post-auth re-dispatch
# ---------------------------------------------------------------------------

COMMAND_REGISTRY: dict[str, callable] = {}


def register_command(func):
    COMMAND_REGISTRY[func.__name__] = func
    return func


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def authorized_only(func):
    async def wrapper(update, context):
        if update.effective_user.id != TELEGRAM_USER_ID:
            return  # silently ignore unauthorized users
        message_text = update.message.text if update.message and update.message.text else ""
        # Every command (including read-only ones) carries this decorator, so
        # it's the one choke point that sees every /command before dispatch —
        # handle_message() never does, since its filter excludes commands.
        # /cancel must stay exempt: it's how a pending auth actually gets
        # resolved, and it isn't itself guarded by require_email_auth.
        if (
            message_text.startswith("/")
            and not message_text.startswith("/cancel")
            and auth_manager.is_awaiting_auth(update.effective_user.id)
        ):
            await reply(update, context,
                "⚠️ You have a pending authorization. Type the 6-digit code to complete it,\n"
                "or /cancel to abort before starting a new command."
            )
            return
        return await func(update, context)
    wrapper.__name__ = func.__name__
    return wrapper


def require_write_access(func):
    async def wrapper(update, context):
        if READONLY_MODE:
            await reply(update, context,
                "🔒 Bot is in read-only mode. "
                "SSH to VPS and set BOT_READONLY=false to enable writes."
            )
            return
        return await func(update, context)
    wrapper.__name__ = func.__name__
    return wrapper


def require_email_auth(operation_fn):
    """
    Decorator factory. operation_fn(update, context) -> str describes the operation.
    Sends auth email, stores pending action, tells user to check email.
    Actual command executes after auth code verified in handle_message().
    """
    def decorator(func):
        @authorized_only
        @require_write_access
        async def wrapper(update, context):
            user_id = update.effective_user.id
            operation = operation_fn(update, context)
            sent = auth_manager.send_email_code(user_id, operation)
            if not sent:
                await reply(update, context,
                    "⚠️ Failed to send auth email. "
                    "Check BOT_EMAIL settings. Operation cancelled."
                )
                return
            context.user_data["awaiting_email_auth"] = {
                "func_name": func.__name__,
                "operation": operation,
            }
            context.user_data["pending_command_args"] = context.args
            await reply(update, context,
                f"🔐 *Authorization required*\n"
                f"Operation: {operation}\n\n"
                f"A 6-digit code has been sent to your email.\n"
                f"Enter the code to proceed, or /cancel to abort.",
                parse_mode="Markdown"
            )
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# start_or_resume_discussion
# ---------------------------------------------------------------------------

async def start_or_resume_discussion(subject: str, update, context, thread_type: str = "ticker") -> None:
    ticker = subject.upper()
    thread_id = f"{thread_type}_{ticker}"
    thread_info = thread_manager.get_thread_info(thread_id)
    system_prompt = advisor.build_system_prompt(thread_subject=ticker, current_thread_id=thread_id)

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    if thread_info is None:
        if thread_type == "ticker":
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"🔍 Researching {ticker} — fetching live fundamentals, "
                     f"analyst consensus, recent events... (~20s)"
            )
            context_str = await run_in_executor(
                advisor.get_ticker_context, ticker
            )
            first_message = (
                f"Please analyze {ticker} for my portfolio consideration. "
                f"Context:\n{context_str}"
            )
        elif thread_type == "topic" and ticker == "PORTFOLIO":
            monitor_data = await run_in_executor(run_portfolio_monitor)
            positions_snapshot = []
            for pos_ticker, data in monitor_data.get("positions", {}).items():
                price = data.get("price_current", "N/A")
                change = data.get("change_1d_pct", 0)
                flag = data.get("move_flag", "")
                positions_snapshot.append(
                    f"{pos_ticker}: ${price} ({change:+.1f}% today) {flag}"
                )
            live_context = "Current portfolio prices (live):\n" + "\n".join(positions_snapshot)
            first_message = (
                f"Portfolio review requested. Current regime: {advisor.get_regime_context()}\n\n"
                f"{live_context}\n\n"
                f"What aspect of the portfolio would you like to review?"
            )
        else:
            regime = advisor.get_regime_context()
            first_message = (
                f"Let us discuss {ticker}. {regime} "
                f"What would you like to explore?"
            )
        if not check_claude_rate_limit():
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⚠️ Claude call limit reached for this hour."
            )
            return
        response = await run_in_executor(
            advisor.chat, thread_id, first_message, system_prompt, ticker
        )
    else:
        msg_count = thread_info.get("message_count", 0)
        last_active = thread_info.get("last_active", "unknown")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Resuming *{ticker}* ({msg_count} messages, last active {last_active})",
            parse_mode="Markdown"
        )
        if thread_type == "ticker":
            from equity.data.monitoring import get_monitoring_for_ticker

            mon_items = get_monitoring_for_ticker(ticker)
            if mon_items:
                mon_summary = "; ".join(f'{m["item"][:50]}' for m in mon_items[:3])
                resume_message = (
                    f"We are resuming our {ticker} discussion. "
                    f"Active monitoring: {mon_summary}. "
                    f"Briefly recap where we left off and check in on each monitoring item."
                )
            else:
                resume_message = (
                    f"We are resuming our {ticker} discussion. "
                    f"Briefly recap where we left off and note anything "
                    f"that may have changed since then."
                )
        else:
            resume_message = (
                f"We are resuming our {ticker} discussion. "
                f"Briefly recap where we left off and note anything "
                f"that may have changed since then."
            )
        if not check_claude_rate_limit():
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⚠️ Claude call limit reached for this hour."
            )
            return
        response = await run_in_executor(
            advisor.chat, thread_id, resume_message, system_prompt, ticker
        )

    thread_manager.set_active_thread(TELEGRAM_USER_ID, thread_id)
    suggestions = advisor.get_follow_up_suggestions(thread_id, response)
    await send_in_parts(context.bot, update.effective_chat.id, response)
    if suggestions:
        context.user_data["suggestions"] = suggestions
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="💡 *You might ask:*",
            parse_mode="Markdown",
            reply_markup=make_suggestions_keyboard(suggestions)
        )
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Reply to continue, or choose an action:",
        reply_markup=make_ticker_actions(ticker)
    )


# ---------------------------------------------------------------------------
# propose_position_save — advisor-drafted thesis/tier update
# ---------------------------------------------------------------------------

async def propose_position_save(update, context, ticker: str) -> None:
    """Drafts a structured position update from a ticker thread's discussion
    and proposes it for confirmation.

    Reachable from /save TICKER, /update TICKER (existing), and the
    [💾 Update Thesis] ticker-actions button — all funnel through here so
    they share one draft-then-confirm flow. Confirmation is the same email
    2FA code used for every other write in this bot (see
    `context.user_data["pending_auth_action"]` / handle_message's
    'position_update' branch) rather than a separate inline button — a
    second, unwired confirm/cancel button here would just be a dead end,
    since the actual write only happens once the emailed code is entered.
    """
    if READONLY_MODE:
        await reply(update, context,
            "🔒 Bot is in read-only mode. "
            "SSH to VPS and set BOT_READONLY=false to enable writes."
        )
        return

    ticker = ticker.upper()
    chat_id = update.effective_chat.id
    thread_id = f"ticker_{ticker}"

    await context.bot.send_message(
        chat_id=chat_id,
        text=f"📋 Drafting {ticker} position update from our discussion..."
    )

    history = thread_manager.get_messages_for_api(thread_id, recent_verbatim=30)
    if not history:
        await reply(update, context,
            f"No discussion found for {ticker}. "
            f"Start one with /discuss {ticker} first."
        )
        return

    update_dict = await run_in_executor(
        advisor.draft_position_update, ticker, history
    )

    if not update_dict or not update_dict.get("thesis"):
        await reply(update, context,
            f"Could not extract clear conclusions from the {ticker} discussion. "
            f"Continue the discussion and try /save {ticker} again when conclusions are clearer."
        )
        return

    from equity.config.market_config import POSITION_TIERS
    tier_key = update_dict.get("tier_v2", "")
    tier_label = POSITION_TIERS.get(tier_key, {}).get("label", tier_key or "unclassified")
    breakers = update_dict.get("thesis_breakers") or []

    review_text = (
        f"📋 *Proposed update for {ticker}*\n\n"
        f"*Thesis:*\n{update_dict.get('thesis', '')}\n\n"
        f"*Thesis-breakers:*\n" +
        "\n".join(f"• {b}" for b in breakers) +
        f"\n\n*Macro thesis:*\n{update_dict.get('macro_thesis', '')}\n\n"
        f"*Exit conditions:*\n{update_dict.get('target_exit_conditions', '')}\n\n"
        f"*Tier:* {tier_label}\n"
        f"*Style:* {update_dict.get('style', '') or 'unspecified'}\n"
        f"*Size target:* {update_dict.get('size_target_pct') or 'not specified'}%\n\n"
        f"Review carefully. Email authorization required to save."
    )

    context.user_data["pending_auth_action"] = {
        "action": "position_update",
        "ticker": ticker,
        "updates": update_dict,
    }

    sent = auth_manager.send_email_code(
        update.effective_user.id,
        f"Update {ticker} thesis and classification"
    )
    if not sent:
        context.user_data.pop("pending_auth_action", None)
        await reply(update, context,
            "⚠️ Failed to send auth email. Check BOT_EMAIL settings."
        )
        return

    await send_safe(context.bot, chat_id, review_text)
    await context.bot.send_message(
        chat_id=chat_id,
        text="📧 Authorization code sent to your email. Enter it to confirm, or /cancel to abort."
    )


# ---------------------------------------------------------------------------
# Read-only handlers
# ---------------------------------------------------------------------------

@authorized_only
async def start(update, context):
    await reply(update, context,
        "👋 Portfolio Advisor online. I help you research, discuss, and track "
        "your equity positions and watchlist. Use /help to see everything I can do.",
        reply_markup=make_main_menu(),
    )


@authorized_only
async def send_help(update, context):
    text = (
        "*📖 Portfolio Advisor — Commands*\n\n"
        "*Read-only*\n"
        "/brief — Full morning brief\n"
        "/screener — Run equity screener\n"
        "/portfolio — Portfolio price action\n"
        "/news — News triage for all positions\n"
        "/watchlist — Watchlist with live prices\n"
        "/threads — List discussion threads\n"
        "/framework — Position tier framework and classification status\n"
        "/audit — Recent config changes and operations\n"
        "/monitoring [TICKER] — Active monitoring items, carried forward across briefs\n\n"
        "*Discussion*\n"
        "/discuss TICKER — Discuss a ticker\n"
        "/macro — Macro and regime discussion\n"
        "/portfolio_review — In-depth portfolio review\n"
        "/switch thread_id — Switch active thread\n"
        "/done — Pause current thread\n\n"
        "*Writes (email 2FA required)*\n"
        "/add TICKER — Add to watchlist\n"
        "/remove TICKER — Remove from watchlist\n"
        "/update TICKER [field] — Update thesis\n"
        "/save TICKER — Save discussion conclusions as a position update\n"
        "/dismiss TICKER — Dismiss monitoring item(s) for a ticker\n"
        "/set FIELD VALUE — Update market config\n"
        "/confirm — Approve pending change\n"
        "/cancel — Cancel pending change/auth\n"
    )
    await reply(update, context, text, parse_mode="Markdown", reply_markup=make_main_menu())


@authorized_only
async def send_brief(update, context):
    chat_id = update.effective_chat.id
    await context.bot.send_message(chat_id=chat_id, text="🌅 Building morning brief...")
    sections = await run_in_executor(build_morning_brief)
    await run_in_executor(save_brief_to_thread, sections, thread_manager)
    for text, keyboard in sections:
        if text and text.strip():
            await send_safe(context.bot, chat_id, text, reply_markup=keyboard)
            await asyncio.sleep(0.3)


@authorized_only
async def send_screener(update, context):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="🔍 Running screener...")
    # Load regime flags from the market snapshot (also 1h-cached) so the
    # screener's RSI dislocation threshold matches today's regime.
    try:
        snapshot = await run_in_executor(fetch_market_snapshot)
        regime_flags = snapshot.get("regime_flags", [])
    except Exception:
        regime_flags = []
    df = await run_in_executor(run_screener, False, regime_flags)
    text = format_screener_output(df)
    passing_tickers = df["ticker"].tolist() if "ticker" in df.columns else []
    await send_in_parts(
        context.bot, update.effective_chat.id, text,
        reply_markup=make_screener_actions(passing_tickers),
    )


@authorized_only
async def send_portfolio(update, context):
    monitor_data = await run_in_executor(run_portfolio_monitor)
    text = format_portfolio_monitor(monitor_data)
    position_tickers = list(monitor_data.get("positions", {}).keys())
    await send_in_parts(
        context.bot, update.effective_chat.id, text,
        reply_markup=make_portfolio_actions(position_tickers),
    )


@authorized_only
async def send_news(update, context):
    tickers = positions_module.get_all_tickers()
    triage = await run_in_executor(run_news_triage, tickers)
    text = format_news_triage(triage)
    alert_tickers = [t for t, d in triage.items() if d.get("has_thesis_alert")]
    await send_in_parts(
        context.bot, update.effective_chat.id, text,
        reply_markup=make_news_actions(alert_tickers),
    )


@authorized_only
async def send_watchlist(update, context):
    lines = ["📋 WATCHLIST"]
    for ticker, cfg in WATCHLIST.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            close = hist["Close"].dropna() if "Close" in hist.columns else None
            if close is not None and len(close) >= 2:
                price = float(close.iloc[-1])
                change_pct = (price / float(close.iloc[-2]) - 1) * 100
                lines.append(f"{ticker:<6} ${price:.2f}  {change_pct:+.1f}%  ({cfg.get('tier', '')})")
            else:
                lines.append(f"{ticker:<6} price unavailable  ({cfg.get('tier', '')})")
        except Exception as exc:
            logger.warning("send_watchlist: price fetch failed for %s: %s", ticker, exc)
            lines.append(f"{ticker:<6} price unavailable  ({cfg.get('tier', '')})")
    await reply(update, context,
        "\n".join(lines), reply_markup=make_discuss_menu(POSITIONS, WATCHLIST)
    )


@authorized_only
async def send_framework(update, context):
    """Shows the current tier framework and classification status of all positions."""
    from equity.config.market_config import CLASSIFICATION_STATUS, POSITION_TIERS

    lines = ["📋 *POSITION FRAMEWORK*", ""]

    lines.append("*TIERS*")
    for tier in POSITION_TIERS.values():
        lines.append(f'*{tier["label"]}* ({tier["min_size_pct"]}-{tier["max_size_pct"]}%)')
        lines.append(f'  {tier["behavior"]}')
        lines.append("")

    lines.append("*CLASSIFICATION STATUS*")
    status_groups: dict[str, list[str]] = {}
    for ticker, pos in {**POSITIONS, **WATCHLIST}.items():
        status = pos.get("classification_status", "unclassified")
        status_groups.setdefault(status, []).append(ticker)

    for status_key, label in CLASSIFICATION_STATUS.items():
        tickers = status_groups.get(status_key, [])
        if tickers:
            lines.append(f'{label}: {", ".join(sorted(tickers))}')

    await send_in_parts(context.bot, update.effective_chat.id,
                         "\n".join(lines), reply_markup=make_main_menu())


def _is_macro_monitoring_item(ticker: str) -> bool:
    """True when a monitoring item's `ticker` is a thematic/macro subject
    (e.g. USDJPY, COPPER, DXY — brief_synthesizer's NEW MONITORING ITEMS
    parser stores these as if they were tickers) rather than one of our
    actual position or watchlist tickers.

    Mirrors intraday_alert_job()'s alert_type-based routing: a macro/
    thematic item's [💬 Discuss] button below goes to the shared MACRO
    topic thread (cmd_macro) instead of a per-ticker thread that would
    never resolve to a real, discussable symbol.
    """
    ticker = ticker.upper()
    return ticker not in POSITIONS and ticker not in WATCHLIST


@authorized_only
async def send_monitoring(update, context):
    """
    /monitoring — shows all active monitoring items, grouped by ticker.
    /monitoring TICKER — shows items for a specific ticker.

    Tickers are sorted by their highest-priority item, then by recency;
    items within a ticker are sorted the same way. Each ticker gets one
    [💬 Discuss] / [✕ Dismiss TICKER] row (capped at 8 tickers to avoid
    keyboard overflow) rather than a button per item — dismissing (or
    resolving) a single item is still available via `/dismiss ID`, using
    the ID printed under each item.
    """
    from equity.data.monitoring import load_monitoring

    ticker_filter = context.args[0].upper() if context.args else None
    all_items = load_monitoring()

    if ticker_filter:
        all_items = [i for i in all_items if i.get("ticker") == ticker_filter]

    if not all_items:
        msg = f'No active monitoring items{f" for {ticker_filter}" if ticker_filter else ""}.'
        await reply(update, context, msg, reply_markup=make_main_menu())
        return

    # Group by ticker.
    grouped = defaultdict(list)
    for item in all_items:
        grouped[item["ticker"]].append(item)

    priority_order = {"high": 0, "medium": 1, "low": 2}

    # Sort tickers: highest-priority tickers first, then most recent item.
    def ticker_sort_key(ticker):
        items = grouped[ticker]
        top_priority = min(priority_order.get(i.get("priority", "medium"), 1) for i in items)
        newest_age = min(i.get("age_days", 0) for i in items)
        return (top_priority, newest_age)

    sorted_tickers = sorted(grouped.keys(), key=ticker_sort_key)

    lines = [f"📋 *MONITORING — {len(all_items)} items across {len(sorted_tickers)} names*", ""]
    priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}

    for ticker in sorted_tickers:
        items = grouped[ticker]
        # Sort items within ticker: high priority first, then most recent.
        items.sort(key=lambda x: (priority_order.get(x.get("priority", "medium"), 1), x.get("age_days", 0)))

        is_macro = _is_macro_monitoring_item(ticker)
        route_label = "🌍 macro" if is_macro else "📈 position"
        lines.append(f"*{ticker}* ({route_label})")

        for item in items:
            emoji = priority_emoji.get(item.get("priority", "medium"), "⚪")
            age = item.get("age_days", 0)
            age_str = "today" if age == 0 else f"{age}d ago"
            source = item.get("added_from", "").replace("_", " ")

            # Truncate at a word boundary rather than mid-word.
            item_text = item["item"]
            if len(item_text) > 80:
                cut = item_text[:80].rfind(" ")
                item_text = item_text[: cut if cut > 0 else 80] + "…"

            lines.append(f"  {emoji} {item_text}")
            lines.append(f'     Added: {age_str} via {source} | ID: `{item["id"]}`')

        lines.append("")

    lines.append("─────────────────────")
    lines.append("`/dismiss TICKER` — dismiss all for ticker")
    lines.append("`/dismiss ID` — dismiss one specific item (ID shown above)")

    kb = _make_monitoring_action_keyboard(sorted_tickers, grouped)

    await send_in_parts(context.bot, update.effective_chat.id,
                         "\n".join(lines), reply_markup=kb)


def _make_monitoring_action_keyboard(sorted_tickers: list, grouped: dict):
    """One row per ticker with [💬 Discuss] and [✕ Dismiss] buttons.
    Capped at 8 tickers to avoid keyboard overflow.
    """
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    buttons = []
    for ticker in sorted_tickers[:8]:
        discuss_cb = "cmd_macro" if _is_macro_monitoring_item(ticker) else f"discuss_{ticker}"
        buttons.append([
            InlineKeyboardButton(f"💬 {ticker}", callback_data=discuss_cb),
            InlineKeyboardButton(f"✕ Dismiss {ticker}", callback_data=f"dismiss_all_{ticker}"),
        ])
    buttons.append([InlineKeyboardButton("🏠 Main Menu", callback_data="cmd_main_menu")])
    return InlineKeyboardMarkup(buttons)


async def _send_dismiss_selection(update, context, ticker: str, items: list[dict]) -> None:
    """Inline keyboard listing each monitoring item for `ticker`, so the user
    can tap the specific one to dismiss rather than clearing all of them.
    Offers [Dismiss All] and [Cancel] too. Called only from `send_dismiss()`
    when more than one item is active for the ticker — not itself a
    registered handler, so it needs no @authorized_only of its own.
    """
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}

    lines = [f"📋 *{ticker} — Select item to dismiss:*", ""]
    buttons = []

    for item in items:
        emoji = priority_emoji.get(item.get("priority", "medium"), "⚪")
        age = item.get("age_days", 0)
        age_str = "today" if age == 0 else f"{age}d ago"
        short_text = item["item"][:60] + ("..." if len(item["item"]) > 60 else "")
        lines.append(f"{emoji} {short_text} ({age_str})")
        buttons.append([InlineKeyboardButton(
            f"{emoji} {short_text[:50]}", callback_data=f'dismiss_item_{item["id"]}'
        )])

    buttons.append([
        InlineKeyboardButton(f"🗑 Dismiss All ({len(items)})", callback_data=f"dismiss_all_{ticker}"),
        InlineKeyboardButton("❌ Cancel", callback_data="cmd_main_menu"),
    ])

    await send_safe(context.bot, update.effective_chat.id,
                     "\n".join(lines), reply_markup=InlineKeyboardMarkup(buttons))


@authorized_only
async def send_threads(update, context):
    threads = thread_manager.list_threads()
    await reply(update, context,
        format_thread_list(threads), reply_markup=make_thread_list_keyboard(threads)
    )


@authorized_only
async def send_discuss(update, context):
    if not context.args:
        await reply(update, context,
            "Choose a ticker:", reply_markup=make_discuss_menu(POSITIONS, WATCHLIST)
        )
        return
    await start_or_resume_discussion(context.args[0], update, context)


@authorized_only
async def send_macro(update, context):
    await start_or_resume_discussion("MACRO", update, context, thread_type="topic")


_PRICE_FILTERS = {
    "all", "yields", "fx", "commodities", "positions",
    "futures", "crypto", "vol", "volatility", "intl", "international", "credit",
}


def _session_suffix(data: dict, show_session: bool = True) -> str:
    """The ' | 3min ago (2026-09-07 14:35 ET)'-style tail `_format_price_line()`
    and the yields section both append — factored out since yields don't fit
    `_format_price_line()`'s $-price/percent-change shape but still carry
    the same fields (see price_cache.py's `_get_session_context()` and
    `_fetch_fred_yields()`).

    Prefers the always-exact `exact_timestamp` (full date + ET time for an
    intraday bar, weekday + calendar date for a daily one) over the vaguer
    `session_label`/`condition_note` bucketing ("Fri close", "3d ago") —
    a viewer should never have to guess which session a price is from.
    `data_lag_minutes` (intraday bars only) leads as the relative "how
    stale" figure when available, in minutes rather than the coarser hour
    buckets; a daily bar (`data_lag_minutes` is None) shows just the exact
    timestamp on its own, since a day's-worth of age isn't meaningful in
    minutes anyway.
    """
    if not show_session:
        return ""
    exact_ts = data.get("exact_timestamp", "")
    lag_min = data.get("data_lag_minutes")
    if lag_min is not None:
        if lag_min < 1:
            label = "<1min ago"
        elif lag_min < 60:
            label = f"{int(lag_min)}min ago"
        else:
            label = f"{int(lag_min // 60)}h {int(lag_min % 60)}min ago"
        suffix = f" | {label}"
        if exact_ts:
            suffix += f" ({exact_ts})"
        return suffix

    if exact_ts:
        return f" | {exact_ts}"

    # Neither field present (e.g. a hand-built dict in a test/caller that
    # predates exact_timestamp) — fall back to the coarser session_label.
    session_label = data.get("session_label", "")
    if not session_label:
        return ""
    condition_note = data.get("condition_note", "")
    suffix = f" | {session_label}"
    if condition_note and condition_note != session_label:
        suffix += f" ({condition_note})"
    return suffix


def _fmt_price_value(price: float) -> str:
    """The magnitude-scaled numeric formatting `_format_price_line()` uses
    for both its headline price and (when present) the AH price — pulled
    out so the two don't drift apart."""
    if price > 10000:
        return f"{price:,.0f}"
    elif price > 100:
        return f"{price:,.2f}"
    elif price > 1:
        return f"{price:.4f}"
    else:
        return f"{price:.6f}"


def _format_price_line(label: str, data: dict, show_session: bool = True) -> str:
    """Formats one price_cache entry as a display line: emoji, price, 1D
    change, a ⚠️ STALE flag when `is_stale`, an after-hours/pre-market
    supplement when one's available, and (unless `show_session` is False)
    session/data-age context.

    The headline price/change prefer `official_close`/`change_1d_pct` — the
    regular-session close and its move off the prior close — over the raw
    `price` field, which becomes an after-hours or pre-market print once
    price_cache has one (see price_cache.py's module docstring). Tickers
    with no `official_close` (crypto/futures/commodities — no cash-session
    concept to split out) just show `price` as before.

    Format examples:
      Intraday (5m bar): 🟢 MSFT: $420.10 (+0.30%) | 3min ago (2026-09-07 14:35 ET)
      Daily close:       🟢 Gold: $4,476.60 (+1.06%) | Fri 2026-09-04 close
      After-hours:        🔴 TSLA: 368.16 (-0.05%) | 🔴 AH: 366.42 (-0.49%) | Wed 2026-09-09 close
      Stale:              🔴 ES Futures: 5,720.00 (+0.00%) ⚠️ STALE | Fri 2026-09-04 close
    """
    official_close = data.get("official_close")
    price = official_close if official_close is not None else (data.get("price", 0) or 0)
    chg = data.get("change_1d_pct", 0) or 0
    stale_flag = " ⚠️ STALE" if data.get("is_stale", False) else ""
    emoji = "🟢" if chg > 0 else "🔴" if chg < 0 else "⚪"

    price_str = _fmt_price_value(price)

    ah_str = ""
    afterhours_price = data.get("afterhours_price")
    # Only worth a line when it differs meaningfully from the close it's
    # being compared to — a flat after-hours print would just repeat the
    # headline number with an extra 0.00% tacked on.
    if afterhours_price and official_close and abs(afterhours_price - official_close) > 0.01:
        ah_chg = (afterhours_price / official_close - 1) * 100
        ah_emoji = "🟢" if ah_chg > 0 else "🔴" if ah_chg < 0 else "⚪"
        ah_label = "AH" if data.get("session_type") == "afterhours" else "PM"
        ah_str = f" | {ah_emoji} {ah_label}: {_fmt_price_value(afterhours_price)} ({ah_chg:+.2f}%)"

    return f"  {emoji} {label}: {price_str} ({chg:+.2f}%){stale_flag}{ah_str}{_session_suffix(data, show_session)}"


@authorized_only
async def send_prices(update, context):
    """
    /prices — live snapshot of every tracked security, read straight from
    the shared price_cache: Treasury yields, FX, commodities, positions,
    equity futures, crypto, volatility, international indices, and credit
    proxies.

    Optional filter: /prices yields | fx | commodities | positions |
    futures | crypto | vol | intl | credit

    Forces a fresh price_cache refresh rather than waiting for its normal
    15/60-minute TTL — the entire point of asking on demand is "what's the
    level right now."
    """
    arg = context.args[0].lower() if context.args else "all"
    if arg not in _PRICE_FILTERS:
        await reply(
            update, context,
            f"Unknown filter '{arg}'. Try: yields, fx, commodities, positions, "
            f"futures, crypto, vol, intl, credit — or no argument for everything.",
            reply_markup=make_main_menu(),
        )
        return

    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="📡 Fetching live prices..."
    )

    from equity.config.positions import POSITIONS, get_position

    await run_in_executor(price_cache.refresh, True)
    lines = [f'📡 *Live Prices* — {datetime.now().strftime("%H:%M ET")}', ""]

    # Stale-data banner — a viewer scanning fast should see this before any
    # individual ⚠️ STALE tag further down.
    all_prices = price_cache.get_prices()
    stale_tickers = [t for t, d in all_prices.items() if d.get("is_stale", False)]
    if stale_tickers:
        stale_by_type: dict[str, list[str]] = {}
        for t in stale_tickers:
            stale_by_type.setdefault(all_prices[t].get("instrument_type", "other"), []).append(t)
        lines.append("⚠️ *Stale data detected:*")
        for itype, tickers in stale_by_type.items():
            more = f" +{len(tickers) - 5} more" if len(tickers) > 5 else ""
            lines.append(f"  {itype}: {', '.join(tickers[:5])}{more}")
        lines.append("_Stale = older than expected for instrument type. Do not action these prices without verification._")
        lines.append("")

    if arg in ("all", "yields"):
        lines.append("*Treasury Yields*")
        for tenor in ("3M", "2Y", "5Y", "10Y", "20Y", "30Y"):
            data = price_cache.get_yield(tenor)
            if data:
                stale_flag = " ⚠️ STALE" if data.get("is_stale", False) else ""
                lines.append(
                    f'  {tenor}: {data["price"]:.3f}% ({data["change_1d_bps"]:+.1f}bp){stale_flag}{_session_suffix(data)}'
                )
        lines.append("")

    if arg in ("all", "fx"):
        lines.append("*FX*")
        for ticker, label in FX_TICKERS.items():
            data = price_cache.get(ticker)
            if data:
                lines.append(_format_price_line(label, data))
        lines.append("")

    if arg in ("all", "commodities"):
        lines.append("*Commodities*")
        for ticker, label in COMMODITY_TICKERS_EXTENDED.items():
            data = price_cache.get(ticker)
            if data:
                lines.append(_format_price_line(label, data))
        lines.append("")

    if arg in ("all", "futures"):
        lines.append("*Equity Futures*")
        for ticker, label in EQUITY_FUTURES.items():
            data = price_cache.get(ticker)
            if data:
                lines.append(_format_price_line(label, data))
        lines.append("")

    if arg in ("all", "crypto"):
        lines.append("*Crypto*")
        for ticker, label in CRYPTO_TICKERS.items():
            data = price_cache.get(ticker)
            if data:
                lines.append(_format_price_line(label, data))
        lines.append("")

    if arg in ("all", "vol", "volatility"):
        lines.append("*Volatility*")
        for ticker, label in VOLATILITY_TICKERS.items():
            data = price_cache.get(ticker)
            if data:
                lines.append(_format_price_line(label, data))
        lines.append("")

    if arg in ("all", "intl", "international"):
        lines.append("*International Indices*")
        for ticker, label in INTERNATIONAL_INDICES.items():
            data = price_cache.get(ticker)
            if data:
                session = _get_market_session_status(ticker)
                lines.append(f"{_format_price_line(label, data)}{session}")
        lines.append("")

    if arg in ("all", "credit"):
        lines.append("*Credit Proxies*")
        for ticker, label in CREDIT_TICKERS.items():
            data = price_cache.get(ticker)
            if data:
                lines.append(_format_price_line(label, data))
        lines.append("")

    if arg in ("all", "positions"):
        lines.append("*Positions (1D move)*")
        # After the cash session closes, the 1D move shown is the official
        # close vs prior close — not a live quote — with an after-hours
        # print appended where price_cache has one. Worth a one-line
        # reminder so a post-close read isn't mistaken for a live tape.
        import pytz

        now_et = datetime.now(pytz.timezone("America/New_York"))
        if now_et.weekday() < 5 and now_et.hour >= 16:
            lines.append(
                "_Showing official closes; after-hours prices shown where available._\n"
                "_Connect IBKR (Module 4) for real-time position prices._\n"
            )
        pos_data = price_cache.get_prices(list(POSITIONS.keys()))
        sorted_pos = sorted(
            pos_data.items(), key=lambda kv: abs(kv[1].get("change_1d_pct", 0) or 0), reverse=True
        )
        for ticker, data in sorted_pos:
            line = _format_price_line(ticker, data)
            pos_config = get_position(ticker)
            avg_cost = pos_config.get("avg_cost") if pos_config else None
            if avg_cost and avg_cost > 0 and data.get("price"):
                pnl = (data["price"] / avg_cost - 1) * 100
                line += f" | cost ${avg_cost:.2f} P&L {pnl:+.1f}%"
            lines.append(line)
        lines.append("")

    await send_safe(
        context.bot, update.effective_chat.id, "\n".join(lines), reply_markup=make_main_menu(),
    )

    # Full-snapshot synthesis — only for the unfiltered /prices, not a
    # filtered one (/prices fx), which is a quick single-section look-up
    # rather than a "what does the whole board say" moment worth a Claude
    # call over.
    if arg == "all":
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text="💡 Generating market synthesis..."
        )
        try:
            from equity.brief.brief_synthesizer import (
                _parse_and_persist_monitoring,
                synthesize_global_signals,
            )
            from equity.brief.market_snapshot import fetch_market_snapshot
            from equity.data.monitoring import load_monitoring

            # fetch_market_snapshot() does its own multi-ticker + 5Y-history
            # + FRED fetch (see that module) — route it through the executor
            # like the price_cache refresh above rather than blocking the
            # event loop for everyone else while it runs.
            snapshot = await run_in_executor(fetch_market_snapshot)
            regime_flags = snapshot.get("regime_flags", [])
            monitoring_items = load_monitoring()

            price_text = "\n".join(lines)
            recent_alert_context = _get_recent_alert_context()
            if recent_alert_context:
                price_text += f"\n\nRECENT ALERTS:\n{recent_alert_context}"

            synthesis = await run_in_executor(
                synthesize_global_signals, price_text, regime_flags, monitoring_items,
            )
            _parse_and_persist_monitoring(synthesis, source="prices_synthesis")
            await send_safe(
                context.bot, update.effective_chat.id,
                f"💡 *Market Synthesis*\n\n{synthesis}", reply_markup=make_main_menu(),
            )
        except Exception as e:
            logger.warning(f"send_prices synthesis failed: {e}")


@authorized_only
async def send_portfolio_review(update, context):
    await start_or_resume_discussion("PORTFOLIO", update, context, thread_type="topic")


async def _send_switch_confirmation(update, context, thread_id: str, subject) -> None:
    """Confirmation shown after switching to `thread_id`.

    topic_BRIEF gets a special case: show the full brief synthesis
    immediately (via send_in_parts, since it can run long) rather than the
    generic "switched" message — so /switch topic_BRIEF doubles as a way
    to review what the brief said without re-running it. Shared by
    send_switch() (/switch command) and handle_callback()'s "switch_"
    inline-button branch.
    """
    if thread_id == "topic_BRIEF":
        last_brief = get_last_brief_synthesis(thread_manager)
        text = (
            last_brief[1] if last_brief else
            "No morning brief has been generated yet. Run /brief to generate one."
        )
    else:
        text = f"✓ Switched to {thread_id}. Reply to continue."
    await send_in_parts(
        context.bot, update.effective_chat.id, text, reply_markup=make_ticker_actions(subject)
    )


@authorized_only
async def send_switch(update, context):
    if not context.args:
        threads = thread_manager.list_threads()
        await reply(update, context,
            "Choose a thread:", reply_markup=make_thread_list_keyboard(threads)
        )
        return
    thread_id = context.args[0]
    thread_manager.set_active_thread(TELEGRAM_USER_ID, thread_id)
    info = thread_manager.get_thread_info(thread_id)
    subject = info["subject"] if info else thread_id
    await _send_switch_confirmation(update, context, thread_id, subject)


@authorized_only
async def send_done(update, context):
    thread_manager.clear_active_thread(TELEGRAM_USER_ID)
    await reply(update, context,
        "✓ Thread paused. Resume anytime with /discuss TICKER",
        reply_markup=make_main_menu(),
    )


@authorized_only
async def send_audit(update, context):
    result = subprocess.run(
        ["git", "log", "--oneline", "-10", "--",
         "equity/config/positions.py",
         "equity/config/positions_override.json",
         "equity/config/market_config.py"],
        capture_output=True, text=True, cwd=os.getcwd()
    )
    try:
        with open("equity/data/logs/security.log") as f:
            recent_ops = "".join(f.readlines()[-5:])
    except FileNotFoundError:
        recent_ops = "No operations logged yet."
    await reply(update, context,
        f"📋 *Recent config changes:*\n```\n{result.stdout or 'None'}\n```\n"
        f"📋 *Recent write operations:*\n```\n{recent_ops}\n```",
        parse_mode="Markdown"
    )


@authorized_only
async def send_logs(update, context):
    """
    Shows recent errors and activity summary from log files.
    Usage: /logs          — last 20 errors from errors.log
           /logs brief    — last 20 lines from brief.log
           /logs advisor  — last 20 lines from advisor.log
           /logs screener — last 20 lines from screener.log
           /logs security — last 20 lines from security.log
    """
    from pathlib import Path

    log_dir = Path("equity/data/logs")
    arg = context.args[0].lower() if context.args else "errors"

    file_map = {
        "errors": "errors.log",
        "brief": "brief.log",
        "advisor": "advisor.log",
        "screener": "screener.log",
        "security": "security.log",
        "app": "app.log",
    }

    filename = file_map.get(arg, "errors.log")
    log_path = log_dir / filename

    if not log_path.exists():
        await reply(update, context, f"No {filename} found yet.")
        return

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        last_lines = lines[-20:]
        content = "".join(last_lines)
        if not content.strip():
            await reply(update, context, f"{filename}: no recent entries.")
            return
        await send_safe(
            context.bot,
            update.effective_chat.id,
            f"📋 *{filename}* (last {len(last_lines)} lines)\n```\n{content}\n```"
        )
    except Exception as e:
        await reply(update, context, f"Error reading {filename}: {e}")


@authorized_only
async def send_status(update, context):
    """
    Shows system health: uptime, last brief, Claude call budget, advisor.db
    size, positions loaded, data source connectivity, and today's error
    count.
    /status
    """
    import re
    from pathlib import Path

    lines = ["🖥️ *SYSTEM STATUS*", ""]

    # Uptime
    uptime_seconds = time.time() - _BOT_START_TIME
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    lines.append(f"⏱ Uptime: {hours}h {minutes}m")

    # Last brief — cross-checked against brief.log's completion marker
    # (see scheduled_morning_brief()), not just the saved-brief file's
    # mtime: if a brief fails partway through, a partial file can still
    # land with a timestamp that looks like a clean run. Whichever source
    # is more recent wins; if they disagree by more than a few minutes,
    # that itself indicates a partial failure worth surfacing.
    briefs_dir = Path("equity/data/briefs")
    brief_files = sorted(briefs_dir.glob("brief_*.txt")) if briefs_dir.exists() else []
    file_mtime = datetime.fromtimestamp(brief_files[-1].stat().st_mtime) if brief_files else None

    last_brief_from_log = None
    try:
        with open("equity/data/logs/brief.log", encoding="utf-8") as f:
            log_content = f.read()
        log_matches = re.findall(
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?(?:Generated|[Bb]rief.*complete)",
            log_content,
        )
        if log_matches:
            last_brief_from_log = datetime.strptime(log_matches[-1], "%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    candidates = [t for t in (file_mtime, last_brief_from_log) if t is not None]
    if candidates:
        latest = max(candidates)
        age_hours = (datetime.now() - latest).total_seconds() / 3600
        age_str = f"{age_hours:.1f}h ago" if age_hours < 24 else f"{age_hours / 24:.1f}d ago"
        lines.append(f"📊 Last brief: {latest.strftime('%Y-%m-%d %H:%M')} ({age_str})")
        if (
            file_mtime and last_brief_from_log
            and abs((file_mtime - last_brief_from_log).total_seconds()) > 300
        ):
            lines.append(
                f"   ⚠️ file mtime {file_mtime.strftime('%Y-%m-%d %H:%M')} vs "
                f"log completion {last_brief_from_log.strftime('%Y-%m-%d %H:%M')} "
                f"— possible partial failure"
            )
    else:
        lines.append("📊 Last brief: never run")

    # Claude API usage — _claude_call_times only ever holds the trailing
    # hour (check_claude_rate_limit prunes it against that same window),
    # so report against the hourly limit it's actually tracking.
    lines.append(f"🤖 Claude calls (last hour): {len(_claude_call_times)}/{MAX_CLAUDE_CALLS_PER_HOUR}")

    # advisor.db size
    db_path = Path("equity/data/advisor.db")
    if db_path.exists():
        db_mb = db_path.stat().st_size / 1024 / 1024
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            thread_count = conn.execute("SELECT COUNT(*) FROM threads").fetchone()[0]
            conn.close()
            lines.append(f"💾 advisor.db: {db_mb:.1f}MB | {thread_count} threads | {msg_count} messages")
        except Exception:
            lines.append(f"💾 advisor.db: {db_mb:.1f}MB")
    else:
        lines.append("💾 advisor.db: not found")

    # Positions loaded
    lines.append(f"📋 Positions: {len(POSITIONS)} held | {len(WATCHLIST)} watchlist")

    from equity.data.monitoring import load_monitoring

    mon_items = load_monitoring()
    if mon_items:
        high = sum(1 for i in mon_items if i.get("priority") == "high")
        med = sum(1 for i in mon_items if i.get("priority") == "medium")
        lines.append(f"📋 Monitoring: {len(mon_items)} items ({high} high, {med} medium)")

    stale_count = sum(1 for d in price_cache.get_prices().values() if d.get("is_stale", False))
    stale_warning = f" | ⚠️ {stale_count} stale" if stale_count > 0 else ""
    lines.append(f"💹 {price_cache.coverage_report()}{stale_warning}")

    # Data source health checks
    lines.append("")
    lines.append("📡 *Data Sources*")

    try:
        test = yf.Ticker("SPY").history(period="1d")
        yf_status = "✓" if len(test) > 0 else "✗ empty response"
    except Exception as e:
        yf_status = f"✗ {str(e)[:50]}"
    lines.append(f"  yfinance: {yf_status}")

    try:
        fred_key = os.getenv("FRED_API_KEY", "")
        if not fred_key:
            fred_status = "✗ FRED_API_KEY not set"
        else:
            from fredapi import Fred
            Fred(api_key=fred_key).get_series("DFF", limit=1)
            fred_status = "✓"
    except Exception as e:
        fred_status = f"✗ {str(e)[:50]}"
    lines.append(f"  FRED: {fred_status}")

    try:
        import requests

        from equity.config.settings import FMP_API_KEY
        r = requests.get(
            "https://financialmodelingprep.com/stable/quote",
            params={"symbol": "AAPL", "apikey": FMP_API_KEY},
            timeout=5,
        )
        fmp_status = "✓" if r.status_code == 200 else f"✗ HTTP {r.status_code}"
    except Exception as e:
        fmp_status = f"✗ {str(e)[:50]}"
    lines.append(f"  FMP: {fmp_status}")

    anthropic_status = "✓ (key present)" if os.getenv("ANTHROPIC_API_KEY") else "✗ ANTHROPIC_API_KEY not set"
    lines.append(f"  Anthropic: {anthropic_status}")

    lines.append("  IBKR: ✗ not connected (Module 4 pending)")

    # Recent errors
    lines.append("")
    log_path = Path("equity/data/logs/errors.log")
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            error_lines = f.readlines()
        today_errors = [l for l in error_lines if datetime.now().strftime("%Y-%m-%d") in l]
        lines.append(f"⚠️ Errors today: {len(today_errors)} — /logs errors for details")
    else:
        lines.append("⚠️ Error log: not yet created")

    await send_in_parts(
        context.bot, update.effective_chat.id,
        "\n".join(lines), reply_markup=make_main_menu()
    )


@require_email_auth(lambda u, c: "Restart bot process")
@authorized_only
@register_command
async def send_restart(update, context):
    """
    /restart — replaces the running process with a fresh one via os.execv.
    Requires email 2FA since it interrupts service.

    Re-execs as `python -m equity.telegram.bot` explicitly — this file's
    own documented invocation (see the "Run with:" note at the top) —
    rather than replaying sys.argv. After a `-m` launch, sys.argv[0] is
    the resolved absolute path to *this file*, not "-m equity.telegram.bot"
    — so `os.execv(sys.executable, [sys.executable] + sys.argv)` doesn't
    restart the same way it was started: it re-launches as a direct
    script instead of a module. That changes what lands on sys.path[0]
    (this file's own directory, equity/telegram/, instead of the repo
    root that CWD supplies to a `-m` launch), and every `from equity.x
    import y` absolute import in the codebase — nearly all of them —
    fails immediately on the new process's startup. Hardcoding the
    correct module path sidesteps that rather than trying to detect and
    reconstruct whatever invocation form the original launch used.

    os.execv replaces the current process image in place (same PID, same
    CWD, same environment — there's nothing else to reconstruct) — this
    works identically whether the bot is running under systemd/a process
    supervisor or as a bare foreground process; no supervisor
    coordination needed, and there's nothing for a supervisor to detect
    or restart because the process never actually exits. That in-place
    swap also means there's never a moment with two processes racing
    Telegram's getUpdates long-poll for the same bot token, which a
    fork-then-exit (spawn the new process, then exit the old one)
    approach would risk during its handover window.
    """
    await reply(update, context, "🔄 Restarting bot... will be back in ~10 seconds.")
    security_logger.warning("WRITE_OP | restart")
    logger.info("Bot restart requested via /restart command")

    await asyncio.sleep(1)  # give Telegram time to deliver the message
    os.execv(sys.executable, [sys.executable, "-m", "equity.telegram.bot"])


@require_email_auth(lambda u, c: "Kill bot process")
@authorized_only
@register_command
async def send_kill(update, context):
    """
    /kill — stops the bot process entirely, no restart. For when /restart
    itself isn't working and the bot needs to be brought back up manually
    (SSH + supervisor restart, or a fresh `python -m equity.telegram.bot`).
    Requires email 2FA since it interrupts service.

    os._exit(0) — an immediate process exit with no Python-level cleanup
    (no atexit handlers, no flushing anything not already flushed). That's
    the point: a plain `sys.exit()` inside a handler wouldn't reliably tear
    down the asyncio event loop the bot is running under.
    """
    await reply(update, context, "🛑 Stopping bot process.")
    security_logger.warning("WRITE_OP | kill")
    logger.info("Bot kill requested via /kill command")

    await asyncio.sleep(1)  # give Telegram time to deliver the message
    os._exit(0)


def _execute_pending_change(change: dict) -> str:
    """Shared execution logic for a confirmed pending change. Returns a result message."""
    change_type = change["change_type"]
    payload = change["payload"]

    if change_type == "add_watchlist":
        ticker = payload["ticker"]
        ok = config_manager.add_to_watchlist(ticker, payload["entry"], "Added via Telegram advisor")
        if ok:
            security_logger.warning(f"WRITE_OP | add_watchlist | ticker={ticker}")
            return f"✓ {ticker} added to watchlist. {config_commands._git_head_hash()}"
        return f"❌ Failed to add {ticker} to watchlist."

    if change_type == "set_config":
        field, value = payload["field"], payload["value"]
        ok = config_manager.update_market_config(field, value, "Updated via Telegram advisor")
        if ok:
            security_logger.warning(f"WRITE_OP | set_config | field={field} value={value}")
            return f"✓ {field} set to {value}. {config_commands._git_head_hash()}"
        return f"❌ Failed to update {field}."

    return f"❌ Unknown change type: {change_type}"


@authorized_only
async def send_confirm(update, context):
    change = thread_manager.get_latest_pending_change()
    if change is None:
        await reply(update, context, "No pending change to confirm (or it has expired).")
        return
    result = _execute_pending_change(change)
    thread_manager.clear_pending_change(change["id"])
    await reply(update, context, result)


@authorized_only
async def send_cancel(update, context):
    auth_manager.cancel_pending_auth(TELEGRAM_USER_ID)
    thread_manager.clear_latest_pending_change()
    await reply(update, context, "❌ Cancelled.", reply_markup=make_main_menu())


# ---------------------------------------------------------------------------
# Write handlers — @require_email_auth (which internally applies
# @authorized_only + @require_write_access). @authorized_only is also
# applied directly here so the security_check.py static-AST scan — which
# requires every add_handler'd function to carry a literal @authorized_only
# — can verify it without parsing through require_email_auth's internals;
# it's a redundant, harmless second check (same predicate, checked twice).
#
# @register_command sits closest to the raw function (applied first) so
# COMMAND_REGISTRY captures the *undecorated* body — handle_message's
# post-auth re-dispatch calls COMMAND_REGISTRY[func_name] directly, and it
# must reach the real logic without re-triggering require_email_auth's
# "send another code" path.
# ---------------------------------------------------------------------------

@require_email_auth(lambda u, c: f'Add {c.args[0] if c.args else "?"} to watchlist')
@authorized_only
@register_command
async def send_add(update, context):
    await config_commands.handle_add_watchlist(update, context, advisor, thread_manager, auth_manager)


@require_email_auth(lambda u, c: f'Remove {c.args[0] if c.args else "?"} from watchlist')
@authorized_only
@register_command
async def send_remove(update, context):
    await config_commands.handle_remove_watchlist(update, context, thread_manager, auth_manager)


@require_email_auth(lambda u, c: f'Update {c.args[0] if c.args else "?"} thesis')
@authorized_only
@register_command
async def send_update(update, context):
    await config_commands.handle_update_thesis(update, context, advisor, thread_manager, auth_manager)


@require_email_auth(lambda u, c: f'Update market config: {" ".join(c.args or [])}')
@authorized_only
@register_command
async def send_set(update, context):
    await config_commands.handle_set_config(update, context, thread_manager, auth_manager)


@require_email_auth(lambda u, c: f'Dismiss monitoring item for {c.args[0] if c.args else "?"}')
@authorized_only
@register_command
async def send_dismiss(update, context):
    """
    /dismiss TICKER — if exactly one monitoring item is active for the
    ticker, dismisses it directly. If multiple are active, shows a
    selection keyboard so the user can choose which one (or dismiss all).
    /dismiss ID — dismisses one specific item by ID (as printed by
    /monitoring). Requires email 2FA since it modifies persistent state.

    Carries @register_command (like send_add/remove/update/set) because it
    goes through @require_email_auth: after the emailed code is verified,
    handle_message() re-dispatches via COMMAND_REGISTRY[func_name] — without
    this decorator the post-auth call would silently no-op.
    """
    from equity.data.monitoring import dismiss_monitoring_item, load_monitoring

    if not context.args:
        await reply(update, context,
            "Usage: /dismiss TICKER or /dismiss ID\nExample: /dismiss TSLA\n\n"
            "Shows active monitoring items for that ticker to dismiss."
        )
        return

    arg = context.args[0]
    active = load_monitoring()

    # Item IDs are lowercase "ticker_date_index" strings (see
    # equity.data.monitoring.add_monitoring_items) and can never collide
    # with an uppercased ticker filter, so checking for an exact ID match
    # first is unambiguous.
    item_by_id = next((i for i in active if i.get("id") == arg), None)
    if item_by_id:
        dismiss_monitoring_item(item_by_id["id"], reason="Dismissed via /dismiss command")
        await reply(update, context,
            f'✓ Dismissed monitoring item for {item_by_id["ticker"]}:\n_{item_by_id["item"]}_',
            reply_markup=make_main_menu()
        )
        return

    ticker = arg.upper()
    items = [i for i in active if i.get("ticker", "").upper() == ticker]

    if not items:
        await reply(update, context,
            f"No active monitoring items for {ticker}.",
            reply_markup=make_main_menu()
        )
        return

    if len(items) == 1:
        item = items[0]
        dismiss_monitoring_item(item["id"], reason="Dismissed via /dismiss command")
        await reply(update, context,
            f'✓ Dismissed monitoring item for {ticker}:\n_{item["item"]}_',
            reply_markup=make_main_menu()
        )
        return

    await _send_dismiss_selection(update, context, ticker, items)


@authorized_only
async def send_save(update, context):
    """/save TICKER — draft a position update from that ticker's discussion thread.

    Unlike /add, /remove, /update, /set, this doesn't go through
    @require_email_auth: propose_position_save() sends the auth email
    itself, after drafting, so the emailed review shows exactly what's
    about to be saved rather than a generic "operation" description sent
    before any content exists.
    """
    if not context.args:
        await reply(update, context,
            "Usage: /save TICKER\nExample: /save MSFT\n\n"
            "Drafts a position update from your current discussion thread."
        )
        return
    ticker = context.args[0].upper()
    await propose_position_save(update, context, ticker)


# ---------------------------------------------------------------------------
# Callback query handler
# ---------------------------------------------------------------------------

@authorized_only
async def handle_callback(update, context):
    query = update.callback_query
    try:
        await query.answer()
    except BadRequest:
        pass  # callback query expired — harmless, continue processing
    data = query.data
    chat_id = query.message.chat_id

    if data == "cmd_brief": await send_brief(update, context)
    elif data == "cmd_screener": await send_screener(update, context)
    elif data == "cmd_portfolio": await send_portfolio(update, context)
    elif data == "cmd_news": await send_news(update, context)
    elif data == "cmd_watchlist": await send_watchlist(update, context)
    elif data == "cmd_threads": await send_threads(update, context)
    elif data == "cmd_performance":
        perf = await run_in_executor(fetch_benchmark_performance)
        rel = await run_in_executor(fetch_position_relative_performance)
        spot = fetch_spot_prices()
        port = fetch_portfolio_performance()
        text = format_performance_section(perf, port, rel, spot)
        await send_in_parts(context.bot, chat_id, text, reply_markup=make_main_menu())
    elif data == "cmd_sectors":
        sector_data = await run_in_executor(fetch_sector_data)
        text = format_sector_section(sector_data)
        await send_in_parts(context.bot, chat_id, text, reply_markup=make_main_menu())
    elif data == "cmd_calendar":
        cal = await run_in_executor(fetch_eco_calendar, 7)
        earnings = await run_in_executor(fetch_earnings_calendar, 14)
        text = _format_section2_eco_calendar(cal, earnings)
        await send_in_parts(context.bot, chat_id, text, reply_markup=make_main_menu())
    elif data == "cmd_status":
        await send_status(update, context)
    elif data == "cmd_monitoring":
        await send_monitoring(update, context)
    elif data == "cmd_discuss_menu":
        await query.message.reply_text(
            "Choose a ticker:", reply_markup=make_discuss_menu(POSITIONS, WATCHLIST)
        )
    elif data == "cmd_main_menu":
        await query.message.reply_text("What would you like to do?", reply_markup=make_main_menu())
    elif data == "cmd_done":
        thread_manager.clear_active_thread(TELEGRAM_USER_ID)
        await query.message.reply_text(
            "✓ Thread paused. Resume anytime with /discuss TICKER", reply_markup=make_main_menu()
        )
    elif data == "cmd_macro":
        await start_or_resume_discussion("MACRO", update, context, thread_type="topic")
    elif data == "cmd_prices":
        await send_prices(update, context)
    elif data == "cmd_portfolio_review":
        await start_or_resume_discussion("PORTFOLIO", update, context, thread_type="topic")
    elif data == "cmd_other_ticker":
        await query.message.reply_text("Type the ticker symbol to discuss:")
        context.user_data["awaiting_ticker_input"] = True
    elif data.startswith("discuss_"):
        ticker = data[len("discuss_"):]
        await start_or_resume_discussion(ticker, update, context)
    elif data.startswith("switch_"):
        thread_id = data[len("switch_"):]
        thread_manager.set_active_thread(TELEGRAM_USER_ID, thread_id)
        info = thread_manager.get_thread_info(thread_id)
        subject = info["subject"] if info else thread_id
        await _send_switch_confirmation(update, context, thread_id, subject)
    elif data.startswith("ticker_news_"):
        ticker = data[len("ticker_news_"):]
        if not _is_valid_ticker(ticker):
            await query.message.reply_text(
                f"{ticker} is a monitoring subject, not a tradable ticker — no news to fetch."
            )
            return
        triage = await run_in_executor(run_news_triage, [ticker])
        text = format_news_triage(triage)
        await send_in_parts(context.bot, chat_id, text, reply_markup=make_ticker_actions(ticker))
    elif data.startswith("ticker_metrics_"):
        ticker = data[len("ticker_metrics_"):]
        if not _is_valid_ticker(ticker):
            await query.message.reply_text(
                f"{ticker} is a monitoring subject, not a tradable ticker — no quality metrics to fetch."
            )
            return
        score = await run_in_executor(score_ticker, ticker, os.getenv("FMP_API_KEY"))
        lines = [
            f"📊 *{ticker} Quality Metrics*",
            f'Score: {score.get("quality_score")}/100 ({score.get("tier")})',
            f'ROIC: {score.get("roic_current", "N/A")}%',
            f'Net Debt/EBITDA: {score.get("net_debt_ebitda", "N/A")}x',
            f'Rev CAGR 3Y: {score.get("revenue_cagr_3y", "N/A")}%',
            f'EBITDA Margin: {score.get("ebitda_margin_3y_avg", "N/A")}%',
            f'CFO≥NI: {score.get("cfo_gte_ni", "N/A")}',
            f'Flags: {", ".join(score.get("red_flags", [])) or "None"}',
        ]
        await context.bot.send_message(
            chat_id=chat_id, text="\n".join(lines), parse_mode="Markdown",
            reply_markup=make_ticker_actions(ticker),
        )
    elif data.startswith("ticker_refresh_"):
        ticker = data[len("ticker_refresh_"):]
        await query.message.reply_text(f"🔄 Refreshing {ticker}...")
        context_str = await run_in_executor(advisor.get_ticker_context, ticker)
        active_thread = f"ticker_{ticker}"
        system_prompt = advisor.build_system_prompt(thread_subject=ticker, current_thread_id=active_thread)
        response = await run_in_executor(
            advisor.chat, active_thread, f"Context refresh for {ticker}:\n{context_str}",
            system_prompt, ticker,
        )
        await send_in_parts(context.bot, chat_id, response, reply_markup=make_ticker_actions(ticker))
    elif data.startswith("ticker_update_thesis_"):
        ticker = data[len("ticker_update_thesis_"):]
        await propose_position_save(update, context, ticker)
    elif data.startswith("ticker_add_watchlist_"):
        ticker = data[len("ticker_add_watchlist_"):]
        context.args = [ticker]
        await send_add(update, context)
    elif data.startswith("ticker_flag_risk_"):
        ticker = data[len("ticker_flag_risk_"):]
        context.args = [ticker, "stop_thesis"]
        await send_update(update, context)
    elif data.startswith("dismiss_item_"):
        item_id = data[len("dismiss_item_"):]
        from equity.data.monitoring import dismiss_monitoring_item, load_monitoring
        # load_monitoring() only returns active items — look the text up
        # before dismissing, since a dismissed item wouldn't be found after.
        item_text = next(
            (i.get("item", item_id) for i in load_monitoring() if i.get("id") == item_id),
            item_id
        )
        success = dismiss_monitoring_item(item_id, reason="Dismissed via selection menu")
        if success:
            await query.message.reply_text(
                f"✓ Dismissed:\n_{item_text[:100]}_",
                parse_mode="Markdown",
                reply_markup=make_main_menu()
            )
        else:
            await query.message.reply_text(
                "Item not found or already dismissed.",
                reply_markup=make_main_menu()
            )
    elif data.startswith("dismiss_all_"):
        ticker = data[len("dismiss_all_"):]
        from equity.data.monitoring import dismiss_monitoring
        count = dismiss_monitoring(ticker, reason="Dismiss all via selection menu")
        await query.message.reply_text(
            f"✓ Dismissed all {count} monitoring item(s) for {ticker}.",
            reply_markup=make_main_menu()
        )
    elif data.startswith("confirm_"):
        await config_commands.handle_confirm_callback(query, context, thread_manager)
    elif data.startswith("cancel_"):
        await config_commands.handle_cancel_callback(query, context, thread_manager)
    elif data.startswith("headlines_"):
        parts = data.split("_")
        ticker = parts[1]
        page = int(parts[2]) if len(parts) > 2 else 0
        triage = await run_in_executor(run_news_triage, [ticker])
        all_headlines = triage.get(ticker, {}).get("all_headlines", [])
        if not all_headlines:
            await query.message.reply_text(f"No headlines available for {ticker}.")
            return
        text = format_headline_page(ticker, all_headlines, page, NEWS_HEADLINE_PAGE_SIZE)
        kb = make_headline_page_keyboard(ticker, page, len(all_headlines), NEWS_HEADLINE_PAGE_SIZE)
        await send_safe(context.bot, chat_id, text, reply_markup=kb)
    elif data == "noop":
        pass
    elif data.startswith("suggest_"):
        idx = int(data.split("_", 1)[1])
        suggestions = context.user_data.get("suggestions", [])
        if idx < len(suggestions):
            active_thread = thread_manager.get_active_thread(TELEGRAM_USER_ID)
            if active_thread and check_claude_rate_limit():
                thread_info = thread_manager.get_thread_info(active_thread)
                subject = thread_info["subject"] if thread_info else None
                system_prompt = advisor.build_system_prompt(thread_subject=subject, current_thread_id=active_thread)
                await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                response = await run_in_executor(
                    advisor.chat, active_thread, suggestions[idx], system_prompt, subject
                )
                new_sugg = advisor.get_follow_up_suggestions(active_thread, response)
                await send_in_parts(context.bot, chat_id, response)
                if new_sugg:
                    context.user_data["suggestions"] = new_sugg
                    await context.bot.send_message(
                        chat_id=chat_id, text="💡 *You might ask:*", parse_mode="Markdown",
                        reply_markup=make_suggestions_keyboard(new_sugg),
                    )
                # Always surface action buttons after an advisor response —
                # same fix as handle_message() below, so the "Reply to
                # continue, or choose an action" affordance isn't lost after
                # a suggestion round-trip either.
                if active_thread.startswith("ticker_"):
                    await context.bot.send_message(
                        chat_id=chat_id, text="Actions:",
                        reply_markup=make_ticker_actions(active_thread.split("_", 1)[1]),
                    )
                else:
                    await context.bot.send_message(chat_id=chat_id, text="Actions:", reply_markup=make_main_menu())


# ---------------------------------------------------------------------------
# Plain-text message handler
# ---------------------------------------------------------------------------

@authorized_only
async def handle_message(update, context):
    user_id = update.effective_user.id
    message_text = update.message.text.strip()

    # Rate limit
    if not check_rate_limit(user_id):
        await reply(update, context,
            "⏱ Too many messages. Wait a moment and try again."
        )
        return

    # Priority 1: pending email auth
    if auth_manager.is_awaiting_auth(user_id):
        if message_text.startswith("/cancel"):
            auth_manager.cancel_pending_auth(user_id)
            await reply(update, context,
                "❌ Authorization cancelled.", reply_markup=make_main_menu()
            )
            return
        success, payload = auth_manager.verify_email_code(user_id, message_text)
        if success:
            await reply(update, context, "✓ Authorized.")
            pending = context.user_data.pop("awaiting_email_auth", {})
            func_name = pending.get("func_name")
            context.args = context.user_data.pop("pending_command_args", [])
            action = context.user_data.pop("pending_auth_action", {})
            if action.get("action") == "add_watchlist":
                await config_commands.execute_add_watchlist(
                    action["ticker"], update, context, advisor, thread_manager
                )
            elif action.get("action") == "position_update":
                ticker = action["ticker"]
                updates = action["updates"]
                from equity.config.config_manager import save_thesis_update, update_position_tier

                thesis_fields = {k: v for k, v in updates.items()
                                  if k in ("thesis", "thesis_breakers", "macro_thesis",
                                           "target_exit_conditions", "last_reviewed")}
                if thesis_fields:
                    save_thesis_update(ticker, thesis_fields,
                                        f'Updated from Telegram discussion {datetime.now().strftime("%Y-%m-%d")}')

                if updates.get("tier_v2"):
                    update_position_tier(
                        ticker,
                        tier_v2=updates["tier_v2"],
                        style=updates.get("style", ""),
                        classification_status=updates.get("classification_status", "complete"),
                        reason=f'Classified from Telegram discussion {datetime.now().strftime("%Y-%m-%d")}'
                    )

                security_logger.warning(f"WRITE_OP | position_update | ticker={ticker}")

                await reply(update, context,
                    f"✓ *{ticker} updated successfully.*\n\n"
                    f"Thesis, tier, and classification saved.\n"
                    f"Changes committed to git.\n\n"
                    f"Use /discuss {ticker} to continue refining.",
                    reply_markup=make_ticker_actions(ticker)
                )
            elif func_name and func_name in COMMAND_REGISTRY:
                await COMMAND_REGISTRY[func_name](update, context)
        else:
            await reply(update, context,
                "❌ Invalid or expired code. Try again or /cancel to abort."
            )
        return

    # Priority 2: typed confirmation for remove
    if context.user_data.get("pending_remove_ticker"):
        expected = context.user_data["pending_remove_ticker"]
        if message_text.upper() == expected.upper():
            context.user_data.pop("pending_remove_ticker")
            config_manager.remove_from_watchlist(expected, "Removed via Telegram")
            security_logger.warning(f"WRITE_OP | remove_watchlist | ticker={expected}")
            await reply(update, context,
                f"✓ {expected} removed from watchlist.", reply_markup=make_main_menu()
            )
        else:
            await reply(update, context,
                f"Type *{expected}* exactly to confirm removal, or /cancel to abort.",
                parse_mode="Markdown",
            )
        return

    # Priority 2b: typed CONFIRM TICKER for stop_thesis
    if context.user_data.get("pending_stop_thesis_ticker"):
        expected = context.user_data["pending_stop_thesis_ticker"]
        if message_text.strip().upper() == f"CONFIRM {expected.upper()}":
            context.user_data.pop("pending_stop_thesis_ticker")
            config_manager.save_thesis_update(
                expected, {"stop_thesis": True}, "Thesis flagged broken via Telegram"
            )
            security_logger.warning(f"WRITE_OP | stop_thesis | ticker={expected}")
            await reply(update, context,
                f"🚨 {expected} thesis flagged as broken.", reply_markup=make_main_menu()
            )
        else:
            await reply(update, context,
                f'Type "CONFIRM {expected}" exactly to confirm, or /cancel to abort.'
            )
        return

    # Priority 2c: awaiting a typed ticker for "Other ticker..." button
    if context.user_data.get("awaiting_ticker_input"):
        context.user_data.pop("awaiting_ticker_input")
        await start_or_resume_discussion(message_text.strip().upper(), update, context)
        return

    active_thread = thread_manager.get_active_thread(user_id)

    # Priority 2d: brief-related keyword outside the BRIEF thread, when no
    # brief has EVER been run — nudge toward /brief rather than letting
    # Claude answer with zero brief context. A brief from a prior day is NOT
    # gated here: the advisor's cross-thread context (get_recent_briefs(),
    # recency-weighted) already surfaces it and can say e.g. "the most
    # recent brief I have is from yesterday" naturally.
    if active_thread != "topic_BRIEF":
        text_lower = message_text.lower()
        if any(kw in text_lower for kw in ("brief", "morning", "synthesis", "snapshot")):
            last_brief = get_last_brief_synthesis(thread_manager)
            if not last_brief:
                await reply(update, context,
                    "No morning brief has been generated yet. Run /brief to generate one."
                )
                return

    # Priority 3: active conversation thread
    if not active_thread:
        await reply(update, context,
            "No active discussion. Choose one:",
            reply_markup=make_discuss_menu(POSITIONS, WATCHLIST),
        )
        return

    if not check_claude_rate_limit():
        await reply(update, context,
            "⚠️ Claude call limit reached for this hour. "
            "Data commands (/portfolio, /screener) still work."
        )
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    thread_info = thread_manager.get_thread_info(active_thread)
    subject = thread_info["subject"] if thread_info else None
    system_prompt = advisor.build_system_prompt(thread_subject=subject, current_thread_id=active_thread)
    response = await run_in_executor(
        advisor.chat, active_thread, message_text, system_prompt, subject
    )
    suggestions = advisor.get_follow_up_suggestions(active_thread, response)
    await send_in_parts(context.bot, update.effective_chat.id, response)
    if suggestions:
        context.user_data["suggestions"] = suggestions
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="💡 *You might ask:*",
            parse_mode="Markdown",
            reply_markup=make_suggestions_keyboard(suggestions),
        )

    # Always show action buttons after an advisor response — previously
    # only start_or_resume_discussion() did this, so the "choose an
    # action" affordance vanished after the first reply in a thread.
    if active_thread.startswith("ticker_"):
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Actions:",
            reply_markup=make_ticker_actions(active_thread.split("_", 1)[1]),
        )
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Actions:",
            reply_markup=make_main_menu(),
        )


KNOWN_COMMANDS = [
    "discuss", "macro", "prices", "portfolio", "portfolio_review",
    "screener", "news", "brief", "watchlist", "threads",
    "switch", "add", "remove", "update", "set", "save", "framework",
    "confirm", "cancel", "done", "audit", "logs", "status", "logout",
    "start", "help", "monitoring", "dismiss", "restart", "kill",
]


@authorized_only
async def handle_unknown_command(update, context):
    import difflib
    cmd = update.message.text.lstrip("/").split()[0].lower()
    matches = difflib.get_close_matches(cmd, KNOWN_COMMANDS, n=1, cutoff=0.6)
    if matches:
        await reply(update, context,
            f"Did you mean */{matches[0]}*? "
            f"Try again or /help for all commands.",
            parse_mode="Markdown",
        )
    else:
        await reply(update, context,
            "Unknown command.", reply_markup=make_main_menu()
        )


# ---------------------------------------------------------------------------
# Scheduled morning brief
# ---------------------------------------------------------------------------

async def scheduled_morning_brief(context):
    try:
        sections = await run_in_executor(build_morning_brief)
        await run_in_executor(save_brief_to_thread, sections, thread_manager)
        for text, keyboard in sections:
            if text and text.strip():
                await send_safe(context.bot, TELEGRAM_USER_ID, text, reply_markup=keyboard)
                await asyncio.sleep(0.3)
        # Completion marker for send_status()'s "last brief" check — logged
        # under the "equity.brief" namespace (not this module's own logger)
        # so it actually lands in brief.log rather than advisor.log; see
        # equity/config/logging_config.py's per-subsystem file routing.
        logging.getLogger("equity.brief.brief_builder").info(
            "Morning brief complete — %d sections delivered", len(sections)
        )
    except Exception as e:
        await context.bot.send_message(
            chat_id=TELEGRAM_USER_ID,
            text=f"⚠️ Morning brief failed: {e}"
        )


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

async def handle_error(update, context) -> None:
    """
    Global error handler — catches any exception left unhandled by an
    individual command/callback handler (registered via
    app.add_error_handler in __main__, so it isn't itself an
    add_handler target and doesn't need @authorized_only).
    Logs the full traceback to errors.log and notifies TELEGRAM_USER_ID
    with a short summary.
    """
    error = context.error
    logger.error("Unhandled exception in handler", exc_info=error)

    update_info = ""
    if isinstance(update, Update) and update.effective_message and update.effective_message.text:
        update_info = f"Command: {update.effective_message.text[:100]}\n"

    message = (
        f"⚠️ *Bot Error*\n\n"
        f"{update_info}"
        f"Error: `{type(error).__name__}: {str(error)[:200]}`\n\n"
        f"Check `/logs errors` for full traceback."
    )

    try:
        await send_safe(context.bot, TELEGRAM_USER_ID, message)
    except Exception:
        # If notification itself fails, at least it's in the log.
        logger.error("Failed to send error notification to Telegram")


# ---------------------------------------------------------------------------
# Intraday price/news alerts
# ---------------------------------------------------------------------------

_alerted_today: dict[str, bool] = {}  # {alert_key: True}; reset when the date rolls over

# How long after startup intraday_alert_job() suppresses all alerts —
# see its docstring. _BOT_START_TIME already exists (module top, for
# /status uptime); reused here rather than duplicated.
ALERT_STARTUP_GRACE_SECONDS = 300  # 5 minutes

# Trailing window of sent alerts, in-memory only (cleared on restart — see
# /restart above) — feeds /prices' "RECENT ALERTS" synthesis context
# (`_get_recent_alert_context()`) with a bit of what's already fired today,
# not a persisted audit log (that's `security_logger`/logs/ for writes;
# alerts aren't writes).
_recent_alerts: list[dict] = []
MAX_RECENT_ALERTS = 20


def _record_alert(alert: dict) -> None:
    """Records a just-sent alert to the trailing in-memory history."""
    _recent_alerts.append({
        "ticker": alert.get("ticker"),
        "type": alert.get("type"),
        "message": alert.get("message", "")[:100],
        "timestamp": datetime.now().isoformat(),
    })
    if len(_recent_alerts) > MAX_RECENT_ALERTS:
        _recent_alerts.pop(0)


def _get_recent_alert_context() -> str:
    """Last 10 recorded alerts as text, for /prices' synthesis prompt."""
    if not _recent_alerts:
        return ""
    lines = []
    for a in _recent_alerts[-10:]:
        ts = a.get("timestamp", "")[:16]  # YYYY-MM-DDTHH:MM
        lines.append(f'[{ts}] {a["ticker"]}: {a["message"]}')
    return "\n".join(lines)


def _get_market_hours_now() -> bool:
    """True if the current time is within US market hours (9:30am-4:00pm ET, Mon-Fri)."""
    import pytz

    et = pytz.timezone("America/New_York")
    now_et = datetime.now(et)
    if now_et.weekday() >= 5:  # Saturday/Sunday
        return False
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close


def _level_crossed(prev: float, curr: float, level: float) -> bool:
    """True if the move from `prev` to `curr` crossed `level`, in either direction."""
    return (prev < level <= curr) or (prev > level >= curr)


def _fx_context_note(fx_ticker: str, move_pct: float) -> str:
    """Portfolio-relevant note for an FX move alert."""
    notes = {
        "USDJPY=X": ("JPY strengthening — risk-off signal, watch SMFG thesis" if move_pct < 0
                     else "JPY weakening — BOJ pressure or risk-on carry trade"),
        "USDCNH=X": ("CNH strengthening — watch BYDDY, TSM, EWW exposure" if move_pct > 0
                     else "CNH weakening — China macro stress, watch BYDDY, TSM, EWW"),
        "EURUSD=X": ("EUR strengthening — watch European exposure, TLT correlation" if move_pct > 0
                     else "EUR weakening — dollar strength, headwind for commodities"),
        "DXY":      ("Dollar weakening — tailwind for gold, commodities, EM" if move_pct < 0
                     else "Dollar strengthening — headwind for commodities, EM positions"),
    }
    return notes.get(fx_ticker, f'{"USD weakening" if move_pct < 0 else "USD strengthening"} — check commodity and EM exposure')


def _commodity_context_note(ticker: str, move_pct: float) -> str:
    """Portfolio-relevant note for a commodity move alert."""
    notes = {
        "GC=F": ("Gold surging — safe-haven or inflation signal, watch TLT relationship" if move_pct > 0
                 else "Gold selling — risk-on or deflation signal, watch real rate direction"),
        "CL=F": ("Oil rising — inflation risk, watch energy positioning" if move_pct > 0
                 else "Oil falling — growth concern or supply, watch FCX/commodity exposure"),
        "BZ=F": ("Brent rising — inflation risk, global demand signal" if move_pct > 0
                 else "Brent falling — demand concern, watch commodity positions"),
        "HG=F": ("Copper surging — strong global growth signal, positive for FCX, CAT, industrials" if move_pct > 0
                 else "Copper falling — growth slowdown signal, watch FCX, CAT, industrials"),
        "URA":  ("Uranium ETF surging — nuclear thesis accelerating, constructive for CCJ/CEG" if move_pct > 0
                 else "Uranium ETF falling — watch CCJ/CEG thesis, check for sector news"),
        "NG=F": ("Nat gas surging — energy cost pressure, watch utility margins" if move_pct > 0
                 else "Nat gas falling — energy cost relief"),
        "SI=F": ("Silver surging — industrial/monetary demand, watch PPLT relationship" if move_pct > 0
                 else "Silver falling — check gold/silver ratio for regime signal"),
    }
    return notes.get(ticker, f'{"Rising" if move_pct > 0 else "Falling"} — check portfolio exposure')


def _data_date_key(entry: dict | None, fallback: str) -> str:
    """The YYYY-MM-DD date the price_cache entry `entry` is actually from,
    for keying alert dedup (`_alerted_today`) to the underlying data bar
    rather than the date the job happened to run.

    `_alerted_today` is in-memory only and gets wiped on every restart —
    without this, a restart (e.g. Monday morning, still showing Friday's
    close) resets `today`'s keys to Monday's date, so a move already
    alerted on before the restart doesn't match its old key and fires
    again. Keying off the bar's own date instead means the same bar can
    never generate the same alert twice, restart or not.

    Reads `entry["last_update"]` (`"YYYY-MM-DD HH:MM UTC"`), not
    `exact_timestamp`: `exact_timestamp` is `"YYYY-MM-DD HH:MM ET"` for an
    intraday bar but `"Weekday YYYY-MM-DD close"` for a daily one (see
    `_get_session_context()`) — slicing *that* string's first 10
    characters would silently collide same-weekday bars a week apart
    (e.g. two different Wednesdays both truncating to `"Wed 2026-0"`).
    `last_update` doesn't have that daily/intraday format split — it's
    always UTC-dated first, both cases — so slicing it is safe.

    Falls back to `fallback` (the run date) when `entry` is missing or
    carries no timestamp, same as before this existed.
    """
    if not entry:
        return fallback
    last_update = entry.get("last_update") or ""
    return last_update[:10] if last_update else fallback


def _check_macro_alerts(today: str) -> list[dict]:
    """Checks Treasury yields, FX, commodities, volatility, crypto,
    international indices, and cross-asset ratios for intraday alerts.

    Returns a list of alert dicts in the same shape `intraday_alert_job()`
    already expects, each carrying a `magnitude` field (absolute move —
    % for most types, bp for yields) that `_should_enrich()` compares
    against `ENRICH_THRESHOLDS`.

    Called in executor — no async. Reads the shared `equity.data.price_cache`
    rather than doing its own fetch — see that module's docstring. 'DXY'
    isn't an FX_TICKERS pair — it resolves to the DX-Y.NYB dollar index,
    also in price_cache, rather than a separate ETF proxy.
    """
    alerts = []

    # --- Treasury yields ---
    for tenor, threshold_bp in YIELD_ALERT_BP.items():
        entry = price_cache.get_yield(tenor)
        if entry is None:
            continue
        curr = entry.get("price")
        move_bp = entry.get("change_1d_bps")
        if curr is None or move_bp is None:
            continue
        prev = curr - move_bp / 100

        alert_key = f"yield_{tenor}_{_data_date_key(entry, today)}"
        if abs(move_bp) >= threshold_bp and alert_key not in _alerted_today:
            direction = "📈" if move_bp > 0 else "📉"
            alerts.append({
                "ticker": f"{tenor} Treasury",
                "raw_ticker": TENOR_TO_CACHE_KEY.get(tenor),
                "type": "macro_yield",
                "magnitude": abs(move_bp),
                "message": (
                    f"{direction} *{tenor} Treasury yield* "
                    f"{move_bp:+.1f}bp intraday → {curr:.2f}%\n"
                    f"{'Rising yields — watch TLT, duration-sensitive positions' if move_bp > 0 else 'Falling yields — watch TLT thesis, rate-sensitive positioning'}"
                ),
                "key": alert_key,
            })

        for level in YIELD_LEVEL_ALERTS.get(tenor, []):
            level_key = f"yield_{tenor}_level_{level}_{_data_date_key(entry, today)}"
            if level_key in _alerted_today or not _level_crossed(prev, curr, level):
                continue
            direction = "broke above" if curr > prev else "broke below"
            emoji = "⚠️" if curr > prev else "✅"
            alerts.append({
                "ticker": f"{tenor} Treasury",
                "raw_ticker": TENOR_TO_CACHE_KEY.get(tenor),
                "type": "macro_yield_level",
                "magnitude": 0,
                "message": (
                    f"{emoji} *{tenor} yield {direction} {level:.2f}%*\n"
                    f"Current: {curr:.3f}% | Prior close: {prev:.3f}%\n"
                    f"Key level breach — reassess duration positioning"
                ),
                "key": level_key,
            })

    # --- FX rates ---
    dxy_entry = price_cache.get("DX-Y.NYB")
    for fx_ticker, threshold in FX_ALERT_PCT.items():
        if fx_ticker == "DXY":
            entry, label = dxy_entry, "DXY Dollar Index"
        else:
            entry = price_cache.get(fx_ticker)
            label = FX_TICKERS.get(fx_ticker, fx_ticker)
        if entry is None:
            continue
        move_pct = entry.get("change_1d_pct")
        if move_pct is None:
            continue

        alert_key = f"fx_{fx_ticker}_{_data_date_key(entry, today)}"
        if abs(move_pct) >= threshold and alert_key not in _alerted_today:
            direction = "📈" if move_pct > 0 else "📉"
            alerts.append({
                "ticker": label,
                "raw_ticker": "DX-Y.NYB" if fx_ticker == "DXY" else fx_ticker,
                "type": "macro_fx",
                "magnitude": abs(move_pct),
                "message": f"{direction} *{label}* {move_pct:+.2f}% intraday\n{_fx_context_note(fx_ticker, move_pct)}",
                "key": alert_key,
            })

    # --- Commodities ---
    for comm_ticker, threshold in COMMODITY_ALERT_PCT.items():
        entry = price_cache.get(comm_ticker)
        if entry is None:
            continue
        move_pct = entry.get("change_1d_pct")
        if move_pct is None:
            continue
        label = COMMODITY_TICKERS_EXTENDED.get(comm_ticker, comm_ticker)

        alert_key = f"commodity_{comm_ticker}_{_data_date_key(entry, today)}"
        if abs(move_pct) >= threshold and alert_key not in _alerted_today:
            direction = "📈" if move_pct > 0 else "📉"
            alerts.append({
                "ticker": label,
                "raw_ticker": comm_ticker,
                "type": "macro_commodity",
                "magnitude": abs(move_pct),
                "message": f"{direction} *{label}* {move_pct:+.2f}% intraday\n{_commodity_context_note(comm_ticker, move_pct)}",
                "key": alert_key,
            })

    # --- Volatility (VIX, VVIX — point moves, not %) ---
    for vol_ticker, threshold_pts in VOLATILITY_ALERT.items():
        entry = price_cache.get(vol_ticker)
        if entry is None:
            continue
        curr = entry.get("price")
        prev = entry.get("prev_close")
        if curr is None or prev is None:
            continue
        move_pts = curr - prev
        label = VOLATILITY_TICKERS.get(vol_ticker, vol_ticker)

        alert_key = f"vol_{vol_ticker}_{_data_date_key(entry, today)}"
        if abs(move_pts) >= threshold_pts and alert_key not in _alerted_today:
            direction = "📈" if move_pts > 0 else "📉"
            alerts.append({
                "ticker": label,
                "raw_ticker": vol_ticker,
                "type": "volatility",
                "magnitude": abs(move_pts),
                "message": f"{direction} *{label}* {move_pts:+.1f}pts intraday → {curr:.1f}",
                "key": alert_key,
            })

    # --- Crypto ---
    for crypto_ticker, threshold in CRYPTO_ALERT_PCT.items():
        entry = price_cache.get(crypto_ticker)
        if entry is None:
            continue
        move_pct = entry.get("change_1d_pct")
        if move_pct is None:
            continue
        label = CRYPTO_TICKERS.get(crypto_ticker, crypto_ticker)

        alert_key = f"crypto_{crypto_ticker}_{_data_date_key(entry, today)}"
        if abs(move_pct) >= threshold and alert_key not in _alerted_today:
            direction = "📈" if move_pct > 0 else "📉"
            alerts.append({
                "ticker": label,
                "raw_ticker": crypto_ticker,
                "type": "crypto",
                "magnitude": abs(move_pct),
                "message": f"{direction} *{label}* {move_pct:+.1f}% intraday",
                "key": alert_key,
            })

    # --- International indices ---
    for intl_ticker, threshold in INTL_ALERT_PCT.items():
        entry = price_cache.get(intl_ticker)
        if entry is None:
            continue
        move_pct = entry.get("change_1d_pct")
        if move_pct is None:
            continue
        label = INTERNATIONAL_INDICES.get(intl_ticker, intl_ticker)

        alert_key = f"intl_{intl_ticker}_{_data_date_key(entry, today)}"
        if abs(move_pct) >= threshold and alert_key not in _alerted_today:
            direction = "📈" if move_pct > 0 else "📉"
            alerts.append({
                "ticker": label,
                "raw_ticker": intl_ticker,
                "type": "international",
                "magnitude": abs(move_pct),
                "message": f"{direction} *{label}* {move_pct:+.2f}% intraday",
                "key": alert_key,
            })

    # --- Cross-asset ratio alerts ---
    ratio_alert_configs = {
        "copper_gold": {
            "move_threshold_pct": 1.5,
            "description": "Copper/Gold ratio",
            "portfolio_note": "Growth signal — affects FCX, CAT, industrials thesis",
        },
        "vix_vvix": {
            "level_threshold": 0.30,
            "description": "VIX/VVIX ratio",
            "portfolio_note": "Elevated = options market pricing tail risk",
        },
        "silver_gold": {
            "move_threshold_pct": 2.0,
            "description": "Silver/Gold ratio",
            "portfolio_note": "Risk appetite signal — affects PPLT, precious metals thesis",
        },
    }
    for ratio_name, config in ratio_alert_configs.items():
        if ratio_name not in CROSS_ASSET_RATIOS:
            continue
        t1, t2, _desc = CROSS_ASSET_RATIOS[ratio_name]
        d1 = price_cache.get(t1)
        d2 = price_cache.get(t2)
        if not d1 or not d2 or not d2.get("price"):
            continue

        curr_ratio = d1["price"] / d2["price"]
        if d1.get("prev_close") and d2.get("prev_close"):
            prev_ratio = d1["prev_close"] / d2["prev_close"]
            ratio_move_pct = (curr_ratio / prev_ratio - 1) * 100 if prev_ratio else 0
        else:
            ratio_move_pct = 0

        # Keyed to the run date (today), not _data_date_key(): a ratio is
        # computed fresh from two other tickers' prices each run, not
        # fetched as its own bar with its own timestamp — there's no
        # single underlying "data date" to key it to instead.
        alert_key = f"ratio_{ratio_name}_{today}"
        should_alert = False
        alert_note = ""

        if "move_threshold_pct" in config and abs(ratio_move_pct) >= config["move_threshold_pct"]:
            should_alert = True
            direction = "↑" if ratio_move_pct > 0 else "↓"
            alert_note = f'{direction} {ratio_move_pct:+.2f}% today → {config["portfolio_note"]}'

        if "level_threshold" in config and curr_ratio > config["level_threshold"]:
            should_alert = True
            alert_note = f'Ratio at {curr_ratio:.3f} (threshold: {config["level_threshold"]}) → {config["portfolio_note"]}'

        if should_alert and alert_key not in _alerted_today:
            alerts.append({
                "ticker": config["description"],
                "type": "ratio",
                "magnitude": abs(ratio_move_pct),
                "message": (
                    f'📊 *{config["description"]}* alert\n'
                    f"Current: {curr_ratio:.4f}\n"
                    f"{alert_note}"
                ),
                "key": alert_key,
            })

    return alerts


# ---------------------------------------------------------------------------
# Alert enrichment (web search via Claude) — only for significant alerts
# ---------------------------------------------------------------------------

# Significance threshold — only enrich alerts at or above this magnitude
# (% for most types, bp for yields, points for VIX/VVIX). 0 means "always
# enrich"; a type with no entry here is never enriched.
ENRICH_THRESHOLDS = {
    "price":             5.0,   # >5% equity move → enrich
    "news":              0,     # all thesis alerts → enrich
    "macro_yield":       12,    # >12bp yield move → enrich
    "macro_yield_level": 0,     # all level breaks → enrich
    "macro_fx":          0.8,   # >0.8% FX move → enrich
    "macro_commodity":   3.0,   # >3% commodity move → enrich
    "volatility":        0,     # all VIX/VVIX alerts → enrich
    "crypto":            7.0,   # >7% crypto move → enrich
    "international":     2.0,   # >2% international index → enrich
}


def _should_enrich(alert: dict) -> bool:
    """True if `alert` is significant enough to warrant web-search enrichment."""
    threshold = ENRICH_THRESHOLDS.get(alert.get("type", ""))
    if threshold is None:
        return False
    if threshold == 0:
        return True
    return abs(alert.get("magnitude", 0)) >= threshold


def _build_enrichment_prompt(alert: dict) -> str | None:
    """The web-search prompt for `alert`, or None if its type isn't enrichable."""
    alert_type = alert.get("type", "")
    ticker = alert.get("ticker", "")
    message = alert.get("message", "")

    if alert_type in ("price", "news"):
        from equity.config.positions import POSITIONS

        pos_data = POSITIONS.get(ticker, {})
        thesis = (pos_data.get("thesis", "") or "")[:200]
        thesis_breakers = (pos_data.get("thesis_breakers", []) or [])[:3]
        return (
            f"Search for the latest news about {ticker} stock today. "
            f"Focus on: earnings, guidance, analyst actions, sector news, "
            f"any company-specific catalyst explaining today's price move. "
            f"Current move: {message}\n"
            f"Position thesis context: {thesis}\n"
            f'Thesis-breakers to watch: {"; ".join(thesis_breakers)}\n\n'
            f"Return a 3-5 sentence briefing: (1) most likely catalyst for the move, "
            f"(2) whether any thesis-breaker conditions are triggered or proximate, "
            f"(3) recommended immediate action: monitor/discuss/act. "
            f"Be direct and specific. If no clear catalyst found, say so."
        )
    if alert_type in ("macro_yield", "macro_yield_level"):
        return (
            f"Search for today's news explaining the following Treasury yield move: {message}\n"
            f"Focus on: Fed speakers, economic data releases, auction results, "
            f"geopolitical events, or technical level breaks.\n\n"
            f"Return a 3-sentence briefing: (1) most likely driver of the move, "
            f"(2) implications for duration positioning (TLT) and rate-sensitive equities, "
            f"(3) whether this changes the rate trajectory thesis. "
            f"Be direct. If no clear driver found, say so."
        )
    if alert_type == "macro_fx":
        return (
            f"Search for today's news explaining this FX move: {message}\n"
            f"Focus on: central bank statements, economic data, political events, "
            f"intervention signals, or carry trade unwinds.\n\n"
            f"Return a 3-sentence briefing: (1) most likely driver, "
            f"(2) portfolio implications (BYDDY/EWW for CNH, SMFG for JPY, etc.), "
            f"(3) whether this is a regime shift or single-day noise."
        )
    if alert_type == "macro_commodity":
        return (
            f"Search for today's news explaining this commodity move: {message}\n"
            f"Focus on: supply/demand news, geopolitical events, inventory data, "
            f"dollar moves, or sector-specific catalysts.\n\n"
            f"Return a 3-sentence briefing: (1) most likely driver, "
            f"(2) implications for relevant portfolio positions (CCJ/CEG for uranium, "
            f"FCX for copper, etc.), "
            f"(3) whether this is consistent with or contradicts current macro thesis."
        )
    if alert_type == "volatility":
        return (
            f"Search for today's news explaining this volatility move: {message}\n"
            f"Focus on: market stress catalysts, options positioning, macro data "
            f"surprises, or geopolitical events.\n\n"
            f"Return a 3-sentence briefing: (1) most likely driver, "
            f"(2) portfolio risk implications, "
            f"(3) whether this is a genuine regime shift or single-day noise."
        )
    if alert_type == "crypto":
        return (
            f"Search for today's news explaining this crypto move: {message}\n"
            f"Focus on: regulatory news, ETF flows, macro liquidity conditions, "
            f"exchange or network-specific events.\n\n"
            f"Return a 3-sentence briefing: (1) most likely driver, "
            f"(2) whether this reflects broader risk sentiment relevant to the "
            f"equity portfolio, (3) monitor/discuss/act recommendation."
        )
    if alert_type == "international":
        return (
            f"Search for today's news explaining this international index move: {message}\n"
            f"Focus on: local central bank action, political events, trade/tariff "
            f"news, or earnings-season effects specific to that market.\n\n"
            f"Return a 3-sentence briefing: (1) most likely driver, "
            f"(2) read-through for US markets and portfolio exposure "
            f"(BYDDY/TSM/EWW/EEM), (3) whether this is contained or could spread."
        )
    return None


def _enrich_alert_with_context(alert: dict) -> str:
    """For significant alerts, fetches news/macro context via Claude with web search.

    Returns enriched alert text; falls back to the plain message on any
    failure, on a missing thesis/prompt mapping, or when the existing
    hourly Claude budget (`check_claude_rate_limit()` — the same guard
    `handle_message`'s chat calls go through) is exhausted, so alert
    enrichment can never itself blow through that budget unnoticed.

    Synchronous and blocking (the web-search-augmented call can take
    several seconds) — call via `run_in_executor()`, not directly from
    an async context.
    """
    message = alert.get("message", "")
    ticker = alert.get("ticker", "")

    search_prompt = _build_enrichment_prompt(alert)
    if search_prompt is None:
        return message

    if not check_claude_rate_limit():
        logger.warning("_enrich_alert_with_context: Claude rate limit reached — skipping enrichment for %s", ticker)
        return message

    try:
        response = advisor.client.messages.create(
            model=ADVISOR_MODEL,
            max_tokens=400,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": search_prompt}],
        )
        text_parts = [block.text for block in response.content if hasattr(block, "text") and block.text]
        enriched = "\n".join(text_parts).strip()
        if enriched:
            return f"{message}\n\n📰 *Context:*\n{enriched}"
    except Exception as e:
        logger.warning(f"_enrich_alert_with_context failed for {ticker}: {e}")

    return message


# Alert-specific freshness thresholds — much tighter than price_cache's own
# display `is_stale` thresholds (up to 75h for equities, sized to cover a
# full weekend/holiday gap without mislabeling a Monday-morning read as
# stale — see price_cache.py's _get_session_context()). Display staleness
# answers "is this too old to even show"; this answers "is this fresh
# enough that a move computed off it is actually worth pinging about right
# now" — a materially tighter bar. Values are hours.
ALERT_MAX_AGE_HOURS = {
    "equity": 4,        # only alert during/shortly after the current session
    "index": 4,         # VIX/VVIX — same
    "futures": 2,       # trade nearly 24/7 — no excuse for a stale read
    "crypto": 1,        # 24/7 — tightest of all
    "fx": 2,             # runs Sun 5pm ET - Fri 5pm ET
    "intl_index": 4,    # only during/shortly after their own local session
    "yield": 6,          # daily-publish cadence — see docstring caveat below
    "default": 4,
}


def _is_alert_data_fresh(alert: dict) -> bool:
    """False if the price data underlying `alert` is too old for the move
    to still be worth alerting on — tight, alert-specific thresholds
    (`ALERT_MAX_AGE_HOURS`), not price_cache's much wider display-staleness
    `is_stale` (see that constant's comment for why the two need to differ).

    `raw_ticker` (the actual price_cache key — see where each alert type
    is built in `_check_macro_alerts()`) is missing for alert types not
    tied to a single price reading (news, ratio) — those pass through
    unchecked, matching how `_should_enrich()` treats unmapped types.

    Checks `data_lag_minutes` (intraday bars — 5m/1h, see price_cache.py)
    when present, else `data_age_hours` (daily bars, which is where every
    `yield` reading and any equity/futures/commodity read outside its
    fetch window falls), else falls back to price_cache's own `is_stale`
    flag if somehow neither is available (see the fallback's own comment).
    Known caveat for `yield`: FRED's 2Y/20Y series
    (`_fetch_fred_yields()`) publish roughly once a day and carry no
    intraday reading at all, so `data_age_hours` for those two tenors
    climbs past the 6h threshold for most of the day even on a fully
    current observation — a real gap between "6h is generous" intent and
    what a daily-cadence series can actually deliver, not something the
    threshold number alone fixes. Verify against a live run before relying
    on 2Y/20Y move/level alerts firing reliably.
    """
    ticker = alert.get("raw_ticker")
    if not ticker:
        return True
    data = price_cache.get(ticker)
    if not data:
        return False

    instrument_type = data.get("instrument_type", "default")
    max_age_hours = ALERT_MAX_AGE_HOURS.get(instrument_type, ALERT_MAX_AGE_HOURS["default"])

    lag_min = data.get("data_lag_minutes")
    if lag_min is not None:
        if lag_min / 60 > max_age_hours:
            logger.info(
                f"Alert suppressed for {ticker}: data is {lag_min:.0f}min old "
                f"(max {max_age_hours * 60:.0f}min for {instrument_type})"
            )
            return False
        return True

    age_hours = data.get("data_age_hours")
    if age_hours is not None:
        if age_hours > max_age_hours:
            logger.info(
                f"Alert suppressed for {ticker}: daily bar is {age_hours:.1f}h old "
                f"(max {max_age_hours}h for {instrument_type})"
            )
            return False
        return True

    # Neither field present — real price_cache entries always carry at
    # least data_age_hours once a timestamp exists, so this only happens
    # when the timestamp itself was unavailable (_get_session_context()'s
    # `last_bar_time is None` branch), which is exactly what its own
    # is_stale=True already flags. Fall back to that rather than default
    # to "fresh" on data we can't actually measure the age of.
    if data.get("is_stale", False):
        logger.info(f"Alert suppressed for {ticker}: no age data and cache marks it stale")
        return False
    return True


def _format_alert_message(alert: dict) -> str:
    """Formats an alert for sending: its own message plus an exact
    timestamp/session-context line straight from price_cache — no web
    search, no Claude call, so this always fires immediately and for free.

    `intraday_alert_job()`'s formatting step — replaces the old
    `_should_enrich()`/`_enrich_alert_with_context()` web-search pass
    there (kept, just unused by the job now — TestShouldEnrich/
    TestBuildEnrichmentPrompt in test_bot_macro_alerts.py still cover
    them, and a future call site may want enrichment for a specific
    alert type again without redoing that work).

    Plain sync — nothing here does I/O beyond a `price_cache.get()` dict
    lookup (same as `_is_alert_data_fresh()`, called the same way from
    the same loop), so there's no need for callers to route it through
    `run_in_executor()`.
    """
    ticker = alert.get("raw_ticker") or alert.get("ticker", "")
    data = price_cache.get(ticker) if ticker else None
    data = data or {}

    exact_ts = data.get("exact_timestamp", "")
    instrument_type = data.get("instrument_type", "")
    lag = data.get("data_lag_minutes")

    if lag is not None and lag < 10:
        session_ctx = f"🕐 {exact_ts} ({int(lag)}min ago)"
    elif exact_ts:
        session_ctx = f"🕐 {exact_ts}"
    else:
        session_ctx = ""
    if instrument_type:
        session_ctx = f"{session_ctx} | {instrument_type}" if session_ctx else instrument_type

    parts = [alert.get("message", "")]
    if session_ctx:
        parts.append(session_ctx)
    return "\n".join(parts)


async def intraday_alert_job(context) -> None:
    """
    Runs every 30 minutes during market hours (see __main__ job_queue
    setup). Checks all positions for a large price move
    (>LARGE_MOVE_THRESHOLD_PCT in either direction) or thesis-breaker news,
    and alerts TELEGRAM_USER_ID with a [💬 Discuss] button. Deduplicates:
    at most one alert per ticker per alert type per data date (see
    `_data_date_key()` — not the run date, so a restart replaying the same
    still-latest bar doesn't re-fire an alert already sent for it).
    """
    # The first job run after a (re)start sees whatever the price cache
    # was just warmed with — which, right after a restart, can be a stale
    # prior-session close before the cache has had a real chance to
    # refresh against live data. `_is_alert_data_fresh()`'s tight
    # thresholds already suppress genuinely stale reads, but this grace
    # period additionally avoids a burst of alerts landing all at once the
    # moment the bot comes back up, restart included.
    if time.time() - _BOT_START_TIME < ALERT_STARTUP_GRACE_SECONDS:
        logger.info(
            f"intraday_alert_job: startup grace period active "
            f"({ALERT_STARTUP_GRACE_SECONDS}s) — skipping this run"
        )
        return

    if not _get_market_hours_now():
        return

    today = datetime.now().strftime("%Y-%m-%d")
    # Reset daily alert tracking once the date rolls over.
    if _alerted_today.get("_date") != today:
        _alerted_today.clear()
        _alerted_today["_date"] = today

    alerts = []
    tickers = list(POSITIONS.keys())

    # Price alerts
    try:
        price_data = yf.download(
            tickers, period="2d", interval="1d",
            auto_adjust=True, progress=False,
        )["Close"]
        if len(tickers) == 1:
            price_data = price_data.to_frame(name=tickers[0])

        for ticker in tickers:
            if ticker not in price_data.columns:
                continue
            closes = price_data[ticker].dropna()
            if len(closes) < 2:
                continue

            prev_close = closes.iloc[-2]
            curr_price = closes.iloc[-1]
            change_pct = (curr_price / prev_close - 1) * 100

            # This block fetches its own yf.download() bars rather than
            # reading price_cache, so unlike _check_macro_alerts()'s alert
            # types there's no already-fetched price_cache `entry` in
            # scope to pass to _data_date_key() — look it up explicitly.
            alert_key = f"{ticker}_price_{_data_date_key(price_cache.get(ticker), today)}"
            if abs(change_pct) >= LARGE_MOVE_THRESHOLD_PCT and alert_key not in _alerted_today:
                direction = "🚀" if change_pct > 0 else "🔴"
                alerts.append({
                    "ticker": ticker,
                    "raw_ticker": ticker,
                    "type": "price",
                    "magnitude": abs(change_pct),
                    "message": f"{direction} *{ticker}* {change_pct:+.1f}% (${curr_price:.2f})",
                    "key": alert_key,
                })
    except Exception as e:
        logger.warning(f"intraday_alert_job: price check failed: {e}")

    # Thesis-breaker news alerts (run for all positions)
    try:
        triage = await run_in_executor(run_news_triage, tickers)
        for ticker, tdata in triage.items():
            if tdata.get("has_thesis_alert"):
                # Kept on the run date, not _data_date_key(): a thesis
                # alert comes from run_news_triage()'s live re-check of
                # today's news against thesis-breakers, not a price bar —
                # there's no price_cache timestamp it's "from".
                alert_key = f"{ticker}_news_{today}"
                if alert_key not in _alerted_today:
                    matched = tdata.get("thesis_alerts", [])
                    alerts.append({
                        "ticker": ticker,
                        "type": "news",
                        "magnitude": 0,
                        "message": (
                            f"⚠️ *{ticker}* thesis alert\n"
                            f"Matched: {matched[0][:80] if matched else 'unknown'}"
                        ),
                        "key": alert_key,
                    })
    except Exception as e:
        logger.warning(f"intraday_alert_job: news check failed: {e}")

    # Macro alerts — Treasury yields, FX rates, commodities
    try:
        macro_alerts = await run_in_executor(_check_macro_alerts, today)
        alerts.extend(macro_alerts)
    except Exception as e:
        logger.warning(f"intraday_alert_job: macro check failed: {e}")

    # Send alerts
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    for alert in alerts:
        if not _is_alert_data_fresh(alert):
            continue

        ticker = alert["ticker"]
        alert_type = alert.get("type", "price")

        # Alerts whose "ticker" is a display label ("10Y Treasury", "DXY
        # Dollar Index", "Copper/Gold ratio"), not a real discussable
        # symbol, route Discuss to the existing MACRO topic thread (same
        # callback as /macro's menu button) rather than
        # make_tickers_keyboard(), which would build a ticker_<label>
        # thread for a symbol that doesn't exist.
        if alert_type.startswith("macro_") or alert_type in ("volatility", "international", "ratio"):
            discuss_label = "💬 Discuss Macro"
            discuss_button = InlineKeyboardButton(discuss_label, callback_data="cmd_macro")
        elif alert_type == "crypto":
            discuss_button = InlineKeyboardButton("💬 Discuss Crypto/Macro", callback_data="cmd_macro")
        else:
            discuss_button = InlineKeyboardButton(f"💬 Discuss {ticker}", callback_data=f"discuss_{ticker}")

        kb = InlineKeyboardMarkup([[
            discuss_button,
            InlineKeyboardButton("📰 /prices", callback_data="cmd_prices"),
        ]])

        formatted_message = _format_alert_message(alert)

        # Surface active high-priority monitoring items for this ticker —
        # falls back to alert["ticker"] for alert types (news) that don't
        # set raw_ticker, and is a harmless no-op for macro alert types
        # whose raw_ticker is a cache-key symbol (e.g. "DX-Y.NYB") rather
        # than a monitoring "ticker" label.
        try:
            from equity.data.monitoring import get_monitoring_for_ticker

            mon_ticker = alert.get("raw_ticker") or ticker
            mon_items = get_monitoring_for_ticker(mon_ticker) if mon_ticker else []
            high_items = [m for m in mon_items if m.get("priority") == "high"]
            if high_items:
                mon_note = "\n".join(f'📋 {m["item"][:60]}' for m in high_items[:2])
                formatted_message += f"\n\n*Active monitoring:*\n{mon_note}"
        except Exception as e:
            logger.warning(f"intraday_alert_job: monitoring lookup failed for {ticker}: {e}")

        try:
            await send_safe(context.bot, TELEGRAM_USER_ID,
                             f"🔔 *ALERT*\n\n{formatted_message}", reply_markup=kb)
            _alerted_today[alert["key"]] = True
            _record_alert(alert)
            logger.info(f"intraday_alert_job: sent {alert_type} alert for {ticker}")
        except Exception as e:
            logger.error(f"intraday_alert_job: failed to send alert for {ticker}: {e}")


# ---------------------------------------------------------------------------
# advisor.db backup
# ---------------------------------------------------------------------------

async def backup_advisor_db(context) -> None:
    """
    Daily backup of advisor.db (see __main__ job_queue setup — runs at
    2:00 AM UTC). Keeps the last 7 daily backups. Silent on success;
    notifies TELEGRAM_USER_ID only on failure.
    """
    import sqlite3
    from pathlib import Path

    db_path = Path("equity/data/advisor.db")
    backup_dir = Path("equity/data/backups")
    backup_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        logger.warning("backup_advisor_db: advisor.db not found, skipping")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    backup_path = backup_dir / f"advisor_{today}.db"

    try:
        # SQLite's own backup API for a safe hot backup — no corruption
        # risk even while the bot is actively writing to advisor.db.
        source = sqlite3.connect(db_path)
        dest = sqlite3.connect(backup_path)
        source.backup(dest)
        source.close()
        dest.close()

        backup_mb = backup_path.stat().st_size / 1024 / 1024
        logger.info(f"backup_advisor_db: backed up to {backup_path.name} ({backup_mb:.1f}MB)")

        # Prune old backups — keep last 7.
        backups = sorted(backup_dir.glob("advisor_*.db"))
        for old_backup in backups[:-7]:
            old_backup.unlink()
            logger.info(f"backup_advisor_db: pruned {old_backup.name}")

    except Exception as e:
        logger.error(f"backup_advisor_db: FAILED: {e}")
        try:
            await send_safe(
                context.bot, TELEGRAM_USER_ID,
                f"⚠️ *advisor.db backup failed*\n{str(e)[:200]}\n\nCheck `/logs errors`"
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Daily thread summarization
# ---------------------------------------------------------------------------

async def daily_thread_summarization(context) -> None:
    """
    Daily job: summarizes exchanges older than 3 days across all active
    threads (see __main__ job_queue setup — runs at 3:00 AM UTC, an hour
    before the morning brief). Complements ThreadManager.auto_summarize_thread()'s
    message-count trigger with a time-based one, so a low-traffic thread
    still gets a compact summary for Advisor._get_cross_thread_context()
    instead of running verbatim-only forever. topic_BRIEF is skipped — its
    history is brief_builder's saved-briefs mechanism, not chat exchanges.
    """
    try:
        all_threads = thread_manager.list_threads()
        summarized_count = 0
        for thread in all_threads:
            thread_id = thread["thread_id"]
            if thread_id == "topic_BRIEF":
                continue
            try:
                await run_in_executor(
                    thread_manager.summarize_old_exchanges, thread_id, advisor.summarize_messages, 3
                )
                summarized_count += 1
            except Exception as e:
                logger.warning(f"daily_thread_summarization: {thread_id} failed: {e}")

        logger.info(f"daily_thread_summarization: processed {summarized_count} threads")
    except Exception as e:
        logger.error(f"daily_thread_summarization failed: {e}")


# ---------------------------------------------------------------------------
# post_init and __main__
# ---------------------------------------------------------------------------

async def post_init(application):
    from telegram import BotCommand
    await application.bot.set_my_commands([
        BotCommand("brief", "Full morning brief"),
        BotCommand("discuss", "Discuss a ticker: /discuss APP"),
        BotCommand("macro", "Macro and regime discussion"),
        BotCommand("prices", "Live prices — yields, FX, commodities, positions"),
        BotCommand("portfolio", "Portfolio price action"),
        BotCommand("portfolio_review", "In-depth portfolio review"),
        BotCommand("screener", "Run equity screener"),
        BotCommand("news", "News triage for all positions"),
        BotCommand("watchlist", "Watchlist with live prices"),
        BotCommand("threads", "List all discussion threads"),
        BotCommand("switch", "Switch thread: /switch ticker_APP"),
        BotCommand("add", "Add to watchlist: /add ONON"),
        BotCommand("remove", "Remove from watchlist"),
        BotCommand("update", "Update thesis: /update APP"),
        BotCommand("set", "Update config: /set VIX_ELEVATED 22"),
        BotCommand("confirm", "Approve pending change"),
        BotCommand("cancel", "Cancel pending change"),
        BotCommand("done", "Pause current thread"),
        BotCommand("save", "Save discussion conclusions: /save MSFT"),
        BotCommand("monitoring", "View active monitoring items: /monitoring or /monitoring TSLA"),
        BotCommand("dismiss", "Dismiss monitoring: /dismiss TSLA"),
        BotCommand("framework", "Position tier framework and classification status"),
        BotCommand("audit", "Recent config changes and operations"),
        BotCommand("logs", "View logs: /logs errors | brief | advisor | screener"),
        BotCommand("status", "System health and data source status"),
        BotCommand("restart", "Restart the bot process"),
        BotCommand("kill", "Stop the bot process (use if /restart fails)"),
        BotCommand("help", "All commands with examples"),
    ])

    # Warm the shared price cache immediately rather than waiting for the
    # first read to trigger it (the first portfolio monitor / macro alert
    # / /prices call would otherwise eat that fetch's latency).
    try:
        await run_in_executor(price_cache.refresh, True)
        logger.info(f"Startup: {price_cache.coverage_report()}")
    except Exception as e:
        logger.warning(f"Price cache warmup failed: {e}")

    await application.bot.send_message(
        chat_id=TELEGRAM_USER_ID,
        text="Portfolio Advisor online. Good morning. 🌅",
        reply_markup=make_main_menu()
    )


if __name__ == "__main__":
    from equity.config.logging_config import setup_logging
    setup_logging()

    # Logged for /restart debugging — confirms the actual invocation this
    # process started with (sys.executable/argv/cwd) independent of
    # whatever /restart itself does (see send_restart()'s docstring on
    # why it doesn't just replay sys.argv).
    logger.info(f"Bot started: {sys.executable} {sys.argv} cwd={os.getcwd()}")

    import subprocess
    import sys

    # Run security check before starting — abort on any failure
    print("Running security check...")
    result = subprocess.run(
        ["python", "equity/telegram/security_check.py"],
        capture_output=False  # prints directly to terminal
    )
    if result.returncode != 0:
        print()
        print("❌ Security check failed. Bot will not start.")
        print("   Fix all FAIL items in equity/telegram/security_check.py first.")
        sys.exit(1)
    print()
    print("✅ Security check passed. Starting bot...")
    print()

    import datetime as dt

    import pytz
    from telegram import Update
    from telegram.ext import (
        Application, CallbackQueryHandler, CommandHandler,
        MessageHandler, filters,
    )

    app = (Application.builder()
           .token(TELEGRAM_BOT_TOKEN)
           .post_init(post_init)
           .build())

    # Registered explicitly (rather than via a loop over a list of
    # (cmd, handler) tuples) so security_check.py's static AST scan can
    # resolve each add_handler call to the literal handler function name —
    # a loop variable name isn't traceable back to any one handler.
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", send_help))
    app.add_handler(CommandHandler("brief", send_brief))
    app.add_handler(CommandHandler("screener", send_screener))
    app.add_handler(CommandHandler("portfolio", send_portfolio))
    app.add_handler(CommandHandler("news", send_news))
    app.add_handler(CommandHandler("watchlist", send_watchlist))
    app.add_handler(CommandHandler("threads", send_threads))
    app.add_handler(CommandHandler("discuss", send_discuss))
    app.add_handler(CommandHandler("macro", send_macro))
    app.add_handler(CommandHandler("prices", send_prices))
    app.add_handler(CommandHandler("portfolio_review", send_portfolio_review))
    app.add_handler(CommandHandler("switch", send_switch))
    app.add_handler(CommandHandler("done", send_done))
    app.add_handler(CommandHandler("save", send_save))
    app.add_handler(CommandHandler("monitoring", send_monitoring))
    app.add_handler(CommandHandler("dismiss", send_dismiss))
    app.add_handler(CommandHandler("framework", send_framework))
    app.add_handler(CommandHandler("audit", send_audit))
    app.add_handler(CommandHandler("logs", send_logs))
    app.add_handler(CommandHandler("status", send_status))
    app.add_handler(CommandHandler("restart", send_restart))
    app.add_handler(CommandHandler("kill", send_kill))
    app.add_handler(CommandHandler("add", send_add))
    app.add_handler(CommandHandler("remove", send_remove))
    app.add_handler(CommandHandler("update", send_update))
    app.add_handler(CommandHandler("set", send_set))
    app.add_handler(CommandHandler("confirm", send_confirm))
    app.add_handler(CommandHandler("cancel", send_cancel))

    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_message
    ))
    app.add_handler(MessageHandler(filters.COMMAND, handle_unknown_command))

    # Global error handler — not an add_handler target, so it's exempt from
    # security_check.py's "every add_handler'd function has @authorized_only"
    # scan; registered last, after every command/message/callback handler.
    app.add_error_handler(handle_error)

    app.job_queue.run_daily(
        scheduled_morning_brief,
        time=dt.time(12, 30, 0, tzinfo=pytz.utc),
        name="morning_brief",
        job_kwargs={
            'misfire_grace_time': 300,  # 5 min grace, brief takes time to run
            'max_instances': 1,
            'coalesce': True,
        }
    )
    app.job_queue.run_repeating(
        intraday_alert_job,
        interval=1800,   # every 30 minutes
        first=60,        # first run 60 seconds after bot starts
        name="intraday_alerts",
        job_kwargs={
            'misfire_grace_time': 120,  # tolerate up to 2 minutes late — suppresses the warning
            'max_instances': 1,         # never run two instances simultaneously
            'coalesce': True,           # if multiple runs were missed, only run once
        }
    )
    app.job_queue.run_daily(
        backup_advisor_db,
        time=dt.time(2, 0, 0, tzinfo=pytz.utc),
        name="db_backup",
        job_kwargs={
            'misfire_grace_time': 600,  # 10 min grace, low priority
            'max_instances': 1,
            'coalesce': True,
        }
    )
    app.job_queue.run_daily(
        daily_thread_summarization,
        time=dt.time(3, 0, 0, tzinfo=pytz.utc),
        name="thread_summarization",
        job_kwargs={
            'misfire_grace_time': 600,
            'max_instances': 1,
            'coalesce': True,
        }
    )

    print("Portfolio Advisor bot started. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
