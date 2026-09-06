"""Telegram portfolio advisor bot — main entry point.

Run with: python -m equity.telegram.bot
Before the first run, always: python equity/telegram/security_check.py
"""

import asyncio
import logging
import os
import subprocess
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

from equity.brief.brief_builder import build_morning_brief, get_last_brief_synthesis, save_brief_to_thread
from equity.brief.market_snapshot import fetch_market_snapshot
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
    FX_ALERT_PCT,
    LARGE_MOVE_THRESHOLD_PCT,
    NEWS_HEADLINE_PAGE_SIZE,
    YIELD_ALERT_BP,
    YIELD_LEVEL_ALERTS,
)
from equity.portfolio.monitor import format_portfolio_monitor, run_portfolio_monitor
from equity.portfolio.news_triage import format_news_triage, run_news_triage
from equity.screener.quality_scorer import score_ticker
from equity.screener.screener import format_screener_output, run_screener
from equity.telegram import config_commands
from equity.telegram.advisor import Advisor
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
    make_tickers_keyboard,
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
                text=f"🔍 Researching {ticker}..."
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


@authorized_only
async def send_monitoring(update, context):
    """
    /monitoring — shows all active monitoring items across the portfolio.
    /monitoring TICKER — shows items for a specific ticker.

    Each item gets inline [💬 Discuss] / [✕ Dismiss] buttons so an item can
    be cleared straight from the list — see the "dismiss_item_" branch in
    handle_callback() — without typing /dismiss TICKER and picking through
    a selection menu.
    """
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    from equity.data.monitoring import load_monitoring

    ticker_filter = context.args[0].upper() if context.args else None
    items = load_monitoring()

    if ticker_filter:
        items = [i for i in items if i.get("ticker") == ticker_filter]

    if not items:
        msg = f'No active monitoring items{f" for {ticker_filter}" if ticker_filter else ""}.'
        await reply(update, context, msg, reply_markup=make_main_menu())
        return

    # Sort by priority then age (oldest first within a priority tier).
    priority_order = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda x: (priority_order.get(x.get("priority"), 1), -x.get("age_days", 0)))

    priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    lines = ["📋 *ACTIVE MONITORING*", ""]
    item_buttons = []
    for item in items:
        emoji = priority_emoji.get(item.get("priority"), "⚪")
        age = item.get("age_days", 0)
        age_str = "today" if age == 0 else f"{age}d ago"
        lines.append(f'{emoji} *{item["ticker"]}* ({age_str})\n  {item["item"]}')
        lines.append("")
        item_buttons.append([
            InlineKeyboardButton(f'💬 {item["ticker"]}', callback_data=f'discuss_{item["ticker"]}'),
            InlineKeyboardButton("✕ Dismiss", callback_data=f'dismiss_item_{item["id"]}'),
        ])
    item_buttons.append([InlineKeyboardButton("🏠 Main Menu", callback_data="cmd_main_menu")])

    await send_in_parts(context.bot, update.effective_chat.id,
                         "\n".join(lines), reply_markup=InlineKeyboardMarkup(item_buttons))


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


@authorized_only
async def send_yields(update, context):
    """
    /yields — on-demand live macro snapshot: Treasury yields, key FX
    pairs, key commodities. Useful when you need current yield levels
    mid-discussion, without opening a MACRO discussion thread (/macro).

    Forces a fresh fetch rather than reusing advisor.get_live_macro_snapshot()'s
    normal 15-minute chat-context cache — the entire point of asking on
    demand is "what's the level right now" (same private-cache-reset
    pattern positions.reload() already uses on _portfolio_context_cache).
    """
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="📡 Fetching live macro data..."
    )

    import equity.telegram.advisor as advisor_module
    advisor_module._macro_snapshot_cache["timestamp"] = 0

    snapshot = await run_in_executor(advisor.get_live_macro_snapshot)
    if not snapshot:
        await reply(
            update, context,
            "⚠️ Could not fetch live macro data. Check /logs errors.",
            reply_markup=make_main_menu(),
        )
        return

    await send_safe(
        context.bot, update.effective_chat.id,
        f"📡 *Live Macro Snapshot*\n\n{snapshot}", reply_markup=make_main_menu(),
    )


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
    from pathlib import Path

    lines = ["🖥️ *SYSTEM STATUS*", ""]

    # Uptime
    uptime_seconds = time.time() - _BOT_START_TIME
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    lines.append(f"⏱ Uptime: {hours}h {minutes}m")

    # Last brief
    briefs_dir = Path("equity/data/briefs")
    brief_files = sorted(briefs_dir.glob("brief_*.txt")) if briefs_dir.exists() else []
    if brief_files:
        latest = brief_files[-1]
        mtime = datetime.fromtimestamp(latest.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        age_str = f"{age_hours:.1f}h ago" if age_hours < 24 else f"{age_hours / 24:.1f}d ago"
        lines.append(f"📊 Last brief: {mtime.strftime('%Y-%m-%d %H:%M')} ({age_str})")
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
    Requires email 2FA since it modifies persistent state.

    Carries @register_command (like send_add/remove/update/set) because it
    goes through @require_email_auth: after the emailed code is verified,
    handle_message() re-dispatches via COMMAND_REGISTRY[func_name] — without
    this decorator the post-auth call would silently no-op.
    """
    from equity.data.monitoring import dismiss_monitoring_item, load_monitoring

    if not context.args:
        await reply(update, context,
            "Usage: /dismiss TICKER\nExample: /dismiss TSLA\n\n"
            "Shows active monitoring items for that ticker to dismiss."
        )
        return

    ticker = context.args[0].upper()
    items = [i for i in load_monitoring() if i.get("ticker", "").upper() == ticker]

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
        triage = await run_in_executor(run_news_triage, [ticker])
        text = format_news_triage(triage)
        await send_in_parts(context.bot, chat_id, text, reply_markup=make_ticker_actions(ticker))
    elif data.startswith("ticker_metrics_"):
        ticker = data[len("ticker_metrics_"):]
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
    "discuss", "macro", "yields", "portfolio", "portfolio_review",
    "screener", "news", "brief", "watchlist", "threads",
    "switch", "add", "remove", "update", "set", "save", "framework",
    "confirm", "cancel", "done", "audit", "logs", "status", "logout",
    "start", "help", "monitoring", "dismiss"
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


def _check_macro_alerts(today: str) -> list[dict]:
    """Checks Treasury yields, FX rates, and commodities for intraday alerts.

    Returns a list of alert dicts in the same shape `intraday_alert_job()`
    already expects. Called in executor — no async, and does its own
    (synchronous) data fetch rather than taking a pre-fetched snapshot.

    Built on `market_snapshot.fetch_market_snapshot()` — already imported
    in this module for the morning brief — rather than a second yfinance/
    FRED fetch of its own: that function already covers every tenor in
    `YIELD_ALERT_BP` (2Y/20Y come from FRED; yfinance alone doesn't carry
    them — see `market_config.TREASURY_FRED_SERIES`) and every FX/commodity
    ticker below, each already paired with its move vs prior close. 'DXY'
    isn't an FX_TICKERS pair — it resolves to the DX-Y.NYB dollar index in
    `snapshot['commodities']`, the same one `market_snapshot` already
    fetches for its own regime inputs, rather than a separate ETF proxy.
    """
    alerts = []

    try:
        snapshot = fetch_market_snapshot()
    except Exception as e:
        logger.warning(f"_check_macro_alerts: snapshot fetch failed: {e}")
        return alerts

    # --- Treasury yields ---
    treasury_curve = snapshot.get("treasury_curve", {})
    for tenor, threshold_bp in YIELD_ALERT_BP.items():
        entry = treasury_curve.get(tenor)
        if entry is None:
            continue
        curr = entry.get("yield_pct")
        move_bp = entry.get("change_1d_bps")
        if curr is None or move_bp is None:
            continue
        prev = curr - move_bp / 100

        alert_key = f"yield_{tenor}_{today}"
        if abs(move_bp) >= threshold_bp and alert_key not in _alerted_today:
            direction = "📈" if move_bp > 0 else "📉"
            alerts.append({
                "ticker": f"{tenor} Treasury",
                "type": "macro_yield",
                "message": (
                    f"{direction} *{tenor} Treasury yield* "
                    f"{move_bp:+.1f}bp intraday → {curr:.2f}%\n"
                    f"{'Rising yields — watch TLT, duration-sensitive positions' if move_bp > 0 else 'Falling yields — watch TLT thesis, rate-sensitive positioning'}"
                ),
                "key": alert_key,
            })

        for level in YIELD_LEVEL_ALERTS.get(tenor, []):
            level_key = f"yield_{tenor}_level_{level}_{today}"
            if level_key in _alerted_today or not _level_crossed(prev, curr, level):
                continue
            direction = "broke above" if curr > prev else "broke below"
            emoji = "⚠️" if curr > prev else "✅"
            alerts.append({
                "ticker": f"{tenor} Treasury",
                "type": "macro_yield_level",
                "message": (
                    f"{emoji} *{tenor} yield {direction} {level:.2f}%*\n"
                    f"Current: {curr:.3f}% | Prior close: {prev:.3f}%\n"
                    f"Key level breach — reassess duration positioning"
                ),
                "key": level_key,
            })

    # --- FX rates ---
    fx_data = snapshot.get("fx", {})
    dxy_entry = snapshot.get("commodities", {}).get("DX-Y.NYB")
    for fx_ticker, threshold in FX_ALERT_PCT.items():
        if fx_ticker == "DXY":
            entry, label = dxy_entry, "DXY Dollar Index"
        else:
            entry = fx_data.get(fx_ticker)
            label = entry.get("label", fx_ticker) if entry else fx_ticker
        if entry is None:
            continue
        move_pct = entry.get("change_1d_pct")
        if move_pct is None:
            continue

        alert_key = f"fx_{fx_ticker}_{today}"
        if abs(move_pct) >= threshold and alert_key not in _alerted_today:
            direction = "📈" if move_pct > 0 else "📉"
            alerts.append({
                "ticker": label,
                "type": "macro_fx",
                "message": f"{direction} *{label}* {move_pct:+.2f}% intraday\n{_fx_context_note(fx_ticker, move_pct)}",
                "key": alert_key,
            })

    # --- Commodities ---
    commodities_ext = snapshot.get("commodities_extended", {})
    for comm_ticker, threshold in COMMODITY_ALERT_PCT.items():
        entry = commodities_ext.get(comm_ticker)
        if entry is None:
            continue
        move_pct = entry.get("change_1d_pct")
        if move_pct is None:
            continue
        label = entry.get("label", comm_ticker)

        alert_key = f"commodity_{comm_ticker}_{today}"
        if abs(move_pct) >= threshold and alert_key not in _alerted_today:
            direction = "📈" if move_pct > 0 else "📉"
            alerts.append({
                "ticker": label,
                "type": "macro_commodity",
                "message": f"{direction} *{label}* {move_pct:+.2f}% intraday\n{_commodity_context_note(comm_ticker, move_pct)}",
                "key": alert_key,
            })

    return alerts


async def intraday_alert_job(context) -> None:
    """
    Runs every 30 minutes during market hours (see __main__ job_queue
    setup). Checks all positions for a large price move
    (>LARGE_MOVE_THRESHOLD_PCT in either direction) or thesis-breaker news,
    and alerts TELEGRAM_USER_ID with a [💬 Discuss] button. Deduplicates:
    at most one alert per ticker per alert type per day.
    """
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

            alert_key = f"{ticker}_price_{today}"
            if abs(change_pct) >= LARGE_MOVE_THRESHOLD_PCT and alert_key not in _alerted_today:
                direction = "🚀" if change_pct > 0 else "🔴"
                alerts.append({
                    "ticker": ticker,
                    "type": "price",
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
                alert_key = f"{ticker}_news_{today}"
                if alert_key not in _alerted_today:
                    matched = tdata.get("thesis_alerts", [])
                    alerts.append({
                        "ticker": ticker,
                        "type": "news",
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
    for alert in alerts:
        ticker = alert["ticker"]
        if alert["type"].startswith("macro_"):
            # Macro alerts' "ticker" is a display label ("10Y Treasury", "DXY
            # Dollar Index"), not a real discussable symbol — route Discuss
            # to the existing MACRO topic thread (same callback as /macro's
            # menu button) rather than make_tickers_keyboard(), which would
            # build a ticker_<label> thread for a symbol that doesn't exist.
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup

            kb = InlineKeyboardMarkup([[InlineKeyboardButton("💬 Discuss Macro", callback_data="cmd_macro")]])
        else:
            kb = make_tickers_keyboard([ticker], label="💬 Discuss")
        try:
            await send_safe(context.bot, TELEGRAM_USER_ID,
                             f"🔔 *INTRADAY ALERT*\n\n{alert['message']}", reply_markup=kb)
            _alerted_today[alert["key"]] = True
            logger.info(f"intraday_alert_job: sent {alert['type']} alert for {ticker}")
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
# post_init and __main__
# ---------------------------------------------------------------------------

async def post_init(application):
    from telegram import BotCommand
    await application.bot.set_my_commands([
        BotCommand("brief", "Full morning brief"),
        BotCommand("discuss", "Discuss a ticker: /discuss APP"),
        BotCommand("macro", "Macro and regime discussion"),
        BotCommand("yields", "Live yields, FX and commodity prices"),
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
        BotCommand("help", "All commands with examples"),
    ])
    await application.bot.send_message(
        chat_id=TELEGRAM_USER_ID,
        text="Portfolio Advisor online. Good morning. 🌅",
        reply_markup=make_main_menu()
    )


if __name__ == "__main__":
    from equity.config.logging_config import setup_logging
    setup_logging()

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
    app.add_handler(CommandHandler("yields", send_yields))
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
        name="morning_brief"
    )
    app.job_queue.run_repeating(
        intraday_alert_job,
        interval=1800,   # every 30 minutes
        first=60,        # first run 60 seconds after bot starts
        name="intraday_alerts"
    )
    app.job_queue.run_daily(
        backup_advisor_db,
        time=dt.time(2, 0, 0, tzinfo=pytz.utc),
        name="db_backup"
    )

    print("Portfolio Advisor bot started. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
