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

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from equity.brief.brief_builder import build_morning_brief
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
from equity.config.market_config import NEWS_HEADLINE_PAGE_SIZE
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
    send_in_parts,
    send_safe,
)
from equity.telegram.threads import ThreadManager

POSITIONS = positions_module.POSITIONS
WATCHLIST = positions_module.WATCHLIST

thread_manager = ThreadManager()
auth_manager = AuthManager()
advisor = Advisor(api_key=os.getenv("ANTHROPIC_API_KEY"), thread_manager=thread_manager)

# ---------------------------------------------------------------------------
# Security logging
# ---------------------------------------------------------------------------

security_logger = logging.getLogger("security")
os.makedirs("equity/data", exist_ok=True)
_sh = logging.FileHandler("equity/data/security.log")
security_logger.addHandler(_sh)
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
        return await func(update, context)
    wrapper.__name__ = func.__name__
    return wrapper


def require_write_access(func):
    async def wrapper(update, context):
        if READONLY_MODE:
            await update.message.reply_text(
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
                await update.message.reply_text(
                    "⚠️ Failed to send auth email. "
                    "Check BOT_EMAIL settings. Operation cancelled."
                )
                return
            context.user_data["awaiting_email_auth"] = {
                "func_name": func.__name__,
                "operation": operation,
            }
            context.user_data["pending_command_args"] = context.args
            await update.message.reply_text(
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
    system_prompt = advisor.build_system_prompt(thread_subject=ticker)

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
# Read-only handlers
# ---------------------------------------------------------------------------

@authorized_only
async def start(update, context):
    await update.message.reply_text(
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
        "/audit — Recent config changes and operations\n\n"
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
        "/set FIELD VALUE — Update market config\n"
        "/confirm — Approve pending change\n"
        "/cancel — Cancel pending change/auth\n"
    )
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=make_main_menu())


@authorized_only
async def send_brief(update, context):
    chat_id = update.effective_chat.id
    await context.bot.send_message(chat_id=chat_id, text="🌅 Building morning brief...")
    sections = await run_in_executor(build_morning_brief)
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
    await update.message.reply_text(
        "\n".join(lines), reply_markup=make_discuss_menu(POSITIONS, WATCHLIST)
    )


@authorized_only
async def send_threads(update, context):
    threads = thread_manager.list_threads()
    await update.message.reply_text(
        format_thread_list(threads), reply_markup=make_thread_list_keyboard(threads)
    )


@authorized_only
async def send_discuss(update, context):
    if not context.args:
        await update.message.reply_text(
            "Choose a ticker:", reply_markup=make_discuss_menu(POSITIONS, WATCHLIST)
        )
        return
    await start_or_resume_discussion(context.args[0], update, context)


@authorized_only
async def send_macro(update, context):
    await start_or_resume_discussion("MACRO", update, context, thread_type="topic")


@authorized_only
async def send_portfolio_review(update, context):
    await start_or_resume_discussion("PORTFOLIO", update, context, thread_type="topic")


@authorized_only
async def send_switch(update, context):
    if not context.args:
        threads = thread_manager.list_threads()
        await update.message.reply_text(
            "Choose a thread:", reply_markup=make_thread_list_keyboard(threads)
        )
        return
    thread_id = context.args[0]
    thread_manager.set_active_thread(TELEGRAM_USER_ID, thread_id)
    info = thread_manager.get_thread_info(thread_id)
    subject = info["subject"] if info else thread_id
    await update.message.reply_text(
        f"✓ Switched to {thread_id}. Reply to continue.",
        reply_markup=make_ticker_actions(subject),
    )


@authorized_only
async def send_done(update, context):
    thread_manager.clear_active_thread(TELEGRAM_USER_ID)
    await update.message.reply_text(
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
        with open("equity/data/security.log") as f:
            recent_ops = "".join(f.readlines()[-5:])
    except FileNotFoundError:
        recent_ops = "No operations logged yet."
    await update.message.reply_text(
        f"📋 *Recent config changes:*\n```\n{result.stdout or 'None'}\n```\n"
        f"📋 *Recent write operations:*\n```\n{recent_ops}\n```",
        parse_mode="Markdown"
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
        await update.message.reply_text("No pending change to confirm (or it has expired).")
        return
    result = _execute_pending_change(change)
    thread_manager.clear_pending_change(change["id"])
    await update.message.reply_text(result)


@authorized_only
async def send_cancel(update, context):
    auth_manager.cancel_pending_auth(TELEGRAM_USER_ID)
    thread_manager.clear_latest_pending_change()
    await update.message.reply_text("❌ Cancelled.", reply_markup=make_main_menu())


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
        await query.message.reply_text(
            f"✓ Switched to {thread_id}. Reply to continue.",
            reply_markup=make_ticker_actions(subject),
        )
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
        system_prompt = advisor.build_system_prompt(thread_subject=ticker)
        response = await run_in_executor(
            advisor.chat, active_thread, f"Context refresh for {ticker}:\n{context_str}",
            system_prompt, ticker,
        )
        await send_in_parts(context.bot, chat_id, response, reply_markup=make_ticker_actions(ticker))
    elif data.startswith("ticker_add_watchlist_"):
        ticker = data[len("ticker_add_watchlist_"):]
        context.args = [ticker]
        await send_add(update, context)
    elif data.startswith("ticker_flag_risk_"):
        ticker = data[len("ticker_flag_risk_"):]
        context.args = [ticker, "stop_thesis"]
        await send_update(update, context)
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
                system_prompt = advisor.build_system_prompt(thread_subject=subject)
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
        await update.message.reply_text(
            "⏱ Too many messages. Wait a moment and try again."
        )
        return

    # Priority 1: pending email auth
    if auth_manager.is_awaiting_auth(user_id):
        if message_text.startswith("/cancel"):
            auth_manager.cancel_pending_auth(user_id)
            await update.message.reply_text(
                "❌ Authorization cancelled.", reply_markup=make_main_menu()
            )
            return
        success, payload = auth_manager.verify_email_code(user_id, message_text)
        if success:
            await update.message.reply_text("✓ Authorized.")
            pending = context.user_data.pop("awaiting_email_auth", {})
            func_name = pending.get("func_name")
            context.args = context.user_data.pop("pending_command_args", [])
            action = context.user_data.pop("pending_auth_action", {})
            if action.get("action") == "add_watchlist":
                await config_commands.execute_add_watchlist(
                    action["ticker"], update, context, advisor, thread_manager
                )
            elif func_name and func_name in COMMAND_REGISTRY:
                await COMMAND_REGISTRY[func_name](update, context)
        else:
            await update.message.reply_text(
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
            await update.message.reply_text(
                f"✓ {expected} removed from watchlist.", reply_markup=make_main_menu()
            )
        else:
            await update.message.reply_text(
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
            await update.message.reply_text(
                f"🚨 {expected} thesis flagged as broken.", reply_markup=make_main_menu()
            )
        else:
            await update.message.reply_text(
                f'Type "CONFIRM {expected}" exactly to confirm, or /cancel to abort.'
            )
        return

    # Priority 2c: awaiting a typed ticker for "Other ticker..." button
    if context.user_data.get("awaiting_ticker_input"):
        context.user_data.pop("awaiting_ticker_input")
        await start_or_resume_discussion(message_text.strip().upper(), update, context)
        return

    # Priority 3: active conversation thread
    active_thread = thread_manager.get_active_thread(user_id)
    if not active_thread:
        await update.message.reply_text(
            "No active discussion. Choose one:",
            reply_markup=make_discuss_menu(POSITIONS, WATCHLIST),
        )
        return

    if not check_claude_rate_limit():
        await update.message.reply_text(
            "⚠️ Claude call limit reached for this hour. "
            "Data commands (/portfolio, /screener) still work."
        )
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    thread_info = thread_manager.get_thread_info(active_thread)
    subject = thread_info["subject"] if thread_info else None
    system_prompt = advisor.build_system_prompt(thread_subject=subject)
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
    "discuss", "macro", "portfolio", "portfolio_review",
    "screener", "news", "brief", "watchlist", "threads",
    "switch", "add", "remove", "update", "set",
    "confirm", "cancel", "done", "audit", "logout",
    "start", "help"
]


@authorized_only
async def handle_unknown_command(update, context):
    import difflib
    cmd = update.message.text.lstrip("/").split()[0].lower()
    matches = difflib.get_close_matches(cmd, KNOWN_COMMANDS, n=1, cutoff=0.6)
    if matches:
        await update.message.reply_text(
            f"Did you mean */{matches[0]}*? "
            f"Try again or /help for all commands.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            "Unknown command.", reply_markup=make_main_menu()
        )


# ---------------------------------------------------------------------------
# Scheduled morning brief
# ---------------------------------------------------------------------------

async def scheduled_morning_brief(context):
    try:
        sections = await run_in_executor(build_morning_brief)
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
# post_init and __main__
# ---------------------------------------------------------------------------

async def post_init(application):
    from telegram import BotCommand
    await application.bot.set_my_commands([
        BotCommand("brief", "Full morning brief"),
        BotCommand("discuss", "Discuss a ticker: /discuss APP"),
        BotCommand("macro", "Macro and regime discussion"),
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
        BotCommand("audit", "Recent config changes and operations"),
        BotCommand("help", "All commands with examples"),
    ])
    await application.bot.send_message(
        chat_id=TELEGRAM_USER_ID,
        text="Portfolio Advisor online. Good morning. 🌅",
        reply_markup=make_main_menu()
    )


if __name__ == "__main__":
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
    app.add_handler(CommandHandler("portfolio_review", send_portfolio_review))
    app.add_handler(CommandHandler("switch", send_switch))
    app.add_handler(CommandHandler("done", send_done))
    app.add_handler(CommandHandler("audit", send_audit))
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

    app.job_queue.run_daily(
        scheduled_morning_brief,
        time=dt.time(12, 30, 0, tzinfo=pytz.utc),
        name="morning_brief"
    )

    print("Portfolio Advisor bot started. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
