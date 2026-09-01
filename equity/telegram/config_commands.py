"""Config change handlers for the Telegram advisor bot.

Every function here executes AFTER email 2FA has already been verified
(see bot.py's `require_email_auth` / `handle_message` flow) — nothing in
this module sends the auth email itself. All actual writes go through
`equity.config.config_manager`, which git-commits every change.
"""

import logging
import subprocess

from equity.config import config_manager
from equity.config import positions as positions_config
from equity.telegram.formatters import make_confirm_cancel, send_safe

logger = logging.getLogger(__name__)
security_logger = logging.getLogger("security")

CONFIG_FIELD_BOUNDS = {
    "VIX_ELEVATED": (10, 50),
    "VIX_HIGH": (20, 80),
    "VIX_EXTREME": (30, 100),
    "LARGE_MOVE_THRESHOLD_PCT": (0.5, 10.0),
    "MEDIUM_MOVE_THRESHOLD_PCT": (0.1, 5.0),
    "FOMC_PROXIMITY_DAYS": (0, 7),
    "MAX_HEADLINES_PER_TICKER": (1, 10),
    "NEWS_KEYWORD_MIN_MATCHES": (1, 5),
    "POSITION_UNDERPERFORM_ALERT_PCT": (0.5, 10.0),
    "CORRELATION_CONCENTRATION_THRESHOLD": (0.5, 0.95),
    "EARNINGS_LOOKAHEAD_DAYS": (1, 30),
    "EARNINGS_ALERT_DAYS": (1, 7),
    "SLV_SI_DIVERGENCE_ALERT_PCT": (5.0, 30.0),
}


def validate_config_value(field: str, value_str: str) -> tuple[bool, str, any]:
    if field not in CONFIG_FIELD_BOUNDS:
        return False, f"{field} is not settable via Telegram", None
    lo, hi = CONFIG_FIELD_BOUNDS[field]
    try:
        val = float(value_str)
    except ValueError:
        return False, "Value must be a number", None
    if not (lo <= val <= hi):
        return False, f"{field} must be between {lo} and {hi}", None
    return True, "", val


def _git_head_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# /add — add to watchlist
# ---------------------------------------------------------------------------

async def execute_add_watchlist(ticker: str, update, context, advisor, thread_manager) -> None:
    """Executes the add after auth confirmed. Same logic as handle_add_watchlist."""
    ticker = ticker.upper()
    chat_id = update.effective_chat.id

    if ticker in positions_config.POSITIONS or ticker in positions_config.WATCHLIST:
        await context.bot.send_message(
            chat_id=chat_id, text=f"{ticker} is already tracked."
        )
        return

    await context.bot.send_message(
        chat_id=chat_id, text=f"Researching {ticker} — drafting thesis (~20s)..."
    )

    from equity.telegram.bot import run_in_executor  # local import — avoids circular import

    draft = await run_in_executor(advisor.draft_thesis, ticker)

    lines = [
        f"📋 *Proposed watchlist entry: {ticker}*",
        f"Tier: {draft.get('tier', '')}",
        f"Sector: {draft.get('sector', '')}",
        f"Thesis: {draft.get('thesis', '')}",
        "Breakers: " + "; ".join(draft.get("thesis_breakers", [])),
        f"Macro thesis: {draft.get('macro_thesis', '')}",
        f"Exit conditions: {draft.get('target_exit_conditions', '')}",
    ]
    description = f"Add {ticker} to watchlist"

    change_id = thread_manager.save_pending_change(
        change_type="add_watchlist",
        payload={"ticker": ticker, "entry": draft},
        description=description,
    )

    await send_safe(
        context.bot,
        chat_id,
        "\n".join(lines),
        reply_markup=make_confirm_cancel(change_id, description),
    )


async def handle_add_watchlist(update, context, advisor, thread_manager, auth_manager) -> None:
    """/add TICKER — called AFTER email auth is verified."""
    if not context.args:
        await update.message.reply_text("Usage: /add TICKER")
        return
    ticker = context.args[0].upper()
    await execute_add_watchlist(ticker, update, context, advisor, thread_manager)


# ---------------------------------------------------------------------------
# /remove — remove from watchlist
# ---------------------------------------------------------------------------

async def handle_remove_watchlist(update, context, thread_manager, auth_manager) -> None:
    """/remove TICKER — called AFTER email auth verified."""
    if not context.args:
        await update.message.reply_text("Usage: /remove TICKER")
        return
    ticker = context.args[0].upper()
    entry = positions_config.WATCHLIST.get(ticker)
    if not entry:
        await update.message.reply_text(f"{ticker} is not on the watchlist.")
        return

    await send_safe(
        context.bot,
        update.effective_chat.id,
        f"📋 *{ticker}*\nTier: {entry.get('tier')}\nThesis: {entry.get('thesis', '')}\n\n"
        f"Type *{ticker}* exactly to confirm removal, or /cancel to abort.",
    )
    context.user_data["pending_remove_ticker"] = ticker


# ---------------------------------------------------------------------------
# /update — update thesis fields
# ---------------------------------------------------------------------------

_UPDATABLE_FIELDS = ["thesis", "macro_thesis", "tier", "thesis_breakers", "stop_thesis"]


async def handle_update_thesis(update, context, advisor, thread_manager, auth_manager) -> None:
    """/update TICKER [field] — called AFTER email auth verified."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    if not context.args:
        await update.message.reply_text("Usage: /update TICKER [field]")
        return

    ticker = context.args[0].upper()
    pos = positions_config.get_position(ticker)
    if not pos:
        await update.message.reply_text(f"{ticker} not found in POSITIONS or WATCHLIST.")
        return

    field = context.args[1] if len(context.args) > 1 else None

    if field is None:
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton(f, callback_data=f"updatefield_{ticker}_{f}")] for f in _UPDATABLE_FIELDS]
        )
        await send_safe(
            context.bot,
            update.effective_chat.id,
            f"Which field would you like to update for *{ticker}*?",
            reply_markup=keyboard,
        )
        return

    if field == "tier":
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(t, callback_data=f"settier_{ticker}_{t}")]
                for t in ("core", "high_conviction", "speculative")
            ]
        )
        await send_safe(
            context.bot,
            update.effective_chat.id,
            f"Current tier: *{pos.get('tier')}*. New tier:",
            reply_markup=keyboard,
        )
        return

    if field == "stop_thesis":
        context.user_data["pending_stop_thesis_ticker"] = ticker
        await update.message.reply_text(
            f'Type "CONFIRM {ticker}" exactly to flag the thesis as broken '
            f"(stop_thesis=True), or /cancel to abort."
        )
        return

    current_value = pos.get(field, "")
    await send_safe(
        context.bot,
        update.effective_chat.id,
        f"Current *{field}* for {ticker}:\n{current_value}\n\n"
        f"Open /discuss {ticker} to collaboratively draft a new value, "
        f"then /confirm to save.",
    )


# ---------------------------------------------------------------------------
# /set — market config field update
# ---------------------------------------------------------------------------

async def handle_set_config(update, context, thread_manager, auth_manager) -> None:
    """/set FIELD VALUE — called AFTER email auth verified."""
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /set FIELD VALUE")
        return

    field, value_str = context.args[0], context.args[1]
    ok, error, val = validate_config_value(field, value_str)
    if not ok:
        await update.message.reply_text(f"❌ {error}")
        return

    description = f"Set {field} = {val}"
    change_id = thread_manager.save_pending_change(
        change_type="set_config",
        payload={"field": field, "value": val},
        description=description,
    )
    await send_safe(
        context.bot,
        update.effective_chat.id,
        f"📋 *Proposed config change*\n{field}: {val}\n\n"
        f"This affects thresholds used in the morning brief and portfolio monitor.",
        reply_markup=make_confirm_cancel(change_id, description),
    )


# ---------------------------------------------------------------------------
# Confirm / cancel callbacks
# ---------------------------------------------------------------------------

async def handle_confirm_callback(query, context, thread_manager) -> None:
    """Called when [✅ Confirm] button tapped."""
    change_id = int(query.data.split("_", 1)[1])
    change = thread_manager.get_pending_change(change_id)
    if change is None:
        await query.edit_message_text("⌛ This change has expired. Please try again.")
        return

    change_type = change["change_type"]
    payload = change["payload"]
    success = False

    if change_type == "add_watchlist":
        ticker = payload["ticker"]
        entry = payload["entry"]
        success = config_manager.add_to_watchlist(
            ticker, entry, f"Added via Telegram advisor"
        )
        if success:
            security_logger.warning(f"WRITE_OP | add_watchlist | ticker={ticker}")
            commit_hash = _git_head_hash()
            await query.edit_message_text(f"✓ {ticker} added to watchlist. {commit_hash}")
        else:
            await query.edit_message_text(f"❌ Failed to add {ticker} to watchlist.")

    elif change_type == "set_config":
        field = payload["field"]
        value = payload["value"]
        success = config_manager.update_market_config(
            field, value, "Updated via Telegram advisor"
        )
        if success:
            security_logger.warning(f"WRITE_OP | set_config | field={field} value={value}")
            commit_hash = _git_head_hash()
            await query.edit_message_text(f"✓ {field} set to {value}. {commit_hash}")
        else:
            await query.edit_message_text(f"❌ Failed to update {field}.")

    else:
        await query.edit_message_text(f"❌ Unknown change type: {change_type}")

    thread_manager.clear_pending_change(change_id)


async def handle_cancel_callback(query, context, thread_manager) -> None:
    """Called when [❌ Cancel] button tapped."""
    change_id = int(query.data.split("_", 1)[1])
    thread_manager.clear_pending_change(change_id)
    await query.edit_message_text("❌ Change cancelled.")
