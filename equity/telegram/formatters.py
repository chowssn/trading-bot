"""Message formatting and inline keyboard builders for the Telegram advisor bot."""

import asyncio

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


# ---------------------------------------------------------------------------
# Message splitting / sending
# ---------------------------------------------------------------------------

def split_message(text: str, max_length: int = 4000) -> list[str]:
    if len(text) <= max_length:
        return [text]

    parts = []
    remaining = text
    while len(remaining) > max_length:
        # Prefer a paragraph boundary, then a line boundary, within the window.
        window = remaining[:max_length]
        split_at = window.rfind("\n\n")
        if split_at == -1:
            split_at = window.rfind("\n")
        if split_at == -1:
            split_at = max_length
        parts.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip("\n")
    if remaining:
        parts.append(remaining)
    return parts


async def send_safe(bot, chat_id: int, text: str,
                    reply_markup=None) -> None:
    '''
    Sends message, trying Markdown first.
    Falls back to plain text if Markdown parsing fails.
    '''
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    except Exception:
        # Strip any Markdown and send as plain text
        clean = text.replace('*', '').replace('_', '').replace('`', '')
        await bot.send_message(
            chat_id=chat_id,
            text=clean,
            reply_markup=reply_markup,
            parse_mode=None
        )


async def send_in_parts(
    bot, chat_id: int, text: str, reply_markup=None, parse_mode: str = "Markdown"
) -> None:
    parts = split_message(text)
    for i, part in enumerate(parts):
        kb = reply_markup if i == len(parts) - 1 else None
        await send_safe(bot, chat_id, part, reply_markup=kb)
        if i < len(parts) - 1:
            await asyncio.sleep(0.3)


# ---------------------------------------------------------------------------
# Keyboards
# ---------------------------------------------------------------------------

def make_main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🌅 Brief", callback_data="cmd_brief"),
                InlineKeyboardButton("📊 Screener", callback_data="cmd_screener"),
                InlineKeyboardButton("📁 Portfolio", callback_data="cmd_portfolio"),
            ],
            [
                InlineKeyboardButton("📰 News", callback_data="cmd_news"),
                InlineKeyboardButton("💬 Discuss...", callback_data="cmd_discuss_menu"),
                InlineKeyboardButton("🧵 Threads", callback_data="cmd_threads"),
            ],
            [
                InlineKeyboardButton("📋 Watchlist", callback_data="cmd_watchlist"),
                InlineKeyboardButton("📈 Performance", callback_data="cmd_performance"),
                InlineKeyboardButton("🏭 Sectors", callback_data="cmd_sectors"),
            ],
        ]
    )


def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def make_discuss_menu(positions: dict, watchlist: dict) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(ticker, callback_data=f"discuss_{ticker}")
        for ticker in positions
    ]
    buttons += [
        InlineKeyboardButton(f"★{ticker}", callback_data=f"discuss_{ticker}")
        for ticker in watchlist
    ]
    rows = _chunk(buttons, 3)
    rows.append(
        [
            InlineKeyboardButton("🔍 Other ticker...", callback_data="cmd_other_ticker"),
            InlineKeyboardButton("🌍 Macro", callback_data="cmd_macro"),
            InlineKeyboardButton("💼 Portfolio Review", callback_data="cmd_portfolio_review"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def make_ticker_actions(ticker: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📰 News", callback_data=f"ticker_news_{ticker}"),
                InlineKeyboardButton("📊 Metrics", callback_data=f"ticker_metrics_{ticker}"),
                InlineKeyboardButton("🔄 Refresh", callback_data=f"ticker_refresh_{ticker}"),
            ],
            [
                InlineKeyboardButton("💾 Update Thesis", callback_data=f"ticker_update_thesis_{ticker}"),
                InlineKeyboardButton("🚨 Flag Risk", callback_data=f"ticker_flag_risk_{ticker}"),
                InlineKeyboardButton("➕ Add to Watchlist", callback_data=f"ticker_add_watchlist_{ticker}"),
            ],
            [
                InlineKeyboardButton("🔚 End Thread", callback_data="cmd_done"),
                InlineKeyboardButton("🏠 Main Menu", callback_data="cmd_main_menu"),
            ],
        ]
    )


def make_screener_actions(tickers: list[str]) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(f"Discuss {t}", callback_data=f"discuss_{t}")
        for t in tickers[:8]
    ]
    rows = _chunk(buttons, 2)
    rows.append([InlineKeyboardButton("🏠 Main Menu", callback_data="cmd_main_menu")])
    return InlineKeyboardMarkup(rows)


def make_portfolio_actions(tickers: list[str]) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(f"📈 {t}", callback_data=f"discuss_{t}") for t in tickers
    ]
    rows = _chunk(buttons, 3)
    rows.append(
        [
            InlineKeyboardButton("📰 News Triage", callback_data="cmd_news"),
            InlineKeyboardButton("🏠 Main Menu", callback_data="cmd_main_menu"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def make_news_actions(alert_tickers: list[str]) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(f"⚠️ Discuss {t}", callback_data=f"discuss_{t}")]
        for t in alert_tickers
    ]
    rows.append([InlineKeyboardButton("🏠 Main Menu", callback_data="cmd_main_menu")])
    return InlineKeyboardMarkup(rows)


def make_confirm_cancel(change_id: int, description: str) -> InlineKeyboardMarkup:
    short_desc = description[:25]
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(f"✅ Confirm — {short_desc}", callback_data=f"confirm_{change_id}"),
                InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_{change_id}"),
            ]
        ]
    )


_THREAD_ICONS = {"MACRO": "🌍", "PORTFOLIO": "💼"}


def _thread_icon(thread_id: str) -> str:
    for key, icon in _THREAD_ICONS.items():
        if key in thread_id.upper():
            return icon
    return "📈"


def make_thread_list_keyboard(threads: list[dict]) -> InlineKeyboardMarkup:
    rows = []
    for t in threads:
        thread_id = t["thread_id"]
        subject = t.get("subject") or thread_id
        icon = _thread_icon(thread_id)
        last_active = _relative_time(t.get("last_active"))
        prefix = "switch_topic_" if t.get("thread_type") == "topic" else "switch_ticker_"
        label = f"{icon} {subject} — {last_active}"
        rows.append([InlineKeyboardButton(label, callback_data=f"{prefix}{subject}")])
    return InlineKeyboardMarkup(rows)


def make_suggestions_keyboard(suggestions: list[str]) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(s, callback_data=f"suggest_{i}")]
        for i, s in enumerate(suggestions[:2])
    ]
    return InlineKeyboardMarkup(rows)


# ---------------------------------------------------------------------------
# Text formatters
# ---------------------------------------------------------------------------

def _relative_time(timestamp: str | None) -> str:
    if not timestamp:
        return "unknown"
    try:
        from datetime import datetime

        then = datetime.fromisoformat(timestamp)
        delta = datetime.now() - then
        seconds = delta.total_seconds()
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            return f"{int(seconds // 60)}m ago"
        if seconds < 86400:
            return f"{int(seconds // 3600)}h ago"
        return f"{int(seconds // 86400)}d ago"
    except (ValueError, TypeError):
        return timestamp


def format_thread_list(threads: list[dict]) -> str:
    if not threads:
        return "No active threads. Start one with /discuss TICKER"

    lines = ["🧵 ACTIVE THREADS"]
    for t in threads:
        icon = _thread_icon(t["thread_id"])
        subject = t.get("subject") or t["thread_id"]
        kind = "ticker" if t.get("thread_type") != "topic" else t.get("thread_type", "topic")
        count = t.get("message_count", 0)
        last_active = _relative_time(t.get("last_active"))
        lines.append(f"{icon} {subject} ({kind}) — {count} messages — last active {last_active}")
    return "\n".join(lines)
