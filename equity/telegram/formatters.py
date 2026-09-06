"""Message formatting and inline keyboard builders for the Telegram advisor bot."""

import asyncio
import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message splitting / sending
# ---------------------------------------------------------------------------

def split_message(text: str, max_length: int = 4000) -> list[str]:
    """
    Splits text into chunks under max_length characters.
    Splits at double newlines (paragraph breaks) first,
    then at single newlines, then hard-cuts as last resort.
    Never returns empty strings.
    """
    if not text:
        return []
    if len(text) <= max_length:
        return [text]

    parts = []
    current = ""

    # Try splitting at paragraph boundaries first.
    for para in text.split("\n\n"):
        if len(current) + len(para) + 2 <= max_length:
            current = f"{current}\n\n{para}" if current else para
            continue

        if current:
            parts.append(current.strip())
            current = ""

        if len(para) <= max_length:
            current = para
            continue

        # Paragraph itself is too long — split at single newlines.
        for line in para.split("\n"):
            if len(current) + len(line) + 1 <= max_length:
                current = f"{current}\n{line}" if current else line
                continue

            if current:
                parts.append(current.strip())
                current = ""

            if len(line) <= max_length:
                current = line
                continue

            # Line itself is too long — hard cut.
            for j in range(0, len(line), max_length):
                parts.append(line[j:j + max_length])
            current = ""

    if current:
        parts.append(current.strip())

    return [p for p in parts if p]


async def send_safe(bot, chat_id: int, text: str,
                    reply_markup=None) -> None:
    """
    Sends text safely, handling both Markdown parse errors and
    messages exceeding Telegram's 4096 character limit.
    Splits long messages at paragraph boundaries before sending.
    Only attaches reply_markup to the last part.
    """
    if not text or not text.strip():
        return

    parts = split_message(text, max_length=4000)

    for i, part in enumerate(parts):
        kb = reply_markup if i == len(parts) - 1 else None
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=part,
                reply_markup=kb,
                parse_mode="Markdown",
            )
        except Exception:
            # Strip any Markdown and retry as plain text.
            clean = (
                part.replace("*", "").replace("_", "")
                .replace("`", "").replace("[", "").replace("]", "")
            )
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=clean,
                    reply_markup=kb,
                    parse_mode=None,
                )
            except Exception:
                # Last resort — truncate. Parts are already <= max_length,
                # so this only fires for non-length errors (bad chat_id,
                # network, etc.); log so a silent failure stays visible.
                logger.warning(
                    "send_safe: plain-text send also failed for chat %s, truncating part %d/%d",
                    chat_id, i + 1, len(parts),
                )
                await bot.send_message(
                    chat_id=chat_id,
                    text=clean[:4000],
                    reply_markup=kb,
                    parse_mode=None,
                )

        if len(parts) > 1 and i < len(parts) - 1:
            await asyncio.sleep(0.3)


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


def make_news_actions(alert_tickers: list[str], all_tickers: list[str] | None = None) -> InlineKeyboardMarkup | None:
    """Discuss buttons for thesis-alert tickers, plus a headline-pagination button per ticker with news."""
    buttons = []
    if alert_tickers:
        row = [InlineKeyboardButton(f"⚠️ {t}", callback_data=f"discuss_{t}") for t in alert_tickers[:3]]
        buttons.append(row)
    if all_tickers:
        row = []
        for t in all_tickers[:4]:
            row.append(InlineKeyboardButton(f"📰 {t}", callback_data=f"headlines_{t}_0"))
            if len(row) == 4:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)
    buttons.append([InlineKeyboardButton("🏠 Main Menu", callback_data="cmd_main_menu")])
    return InlineKeyboardMarkup(buttons) if buttons else None


def make_tickers_keyboard(tickers: list[str], label: str = "Discuss") -> InlineKeyboardMarkup:
    """Discuss buttons for up to 8 tickers, 3 per row — used by brief_builder for alert/screener call-outs."""
    buttons = []
    row = []
    for ticker in tickers[:8]:
        row.append(InlineKeyboardButton(f"{label} {ticker}", callback_data=f"discuss_{ticker}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    return InlineKeyboardMarkup(buttons)


def make_article_keyboard(articles: list[dict]) -> InlineKeyboardMarkup:
    """Read/Discuss button pair for up to 3 thesis-alert articles (each `{'url', 'ticker'}`)."""
    buttons = []
    for article in articles[:3]:
        row = []
        if article.get("url"):
            row.append(InlineKeyboardButton("🔗 Read", url=article["url"]))
        row.append(InlineKeyboardButton("💬 Discuss", callback_data=f"discuss_{article.get('ticker', 'MACRO')}"))
        buttons.append(row)
    return InlineKeyboardMarkup(buttons)


def make_headline_page_keyboard(
    ticker: str, page: int, total_headlines: int, page_size: int = 10
) -> InlineKeyboardMarkup:
    """Prev/page-indicator/Next row plus a Discuss-ticker row, for `format_headline_page()`'s output."""
    total_pages = max(1, (total_headlines + page_size - 1) // page_size)
    nav_row = []
    if page > 0:
        nav_row.append(InlineKeyboardButton("◀ Prev", callback_data=f"headlines_{ticker}_{page - 1}"))
    nav_row.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="noop"))
    if (page + 1) * page_size < total_headlines:
        nav_row.append(InlineKeyboardButton("Next ▶", callback_data=f"headlines_{ticker}_{page + 1}"))
    action_row = [InlineKeyboardButton(f"💬 Discuss {ticker}", callback_data=f"discuss_{ticker}")]
    return InlineKeyboardMarkup([nav_row, action_row])


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


def format_headline_page(ticker: str, headlines: list[dict], page: int, page_size: int = 10) -> str:
    """One page of `ticker`'s full (untruncated) headline list — see `news_triage.run_news_triage()`'s `all_headlines`."""
    start = page * page_size
    page_headlines = headlines[start:start + page_size]
    lines = [f"📰 {ticker} — All Headlines (page {page + 1})"]
    for h in page_headlines:
        tier = h.get("source_tier", 3)
        pub = h.get("publisher", "")
        url = h.get("url", "")
        title = h.get("title", "")
        unverified = " [unverified]" if tier == 3 else ""
        thesis = " ⚠️" if h.get("thesis_breaker_match") else ""
        if url:
            lines.append(f"• [{title}]({url}) [{pub}]{unverified}{thesis}")
        else:
            lines.append(f"• {title} [{pub}]{unverified}{thesis}")
    return "\n".join(lines)
