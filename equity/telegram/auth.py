"""Two-tier authentication for the Telegram portfolio advisor.

Tier 1: `authorized_only` in bot.py — the Telegram user ID must match
`TELEGRAM_USER_ID`. Tier 2 (this module): every write operation additionally
requires a 6-digit code emailed to `YOUR_EMAIL` and typed back into the bot.
There is no persistent session token — every write re-authenticates.
"""

import logging
import os
import secrets
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

BOT_EMAIL = os.getenv("BOT_EMAIL")
BOT_EMAIL_PASSWORD = os.getenv("BOT_EMAIL_PASSWORD")
YOUR_EMAIL = os.getenv("YOUR_EMAIL")


def _send_auth_email(operation: str, code: str) -> bool:
    """Sends auth code via Gmail SMTP. Returns True on success."""
    try:
        msg = MIMEText(
            f"Portfolio Bot authorization required.\n\n"
            f"Operation: {operation}\n\n"
            f"Authorization code: {code}\n\n"
            f"This code expires in 10 minutes.\n\n"
            f"If you did not request this, your Telegram account "
            f"may be compromised. SSH to your VPS and set "
            f"BOT_READONLY=true in .env and restart the bot immediately."
        )
        msg["Subject"] = f"[Portfolio Bot] Auth Code: {code}"
        msg["From"] = BOT_EMAIL
        msg["To"] = YOUR_EMAIL
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(BOT_EMAIL, BOT_EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        logging.error(f"Auth email failed: {e}")
        return False


class AuthManager:
    def __init__(self):
        self._pending_email_auths: dict[int, dict] = {}
        # {user_id: {code, expires, operation}}

    def send_email_code(self, user_id: int, operation_description: str) -> bool:
        """
        Generates 6-digit code, emails it to YOUR_EMAIL,
        stores in _pending_email_auths with 10-minute expiry.
        Returns True if email sent successfully, False on failure.
        """
        code = str(secrets.randbelow(900000) + 100000)
        expires = datetime.now() + timedelta(minutes=10)
        self._pending_email_auths[user_id] = {
            "code": code,
            "expires": expires,
            "operation": operation_description,
        }
        return _send_auth_email(operation_description, code)

    def verify_email_code(self, user_id: int, code: str) -> tuple[bool, dict | None]:
        """
        Returns (True, payload) on success, (False, None) on failure/expiry.
        Clears pending auth on success or expiry.
        """
        pending = self._pending_email_auths.get(user_id)
        if not pending:
            return False, None
        if datetime.now() > pending["expires"]:
            del self._pending_email_auths[user_id]
            return False, None
        if code.strip() == pending["code"]:
            payload = pending.copy()
            del self._pending_email_auths[user_id]
            return True, payload
        return False, None

    def is_awaiting_auth(self, user_id: int) -> bool:
        pending = self._pending_email_auths.get(user_id)
        if not pending:
            return False
        if datetime.now() > pending["expires"]:
            del self._pending_email_auths[user_id]
            return False
        return True

    def cancel_pending_auth(self, user_id: int) -> None:
        self._pending_email_auths.pop(user_id, None)

    def get_pending_operation(self, user_id: int) -> str | None:
        pending = self._pending_email_auths.get(user_id)
        if not pending or datetime.now() > pending["expires"]:
            return None
        return pending["operation"]
