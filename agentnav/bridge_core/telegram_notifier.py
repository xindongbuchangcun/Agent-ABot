# agentnav/bridge_core/telegram_notifier.py
"""
Telegram progress notifier for navigation tasks.

Sends one message when navigation starts, then edits it in place as the
robot moves, and updates it a final time on arrival or failure.

Configuration (env vars passed to the MCP bridge subprocess):
    TELEGRAM_BOT_TOKEN  — bot token from @BotFather
    MY_TELEGRAM_ID      — numeric chat/user ID to message

If either variable is absent, all methods are silent no-ops so navigation
is never blocked by a missing Telegram configuration.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum seconds between throttled progress edits
_THROTTLE_S = 8.0


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self._token = bot_token
        self._chat_id = str(chat_id)
        self._message_id: Optional[int] = None
        self._last_sent: float = 0.0
        self._lock = threading.Lock()   # serialise send/edit across threads

    @classmethod
    def from_env(cls) -> "TelegramNotifier | None":
        """Return a configured notifier from env vars, or None if not set."""
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("MY_TELEGRAM_ID", "").strip()
        if not token or not chat_id:
            logger.debug("TelegramNotifier: TELEGRAM_BOT_TOKEN/MY_TELEGRAM_ID not set — disabled")
            return None
        logger.info("TelegramNotifier: enabled for chat_id=%s", chat_id)
        return cls(token, chat_id)

    def reset(self) -> None:
        """Call at the start of each navigation task to force a fresh message."""
        with self._lock:
            self._message_id = None
            self._last_sent = 0.0

    def send(self, text: str, force: bool = False) -> None:
        """
        Send or edit the navigation progress message (fire-and-forget thread).

        Args:
            text:  plain-text message body
            force: bypass throttle — use for start, arrived, failed, cancelled
        """
        now = time.monotonic()
        if not force and now - self._last_sent < _THROTTLE_S:
            return
        self._last_sent = now
        # Run in daemon thread so rclpy feedback callbacks are never blocked
        threading.Thread(target=self._send_or_edit, args=(text,), daemon=True).start()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _send_or_edit(self, text: str) -> None:
        with self._lock:
            if self._message_id is None:
                result = self._api("sendMessage", {
                    "chat_id": self._chat_id,
                    "text": text,
                })
                msg_id = result.get("result", {}).get("message_id")
                if msg_id:
                    self._message_id = msg_id
            else:
                self._api("editMessageText", {
                    "chat_id": self._chat_id,
                    "message_id": self._message_id,
                    "text": text,
                })

    def _api(self, method: str, params: dict) -> dict:
        url = f"https://api.telegram.org/bot{self._token}/{method}"
        data = json.dumps(params).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            logger.warning("TelegramNotifier: %s failed — %s", method, exc)
            return {}
