"""
Telegram Bot Integration
========================
Send notifications and reports to Telegram.

Environment Variables:
    TELEGRAM_BOT_TOKEN: Bot token from @BotFather
    TELEGRAM_CHAT_ID: Chat ID to send messages to

Usage:
    from notifications.telegram_bot import send_message, send_performance_summary

    # Simple message
    send_message("Bot started!")

    # Performance report
    send_performance_summary()
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.parse
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.logging_config import get_logger

logger = get_logger(__name__)


class TelegramNotifier:
    """
    Telegram notification client.

    Uses the Telegram Bot API to send messages.
    No external dependencies required (uses urllib).
    """

    API_BASE = "https://api.telegram.org/bot"
    MAX_MESSAGE_LENGTH = 4096

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        """
        Initialize Telegram notifier.

        Args:
            bot_token: Bot token (or TELEGRAM_BOT_TOKEN env var)
            chat_id: Chat ID (or TELEGRAM_CHAT_ID env var)

        Raises:
            ValueError: If credentials are missing
        """
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')

        if not self.bot_token or not self.chat_id:
            raise ValueError(
                "Telegram credentials required. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."
            )

        self.enabled = os.getenv('TELEGRAM_ENABLED', 'true').lower() == 'true'

    def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False,
    ) -> bool:
        """
        Send a text message.

        Args:
            text: Message text (max 4096 chars)
            parse_mode: "HTML" or "Markdown"
            disable_notification: Send silently

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.info("Telegram disabled, skipping message")
            return True

        # Truncate if too long
        if len(text) > self.MAX_MESSAGE_LENGTH:
            text = text[:self.MAX_MESSAGE_LENGTH - 50] + "\n\n... (truncated)"

        url = f"{self.API_BASE}{self.bot_token}/sendMessage"

        data = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': parse_mode,
            'disable_notification': disable_notification,
        }

        try:
            encoded_data = urllib.parse.urlencode(data).encode('utf-8')
            request = urllib.request.Request(url, data=encoded_data, method='POST')

            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))

                if result.get('ok'):
                    logger.info("Telegram message sent successfully")
                    return True
                else:
                    logger.error(f"Telegram API error: {result}")
                    return False

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_trade_alert(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        pnl: Optional[float] = None,
    ) -> bool:
        """
        Send a trade alert.

        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            qty: Quantity
            price: Execution price
            pnl: P&L if closing trade

        Returns:
            True if sent successfully
        """
        emoji = "🟢" if side == 'buy' else "🔴"

        message = f"{emoji} <b>TRADE ALERT</b>\n\n"
        message += f"<b>{side.upper()}</b> {qty} {symbol}\n"
        message += f"Price: ${price:,.2f}\n"
        message += f"Value: ${qty * price:,.2f}\n"

        if pnl is not None:
            pnl_emoji = "✅" if pnl >= 0 else "❌"
            message += f"\n{pnl_emoji} P&L: ${pnl:,.2f}"

        message += f"\n\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_message(message)

    def send_daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        num_trades: int,
        win_rate: float,
    ) -> bool:
        """
        Send daily summary.

        Args:
            equity: Current account equity
            daily_pnl: Day's P&L
            daily_pnl_pct: Day's P&L percentage
            num_trades: Number of trades today
            win_rate: Win rate percentage

        Returns:
            True if sent successfully
        """
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"

        message = f"📊 <b>DAILY SUMMARY</b>\n\n"
        message += f"💰 Equity: ${equity:,.2f}\n"
        message += f"{pnl_emoji} Daily P&L: ${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%)\n"
        message += f"📈 Trades: {num_trades}\n"
        message += f"🎯 Win Rate: {win_rate:.1f}%\n"
        message += f"\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_message(message)

    def send_error_alert(self, error_message: str, context: str = "") -> bool:
        """
        Send error alert.

        Args:
            error_message: Error description
            context: Additional context

        Returns:
            True if sent successfully
        """
        message = f"🚨 <b>ERROR ALERT</b>\n\n"
        message += f"<code>{error_message}</code>\n"

        if context:
            message += f"\n<b>Context:</b> {context}\n"

        message += f"\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_message(message)

    def send_startup_notification(self, equity: float) -> bool:
        """Send bot startup notification."""
        message = f"🤖 <b>TRADING BOT STARTED</b>\n\n"
        message += f"💰 Account Equity: ${equity:,.2f}\n"
        message += f"✅ Connected to Alpaca Paper Trading\n"
        message += f"\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_message(message)

    def send_shutdown_notification(self, reason: str = "Normal shutdown") -> bool:
        """Send bot shutdown notification."""
        message = f"🛑 <b>TRADING BOT STOPPED</b>\n\n"
        message += f"Reason: {reason}\n"
        message += f"\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_message(message)


# Convenience functions
_notifier: Optional[TelegramNotifier] = None


def _get_notifier() -> TelegramNotifier:
    """Get or create singleton notifier."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


def send_message(text: str) -> bool:
    """Send a message to Telegram."""
    try:
        return _get_notifier().send_message(text)
    except ValueError as e:
        logger.warning(f"Telegram not configured: {e}")
        return False


def send_performance_summary(hours: int = 2) -> bool:
    """
    Generate and send performance summary.

    Args:
        hours: Hours to look back (0 = all time)

    Returns:
        True if sent successfully
    """
    try:
        from reports.performance_report import PerformanceReporter

        reporter = PerformanceReporter()

        since = None
        if hours > 0:
            since = datetime.now() - timedelta(hours=hours)

        summary = reporter.generate_summary(since=since)

        return send_message(summary)

    except Exception as e:
        logger.error(f"Failed to send performance summary: {e}")
        return False


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Test message
    try:
        notifier = TelegramNotifier()

        print("Sending test message...")
        success = notifier.send_message("🧪 Test message from trading bot!")

        if success:
            print("✓ Message sent successfully!")
        else:
            print("✗ Failed to send message")

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nSet these environment variables:")
        print("  TELEGRAM_BOT_TOKEN=your_bot_token")
        print("  TELEGRAM_CHAT_ID=your_chat_id")
