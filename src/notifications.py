"""
Notification system for the Alpaca Trading Bot.
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import requests
from config import config
from src.logger import get_logger

logger = get_logger()

class NotificationSystem:
    """
    Class for sending notifications about trades and portfolio status.
    """
    def __init__(self):
        """
        Initialize the notification system.
        """
        self.enabled = config.ENABLE_NOTIFICATIONS
        self.notification_type = config.NOTIFICATION_TYPE
        logger.info(f"Notification system initialized (enabled: {self.enabled}, type: {self.notification_type})")
        
    def send_notification(self, subject, message):
        """
        Send a notification.
        
        Args:
            subject (str): Notification subject
            message (str): Notification message
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug("Notifications are disabled")
            return False
            
        try:
            if self.notification_type == 'email':
                return self._send_email(subject, message)
            elif self.notification_type == 'telegram':
                return self._send_telegram(subject, message)
            elif self.notification_type == 'webhook':
                return self._send_webhook(subject, message)
            else:
                logger.warning(f"Unknown notification type: {self.notification_type}")
                return False
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False
            
    def _send_email(self, subject, message):
        """
        Send an email notification.
        
        Args:
            subject (str): Email subject
            message (str): Email message
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        if not config.NOTIFICATION_EMAIL or not config.SMTP_SERVER:
            logger.warning("Email notification settings are incomplete")
            return False
            
        try:
            msg = MIMEMultipart()
            msg['From'] = config.SMTP_USERNAME
            msg['To'] = config.NOTIFICATION_EMAIL
            msg['Subject'] = f"Alpaca Trader: {subject}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT)
            server.starttls()
            server.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
            
    def _send_telegram(self, subject, message):
        """
        Send a Telegram notification.
        
        Args:
            subject (str): Notification subject
            message (str): Notification message
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        # This is a placeholder for Telegram integration
        # You would need to add Telegram bot token and chat ID to config
        logger.warning("Telegram notifications not implemented yet")
        return False
        
    def _send_webhook(self, subject, message):
        """
        Send a webhook notification.
        
        Args:
            subject (str): Notification subject
            message (str): Notification message
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        # This is a placeholder for webhook integration
        # You would need to add webhook URL to config
        logger.warning("Webhook notifications not implemented yet")
        return False
        
    def notify_trade(self, order):
        """
        Send a notification about a trade.
        
        Args:
            order (dict): Order information
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        subject = f"Trade Executed: {order['side'].upper()} {order['symbol']}"
        message = f"""
Trade Executed:
Symbol: {order['symbol']}
Side: {order['side'].upper()}
Quantity: {order['qty']}
Type: {order['type']}
Status: {order['status']}
Time: {order['created_at']}
        """
        
        return self.send_notification(subject, message)
        
    def notify_portfolio_status(self, account_info, positions):
        """
        Send a notification about portfolio status.
        
        Args:
            account_info (dict): Account information
            positions (list): List of positions
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        subject = "Portfolio Status Update"
        
        # Format positions
        positions_str = ""
        for pos in positions:
            # Handle both dictionary and namedtuple positions
            try:
                # Try dictionary access first
                symbol = pos['symbol']
                qty = pos['qty']
                market_value = float(pos['market_value'])
                avg_entry_price = float(pos['avg_entry_price'])
                current_price = float(pos['current_price'])
                unrealized_pl = float(pos['unrealized_pl'])
            except (TypeError, KeyError):
                # Fall back to attribute access for namedtuples
                symbol = pos.symbol
                qty = pos.qty
                market_value = float(pos.market_value)
                avg_entry_price = float(pos.avg_entry_price)
                current_price = float(pos.current_price)
                unrealized_pl = float(pos.unrealized_pl)
                
            positions_str += f"""
Symbol: {symbol}
Quantity: {qty}
Market Value: ${market_value:.2f}
Avg Entry: ${avg_entry_price:.2f}
Current Price: ${current_price:.2f}
Unrealized P/L: ${unrealized_pl:.2f}
-------------------"""
        
        if not positions_str:
            positions_str = "No open positions"
            
        # Handle both dictionary and object account_info
        try:
            # Try dictionary access first
            equity = float(account_info['equity'])
            cash = float(account_info['cash'])
            buying_power = float(account_info['buying_power'])
        except (TypeError, KeyError):
            # Fall back to attribute access
            equity = float(account_info.equity)
            cash = float(account_info.cash)
            buying_power = float(account_info.buying_power)
            
        message = f"""
Portfolio Status:
Equity: ${equity:.2f}
Cash: ${cash:.2f}
Buying Power: ${buying_power:.2f}

Positions:
{positions_str}
        """
        
        return self.send_notification(subject, message)
        
    def notify_error(self, error_message):
        """
        Send a notification about an error.
        
        Args:
            error_message (str): Error message
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        subject = "Error Alert"
        message = f"Error in Alpaca Trader:\n{error_message}"
        
        return self.send_notification(subject, message)