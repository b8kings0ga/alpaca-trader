#!/usr/bin/env python
"""
Script to run the optimized ensemble ML trading bot during market hours.
"""
import os
import time
import logging
import argparse
import subprocess
import signal
import sys
from datetime import datetime, timedelta
import pytz
import alpaca_trade_api as tradeapi
from config.config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/market_hours_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

class MarketHoursBot:
    """
    Bot that runs the optimized ensemble ML trading bot during market hours.
    """
    
    def __init__(self, symbols, interval_minutes=15, pre_market_minutes=30, post_market_minutes=30):
        """
        Initialize the MarketHoursBot.
        
        Args:
            symbols (list): List of stock symbols to trade
            interval_minutes (int): Interval between trading bot runs in minutes
            pre_market_minutes (int): Minutes before market open to start the bot
            post_market_minutes (int): Minutes after market close to stop the bot
        """
        self.symbols = symbols
        self.interval_minutes = interval_minutes
        self.pre_market_minutes = pre_market_minutes
        self.post_market_minutes = post_market_minutes
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            ALPACA_API_KEY,
            ALPACA_API_SECRET,
            ALPACA_BASE_URL,
            api_version='v2'
        )
        
        # Bot process
        self.bot_process = None
        
        logger.info(f"Initialized MarketHoursBot with {len(symbols)} symbols")
        logger.info(f"Interval: {interval_minutes} minutes")
        logger.info(f"Pre-market: {pre_market_minutes} minutes")
        logger.info(f"Post-market: {post_market_minutes} minutes")
    
    def is_market_open(self):
        """
        Check if the market is open, including pre-market and post-market hours.
        
        Returns:
            bool: True if the market is open, False otherwise
        """
        try:
            # Get the current market clock
            clock = self.api.get_clock()
            
            # Get the current time in US/Eastern timezone
            eastern = pytz.timezone('US/Eastern')
            now = datetime.now(eastern)
            
            # Get market open and close times
            market_open = clock.next_open.astimezone(eastern)
            market_close = clock.next_close.astimezone(eastern)
            
            # If the market is already open, use the current day's open time
            if clock.is_open:
                market_open = clock.next_open.astimezone(eastern) - timedelta(days=1)
            
            # Calculate pre-market and post-market times
            pre_market_open = market_open - timedelta(minutes=self.pre_market_minutes)
            post_market_close = market_close + timedelta(minutes=self.post_market_minutes)
            
            # Check if current time is within trading hours (including pre and post market)
            is_open = pre_market_open <= now <= post_market_close
            
            if is_open:
                if now < market_open:
                    logger.info(f"Pre-market hours (Market opens in {(market_open - now).seconds // 60} minutes)")
                elif now > market_close:
                    logger.info(f"Post-market hours (Market closed {(now - market_close).seconds // 60} minutes ago)")
                else:
                    logger.info("Regular market hours")
            else:
                next_open = pre_market_open
                if now > post_market_close:
                    # Get the next trading day
                    calendar = self.api.get_calendar(start=now.date(), end=(now + timedelta(days=7)).date())
                    if calendar:
                        next_market_open = eastern.localize(datetime.combine(calendar[0].date, calendar[0].open))
                        next_open = next_market_open - timedelta(minutes=self.pre_market_minutes)
                
                time_until_open = (next_open - now).total_seconds() / 60
                logger.info(f"Market is closed. Next trading session in {time_until_open:.0f} minutes")
            
            return is_open
        
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False
    
    def start_bot(self):
        """
        Start the optimized ensemble ML trading bot.
        """
        if self.bot_process is not None and self.bot_process.poll() is None:
            logger.info("Bot is already running")
            return
        
        try:
            # Construct the command to run the bot
            cmd = [
                "python", "run_optimized_bot.py",
                "--symbols"] + self.symbols + [
                "--interval", str(self.interval_minutes)
            ]
            
            # Start the bot process
            logger.info(f"Starting bot with command: {' '.join(cmd)}")
            self.bot_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Start a thread to read and log the bot's output
            import threading
            def log_output():
                for line in self.bot_process.stdout:
                    logger.info(f"Bot: {line.strip()}")
            
            threading.Thread(target=log_output, daemon=True).start()
            
            logger.info("Bot started")
        
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
    
    def stop_bot(self):
        """
        Stop the optimized ensemble ML trading bot.
        """
        if self.bot_process is None or self.bot_process.poll() is not None:
            logger.info("Bot is not running")
            return
        
        try:
            # Send SIGTERM to the bot process
            logger.info("Stopping bot")
            self.bot_process.terminate()
            
            # Wait for the process to terminate
            try:
                self.bot_process.wait(timeout=10)
                logger.info("Bot stopped")
            except subprocess.TimeoutExpired:
                # If the process doesn't terminate within the timeout, kill it
                logger.warning("Bot did not terminate gracefully, killing it")
                self.bot_process.kill()
                self.bot_process.wait()
                logger.info("Bot killed")
            
            self.bot_process = None
        
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
    
    def run(self):
        """
        Run the MarketHoursBot.
        """
        logger.info("Starting MarketHoursBot")
        
        # Register signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down")
            self.stop_bot()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while True:
                # Check if the market is open
                if self.is_market_open():
                    # Start the bot if it's not running
                    self.start_bot()
                else:
                    # Stop the bot if it's running
                    self.stop_bot()
                
                # Sleep for a while before checking again
                logger.info(f"Sleeping for 5 minutes before checking market hours again")
                time.sleep(5 * 60)
        
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down")
            self.stop_bot()
        
        except Exception as e:
            logger.error(f"Error in MarketHoursBot: {e}")
            self.stop_bot()

def main():
    """Main function to run the MarketHoursBot."""
    parser = argparse.ArgumentParser(description='Run the optimized ensemble ML trading bot during market hours')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA'],
                        help='List of stock symbols to trade')
    parser.add_argument('--interval', type=int, default=15,
                        help='Interval between trading bot runs in minutes')
    parser.add_argument('--pre-market', type=int, default=30,
                        help='Minutes before market open to start the bot')
    parser.add_argument('--post-market', type=int, default=30,
                        help='Minutes after market close to stop the bot')
    
    args = parser.parse_args()
    
    # Initialize and run the MarketHoursBot
    bot = MarketHoursBot(
        symbols=args.symbols,
        interval_minutes=args.interval,
        pre_market_minutes=args.pre_market,
        post_market_minutes=args.post_market
    )
    
    bot.run()

if __name__ == "__main__":
    main()