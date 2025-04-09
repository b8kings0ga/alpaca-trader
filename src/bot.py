"""
Main bot logic for the Alpaca Trading Bot.
"""
import os
import time
from datetime import datetime
import pytz
from config import config
from src.logger import get_logger
from src.data import MarketData
from src.strategies import get_strategy
from src.trader import Trader
from src.notifications import NotificationSystem
from src.scheduler import TradingScheduler

logger = get_logger()

class AlpacaBot:
    """
    Main trading bot class that orchestrates the trading process.
    """
    def __init__(self, strategy_name='moving_average_crossover'):
        """
        Initialize the trading bot.
        
        Args:
            strategy_name (str): Name of the strategy to use
        """
        logger.info(f"Initializing AlpacaBot with strategy: {strategy_name}")
        
        # Check if API credentials are set
        if not config.ALPACA_API_KEY or not config.ALPACA_API_SECRET:
            raise ValueError("Alpaca API credentials not set")
            
        # Initialize components
        self.market_data = MarketData()
        self.strategy = get_strategy(strategy_name)
        self.trader = Trader()
        self.notification = NotificationSystem()
        self.scheduler = TradingScheduler()
        
        # Initialize state
        self.last_run_time = None
        self.is_running = False
        
        logger.info("AlpacaBot initialized successfully")
        
    def run(self):
        """
        Run the trading bot once.
        """
        if self.is_running:
            logger.warning("Bot is already running")
            return
            
        self.is_running = True
        start_time = time.time()
        logger.info("Starting trading run")
        
        try:
            # Check if market is open
            is_market_open = self.market_data.is_market_open()
            logger.info(f"Market is {'open' if is_market_open else 'closed'}")
            
            if not is_market_open:
                market_hours = self.market_data.get_market_hours()
                if market_hours[0]:
                    next_open = market_hours[0].astimezone(pytz.timezone(config.TIMEZONE))
                    logger.info(f"Next market open: {next_open}")
                    
                # We can still run in paper trading mode even if market is closed
                logger.info("Running in paper trading mode despite market being closed")
            
            # Get account information
            account_info = self.market_data.get_account()
            logger.info(f"Account equity: ${float(account_info.get('equity', 0)):.2f}")
            
            # Get current positions
            positions = self.market_data.get_positions()
            logger.info(f"Current positions: {len(positions)}")
            
            # Fetch market data
            data = self.market_data.get_bars(config.SYMBOLS)
            if not data:
                logger.error("Failed to fetch market data")
                self.is_running = False
                return
                
            logger.info(f"Fetched data for {len(data)} symbols")
            
            # Generate signals
            signals = self.strategy.generate_signals(data)
            logger.info(f"Generated signals for {len(signals)} symbols")
            
            # Log signals
            for symbol, signal_data in signals.items():
                logger.info(f"Signal for {symbol}: {signal_data['action']} (changed: {signal_data.get('signal_changed', False)})")
            
            # Execute signals
            executed_orders = self.trader.execute_signals(signals, account_info)
            logger.info(f"Executed {len(executed_orders)} orders")
            
            # Send notifications for executed orders
            for order in executed_orders:
                self.notification.notify_trade(order)
            
            # Update positions after trading
            updated_positions = self.market_data.get_positions()
            updated_account = self.market_data.get_account()
            
            # Send portfolio status notification
            self.notification.notify_portfolio_status(updated_account, updated_positions)
            
            # Update last run time
            self.last_run_time = datetime.now()
            
            # Log execution time
            execution_time = time.time() - start_time
            logger.info(f"Trading run completed in {execution_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during trading run: {e}", exc_info=True)
            self.notification.notify_error(str(e))
        finally:
            self.is_running = False
            
    def start_scheduled(self):
        """
        Start the bot with scheduling.
        """
        logger.info("Starting scheduled trading")
        
        # Add job to scheduler
        job_id = self.scheduler.add_job(self.run)
        
        # Start the scheduler
        self.scheduler.start()
        
        # Get next run time
        next_run = self.scheduler.get_next_run_time(job_id)
        if next_run:
            logger.info(f"Next scheduled run: {next_run}")
            
        # Run once immediately
        self.scheduler.run_once_now(self.run)
        
        return job_id
        
    def stop_scheduled(self):
        """
        Stop the scheduled trading.
        """
        logger.info("Stopping scheduled trading")
        self.scheduler.stop()
        
    def run_backtest(self, start_date, end_date=None):
        """
        Run a backtest for the strategy.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            dict: Backtest results
        """
        logger.info(f"Running backtest from {start_date} to {end_date or 'now'}")
        
        # This is a placeholder for backtesting functionality
        # In a real implementation, you would:
        # 1. Fetch historical data for the date range
        # 2. Run the strategy on each day
        # 3. Track virtual trades and portfolio value
        # 4. Calculate performance metrics
        
        logger.warning("Backtesting not fully implemented yet")
        
        return {
            "status": "not_implemented",
            "message": "Backtesting functionality is not fully implemented yet"
        }


def main():
    """
    Main entry point for the trading bot.
    """
    try:
        # Create logs directory
        os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
        
        logger.info("Starting Alpaca Trading Bot")
        
        # Initialize the bot
        bot = AlpacaBot()
        
        # Start scheduled trading
        bot.start_scheduled()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        

if __name__ == "__main__":
    main()