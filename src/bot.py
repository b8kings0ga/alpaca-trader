"""
Main bot logic for the Alpaca Trading Bot.
"""
import os
import time
from datetime import datetime, timedelta
import pytz
from config import config
from src.logger import get_logger
from src.data import MarketData
from src.yfinance_data import YFinanceData
from src.yfinance_db_client import YFinanceDBClient
from src.strategies import get_strategy
from src.trader import Trader
from src.notifications import NotificationSystem
from src.scheduler import TradingScheduler

logger = get_logger()

class AlpacaBot:
    """
    Main trading bot class that orchestrates the trading process.
    """
    def __init__(self, strategy_name='moving_average_crossover', data_source='yfinance'):
        """
        Initialize the trading bot.
        
        Args:
            strategy_name (str): Name of the strategy to use
        """
        logger.info(f"Initializing AlpacaBot with strategy: {strategy_name}, data source: {data_source}")
        
        # Check if API credentials are set
        if not config.ALPACA_API_KEY or not config.ALPACA_API_SECRET:
            raise ValueError("Alpaca API credentials not set")
            
        # Initialize components
        if data_source == 'yfinance':
            self.market_data = YFinanceData()
            logger.info("Using YFinanceData for market data")
        elif data_source == 'yfinance-db':
            self.market_data = YFinanceDBClient()
            # Check if the service is available
            if not self.market_data.is_service_available():
                logger.warning("YFinance DB service is not available, falling back to YFinanceData")
                self.market_data = YFinanceData()
            else:
                logger.info("Using YFinance DB service for market data")
        else:
            self.market_data = MarketData()
            logger.info("Using Alpaca MarketData for market data")
            
        self.strategy = get_strategy(strategy_name)
        self.trader = Trader()
        self.notification = NotificationSystem()
        self.scheduler = TradingScheduler()
        
        # Initialize state
        self.last_run_time = None
        self.is_running = False
        
        # Create data directory if it doesn't exist
        import os
        os.makedirs('data/cache', exist_ok=True)
        
        logger.info("AlpacaBot initialized successfully")
        
    def run(self, force_initial_positions=True):
        """
        Run the trading bot once.
        
        Args:
            force_initial_positions (bool): Whether to force creating initial positions if none exist
        """
        # Generate a unique run ID for this execution
        import uuid
        run_id = str(uuid.uuid4())[:8]
        
        if self.is_running:
            logger.warning(f"[Run ID: {run_id}] Bot is already running - skipping this run")
            # Add force-kill logic for running instances that have been running for too long
            if self.last_run_time and (datetime.now() - self.last_run_time).total_seconds() > 300:  # 5 minute timeout
                logger.warning(f"[Run ID: {run_id}] Previous job has been running for too long (>5 minutes), force-killing it")
                self.is_running = False
                logger.info(f"[Run ID: {run_id}] Reset is_running flag to False due to timeout")
            else:
                return
        
        self.is_running = True
        start_time = time.time()
        logger.info(f"[Run ID: {run_id}] Starting trading run at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        
        try:
            # Check if market is open
            is_market_open = self.market_data.is_market_open()
            logger.info(f"Market is {'open' if is_market_open else 'closed'}")
            
            if not is_market_open:
                try:
                    market_hours = self.market_data.get_market_hours()
                    logger.info(f"Market hours: {market_hours}")
                    
                    if market_hours[0]:
                        try:
                            # Check if market_hours[0] is a datetime.time object
                            if hasattr(market_hours[0], 'astimezone'):
                                # It's already a datetime object
                                next_open = market_hours[0].astimezone(pytz.timezone(config.TIMEZONE))
                            else:
                                # It's a time object, convert it to datetime first
                                today = datetime.now().date()
                                next_open_datetime = datetime.combine(today, market_hours[0])
                                next_open = next_open_datetime.astimezone(pytz.timezone(config.TIMEZONE))
                            
                            logger.info(f"Next market open: {next_open}")
                        except Exception as e:
                            logger.error(f"Error converting market open time: {e}")
                            logger.error(f"Market open time type: {type(market_hours[0])}")
                except Exception as e:
                    logger.error(f"Error getting market hours: {e}")
                    
                # We can still run in paper trading mode even if market is closed
                logger.info("Running in paper trading mode despite market being closed")
            
            # Get account information
            account_info = self.market_data.get_account()
            
            # All data source classes now return a consistent dictionary
            logger.info(f"Account equity: ${float(account_info.get('equity', 0)):.2f}")
            
            # Get current positions
            positions = self.market_data.get_positions()
            logger.info(f"Current positions: {len(positions)}")
            # Fetch market data
            data = None
            
            # Try to get recent data first (last 15 minutes)
            try:
                logger.info("Attempting to fetch recent data (last 15 minutes)")
                recent_data = self.market_data.get_recent_data(config.SYMBOLS, minutes=15, interval="1m")
                if recent_data and len(recent_data) > 0:
                    logger.info(f"Successfully fetched recent data for {len(recent_data)} symbols")
                    data = recent_data
                else:
                    logger.warning("Failed to fetch recent data, falling back to historical data")
            except Exception as e:
                logger.warning(f"Error fetching recent data: {e}")
                logger.warning("Falling back to historical data")
            
            # If recent data fetch failed, try historical data
            if data is None:
                if hasattr(self.strategy, 'fetch_data') and callable(getattr(self.strategy, 'fetch_data')):
                    logger.info("Using yfinance to fetch market data")
                    if hasattr(self.strategy, 'fetch_data') and callable(getattr(self.strategy, 'fetch_data')):
                        data = self.strategy.fetch_data(config.SYMBOLS)
                    else:
                        # If strategy doesn't have fetch_data but use_yfinance is True,
                        # switch to a strategy that does have fetch_data
                        logger.info("Strategy doesn't have fetch_data method, switching to DualMovingAverageYF")
                        from src.strategies import get_strategy
                        temp_strategy = get_strategy('dual_ma_yf')
                        data = temp_strategy.fetch_data(config.SYMBOLS)
                else:
                    logger.info("Using Alpaca API to fetch market data")
                    data = self.market_data.get_bars(config.SYMBOLS)
                
            if not data:
                logger.error("Failed to fetch market data")
                self.is_running = False
                return
                
            logger.info(f"Fetched data for {len(data)} symbols")
            logger.info(f"Fetched data for {len(data)} symbols")
            
            # Generate signals
            signals = self.strategy.generate_signals(data)
            logger.info(f"Generated signals for {len(signals)} symbols")
            
            # Count signal types
            buy_signals = sum(1 for s in signals.values() if s['action'] == 'buy')
            sell_signals = sum(1 for s in signals.values() if s['action'] == 'sell')
            hold_signals = sum(1 for s in signals.values() if s['action'] == 'hold')
            logger.info(f"Signal summary: {buy_signals} buy, {sell_signals} sell, {hold_signals} hold")
            
            # Log signals with more details
            for symbol, signal_data in signals.items():
                action = signal_data['action']
                price = signal_data.get('price', 0)
                signal_changed = signal_data.get('signal_changed', False)
                short_ma = signal_data.get('short_ma', None)
                long_ma = signal_data.get('long_ma', None)
                
                ma_info = ""
                if short_ma is not None and long_ma is not None:
                    ma_info = f", short_ma: {short_ma:.2f}, long_ma: {long_ma:.2f}, diff: {short_ma - long_ma:.2f}"
                
                logger.info(f"Signal for {symbol}: {action} (changed: {signal_changed}, price: ${price:.2f}{ma_info})")
            
            # Execute signals with option to force initial positions
            # Pass the run_id to the trader for consistent logging
            self.trader.run_id = run_id
            executed_orders = self.trader.execute_signals(signals, account_info, force_initial_positions)
            logger.info(f"[Run ID: {run_id}] Executed {len(executed_orders)} orders")
            
            # Send notifications for executed orders
            for order in executed_orders:
                self.notification.notify_trade(order)
            
            # Update positions after trading
            updated_positions = self.market_data.get_positions()
            updated_account = self.market_data.get_account()
            
            # Log updated account information
            logger.info(f"Updated account equity: ${float(updated_account.get('equity', 0)):.2f}")
            
            # Convert positions list to a dictionary for easier lookup in notifications
            positions_dict = {}
            logger.info(f"Converting positions to dictionary. Type of updated_positions: {type(updated_positions)}")
            if updated_positions:
                logger.info(f"First position type: {type(updated_positions[0])}")
                logger.info(f"First position dir: {dir(updated_positions[0])}")
                logger.info(f"First position repr: {repr(updated_positions[0])}")
            
            # Convert positions to a list of dictionaries instead of a dictionary
            positions_list = []
            
            for position in updated_positions:
                try:
                    # Try dictionary-like access first (for Alpaca API)
                    symbol = position['symbol']
                    # Convert position to dictionary explicitly
                    if hasattr(position, '_asdict'):  # namedtuple
                        position_dict = position._asdict()
                    else:  # already a dict
                        position_dict = position
                    # Add to both the dictionary and the list
                    positions_dict[symbol] = position_dict
                    positions_list.append(position_dict)
                    logger.info(f"Added position for {symbol} using dictionary access")
                except (TypeError, KeyError) as e:
                    logger.info(f"Dictionary access failed: {e}, trying attribute access")
                    # Fall back to attribute access (for namedtuples)
                    try:
                        symbol = position.symbol
                        # Convert namedtuple to dictionary
                        if hasattr(position, '_asdict'):
                            position_dict = position._asdict()
                        else:
                            # Create a dictionary manually
                            position_dict = {
                                'symbol': symbol,
                                'qty': position.qty,
                                'market_value': position.market_value,
                                'avg_entry_price': position.avg_entry_price,
                                'current_price': position.current_price,
                                'unrealized_pl': position.unrealized_pl
                            }
                        logger.info(f"Added position for {symbol} using attribute access")
                        # Add to both the dictionary and the list
                        positions_dict[symbol] = position_dict
                        positions_list.append(position_dict)
                    except AttributeError as e:
                        logger.info(f"Attribute access failed: {e}, trying tuple access")
                        # If it's a regular tuple, try to access by index
                        try:
                            logger.info(f"Position appears to be a regular tuple: {position}")
                            symbol = str(position[0])  # Assuming symbol is the first element
                            # Create a dictionary from the tuple
                            position_dict = {
                                'symbol': symbol,
                                'qty': position[1],
                                'market_value': float(position[2]),
                                'avg_entry_price': float(position[3]),
                                'current_price': float(position[4]),
                                'unrealized_pl': float(position[5])
                            }
                            positions_dict[symbol] = position_dict
                            positions_list.append(position_dict)
                            logger.info(f"Added position for {symbol} using tuple access")
                        except (IndexError, TypeError) as e:
                            logger.error(f"Unable to extract position data from: {position}, error: {e}")
                            continue
            
            # Send portfolio status notification
            self.notification.notify_portfolio_status(updated_account, positions_list)
            
            # Update last run time
            self.last_run_time = datetime.now()
            
            # Log execution time
            execution_time = time.time() - start_time
            logger.info(f"[Run ID: {run_id}] Trading run completed in {execution_time:.2f} seconds at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            
            # Check if execution time is close to or exceeds the scheduler interval
            if execution_time > (config.RUN_INTERVAL * 60 * 0.8):  # If execution time is >80% of the interval
                logger.warning(f"[Run ID: {run_id}] Execution time ({execution_time:.2f}s) is approaching or exceeding the scheduler interval ({config.RUN_INTERVAL * 60}s). This may cause overlapping runs!")
            
        except Exception as e:
            import traceback
            logger.error(f"Error during trading run: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.notification.notify_error(str(e))
        finally:
            self.is_running = False
            logger.info(f"[Run ID: {run_id}] Reset is_running flag to False at end of run")
            
    def wait_for_market_open(self):
        """
        Wait for the market to open.
        
        This method checks if the market is currently open. If not, it calculates
        the time until the next market open and waits until then.
        
        Returns:
            bool: True if waiting was successful, False otherwise
        """
        logger.info("Checking if market is open...")
        is_market_open = self.market_data.is_market_open()
        
        if is_market_open:
            logger.info("Market is already open. Starting trading immediately.")
            return True
            
        try:
            # Get market hours
            market_hours = self.market_data.get_market_hours()
            
            if not market_hours[0]:
                logger.error("Failed to get market hours")
                return False
                
            # Get next market open time
            next_open = None
            if hasattr(market_hours[0], 'astimezone'):
                # It's already a datetime object
                next_open = market_hours[0].astimezone(pytz.timezone(config.TIMEZONE))
            else:
                # It's a time object, convert it to datetime first
                today = datetime.now().date()
                next_open_datetime = datetime.combine(today, market_hours[0])
                next_open = next_open_datetime.astimezone(pytz.timezone(config.TIMEZONE))
            
            # Check if next_open is in the past (market already opened today but is now closed)
            now = datetime.now(pytz.timezone(config.TIMEZONE))
            if next_open < now:
                # Add a day to get tomorrow's opening time
                next_open = next_open + timedelta(days=1)
                
            # Calculate wait time
            wait_seconds = (next_open - now).total_seconds()
            
            if wait_seconds <= 0:
                logger.info("Market should be open now. Starting trading immediately.")
                return True
                
            # Format wait time for display
            wait_hours = int(wait_seconds // 3600)
            wait_minutes = int((wait_seconds % 3600) // 60)
            
            logger.info(f"Market is closed. Next market open: {next_open}")
            logger.info(f"Waiting {wait_hours} hours and {wait_minutes} minutes for market to open...")
            
            # Wait until market opens
            time.sleep(wait_seconds)
            
            logger.info("Market should be open now. Starting trading.")
            return True
            
        except Exception as e:
            logger.error(f"Error waiting for market open: {e}")
            return False
    
    def start_scheduled(self, wait_for_open=False):
        """
        Start the bot with scheduling.
        
        Args:
            wait_for_open (bool): Whether to wait for market open before starting
        Returns:
            str: Job ID
        """
        logger.info("Starting scheduled trading")
        
        
        # Wait for market open if requested
        if wait_for_open:
            logger.info("Waiting for market to open before starting scheduled trading")
            if not self.wait_for_market_open():
                logger.warning("Failed to wait for market open. Starting scheduled trading anyway.")
        
        # Create a partial function for running
        import functools
        run_with_params = functools.partial(self.run)
        
        # Get interval in minutes - run more frequently
        interval = getattr(config, 'RUN_INTERVAL', 5)  # Default to 5 minutes if not specified
        logger.info(f"Setting up scheduler to run every {interval} minutes")
        
        # Add job to scheduler with more frequent interval
        job_id = self.scheduler.add_job(run_with_params, interval_minutes=interval)
        
        # Start the scheduler
        self.scheduler.start()
        
        # Get next run time
        next_run = self.scheduler.get_next_run_time(job_id)
        if next_run:
            logger.info(f"Next scheduled run: {next_run}")
            
        # Run once immediately
        self.scheduler.run_once_now(run_with_params)
        
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


def setup_logging():
    """
    Set up logging directories.
    """
    # Create logs directory
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    os.makedirs('logs/plots', exist_ok=True)
    
    logger.info("Logging directories created")
        

if __name__ == "__main__":
    main()