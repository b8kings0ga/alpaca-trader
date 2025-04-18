"""
Order execution for the Alpaca Trading Bot.
"""
import os
import time
import alpaca_trade_api as tradeapi
import requests
from config import config
from src.logger import get_logger
from datetime import datetime

logger = get_logger()

class Trader:
    # YFinance DB service URL
    YFINANCE_DB_HOST = os.getenv("YFINANCE_DB_HOST", "localhost")
    YFINANCE_DB_PORT = int(os.getenv("YFINANCE_DB_PORT", "8001"))
    
    # Check if we're running in Docker or directly
    # If YFINANCE_DB_HOST is set to 'yfinance-db' but we can't connect to it,
    # fall back to localhost
    @classmethod
    def _get_yfinance_db_url(cls):
        """
        Get the YFinance DB URL with proper host and port.
        This is a class method so it can be called before any instances are created.
        
        Returns:
            str: The YFinance DB URL
        """
        # Always use yfinance-db as the host when running in Docker
        host = os.getenv("YFINANCE_DB_HOST", "yfinance-db")
        port = int(os.getenv("YFINANCE_DB_PORT", "8001"))
        
        # Log environment variables for debugging
        logger.info(f"Environment variables for YFinance DB connection:")
        logger.info(f"YFINANCE_DB_HOST: {os.getenv('YFINANCE_DB_HOST', 'not set')}")
        logger.info(f"YFINANCE_DB_PORT: {os.getenv('YFINANCE_DB_PORT', 'not set')}")
        
        # Check if we're running in Docker
        is_docker = os.path.exists('/.dockerenv')
        logger.info(f"Running in Docker container: {is_docker}")
        
        # Always use the configured host without falling back to localhost
        # This ensures we use the Docker service name when running in Docker
        logger.info(f"Using YFinance DB host: {host}:{port}")
        
        # Add retry logic to wait for the service to be available
        max_retries = 5
        retry_delay = 2  # seconds
        
        for retry in range(max_retries):
            try:
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                result = s.connect_ex((host, port))
                s.close()
                
                if result == 0:
                    logger.info(f"Successfully connected to {host}:{port}")
                    break
                else:
                    logger.warning(f"Cannot connect to {host}:{port}, retry {retry+1}/{max_retries}")
                    import time
                    time.sleep(retry_delay)
            except Exception as e:
                logger.warning(f"Error checking connection to {host}:{port}: {e}, retry {retry+1}/{max_retries}")
                import time
                time.sleep(retry_delay)
        
        url = f"http://{host}:{port}"
        logger.info(f"Constructed YFinance DB URL: {url}")
        return url
    
    # Set the YFinance DB URL as a class variable
    YFINANCE_DB_URL = None  # Will be set properly when class is loaded
    
    """
    Class for executing trades with Alpaca.
    """
    
    # Initialize the URL using the classmethod
    @classmethod
    def _initialize_class(cls):
        cls.YFINANCE_DB_URL = cls._get_yfinance_db_url()
        logger.info(f"Using YFinance DB URL: {cls.YFINANCE_DB_URL}")

    def __init__(self):
        """
        Initialize the Trader class with Alpaca API.
        """
        # Initialize run_id attribute
        self.run_id = None
        
        # Dictionary to track last order time for each symbol
        self.last_order_times = {}
        
        # Log the API URLs for debugging
        logger.info(f"Initializing Trader with base URL: {config.ALPACA_BASE_URL}")
        
        # Remove trailing /v2 from base URL if present to avoid duplicate version in path
        base_url = config.ALPACA_BASE_URL
        if base_url.endswith('/v2'):
            base_url = base_url[:-3]
            logger.info(f"Adjusted base URL to avoid duplicate version: {base_url}")
        
        self.api = tradeapi.REST(
            config.ALPACA_API_KEY,
            config.ALPACA_API_SECRET,
            base_url,
            api_version='v2'
        )
        
        # Verify Alpaca API connection
        try:
            account = self.api.get_account()
            logger.info(f"Successfully connected to Alpaca API. Account status: {account.status}")
            logger.info(f"Account details: ID={account.id}, Cash=${float(account.cash):.2f}, Equity=${float(account.equity):.2f}")
        except Exception as e:
            logger.error(f"Error connecting to Alpaca API: {e}")
            logger.error("This may cause trades to be recorded in the database but not executed in Alpaca")
            
        logger.info("Trader initialized")
        
    def execute_signals(self, signals, account_info, force_initial_positions=False):
        """
        Execute trades based on signals.
        
        Args:
            signals (dict): Dictionary of signals for each symbol
            account_info (dict): Account information
            
        Returns:
            list: List of executed orders
        """
        executed_orders = []
        
        # If force_initial_positions is True and we have no positions, create initial positions
        current_positions = {p.symbol: p for p in self.api.list_positions()}
        # Get portfolio value from account info
        portfolio_value = float(account_info.get('portfolio_value', 100000.0))
        
        # Convert positions list to a dictionary for easier lookup
        # This handles both Alpaca Position objects and YFinanceDBClient Position namedtuples
        positions_dict = {}
        for position in current_positions:
            try:
                # Try dictionary-like access first (for Alpaca API)
                symbol = position['symbol']
                positions_dict[symbol] = position
            except (TypeError, KeyError):
                # Fall back to attribute access (for namedtuples)
                try:
                    symbol = position.symbol
                    positions_dict[symbol] = position
                except AttributeError:
                    logger.error(f"Unable to get symbol from position: {position}")
                    continue
        
        # Replace current_positions with the dictionary for easier lookup
        current_positions = positions_dict
        
        # Log detailed information about current positions
        logger.info(f"Current positions details: {current_positions}")
        for symbol, position in current_positions.items():
            logger.info(f"Position for {symbol}: {position.qty} shares at avg price ${position.avg_entry_price}")
        
        
        if force_initial_positions and len(current_positions) == 0:
            logger.info("No current positions found. Creating initial positions based on current signals.")
            logger.info(f"Signals available: {list(signals.keys())}")
            
            # Filter for buy signals or create balanced portfolio if no buy signals
            buy_signals = {symbol: data for symbol, data in signals.items()
                          if data['action'] == 'buy' or data['signal'] > 0}
            
            logger.info(f"Buy signals found: {list(buy_signals.keys())}")
            
            # If no buy signals, use all symbols with a balanced approach
            if not buy_signals:
                logger.info("No buy signals found. Creating balanced initial positions.")
                # Use all symbols with equal weight
                symbols_to_buy = list(signals.keys())
                
                logger.info(f"Symbols to buy (all signals): {symbols_to_buy}")
                
                # Check if symbols_to_buy is empty to avoid division by zero
                if not symbols_to_buy:
                    logger.warning("No symbols available for initial positions")
                    logger.warning(f"Signals dictionary: {signals}")
                    logger.warning("This is likely due to insufficient data for signal generation")
                    return executed_orders
                    
                # Use default symbols if none are available in signals
                if len(symbols_to_buy) == 0:
                    logger.info("Using default symbols for initial positions")
                    symbols_to_buy = config.SYMBOLS
                    
                # Log the symbols we're going to buy
                logger.info(f"Creating initial positions for: {symbols_to_buy}")
                
                # Calculate position size per symbol
                position_size_per_symbol = portfolio_value * config.POSITION_SIZE / len(symbols_to_buy)
                
                for symbol in symbols_to_buy:
                    if symbol in current_positions:
                        logger.info(f"Already have position in {symbol}, skipping")
                        continue
                        
                    # Check if price exists and is valid
                    if 'price' not in signals[symbol] or signals[symbol]['price'] <= 0:
                        logger.warning(f"Invalid or missing price for {symbol}")
                        # Use a default price based on historical data or a fixed value
                        try:
                            # Try to get the latest price from Alpaca
                            latest_bar = self.api.get_latest_bar(symbol)
                            price = latest_bar.c  # Close price
                            logger.info(f"Using latest price from Alpaca for {symbol}: ${price:.2f}")
                        except Exception as e:
                            logger.error(f"Error getting latest price for {symbol}: {e}")
                            # Use a default price as fallback
                            price = 100.0  # Default price
                            logger.info(f"Using default price for {symbol}: ${price:.2f}")
                    else:
                        price = signals[symbol]['price']
                    
                    # Calculate quantity based on position size and price
                    qty = int(position_size_per_symbol / price)
                    
                    # Ensure minimum quantity
                    if qty <= 0:
                        qty = 1  # Minimum quantity
                        logger.warning(f"Adjusted quantity for {symbol} to minimum: {qty}")
                    
                    logger.info(f"Calculated quantity for {symbol}: {qty} shares at ${price:.2f}")
                        
                    try:
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        logger.info(f"Initial position: Buy order placed for {qty} shares of {symbol} at ~${price:.2f}")
                        executed_orders.append(self._format_order(order))
                    except Exception as e:
                        logger.error(f"Error creating initial position for {symbol}: {e}")
            else:
                # Use buy signals
                # Check if buy_signals is empty to avoid division by zero
                if not buy_signals:
                    logger.warning("No buy signals available for initial positions")
                    return executed_orders
                    
                # Log the symbols we're going to buy
                logger.info(f"Creating initial positions based on buy signals for: {list(buy_signals.keys())}")
                
                # Calculate position size per symbol
                position_size_per_symbol = portfolio_value * config.POSITION_SIZE / len(buy_signals)
                
                for symbol, signal_data in buy_signals.items():
                    if symbol in current_positions:
                        continue
                        
                    price = signal_data['price']
                    qty = int(position_size_per_symbol / price)
                    
                    if qty <= 0 or price <= 0:
                        logger.warning(f"Invalid quantity ({qty}) or price ({price}) for {symbol}")
                        continue
                        
                    try:
                        # Store the signal in the database
                        signal_id = self.store_signal(symbol, signal_data)
                        
                        # Place buy order
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        
                        formatted_order = self._format_order(order)
                        
                        # Check if the order was actually accepted by Alpaca
                        if order.status == 'accepted' or order.status == 'filled':
                            logger.info(f"[Run ID: {self.run_id or 'unknown'}] Initial position order was accepted by Alpaca, storing in database")
                            # Store the executed trade in the database
                            trade_id = self.store_trade(formatted_order)
                            logger.info(f"[Run ID: {self.run_id or 'unknown'}] Trade stored in database with ID: {trade_id}")
                        else:
                            logger.warning(f"[Run ID: {self.run_id or 'unknown'}] Initial position order was not accepted by Alpaca (status: {order.status}), not storing in database")
                        
                        logger.info(f"Initial position: Buy order placed for {qty} shares of {symbol} at ~${price:.2f}")
                        executed_orders.append(formatted_order)
                    except Exception as e:
                        logger.error(f"Error creating initial position for {symbol}: {e}")
        
        # Check if we have enough buying power
        buying_power = float(account_info.get('buying_power', 0))
        if buying_power <= 0:
            logger.warning(f"Not enough buying power: {buying_power}")
            return executed_orders
            
        # Calculate position size
        portfolio_value = float(account_info.get('portfolio_value', 0))
        position_value = portfolio_value * config.POSITION_SIZE
        
        # Get current positions (refresh after potential initial positions)
        if force_initial_positions and executed_orders:
            # Wait a moment for orders to process
            import time
            time.sleep(2)
            
        # Refresh positions
        current_positions = {p.symbol: p for p in self.api.list_positions()}
        
        # Process signals
        logger.info(f"Processing {len(signals)} signals")
        logger.info(f"Detailed signals: {signals}")
        buy_signals = 0
        sell_signals = 0
        hold_signals = 0
        
        for symbol, signal_data in signals.items():
            action = signal_data['action']
            price = signal_data['price']
            signal_changed = signal_data.get('signal_changed', False)
            
            # Count signal types
            if action == 'buy':
                buy_signals += 1
            elif action == 'sell':
                sell_signals += 1
            elif action == 'hold':
                hold_signals += 1
            
            logger.info(f"Signal for {symbol}: {action} (changed: {signal_changed}, price: ${price:.2f})")
            
            # Skip if no action
            if action == 'hold':
                continue
                
            # Only execute signals if they've changed or if it's the first time we're seeing them
            # This prevents duplicate orders from being placed for the same signal
            if action == 'buy':
                # Use the run_id passed from the bot, or generate one if not available
                run_id = self.run_id or f"trader-{id(self)}"
                logger.info(f"[Run ID: {run_id}] Processing buy signal for {symbol} (signal_changed: {signal_changed})")
                
                # Skip if the signal hasn't changed - this prevents duplicate orders
                if not signal_changed:
                    logger.info(f"[Run ID: {run_id}] Skipping buy signal for {symbol} because signal hasn't changed")
                    continue
                    
                logger.info(f"[Run ID: {run_id}] Processing {action} signal for {symbol} (signal_changed: {signal_changed})")
                    
            try:
                if action == 'buy':
                    # Check if we already have a position, but don't skip - update it if needed
                    if symbol in current_positions:
                        position = current_positions[symbol]
                        current_qty = abs(int(position.qty))
                        logger.info(f"Already have position in {symbol} with {current_qty} shares, evaluating if adjustment needed")
                        
                        # If signal has changed, we might want to increase our position
                        if signal_changed:
                            # Calculate additional quantity to buy
                            additional_qty = int(position_value / price) - current_qty
                            if additional_qty > 0:
                                logger.info(f"Increasing position in {symbol} by {additional_qty} shares")
                                try:
                                    order = self.api.submit_order(
                                        symbol=symbol,
                                        qty=additional_qty,
                                        side='buy',
                                        type='market',
                                        time_in_force='day'
                                    )
                                    
                                    formatted_order = self._format_order(order)
                                    
                                    # Check if the order was actually accepted by Alpaca
                                    if order.status == 'accepted' or order.status == 'filled':
                                        logger.info(f"[Run ID: {self.run_id or 'unknown'}] Additional buy order was accepted by Alpaca, storing in database")
                                        # Store the executed trade in the database
                                        trade_id = self.store_trade(formatted_order)
                                        logger.info(f"[Run ID: {self.run_id or 'unknown'}] Trade stored in database with ID: {trade_id}")
                                    else:
                                        logger.warning(f"[Run ID: {self.run_id or 'unknown'}] Additional buy order was not accepted by Alpaca (status: {order.status}), not storing in database")
                                    
                                    logger.info(f"Additional buy order placed for {additional_qty} shares of {symbol} at ~${price:.2f}")
                                    executed_orders.append(formatted_order)
                                except Exception as e:
                                    logger.error(f"Error placing additional buy order for {symbol}: {e}")
                        else:
                            logger.info(f"Signal hasn't changed for {symbol}, maintaining current position")
                        continue
                        
                    # Calculate quantity based on position size and current price
                    qty = int(position_value / price)
                    
                    # Skip if quantity is too small
                    if qty <= 0:
                        logger.warning(f"Calculated quantity for {symbol} is {qty}, skipping buy")
                        continue
                        
                    # Place buy order
                    # Log detailed information about the order being placed
                    logger.info(f"[Run ID: {run_id}] About to place buy order for {symbol}: {qty} shares at ~${price:.2f}")
                    
                    # Track the last order time for this symbol to detect potential duplicates
                    import time
                    current_time = time.time()
                    last_order_time = self.last_order_times.get(symbol, 0)
                    time_since_last_order = current_time - last_order_time
                    
                    if time_since_last_order < 60:  # If less than 60 seconds since last order
                        logger.warning(f"[Run ID: {run_id}] POTENTIAL DUPLICATE ORDER DETECTED: Placing order for {symbol} only {time_since_last_order:.2f} seconds after previous order")
                    
                    # Update the last order time for this symbol
                    self.last_order_times[symbol] = current_time
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    
                    formatted_order = self._format_order(order)
                    logger.info(f"[Run ID: {self.run_id or 'unknown'}] Buy order placed for {qty} shares of {symbol} at ~${price:.2f}")
                    logger.info(f"[Run ID: {self.run_id or 'unknown'}] Order details: ID={order.id}, Status={order.status}")
                    
                    # Verify order status by fetching it again from Alpaca
                    try:
                        # Wait a moment for the order to be processed
                        time.sleep(1)
                        fetched_order = self.api.get_order(order.id)
                        logger.info(f"[Run ID: {self.run_id or 'unknown'}] Verified order status from Alpaca: ID={fetched_order.id}, Status={fetched_order.status}")
                        
                        # Update the order status in our formatted order
                        formatted_order['status'] = fetched_order.status
                        
                        # Only store the trade if the order is executable
                        if self._is_order_executable(fetched_order):
                            logger.info(f"[Run ID: {self.run_id or 'unknown'}] Order is executable (status: {fetched_order.status}), storing in database")
                            trade_id = self.store_trade(formatted_order)
                            logger.info(f"[Run ID: {self.run_id or 'unknown'}] Trade stored in database with ID: {trade_id}")
                        else:
                            logger.warning(f"[Run ID: {self.run_id or 'unknown'}] Order is not executable (status: {fetched_order.status}), not storing in database")
                    except Exception as e:
                        logger.error(f"[Run ID: {self.run_id or 'unknown'}] Error verifying order status from Alpaca: {e}")
                    # Verify order status by fetching it again from Alpaca
                    try:
                        fetched_order = self.api.get_order(order.id)
                        logger.info(f"[Run ID: {self.run_id or 'unknown'}] Verified order status from Alpaca: ID={fetched_order.id}, Status={fetched_order.status}")
                        
                        # Update the order status in our formatted order
                        formatted_order['status'] = fetched_order.status
                    except Exception as e:
                        logger.error(f"[Run ID: {self.run_id or 'unknown'}] Error verifying order status from Alpaca: {e}")
                    
                    
                    # Check if the order was actually accepted by Alpaca
                    if order.status == 'accepted' or order.status == 'filled':
                        logger.info(f"[Run ID: {self.run_id or 'unknown'}] Order was accepted by Alpaca, storing in database")
                        # Store the executed trade in the database
                        trade_id = self.store_trade(formatted_order)
                        logger.info(f"[Run ID: {self.run_id or 'unknown'}] Trade stored in database with ID: {trade_id}")
                    else:
                        logger.warning(f"[Run ID: {self.run_id or 'unknown'}] Order was not accepted by Alpaca (status: {order.status}), not storing in database")
                    
                    executed_orders.append(formatted_order)
                    
                elif action == 'sell':
                    # Skip if we don't have a position
                    if symbol not in current_positions:
                        logger.info(f"No position in {symbol}, skipping sell")
                        logger.info(f"Current positions keys: {list(current_positions.keys())}")
                        logger.info(f"Sell signal details: action={action}, signal={signal_data.get('signal')}, signal_changed={signal_changed}")
                        continue
                    
                    
                    # Get position quantity
                    position = current_positions[symbol]
                    logger.info(f"Found position for {symbol} with {position.qty} shares, proceeding with sell")
                    qty = abs(int(position.qty))
                    
                    logger.info(f"Attempting to sell {qty} shares of {symbol} at ~${price:.2f}")
                    
                    try:
                        # Store the signal in the database
                        signal_id = self.store_signal(symbol, signal_data)
                        
                        # Place sell order
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                        
                        # Format the order for storage and logging
                        formatted_order = self._format_order(order)
                        
                        logger.info(f"[Run ID: {self.run_id or 'unknown'}] Sell order successfully placed for {qty} shares of {symbol} at ~${price:.2f}")
                        logger.info(f"[Run ID: {self.run_id or 'unknown'}] Order details: ID={order.id}, Status={order.status}")
                        
                        # Verify order status by fetching it again from Alpaca
                        try:
                            # Wait a moment for the order to be processed
                            time.sleep(1)
                            fetched_order = self.api.get_order(order.id)
                            logger.info(f"[Run ID: {self.run_id or 'unknown'}] Verified sell order status from Alpaca: ID={fetched_order.id}, Status={fetched_order.status}")
                            
                            # Update the order status in our formatted order
                            formatted_order['status'] = fetched_order.status
                            
                            # Only store the trade if the order is executable
                            if self._is_order_executable(fetched_order):
                                logger.info(f"[Run ID: {self.run_id or 'unknown'}] Sell order is executable (status: {fetched_order.status}), storing in database")
                                trade_id = self.store_trade(formatted_order)
                                logger.info(f"[Run ID: {self.run_id or 'unknown'}] Trade stored in database with ID: {trade_id}")
                            else:
                                logger.warning(f"[Run ID: {self.run_id or 'unknown'}] Sell order is not executable (status: {fetched_order.status}), not storing in database")
                        except Exception as e:
                            logger.error(f"[Run ID: {self.run_id or 'unknown'}] Error verifying sell order status from Alpaca: {e}")
                        
                        executed_orders.append(formatted_order)
                    except Exception as e:
                        logger.error(f"Error placing sell order for {symbol}: {e}")
            
            except Exception as e:
                logger.error(f"Error executing {action} order for {symbol}: {e}")
        
        # Log summary of processed signals
        logger.info(f"Signal processing summary: {buy_signals} buy, {sell_signals} sell, {hold_signals} hold")
        logger.info(f"Executed orders: {len(executed_orders)}")
                
        return executed_orders
        
    def store_signal(self, symbol, signal_data, strategy_name="default"):
        """
        Store a trading signal in the database.
        
        Args:
            symbol: Stock symbol
            signal_data: Dictionary with signal data
            strategy_name: Name of the strategy that generated the signal
            
        Returns:
            int: ID of the inserted signal or None if failed
        """
        try:
            # Prepare data for API request
            url = f"{Trader.YFINANCE_DB_URL}/signals"
            
            # Extract signal data
            action = signal_data.get('action', 'hold')
            price = signal_data.get('price', 0)
            signal_value = signal_data.get('signal', 0)
            signal_changed = signal_data.get('signal_changed', False)
            short_ma = signal_data.get('short_ma')
            long_ma = signal_data.get('long_ma')
            rsi = signal_data.get('rsi')
            
            # Build request params
            params = {
                'symbol': symbol,
                'action': action,
                'price': price,
                'signal_value': signal_value,
                'signal_changed': signal_changed,
                'strategy': strategy_name
            }
            
            # Add optional parameters if they exist
            if short_ma is not None:
                params['short_ma'] = short_ma
            if long_ma is not None:
                params['long_ma'] = long_ma
            if rsi is not None:
                params['rsi'] = rsi
                
            # Send request
            response = requests.post(url, params=params, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                signal_id = result.get('id')
                logger.info(f"Stored trading signal for {symbol}: {action} (ID: {signal_id})")
                return signal_id
            else:
                logger.error(f"Error storing trading signal for {symbol}: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error storing trading signal for {symbol}: {e}")
            return None
    def _is_order_executable(self, order):
        """
        Check if an order is in a state where it can be considered executable.
        
        Args:
            order: Order object from Alpaca API
            
        Returns:
            bool: True if the order is filled or in a state that will lead to execution
        """
        # Consider these statuses as executable
        executable_statuses = ['filled', 'accepted', 'partially_filled']
        
        # Get the status from either an object or dictionary
        if hasattr(order, 'status'):
            status = order.status
        else:
            status = order.get('status', '')
            
        logger.info(f"Checking if order is executable. Status: {status}")
        return status.lower() in executable_statuses
                
    def store_trade(self, order):
        """
        Store an executed trade in the database.
        
        Args:
            order: Dictionary with order information
            
        Returns:
            int: ID of the inserted trade or None if failed
        """
        # Check if the order is executable
        if not self._is_order_executable(order):
            logger.warning(f"Order {order.get('id')} is not executable, skipping storage")
            return None
            
        # Log the order status to help diagnose issues
        logger.info(f"Storing trade with order status: {order.get('status', 'unknown')}")
        try:
            # Prepare data for API request
            # Ensure we're using the class-level YFINANCE_DB_URL
            url = f"{Trader.YFINANCE_DB_URL}/trades"
            logger.info(f"Using YFinance DB URL: {Trader.YFINANCE_DB_URL}")
            logger.info(f"Attempting to store trade at URL: {url}")
            logger.info(f"Order details for storage: ID={order.get('id')}, Symbol={order.get('symbol')}, Side={order.get('side')}, Status={order.get('status')}")
            
            # Extract order data
            order_id = order.get('id')
            symbol = order.get('symbol')
            side = order.get('side')
            quantity = float(order.get('qty', 0))
            price = float(order.get('price', 0))
            status = order.get('status')
            
            # Build request params
            params = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'status': status
            }
            
            logger.info(f"Trade parameters: {params}")
            
            # Add more detailed logging
            logger.info(f"YFINANCE_DB_HOST: {os.getenv('YFINANCE_DB_HOST', 'not set')}")
            logger.info(f"YFINANCE_DB_PORT: {os.getenv('YFINANCE_DB_PORT', 'not set')}")
            logger.info(f"Class YFINANCE_DB_URL: {Trader.YFINANCE_DB_URL}")
            logger.info(f"Running in Docker: {os.path.exists('/.dockerenv')}")
            
            # Send request
            logger.info(f"Sending POST request to {url} with params: {params}")
            
            try:
                logger.info(f"Attempting to connect to YFinance DB at {url}")
                response = requests.post(url, params=params, timeout=10)
                logger.info(f"Response from YFinance DB: Status={response.status_code}, Content={response.text[:100]}")
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error when storing trade: {e}")
                # Try with an alternative URL if the main one fails
                # First try with the Docker service name
                alt_url = f"http://yfinance-db:8001/trades"
                logger.info(f"Trying alternative URL: {alt_url}")
                try:
                    response = requests.post(alt_url, params=params, timeout=10)
                    logger.info(f"Response from alternative URL: Status={response.status_code}, Content={response.text[:100]}")
                except requests.exceptions.ConnectionError as e:
                    logger.error(f"Connection error with alternative URL: {e}")
                    # Try with localhost as a last resort
                    last_resort_url = f"http://localhost:8001/trades"
                    logger.info(f"Trying last resort URL: {last_resort_url}")
                    try:
                        response = requests.post(last_resort_url, params=params, timeout=10)
                        logger.info(f"Response from last resort URL: Status={response.status_code}, Content={response.text[:100]}")
                    except requests.exceptions.ConnectionError as e:
                        logger.error(f"Connection error with last resort URL: {e}")
                        logger.error(f"All connection attempts to YFinance DB failed. Trade will not be stored.")
                        return None
            
            if response.status_code == 200:
                result = response.json()
                trade_id = result.get('id')
                logger.info(f"Stored executed trade for {symbol}: {side} {quantity} shares at ${price:.2f} (ID: {trade_id})")
                return trade_id
            else:
                logger.error(f"Error storing executed trade for {symbol}: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error storing executed trade: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"YFINANCE_DB_URL: {Trader.YFINANCE_DB_URL}")
            logger.error(f"YFINANCE_DB_HOST: {self.YFINANCE_DB_HOST}")
            logger.error(f"YFINANCE_DB_PORT: {self.YFINANCE_DB_PORT}")
            logger.error(f"Environment variables:")
            logger.error(f"  YFINANCE_DB_HOST: {os.getenv('YFINANCE_DB_HOST', 'not set')}")
            logger.error(f"  YFINANCE_DB_PORT: {os.getenv('YFINANCE_DB_PORT', 'not set')}")
            logger.error(f"Running in Docker: {os.path.exists('/.dockerenv')}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
            
    def _format_order(self, order):
        """
        Format order information.
        
        Args:
            order: Alpaca order object
            
        Returns:
            dict: Formatted order information
        """
        return {
            'id': order.id,
            'symbol': order.symbol,
            'qty': order.qty,
            'side': order.side,
            'type': order.type,
            'status': order.status,
            'created_at': order.created_at
        }
        
    def cancel_all_orders(self):
        """
        Cancel all open orders.
        
        Returns:
            int: Number of orders canceled
        """
        try:
            canceled = self.api.cancel_all_orders()
            logger.info(f"Canceled {len(canceled)} orders")
            return len(canceled)
        except Exception as e:
            logger.error(f"Error canceling orders: {e}")
            return 0
            
    def liquidate_all_positions(self):
        """
        Liquidate all positions.
        
        Returns:
            int: Number of positions liquidated
        """
        try:
            positions = self.api.list_positions()
            for position in positions:
                self.api.submit_order(
                    symbol=position.symbol,
                    qty=position.qty,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                logger.info(f"Liquidated position in {position.symbol}")
            
            return len(positions)
        except Exception as e:
            logger.error(f"Error liquidating positions: {e}")
            return 0
            
    def get_account_info(self):
        """
        Get account information from Alpaca.
        
        Returns:
            dict: Account information
        """
        try:
            account = self.api.get_account()
            return {
                'id': account.id,
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'status': account.status
            }
        except Exception as e:
            logger.error(f"Error getting account information: {e}")
            return None

# Call the initialization method when the class is loaded
Trader._initialize_class()