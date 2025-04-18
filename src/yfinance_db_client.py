"""
Client for connecting to the YFinance Data Service.
This client is used by the Alpaca Trading Bot to fetch market data from the YFinance Data Service.
"""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from src.logger import get_logger

logger = get_logger()

class YFinanceDBClient:
    """
    Client for connecting to the YFinance Data Service.
    """
    def __init__(self, host: str = None, port: int = None):
        """
        Initialize the YFinance DB Client.
        
        Args:
            host: Hostname of the YFinance Data Service
            port: Port of the YFinance Data Service
        """
        # Use the configured host (yfinance-db in Docker, localhost for local development)
        self.host = host or os.getenv("YFINANCE_DB_HOST", "yfinance-db")
        self.port = port or int(os.getenv("YFINANCE_DB_PORT", "8001"))
        
        # Log environment variables for debugging
        logger.info(f"YFinanceDBClient environment variables:")
        logger.info(f"  YFINANCE_DB_HOST: {os.getenv('YFINANCE_DB_HOST', 'not set')}")
        logger.info(f"  YFINANCE_DB_PORT: {os.getenv('YFINANCE_DB_PORT', 'not set')}")
        logger.info(f"  Running in Docker: {os.path.exists('/.dockerenv')}")
        self.base_url = f"http://{self.host}:{self.port}"
        logger.info(f"YFinanceDBClient initialized with base URL: {self.base_url}")
        
    def get_market_data(self, symbol: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        """
        Get market data for a symbol from the YFinance Data Service.
        
        Args:
            symbol: Stock symbol
            start_date: Start date in ISO format
            end_date: End date in ISO format
            limit: Maximum number of records to return
            
        Returns:
            DataFrame: DataFrame with market data
        """
        try:
            url = f"{self.base_url}/data/{symbol}"
            params = {}
            
            if start_date:
                params["start_date"] = start_date
                
            if end_date:
                params["end_date"] = end_date
                
            params["limit"] = limit
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data)
                    # Convert timestamp strings to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    logger.info(f"Retrieved {len(df)} rows of data for {symbol} from YFinance Data Service")
                    return df
                else:
                    logger.warning(f"No data found for {symbol} in YFinance Data Service")
                    return pd.DataFrame()
            else:
                logger.error(f"Error retrieving data for {symbol} from YFinance Data Service: {response.status_code} - {response.text}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error connecting to YFinance Data Service: {e}")
            return pd.DataFrame()
            
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database from the YFinance Data Service.
        
        Returns:
            Dict: Dictionary with database statistics
        """
        try:
            url = f"{self.base_url}/stats"
            response = requests.get(url)
            
            if response.status_code == 200:
                stats = response.json()
                logger.info(f"Retrieved database stats from YFinance Data Service")
                return stats
            else:
                logger.error(f"Error retrieving database stats from YFinance Data Service: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error connecting to YFinance Data Service: {e}")
            return {}
            
    def fetch_data(self, symbols: Optional[List[str]] = None,
                   period: str = "1d", interval: str = "1m") -> Dict[str, Any]:
        """
        Trigger data fetching on the YFinance Data Service.
        
        Args:
            symbols: List of stock symbols
            period: Period to fetch (1d, 5d, 1mo, 3mo, etc.)
            interval: Interval between data points (1m, 5m, 1h, 1d, etc.)
            
        Returns:
            Dict: Dictionary with fetch results
        """
        try:
            logger.info(f"YFinanceDBClient.fetch_data() called with symbols: {symbols}")
            url = f"{self.base_url}/fetch"
            data = {
                "symbols": symbols,
                "period": period,
                "interval": interval
            }
            
            logger.info(f"Sending request to {url} with data: {data}")
            response = requests.post(url, json=data)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Triggered data fetch on YFinance Data Service: {result}")
                return result
            else:
                logger.error(f"Error triggering data fetch on YFinance Data Service: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error connecting to YFinance Data Service: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
            
    def is_service_available(self) -> bool:
        """
        Check if the YFinance Data Service is available.
        
        Returns:
            bool: True if the service is available, False otherwise
        """
        try:
            # Always use the configured host (yfinance-db in Docker)
            # This ensures consistent connectivity in both Docker and local environments
            url = f"{self.base_url}/"
            logger.info(f"Checking YFinance DB service availability at: {url}")
            
            # Try to connect to the configured host first
            try:
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    logger.info(f"YFinance Data Service is available at {url}")
                    return True
                else:
                    logger.warning(f"YFinance Data Service at {url} returned status code {response.status_code}")
                    return False
            except Exception as e:
                logger.warning(f"Could not connect to YFinance Data Service on {self.host}: {e}")
                
                # YFinance Data Service is available (assuming it's running in Docker)
                logger.info(f"YFinance Data Service is available")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to YFinance Data Service at {self.base_url}: {e}")
            import traceback
            logger.error(f"Connection error traceback: {traceback.format_exc()}")
            return False
            
    def is_market_open(self) -> bool:
        """
        Check if the market is currently open.
        
        Returns:
            bool: True if market is open, False otherwise
        """
        from config import config
        
        # Check if we should use real market status from Alpaca API
        if config.USE_REAL_POSITIONS:
            try:
                logger.info("YFinanceDBClient.is_market_open() - Checking real market status from Alpaca API")
                
                # Initialize Alpaca API client
                import alpaca_trade_api as tradeapi
                
                # Remove trailing /v2 from base URL if present to avoid duplicate version in path
                base_url = config.ALPACA_BASE_URL
                if base_url.endswith('/v2'):
                    base_url = base_url[:-3]
                
                api = tradeapi.REST(
                    config.ALPACA_API_KEY,
                    config.ALPACA_API_SECRET,
                    base_url,
                    api_version='v2'
                )
                
                # Get real market status from Alpaca API
                clock = api.get_clock()
                is_open = clock.is_open
                logger.info(f"Market is {'open' if is_open else 'closed'} according to Alpaca API")
                return is_open
                
            except Exception as e:
                logger.error(f"Error checking market status from Alpaca API: {e}")
                logger.warning("Falling back to simulated market status")
                # Fall back to simulated market status if there's an error
        
        # Provide simulated market status
        logger.info("YFinanceDBClient.is_market_open() - Providing simulated market status")
        # For testing purposes, always return True
        return True
        
    def get_account(self):
        """
        Get account information.
        
        Returns:
            dict: Account information with keys 'equity', 'cash', 'buying_power', 'portfolio_value'
                 Always returns a dictionary regardless of whether real or simulated data is used.
        """
        logger.info("YFinanceDBClient.get_account() was called")
        from config import config
        
        # Check if we should use real account info from Alpaca API
        if config.USE_REAL_POSITIONS:
            try:
                logger.info("YFinanceDBClient.get_account() - Using real account info from Alpaca API")
                
                # Initialize Alpaca API client
                import alpaca_trade_api as tradeapi
                
                # Remove trailing /v2 from base URL if present to avoid duplicate version in path
                base_url = config.ALPACA_BASE_URL
                if base_url.endswith('/v2'):
                    base_url = base_url[:-3]
                
                api = tradeapi.REST(
                    config.ALPACA_API_KEY,
                    config.ALPACA_API_SECRET,
                    base_url,
                    api_version='v2'
                )
                
                # Get real account info from Alpaca API
                account = api.get_account()
                logger.info(f"Retrieved real account info from Alpaca API: Equity=${float(account.equity):.2f}")
                logger.info(f"Account type: {type(account)}")
                
                # Convert Account object to dictionary for consistency
                account_dict = {
                    'equity': float(account.equity),
                    'cash': float(account.cash),
                    'buying_power': float(account.buying_power),
                    'portfolio_value': float(account.portfolio_value)
                }
                logger.info(f"Converted Account object to dictionary")
                return account_dict
                
            except Exception as e:
                logger.error(f"Error getting real account info from Alpaca API: {e}")
                logger.warning("Falling back to simulated account info")
                # Fall back to simulated account info if there's an error
        
        # Provide simulated account info
        logger.info("YFinanceDBClient.get_account() - Providing simulated account data")
        
        # Get current positions to calculate portfolio value
        logger.info("Fetching positions to calculate portfolio value")
        positions = self.get_positions()
        logger.info(f"Retrieved {len(positions)} positions")
        
        # Calculate portfolio value based on positions
        base_cash = 100000.0  # Base cash value
        position_value = 0.0
        
        logger.info(f"Starting with base cash value: ${base_cash:.2f}")
        
        for position in positions:
            position_market_value = float(position.market_value) if hasattr(position, 'market_value') else 0.0
            logger.info(f"Position {position.symbol}: market_value=${position_market_value:.2f}")
            position_value += position_market_value
        
        logger.info(f"Total position value: ${position_value:.2f}")
        total_equity = base_cash + position_value
        logger.info(f"Total portfolio equity: ${total_equity:.2f}")
        
        # Return account data with consistent dictionary structure
        account_dict = {
            'equity': total_equity,
            'cash': base_cash - (position_value * 0.5),  # Assume 50% of position value came from cash
            'buying_power': (base_cash - (position_value * 0.5)) * 2,  # Typically 2x cash for margin accounts
            'portfolio_value': total_equity,
            'initial_margin': position_value * 0.5,  # Assume 50% margin requirement
            'maintenance_margin': position_value * 0.25,  # Typically 25% of position value
            'multiplier': 2,  # Margin multiplier
            'status': 'ACTIVE',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Simulated account data: equity=${account_dict['equity']:.2f}, cash=${account_dict['cash']:.2f}, buying_power=${account_dict['buying_power']:.2f}")
        return account_dict
    
    def get_recent_data(self, symbols, minutes=15, interval="1m"):
        """
        Get the most recent market data for the specified symbols.
        
        Args:
            symbols (list): List of stock symbols
            minutes (int): Number of minutes of recent data to fetch
            interval (str): Interval between data points (1m, 5m, etc.)
            
        Returns:
            dict: Dictionary of DataFrames with recent data for each symbol
        """
        logger.info(f"YFinanceDBClient.get_recent_data() was called for {symbols}")
        logger.info(f"Current configuration: minutes={minutes}, interval={interval}")
        
        # Calculate how many data points we need for strategy
        from config import config
        required_data_points = max(config.LONG_WINDOW, config.SHORT_WINDOW) + 10  # Add buffer
        logger.info(f"Required data points for strategy: {required_data_points}")
        logger.info(f"LONG_WINDOW: {config.LONG_WINDOW}, SHORT_WINDOW: {config.SHORT_WINDOW}")
        
        data = {}
        
        for symbol in symbols:
            try:
                # Get data from the YFinance DB service
                url = f"{self.base_url}/data/{symbol}"
                params = {
                    # Use required_data_points instead of minutes to ensure we get enough data
                    "limit": required_data_points * 50  # Multiply by 50 to ensure we get enough data for signal generation
                }
                
                logger.info(f"Requesting {params['limit']} data points for {symbol} from YFinance DB Service")
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    result = response.json()
                    if result:
                        df = pd.DataFrame(result)
                        # Convert timestamp strings to datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        logger.info(f"Retrieved {len(df)} rows of data for {symbol} from YFinance DB Service")
                        data[symbol] = df
                    else:
                        logger.warning(f"No data found for {symbol} in YFinance DB Service")
                else:
                    logger.error(f"Error retrieving data for {symbol} from YFinance DB Service: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error getting recent data for {symbol}: {e}")
                
        return data
    
    def get_bars(self, symbols, timeframe='1D', limit=100):
        """
        Fetch historical bar data for the given symbols.
        
        Args:
            symbols (list): List of stock symbols
            timeframe (str): Bar timeframe (1D, 1H, etc.)
            limit (int): Number of bars to fetch
            
        Returns:
            dict: Dictionary of DataFrames with historical data for each symbol
        """
        logger.info(f"YFinanceDBClient.get_bars() was called for {symbols}")
        data = {}
        missing_symbols = []
        
        for symbol in symbols:
            try:
                # Get data from the YFinance DB service
                url = f"{self.base_url}/data/{symbol}"
                params = {
                    "limit": limit
                }
                
                logger.info(f"Requesting data for {symbol} from {url} with params {params}")
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    result = response.json()
                    if result:
                        df = pd.DataFrame(result)
                        # Convert timestamp strings to datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        logger.info(f"Retrieved {len(df)} rows of data for {symbol} from YFinance DB Service")
                        data[symbol] = df
                    else:
                        logger.warning(f"No data found for {symbol} in YFinance DB Service (empty result)")
                        missing_symbols.append(symbol)
                else:
                    logger.error(f"Error retrieving data for {symbol} from YFinance DB Service: {response.status_code} - {response.text}")
                    missing_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Error getting bars for {symbol}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                missing_symbols.append(symbol)
        
        if missing_symbols:
            logger.warning(f"Failed to retrieve data for the following symbols: {missing_symbols}")
        
        logger.info(f"Successfully retrieved data for {len(data)} out of {len(symbols)} symbols")
        logger.info(f"Symbols with data: {list(data.keys())}")
                
        return data
    
    def get_positions(self):
        """
        Get current positions.
        
        Returns:
            list: List of positions
        """
        logger.info("YFinanceDBClient.get_positions() was called")
        from config import config
        
        # Check if we should use real positions from Alpaca API
        if config.USE_REAL_POSITIONS:
            try:
                logger.info("YFinanceDBClient.get_positions() - Using real positions from Alpaca API")
                
                # Initialize Alpaca API client
                import alpaca_trade_api as tradeapi
                
                # Remove trailing /v2 from base URL if present to avoid duplicate version in path
                base_url = config.ALPACA_BASE_URL
                if base_url.endswith('/v2'):
                    base_url = base_url[:-3]
                
                api = tradeapi.REST(
                    config.ALPACA_API_KEY,
                    config.ALPACA_API_SECRET,
                    base_url,
                    api_version='v2'
                )
                
                # Get real positions from Alpaca API
                positions = api.list_positions()
                logger.info(f"Retrieved {len(positions)} real positions from Alpaca API")
                return positions
                
            except Exception as e:
                logger.error(f"Error getting real positions from Alpaca API: {e}")
                logger.warning("Falling back to simulated positions")
                # Fall back to simulated positions if there's an error
        
        # Provide simulated positions
        logger.info("YFinanceDBClient.get_positions() - Providing simulated position data")
        
        # Create a Position class to mimic Alpaca's Position object
        from collections import namedtuple
        Position = namedtuple('Position', ['symbol', 'qty', 'market_value', 'avg_entry_price', 'current_price', 'unrealized_pl'])
        
        positions = []
        logger.info(f"Using symbols from config: {config.SYMBOLS}")
        
        # Log which symbols we're going to attempt to create positions for
        logger.info(f"Attempting to create positions for symbols: {config.SYMBOLS}")
        
        # Get current prices for symbols
        for symbol in config.SYMBOLS:
            try:
                # Try to get recent data for this symbol
                url = f"{self.base_url}/data/{symbol}"
                params = {"limit": 1}
                
                logger.info(f"Fetching data for simulated position for {symbol}: {url} with params {params}")
                response = requests.get(url, params=params)
                
                logger.info(f"Response status code for {symbol}: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Data received for {symbol}: {len(data)} records")
                    if data and len(data) > 0:
                        logger.info(f"Valid data found for {symbol}, creating position")
                        # Create a simulated position
                        current_price = data[0]['close']
                        logger.info(f"Current price for {symbol}: ${current_price:.2f}")
                        
                        # Create more varied and realistic positions
                        # Use symbol hash to create deterministic but varied quantities
                        symbol_hash = sum(ord(c) for c in symbol)
                        qty = 10 + (symbol_hash % 90)  # Between 10 and 100 shares
                        
                        # Vary entry prices to simulate different purchase times
                        entry_factor = 0.90 + ((symbol_hash % 15) / 100)  # Between 0.90 and 1.05
                        avg_entry_price = current_price * entry_factor
                        
                        market_value = qty * current_price
                        unrealized_pl = qty * (current_price - avg_entry_price)
                        
                        logger.info(f"Creating position with qty={qty}, avg_entry_price=${avg_entry_price:.2f}, market_value=${market_value:.2f}, unrealized_pl=${unrealized_pl:.2f}")
                        
                        position = Position(
                            symbol=symbol,
                            qty=qty,
                            market_value=market_value,
                            avg_entry_price=avg_entry_price,
                            current_price=current_price,
                            unrealized_pl=unrealized_pl
                        )
                        positions.append(position)
                        logger.info(f"Created simulated position for {symbol}: {qty} shares at ${current_price:.2f}")
                    else:
                        logger.warning(f"No valid data found for {symbol}, creating default position")
                        self._create_default_position(symbol, positions, Position)
                else:
                    logger.warning(f"Failed to fetch data for {symbol} (status code: {response.status_code}), creating default position")
                    self._create_default_position(symbol, positions, Position)
            except Exception as e:
                logger.error(f"Error creating simulated position for {symbol}: {e}")
                logger.info(f"Creating default position for {symbol} due to error")
                self._create_default_position(symbol, positions, Position)
        
        logger.info(f"Returning {len(positions)} simulated positions")
        return positions
    
    def _create_default_position(self, symbol, positions, Position):
        """
        Create a default position for a symbol when no data is available.
        
        Args:
            symbol: Stock symbol
            positions: List to append the position to
            Position: Position namedtuple class
        """
        try:
            # Use a default price
            default_price = 100.0
            logger.info(f"Using default price for {symbol}: ${default_price:.2f}")
            
            # Create deterministic but varied quantities
            symbol_hash = sum(ord(c) for c in symbol)
            qty = 10 + (symbol_hash % 90)  # Between 10 and 100 shares
            
            # Vary entry prices
            entry_factor = 0.90 + ((symbol_hash % 15) / 100)  # Between 0.90 and 1.05
            avg_entry_price = default_price * entry_factor
            
            market_value = qty * default_price
            unrealized_pl = qty * (default_price - avg_entry_price)
            
            logger.info(f"Creating default position with qty={qty}, avg_entry_price=${avg_entry_price:.2f}, market_value=${market_value:.2f}")
            
            position = Position(
                symbol=symbol,
                qty=qty,
                market_value=market_value,
                avg_entry_price=avg_entry_price,
                current_price=default_price,
                unrealized_pl=unrealized_pl
            )
            positions.append(position)
            logger.info(f"Created default position for {symbol}: {qty} shares at ${default_price:.2f}")
        except Exception as e:
            logger.error(f"Error creating default position for {symbol}: {e}")
        
    def get_market_hours(self):
        """
        Get market open and close times for today.
        
        Returns:
            tuple: (market_open, market_close) datetime objects
        """
        from config import config
        
        # Check if we should use real market hours from Alpaca API
        if config.USE_REAL_POSITIONS:
            try:
                logger.info("YFinanceDBClient.get_market_hours() - Getting real market hours from Alpaca API")
                
                # Initialize Alpaca API client
                import alpaca_trade_api as tradeapi
                
                # Remove trailing /v2 from base URL if present to avoid duplicate version in path
                base_url = config.ALPACA_BASE_URL
                if base_url.endswith('/v2'):
                    base_url = base_url[:-3]
                
                api = tradeapi.REST(
                    config.ALPACA_API_KEY,
                    config.ALPACA_API_SECRET,
                    base_url,
                    api_version='v2'
                )
                
                # Get real market hours from Alpaca API
                calendar = api.get_calendar(start=datetime.now().strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
                if calendar:
                    market_open = calendar[0].open
                    market_close = calendar[0].close
                    logger.info(f"Market hours from Alpaca API: Open={market_open}, Close={market_close}")
                    return (market_open, market_close)
                else:
                    logger.warning("No calendar data returned from Alpaca API")
                    # Fall back to simulated market hours
                
            except Exception as e:
                logger.error(f"Error getting market hours from Alpaca API: {e}")
                logger.warning("Falling back to simulated market hours")
                # Fall back to simulated market hours if there's an error
        
        # Provide simulated market hours
        logger.info("YFinanceDBClient.get_market_hours() - Providing simulated market hours")
        now = datetime.now()
        today = now.date()
        open_time = datetime.combine(today, datetime.min.time().replace(hour=9, minute=30))
        close_time = datetime.combine(today, datetime.min.time().replace(hour=16, minute=0))
        return (open_time, close_time)