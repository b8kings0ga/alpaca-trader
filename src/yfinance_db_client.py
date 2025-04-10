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
        # Use localhost instead of yfinance-db to ensure we can connect to the service
        self.host = host or os.getenv("YFINANCE_DB_HOST", "localhost")
        self.port = port or int(os.getenv("YFINANCE_DB_PORT", "8001"))
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
            url = f"{self.base_url}/fetch"
            data = {
                "symbols": symbols,
                "period": period,
                "interval": interval
            }
            
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
            return {}
            
    def is_service_available(self) -> bool:
        """
        Check if the YFinance Data Service is available.
        
        Returns:
            bool: True if the service is available, False otherwise
        """
        try:
            # Try to connect to localhost first, then fall back to the configured host
            try:
                response = requests.get("http://localhost:8001/", timeout=5)
                if response.status_code == 200:
                    logger.info("Successfully connected to YFinance Data Service on localhost")
                    # Update the base_url to use localhost
                    self.host = "localhost"
                    self.base_url = f"http://{self.host}:{self.port}"
                    return True
            except Exception as e:
                logger.warning(f"Could not connect to YFinance Data Service on localhost: {e}")
                
            # Try the configured host
            url = f"{self.base_url}/"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                logger.info("YFinance Data Service is available")
                return True
            else:
                logger.warning(f"YFinance Data Service returned status code {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to YFinance Data Service: {e}")
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
        portfolio_value = 100000.0  # Base value
        position_value = 0.0
        
        logger.info(f"Starting with base portfolio value: ${portfolio_value:.2f}")
        
        for position in positions:
            logger.info(f"Position {position.symbol}: market_value=${position.market_value:.2f}")
            position_value += position.market_value
        
        logger.info(f"Total position value: ${position_value:.2f}")
        total_value = portfolio_value + position_value
        logger.info(f"Total portfolio value: ${total_value:.2f}")
        
        # Return account data with consistent dictionary structure
        account_dict = {
            'equity': total_value,
            'cash': portfolio_value,
            'buying_power': portfolio_value,
            'portfolio_value': total_value
        }
        
        logger.debug(f"Simulated account data: {account_dict}")
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
                    "limit": required_data_points * 5  # Multiply by 5 to ensure we get enough data
                }
                
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
        
        for symbol in symbols:
            try:
                # Get data from the YFinance DB service
                url = f"{self.base_url}/data/{symbol}"
                params = {
                    "limit": limit
                }
                
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
                logger.error(f"Error getting bars for {symbol}: {e}")
                
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
        
        # Get current prices for symbols
        for symbol in config.SYMBOLS:
            try:
                # Try to get recent data for this symbol
                url = f"{self.base_url}/data/{symbol}"
                params = {"limit": 1}
                
                logger.info(f"Fetching data for simulated position: {url} with params {params}")
                response = requests.get(url, params=params)
                
                logger.info(f"Response status code: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Data received for {symbol}: {len(data)} records")
                    if data and len(data) > 0:
                        # Create a simulated position
                        current_price = data[0]['close']
                        logger.info(f"Current price for {symbol}: ${current_price:.2f}")
                        
                        qty = 10  # Simulated quantity
                        avg_entry_price = current_price * 0.95  # Simulated entry price (5% below current)
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
                        logger.warning(f"No data available for {symbol} to create simulated position")
            except Exception as e:
                logger.error(f"Error creating simulated position for {symbol}: {e}")
        
        logger.info(f"Returning {len(positions)} simulated positions")
        return positions
        
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