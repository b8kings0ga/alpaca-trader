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
        logger.info("YFinanceDBClient.is_market_open() was called but is not implemented")
        # For now, return a default value to avoid errors
        return True
        
    def get_account(self):
        """
        Get account information.
        
        Returns:
            dict: Account information
        """
        logger.info("YFinanceDBClient.get_account() - Providing simulated account data")
        
        # Get current positions to calculate portfolio value
        positions = self.get_positions()
        
        # Calculate portfolio value based on positions
        portfolio_value = 100000.0  # Base value
        position_value = 0.0
        
        for position in positions:
            position_value += position.market_value
        
        total_value = portfolio_value + position_value
        
        # Return account data
        return {
            'equity': total_value,
            'cash': portfolio_value,
            'buying_power': portfolio_value,
            'portfolio_value': total_value
        }
    
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
        
        data = {}
        
        for symbol in symbols:
            try:
                # Get data from the YFinance DB service
                url = f"{self.base_url}/data/{symbol}"
                params = {
                    # Use required_data_points instead of minutes to ensure we get enough data
                    "limit": required_data_points
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
        logger.info("YFinanceDBClient.get_positions() - Providing simulated position data")
        
        # Create a Position class to mimic Alpaca's Position object
        from collections import namedtuple
        Position = namedtuple('Position', ['symbol', 'qty', 'market_value', 'avg_entry_price', 'current_price', 'unrealized_pl'])
        
        positions = []
        
        # Get current prices for symbols
        for symbol in ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']:
            try:
                # Try to get recent data for this symbol
                url = f"{self.base_url}/data/{symbol}"
                params = {"limit": 1}
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        # Create a simulated position
                        current_price = data[0]['close']
                        qty = 10  # Simulated quantity
                        avg_entry_price = current_price * 0.95  # Simulated entry price (5% below current)
                        market_value = qty * current_price
                        unrealized_pl = qty * (current_price - avg_entry_price)
                        
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
        logger.info("YFinanceDBClient.get_market_hours() was called but is not implemented")
        # For now, return default values to avoid errors
        now = datetime.now()
        today = now.date()
        open_time = datetime.combine(today, datetime.min.time().replace(hour=9, minute=30))
        close_time = datetime.combine(today, datetime.min.time().replace(hour=16, minute=0))
        return (open_time, close_time)