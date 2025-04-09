"""
Market data handling for the Alpaca Trading Bot.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from config import config
from src.logger import get_logger

logger = get_logger()

class MarketData:
    """
    Class for fetching and processing market data from Alpaca.
    """
    def __init__(self):
        """
        Initialize the MarketData class with Alpaca API.
        """
        self.api = tradeapi.REST(
            config.ALPACA_API_KEY,
            config.ALPACA_API_SECRET,
            config.ALPACA_BASE_URL,
            api_version='v2'
        )
        logger.info("MarketData initialized")

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
        logger.info(f"Fetching {timeframe} bars for {symbols}")
        
        # Calculate start and end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=limit)
        
        try:
            # Fetch bars for all symbols
            bars = {}
            for symbol in symbols:
                df = self.api.get_bars(
                    symbol,
                    timeframe,
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    adjustment='raw'
                ).df
                
                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                    
                # Reset index to make timestamp a column
                df = df.reset_index()
                
                # Add technical indicators
                df = self.add_indicators(df)
                
                bars[symbol] = df
                
            return bars
        except Exception as e:
            logger.error(f"Error fetching bars: {e}")
            return {}

    def add_indicators(self, df):
        """
        Add technical indicators to the DataFrame.
        
        Args:
            df (DataFrame): DataFrame with OHLCV data
            
        Returns:
            DataFrame: DataFrame with added indicators
        """
        # Calculate moving averages
        df['sma_short'] = df['close'].rolling(window=config.SHORT_WINDOW).mean()
        df['sma_long'] = df['close'].rolling(window=config.LONG_WINDOW).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=config.RSI_PERIOD).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=config.RSI_PERIOD).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df

    def get_account(self):
        """
        Get account information.
        
        Returns:
            dict: Account information
        """
        try:
            account = self.api.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value)
            }
        except Exception as e:
            logger.error(f"Error fetching account: {e}")
            return {}

    def get_positions(self):
        """
        Get current positions.
        
        Returns:
            list: List of positions
        """
        try:
            positions = self.api.list_positions()
            return [{
                'symbol': position.symbol,
                'qty': int(position.qty),
                'market_value': float(position.market_value),
                'avg_entry_price': float(position.avg_entry_price),
                'unrealized_pl': float(position.unrealized_pl),
                'current_price': float(position.current_price)
            } for position in positions]
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    def is_market_open(self):
        """
        Check if the market is currently open.
        
        Returns:
            bool: True if market is open, False otherwise
        """
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False

    def get_market_hours(self):
        """
        Get market open and close times for today.
        
        Returns:
            tuple: (market_open, market_close) datetime objects
        """
        try:
            calendar = self.api.get_calendar(start=datetime.now().date().isoformat())
            if calendar:
                return (calendar[0].open, calendar[0].close)
            return (None, None)
        except Exception as e:
            logger.error(f"Error fetching market hours: {e}")
            return (None, None)