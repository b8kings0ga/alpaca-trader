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
        # Log the API URLs for debugging
        logger.info(f"Initializing MarketData with base URL: {config.ALPACA_BASE_URL}")
        logger.info(f"Data URL: {config.ALPACA_DATA_URL}")
        
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
                logger.info(f"Attempting to fetch bars for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                df = None
                
                # First try with Alpaca API
                try:
                    df = self.api.get_bars(
                        symbol,
                        timeframe,
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        adjustment='raw'
                    ).df
                    logger.info(f"Successfully fetched {len(df)} bars for {symbol} from Alpaca API")
                except Exception as e:
                    logger.warning(f"Error fetching bars for {symbol} from Alpaca API: {e}")
                    logger.info(f"Attempting to fetch data for {symbol} using yfinance as fallback")
                    df = None
                
                # If Alpaca API failed or returned empty data, try with yfinance
                if df is None or df.empty:
                    try:
                        import yfinance as yf
                        # Convert timeframe to yfinance interval format
                        interval = '1d' if timeframe == '1D' else '1h' if timeframe == '1H' else '1m'
                        ticker = yf.Ticker(symbol)
                        yf_df = ticker.history(
                            start=start_date.strftime('%Y-%m-%d'),
                            end=end_date.strftime('%Y-%m-%d'),
                            interval=interval
                        )
                        
                        if not yf_df.empty:
                            # Rename columns to match Alpaca format
                            yf_df = yf_df.rename(columns={
                                'Open': 'open',
                                'High': 'high',
                                'Low': 'low',
                                'Close': 'close',
                                'Volume': 'volume'
                            })
                            # Reset index to make Date a column named 'timestamp'
                            yf_df = yf_df.reset_index()
                            yf_df = yf_df.rename(columns={'Date': 'timestamp'})
                            df = yf_df
                            logger.info(f"Successfully fetched {len(df)} bars for {symbol} from yfinance")
                        else:
                            logger.warning(f"No data found for {symbol} from yfinance")
                            continue
                    except Exception as e:
                        logger.error(f"Error fetching data for {symbol} from yfinance: {e}")
                        continue
                
                if df.empty:
                    logger.warning(f"No data found for {symbol} from any source")
                    continue
                
                # If df is from Alpaca, reset index to make timestamp a column
                if 'timestamp' not in df.columns and df.index.name == 'timestamp':
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
        
        # Check for zero values in loss to avoid division by zero
        # This can happen when there are no price decreases in the window
        zero_loss_mask = loss == 0
        if zero_loss_mask.any():
            logger.warning(f"Found {zero_loss_mask.sum()} zero values in loss calculation for RSI")
            
        # Replace zeros with a small value to avoid division by zero
        loss = loss.replace(0, 1e-10)
        
        rs = gain / loss
        
        # Check for infinity values
        inf_mask = np.isinf(rs)
        if inf_mask.any():
            logger.warning(f"Found {inf_mask.sum()} infinity values in RS calculation for RSI")
            # Replace infinity with a large value
            rs = rs.replace([np.inf, -np.inf], 100)
            
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Check for NaN values
        nan_mask = np.isnan(df['rsi'])
        if nan_mask.any():
            logger.warning(f"Found {nan_mask.sum()} NaN values in RSI calculation")
            # Replace NaN with 50 (neutral RSI value)
            df['rsi'] = df['rsi'].fillna(50)
            
        return df

    def get_account(self):
        """
        Get account information.
        
        Returns:
            dict: Account information with keys 'equity', 'cash', 'buying_power', 'portfolio_value'
        """
        try:
            account = self.api.get_account()
            logger.info(f"MarketData.get_account() - Retrieved account info from Alpaca API")
            logger.debug(f"Account type: {type(account)}")
            
            # Always return a dictionary with consistent keys
            account_dict = {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value)
            }
            return account_dict
        except Exception as e:
            logger.error(f"Error fetching account: {e}")
            return {
                'equity': 0.0,
                'cash': 0.0,
                'buying_power': 0.0,
                'portfolio_value': 0.0
            }

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
            today = datetime.now().date()
            calendar = self.api.get_calendar(start=today.isoformat())
            logger.info(f"Calendar data: {calendar}")
            
            if calendar:
                # Log the types and values
                open_time = calendar[0].open
                close_time = calendar[0].close
                logger.info(f"Open time type: {type(open_time)}, value: {open_time}")
                logger.info(f"Close time type: {type(close_time)}, value: {close_time}")
                
                # Convert time objects to datetime objects
                open_datetime = datetime.combine(today, open_time)
                close_datetime = datetime.combine(today, close_time)
                
                logger.info(f"Open datetime type: {type(open_datetime)}, value: {open_datetime}")
                logger.info(f"Close datetime type: {type(close_datetime)}, value: {close_datetime}")
                
                return (open_datetime, close_datetime)
            return (None, None)
        except Exception as e:
            import traceback
            logger.error(f"Error fetching market hours: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return (None, None)
            
    def get_recent_data(self, symbols, minutes=15, timeframe='1Min'):
        """
        Get the most recent market data for the specified symbols.
        
        Args:
            symbols (list): List of stock symbols
            minutes (int): Number of minutes of recent data to fetch
            timeframe (str): Bar timeframe (1Min, 5Min, etc.)
            
        Returns:
            dict: Dictionary of DataFrames with recent data for each symbol
        """
        logger.info(f"Fetching recent {minutes} minutes of {timeframe} data for {symbols}")
        
        # Calculate start and end times
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)
        
        # Initialize storage for the data
        recent_data = {}
        
        for symbol in symbols:
            logger.info(f"Attempting to fetch recent data for {symbol}")
            df = None
            
            # First try with Alpaca API
            try:
                df = self.api.get_bars(
                    symbol,
                    timeframe,
                    start=start_time.isoformat(),
                    end=end_time.isoformat(),
                    adjustment='raw'
                ).df
                logger.info(f"Successfully fetched {len(df)} recent bars for {symbol} from Alpaca API")
            except Exception as e:
                logger.warning(f"Error fetching recent data for {symbol} from Alpaca API: {e}")
                df = None
            
            # If Alpaca API failed or returned empty data, try with yfinance
            if df is None or df.empty:
                try:
                    import yfinance as yf
                    # Convert timeframe to yfinance interval format
                    interval = '1m'  # Default to 1 minute for recent data
                    if timeframe == '5Min':
                        interval = '5m'
                    elif timeframe == '15Min':
                        interval = '15m'
                    
                    ticker = yf.Ticker(symbol)
                    # For very recent data, we need to use a period instead of start/end dates
                    period = "1d"  # Get 1 day of data and filter later
                    yf_df = ticker.history(period=period, interval=interval)
                    
                    if not yf_df.empty:
                        # Filter to only include the last 'minutes' minutes
                        now = pd.Timestamp.now(tz=yf_df.index.tz)
                        cutoff = now - pd.Timedelta(minutes=minutes)
                        yf_df = yf_df[yf_df.index >= cutoff]
                        
                        # Rename columns to match Alpaca format
                        yf_df = yf_df.rename(columns={
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume'
                        })
                        # Reset index to make Date a column named 'timestamp'
                        yf_df = yf_df.reset_index()
                        yf_df = yf_df.rename(columns={'Date': 'timestamp', 'Datetime': 'timestamp'})
                        df = yf_df
                        logger.info(f"Successfully fetched {len(df)} recent bars for {symbol} from yfinance")
                    else:
                        logger.warning(f"No recent data found for {symbol} from yfinance")
                        continue
                except Exception as e:
                    logger.error(f"Error fetching recent data for {symbol} from yfinance: {e}")
                    continue
            
            if df is None or df.empty:
                logger.warning(f"No recent data found for {symbol} from any source")
                continue
            
            # If df is from Alpaca, reset index to make timestamp a column
            if 'timestamp' not in df.columns and df.index.name == 'timestamp':
                df = df.reset_index()
            
            # Add technical indicators
            df = self.add_indicators(df)
            
            # Store the data
            recent_data[symbol] = df
            
            # Save to local cache
            self._save_to_cache(symbol, df)
        
        return recent_data
    
    def _save_to_cache(self, symbol, df):
        """
        Save market data to a local cache file.
        
        Args:
            symbol (str): Stock symbol
            df (DataFrame): DataFrame with market data
        """
        try:
            import os
            cache_dir = os.path.join('data', 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_file = os.path.join(cache_dir, f"{symbol}_recent.csv")
            df.to_csv(cache_file, index=False)
            logger.info(f"Saved recent data for {symbol} to cache: {cache_file}")
        except Exception as e:
            logger.error(f"Error saving data to cache for {symbol}: {e}")