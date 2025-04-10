"""
Market data handling using yfinance for the Alpaca Trading Bot.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import sqlite3
from config import config
from src.logger import get_logger

logger = get_logger()

class YFinanceData:
    """
    Class for fetching and processing market data from yfinance.
    """
    def __init__(self, db_path='data/market_data.db'):
        """Initialize the YFinanceData class."""
        self.db_path = db_path
        self._ensure_db_exists()
        logger.info(f"YFinanceData initialized with database at {db_path}")
    
    def _ensure_db_exists(self):
        """Ensure the database directory and file exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            symbol TEXT, timestamp TEXT, open REAL, high REAL, low REAL,
            close REAL, volume INTEGER, sma_short REAL, sma_long REAL, rsi REAL,
            PRIMARY KEY (symbol, timestamp)
        )''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            symbol TEXT, timestamp TEXT, action TEXT, price REAL,
            signal_value INTEGER, signal_changed BOOLEAN,
            PRIMARY KEY (symbol, timestamp)
        )''')
        conn.commit()
        conn.close()
        logger.info("Database tables created if they didn't exist")
    
    def get_historical_data(self, symbols, period="3mo", interval="1d"):
        """Fetch historical data for the given symbols."""
        logger.info(f"Fetching historical data for {symbols}")
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                df = df.reset_index()
                df = df.rename(columns={
                    'Date': 'timestamp', 'Datetime': 'timestamp',
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Volume': 'volume'
                })
                df = self.add_indicators(df)
                self._store_market_data(symbol, df)
                data[symbol] = df
                logger.info(f"Successfully fetched {len(df)} bars for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        return data
    
    def get_recent_data(self, symbols, minutes=60, interval="1m", use_cached=True):
        """
        Get the most recent market data for the specified symbols.
        Fetches enough data to generate signals (default: 60 minutes).
        
        Note: Yahoo Finance only allows fetching 7 days of minute data at a time.
        
        Args:
            symbols (list): List of stock symbols
            minutes (int): Number of minutes of recent data to fetch
            interval (str): Interval between data points (1m, 5m, etc.)
            use_cached (bool): Whether to use cached data if available
            
        Returns:
            dict: Dictionary of DataFrames with recent data for each symbol
        """
        logger.info(f"Fetching recent {minutes} minutes of data for {symbols}")
        data = {}
        
        # First, try to load data from the database
        if use_cached:
            for symbol in symbols:
                try:
                    # Load from database
                    conn = sqlite3.connect(self.db_path)
                    query = f"""
                    SELECT * FROM market_data
                    WHERE symbol = '{symbol}'
                    ORDER BY timestamp DESC
                    LIMIT 100
                    """
                    df = pd.read_sql_query(query, conn)
                    conn.close()
                    
                    if not df.empty:
                        logger.info(f"Loaded {len(df)} rows of cached data for {symbol}")
                        # Convert timestamp strings back to datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        # Sort by timestamp
                        df = df.sort_values('timestamp')
                        data[symbol] = df
                except Exception as e:
                    logger.error(f"Error loading cached data for {symbol}: {e}")
        for symbol in symbols:
            try:
                # If we don't have cached data or need fresh data, fetch from yfinance
                if symbol not in data or data[symbol].empty:
                    # First get historical data to have enough for indicators
                    ticker = yf.Ticker(symbol)
                    
                    # Get daily data for better indicator calculation (3 months)
                    historical_df = ticker.history(period="3mo", interval="1d")
                    if not historical_df.empty:
                        logger.info(f"Fetched {len(historical_df)} days of historical data for {symbol}")
                        
                        # Process and store historical data
                        hist_df = historical_df.reset_index()
                        hist_df = hist_df.rename(columns={
                            'Date': 'timestamp', 'Datetime': 'timestamp',
                            'Open': 'open', 'High': 'high', 'Low': 'low',
                            'Close': 'close', 'Volume': 'volume'
                        })
                        hist_df = self.add_indicators(hist_df)
                        self._store_market_data(symbol, hist_df, if_exists='replace')
                # Initialize ticker if not already done
                if 'ticker' not in locals() or ticker is None:
                    ticker = yf.Ticker(symbol)
                
                # Initialize historical_df if not already done
                if 'historical_df' not in locals() or historical_df is None or historical_df.empty:
                    historical_df = ticker.history(period="3mo", interval="1d")
                    if not historical_df.empty:
                        logger.info(f"Fetched {len(historical_df)} days of historical data for {symbol}")
                
                # Then get recent minute data - limit to 7 days to avoid Yahoo Finance API limitations
                try:
                    # First try with a shorter period for minute data (7 days max for minute data)
                    df = ticker.history(period="7d", interval=interval)
                    logger.info(f"Attempting to fetch 7 days of minute data for {symbol}")
                    
                    if df.empty:
                        # If minute data fails, try hourly data
                        logger.warning(f"No minute data found for {symbol}, trying hourly data")
                        df = ticker.history(period="7d", interval="60m")
                        
                    if df.empty and not historical_df.empty:
                        # If hourly data fails, use daily data
                        logger.warning(f"No hourly data found for {symbol}, using daily data")
                        df = historical_df
                    elif df.empty:
                        logger.warning(f"No data found for {symbol} from any source")
                        continue
                except Exception as e:
                    logger.warning(f"Error fetching minute data for {symbol}: {e}")
                    if not historical_df.empty:
                        logger.warning(f"Falling back to daily data for {symbol}")
                        df = historical_df
                    else:
                        logger.warning(f"No historical data available for {symbol}")
                        continue
                    df = historical_df
                
                if df.empty:
                    logger.warning(f"No data found for {symbol} from any source")
                    continue
                
                # Filter to the requested minutes if we have minute data
                if interval == "1m" and not df.empty and len(df) > minutes:
                    now = pd.Timestamp.now(tz=df.index.tz)
                    cutoff = now - pd.Timedelta(minutes=minutes)
                    recent_df = df[df.index >= cutoff]
                    
                    # Only use the filtered data if it's not empty
                    if not recent_df.empty:
                        df = recent_df
                        logger.info(f"Filtered to the most recent {len(df)} bars for {symbol}")
                
                # Make sure we have enough data for indicators
                min_data_points = max(config.SHORT_WINDOW, config.LONG_WINDOW, config.RSI_PERIOD)
                if len(df) < min_data_points:
                    logger.warning(f"Not enough data for {symbol} to calculate indicators properly, need at least {min_data_points} bars")
                    # If we don't have enough data, use the historical daily data
                    if not historical_df.empty and len(historical_df) >= min_data_points:
                        logger.info(f"Using historical daily data for {symbol} to calculate indicators")
                        df = historical_df
                    else:
                        logger.warning(f"Not enough historical data for {symbol} either")
                df = df.reset_index()
                df = df.rename(columns={
                    'Date': 'timestamp', 'Datetime': 'timestamp',
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Volume': 'volume'
                })
                df = self.add_indicators(df)
                self._store_market_data(symbol, df)
                data[symbol] = df
                logger.info(f"Successfully fetched {len(df)} recent bars for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching recent data for {symbol}: {e}")
        return data
    
    def add_indicators(self, df):
        """Add technical indicators to the DataFrame."""
        df['sma_short'] = df['close'].rolling(window=config.SHORT_WINDOW).mean()
        df['sma_long'] = df['close'].rolling(window=config.LONG_WINDOW).mean()
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
    
    def _store_market_data(self, symbol, df, if_exists='append'):
        """Store market data in the SQLite database."""
        try:
            # Check if DataFrame is empty
            if df.empty:
                logger.warning(f"Cannot store empty DataFrame for {symbol}")
                return
                
            conn = sqlite3.connect(self.db_path)
            df_to_store = df.copy()
            
            # Check if timestamp column exists
            if 'timestamp' not in df_to_store.columns:
                logger.warning(f"No timestamp column in DataFrame for {symbol}")
                # If index is a DatetimeIndex, reset it to create a timestamp column
                if isinstance(df_to_store.index, pd.DatetimeIndex):
                    df_to_store = df_to_store.reset_index()
                    df_to_store = df_to_store.rename(columns={'index': 'timestamp'})
                else:
                    logger.error(f"Cannot create timestamp column for {symbol}")
                    return
            
            # Check if DataFrame is still empty after processing
            if df_to_store.empty:
                logger.warning(f"DataFrame is empty after processing for {symbol}")
                return
                
            # Convert timestamp to string if it's a Timestamp object
            if isinstance(df_to_store['timestamp'].iloc[0], pd.Timestamp):
                df_to_store['timestamp'] = df_to_store['timestamp'].astype(str)
                
            # Add symbol column
            df_to_store['symbol'] = symbol
            
            # Ensure all required columns exist
            required_columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close',
                               'volume', 'sma_short', 'sma_long', 'rsi']
            
            # Check if all required columns exist
            missing_columns = [col for col in required_columns if col not in df_to_store.columns]
            if missing_columns:
                logger.warning(f"Missing columns in DataFrame for {symbol}: {missing_columns}")
                # Add missing columns with NaN values
                for col in missing_columns:
                    if col != 'symbol' and col != 'timestamp':  # We already handled these
                        df_to_store[col] = np.nan
            
            # Select only the required columns
            df_to_store = df_to_store[required_columns]
            
            # Store in database
            df_to_store.to_sql('market_data', conn, if_exists=if_exists, index=False)
            conn.close()
            logger.info(f"Stored {len(df)} rows for {symbol} in database")
        except Exception as e:
            logger.error(f"Error storing data for {symbol}: {e}")
            logger.error(f"DataFrame info: {df.info() if not df.empty else 'Empty DataFrame'}")
    
    def generate_signals(self, data, strategy='moving_average_crossover'):
        """Generate trading signals based on the specified strategy."""
        logger.info(f"Generating signals using {strategy} strategy")
        signals = {}
        for symbol, df in data.items():
            if df.empty:
                continue
            df = df.copy()
            df['signal'] = 0
            if strategy == 'moving_average_crossover':
                df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1
                df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1
            elif strategy == 'rsi':
                df.loc[df['rsi'] < config.RSI_OVERSOLD, 'signal'] = 1
                df.loc[df['rsi'] > config.RSI_OVERBOUGHT, 'signal'] = -1
            latest_signal = df['signal'].iloc[-1]
            signal_changed = False
            if len(df) >= 2:
                prev_signal = df['signal'].iloc[-2]
                signal_changed = prev_signal != latest_signal and latest_signal != 0
            self._store_signal(symbol, df.iloc[-1]['timestamp'], 
                              self._get_action(latest_signal), 
                              df.iloc[-1]['close'], latest_signal, signal_changed)
            signals[symbol] = {
                'action': self._get_action(latest_signal),
                'signal': latest_signal,
                'signal_changed': signal_changed,
                'price': df['close'].iloc[-1],
                'timestamp': df['timestamp'].iloc[-1]
            }
        return signals
    
    def _get_action(self, signal):
        """Convert signal to action string."""
        if signal == 1:
            return 'buy'
        elif signal == -1:
            return 'sell'
        else:
            return 'hold'
    
    def _store_signal(self, symbol, timestamp, action, price, signal_value, signal_changed):
        """Store signal in the SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            if not isinstance(timestamp, str):
                timestamp = str(timestamp)
            cursor.execute('''
            INSERT OR REPLACE INTO signals 
            (symbol, timestamp, action, price, signal_value, signal_changed)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (symbol, timestamp, action, price, signal_value, signal_changed))
            conn.commit()
            conn.close()
            logger.info(f"Stored signal for {symbol}: {action}")
        except Exception as e:
            logger.error(f"Error storing signal for {symbol}: {e}")
    
    def get_account(self):
        """Get mock account information."""
        return {
            'equity': 100000.0, 'cash': 100000.0,
            'buying_power': 100000.0, 'portfolio_value': 100000.0
        }
    
    def get_positions(self):
        """Get mock positions."""
        return []
    
    def is_market_open(self):
        """Check if the market is currently open."""
        now = datetime.now()
        eastern_time = now - timedelta(hours=12)  # Approximation for Eastern Time
        if eastern_time.weekday() >= 5:  # Weekend
            return False
        market_open = eastern_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = eastern_time.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= eastern_time <= market_close
    
    def get_market_hours(self):
        """Get market open and close times for today."""
        now = datetime.now()
        today = now.date()
        open_time = datetime.combine(today, datetime.min.time().replace(hour=9, minute=30))
        close_time = datetime.combine(today, datetime.min.time().replace(hour=16, minute=0))
        return (open_time, close_time)
        
    def get_cached_data(self, symbol, limit=100):
        """
        Get cached data for a symbol from the database.
        
        Args:
            symbol (str): Stock symbol
            limit (int): Maximum number of rows to return
            
        Returns:
            DataFrame: DataFrame with cached data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"""
            SELECT * FROM market_data
            WHERE symbol = '{symbol}'
            ORDER BY timestamp DESC
            LIMIT {limit}
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                # Convert timestamp strings back to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Sort by timestamp
                df = df.sort_values('timestamp')
                logger.info(f"Retrieved {len(df)} rows of cached data for {symbol}")
                return df
            else:
                logger.warning(f"No cached data found for {symbol}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error retrieving cached data for {symbol}: {e}")
            return pd.DataFrame()
