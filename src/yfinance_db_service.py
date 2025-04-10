
"""
YFinance Data Service for fetching and storing market data.
This service runs independently and provides data to the Alpaca Trading Bot.
"""
import os
import time
import logging
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import uvicorn
import json
from fastapi.encoders import jsonable_encoder
import shutil
from apscheduler.schedulers.background import BackgroundScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/yfinance_db_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("yfinance_db_service")

# Create FastAPI app
app = FastAPI(title="YFinance Data Service", description="Service for fetching and storing market data from Yahoo Finance")

# Configuration from environment variables
MAX_DB_SIZE_GB = float(os.getenv("MAX_DB_SIZE_GB", "10"))  # Maximum database size in GB
FETCH_INTERVAL_MINUTES = int(os.getenv("FETCH_INTERVAL_MINUTES", "60"))  # How often to fetch data
SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,AMZN,GOOGL,META").split(",")  # Symbols to fetch
DB_PATH = os.getenv("DB_PATH", "data/market_data.db")  # Database path

# Create necessary directories
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Custom JSON encoder to handle NaN, Infinity values, and pandas Timestamp objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if np.isnan(obj):
                return None
            if np.isinf(obj):
                return None
        # Handle pandas Timestamp objects
        if pd.api.types.is_datetime64_any_dtype(obj) or isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)

class YFinanceDataService:
    """
    Service for fetching and storing market data from Yahoo Finance.
    """
    def __init__(self, db_path: str = DB_PATH):
        """Initialize the YFinanceDataService."""
        self.db_path = db_path
        self._ensure_db_exists()
        logger.info(f"YFinanceDataService initialized with database at {db_path}")
        
    def _ensure_db_exists(self):
        """Ensure the database directory and file exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            symbol TEXT, 
            timestamp TEXT, 
            open REAL, 
            high REAL, 
            low REAL,
            close REAL, 
            volume INTEGER, 
            sma_short REAL, 
            sma_long REAL, 
            rsi REAL,
            PRIMARY KEY (symbol, timestamp)
        )''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_metadata (
            id INTEGER PRIMARY KEY,
            last_fetch_time TEXT,
            total_records INTEGER,
            db_size_bytes INTEGER
        )''')
        conn.commit()
        conn.close()
        logger.info("Database tables created if they didn't exist")
        
    def fetch_and_store_data(self, symbols: List[str], period: str = "1d", interval: str = "1m"):
        """
        Fetch data from Yahoo Finance and store it in the database.
        
        Args:
            symbols: List of stock symbols
            period: Period to fetch (1d, 5d, 1mo, 3mo, etc.)
            interval: Interval between data points (1m, 5m, 1h, 1d, etc.)
        """
        logger.info(f"Fetching data for {len(symbols)} symbols with period={period}, interval={interval}")
        
        total_records = 0
        for symbol in symbols:
            try:
                # Fetch data from Yahoo Finance
                ticker = yf.Ticker(symbol)
                
                # For minute data, we need to be careful about Yahoo Finance's limitations
                if interval.endswith('m'):
                    # Yahoo Finance only allows 7 days of minute data
                    if period == "1d":
                        # Fetch more data for 1-minute intervals to ensure we have enough for strategies
                        logger.info(f"Fetching 7 days of minute data for {symbol} instead of 1 day to ensure sufficient data points")
                        df = ticker.history(period="7d", interval=interval)
                    else:
                        df = ticker.history(period="7d", interval=interval)
                else:
                    # For daily data, we can fetch more history
                    # Use a longer period to ensure we have enough data for strategies
                    logger.info(f"Fetching data for {symbol} with period={period}, interval={interval}")
                    df = ticker.history(period=period, interval=interval)
                
                if df.empty:
                    logger.warning(f"No data found for {symbol} with period={period}, interval={interval}")
                    continue
                    
                # Process the data
                df = df.reset_index()
                df = df.rename(columns={
                    'Date': 'timestamp', 'Datetime': 'timestamp',
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Volume': 'volume'
                })
                
                # Add technical indicators
                df = self.add_indicators(df)
                
                # Store in database
                records_added = self._store_market_data(symbol, df)
                total_records += records_added
                
                logger.info(f"Successfully fetched and stored {records_added} records for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        # Update metadata
        self._update_metadata(total_records)
        
        # Check database size and prune if necessary
        self._check_and_prune_database()
        
        return total_records
        
    def add_indicators(self, df):
        """Add technical indicators to the DataFrame."""
        # Check if we have enough data for meaningful indicators
        if len(df) < 50:
            logger.info(f"Limited data available ({len(df)} rows) - using simplified indicator calculations")
            
        # Add placeholder values for indicators if we don't have enough data
        if len(df) < 2:
            # For a single data point, we can't calculate any indicators
            # Set default neutral values
            df['sma_short'] = df['close']
            df['sma_long'] = df['close']
            df['rsi'] = 50
            logger.info("Only one data point available - using default indicator values")
            return df
            
        # Short-term SMA (use min of 20 or available data points)
        short_window = min(20, len(df) - 1)
        df['sma_short'] = df['close'].rolling(window=short_window).mean()
        
        # Long-term SMA (use min of 50 or available data points)
        long_window = min(50, len(df) - 1)
        df['sma_long'] = df['close'].rolling(window=long_window).mean()
        
        # RSI (use min of 14 or available data points)
        rsi_window = min(14, len(df) - 1)
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_window).mean()
        
        # Check for zero values in loss to avoid division by zero
        zero_loss_mask = loss == 0
        if zero_loss_mask.any():
            logger.debug(f"Found {zero_loss_mask.sum()} zero values in loss calculation for RSI - this is normal for limited data")
            
        # Replace zeros with a small value to avoid division by zero
        loss = loss.replace(0, 1e-10)
        
        rs = gain / loss
        
        # Check for infinity values
        inf_mask = np.isinf(rs)
        if inf_mask.any():
            logger.debug(f"Found {inf_mask.sum()} infinity values in RS calculation for RSI - this is normal for limited data")
            # Replace infinity with a large value
            rs = rs.replace([np.inf, -np.inf], 100)
            
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Fill NaN values with appropriate defaults
        df['sma_short'] = df['sma_short'].fillna(df['close'])
        df['sma_long'] = df['sma_long'].fillna(df['close'])
        
        # Check for NaN values in RSI
        nan_mask = np.isnan(df['rsi'])
        if nan_mask.any():
            nan_count = nan_mask.sum()
            if nan_count > 1:
                logger.debug(f"Found {nan_count} NaN values in RSI calculation - this is normal for the first {rsi_window} data points")
            # Replace NaN with 50 (neutral RSI value)
            df['rsi'] = df['rsi'].fillna(50)
            
        return df
        
    def _store_market_data(self, symbol, df):
        """
        Store market data in the SQLite database.
        
        Returns:
            int: Number of records added
        """
        try:
            # Check if DataFrame is empty
            if df.empty:
                logger.warning(f"Cannot store empty DataFrame for {symbol}")
                return 0
                
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
                    return 0
            
            # Check if DataFrame is still empty after processing
            if df_to_store.empty:
                logger.warning(f"DataFrame is empty after processing for {symbol}")
                return 0
                
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
            
            # Check for potential duplicates before insertion
            cursor = conn.cursor()
            
            # Get the min and max timestamps from the data to be inserted
            min_timestamp = df_to_store['timestamp'].min()
            max_timestamp = df_to_store['timestamp'].max()
            
            logger.info(f"Checking for potential duplicates for {symbol} between {min_timestamp} and {max_timestamp}")
            
            # Query existing records in this time range
            cursor.execute(
                "SELECT timestamp FROM market_data WHERE symbol = ? AND timestamp BETWEEN ? AND ?",
                (symbol, min_timestamp, max_timestamp)
            )
            existing_timestamps = set(row[0] for row in cursor.fetchall())
            
            if existing_timestamps:
                logger.info(f"Found {len(existing_timestamps)} existing records for {symbol} in the time range")
                
                # Check for duplicates in the data to be inserted
                new_timestamps = set(df_to_store['timestamp'])
                duplicate_timestamps = new_timestamps.intersection(existing_timestamps)
                
                if duplicate_timestamps:
                    logger.warning(f"Found {len(duplicate_timestamps)} duplicate timestamps for {symbol}")
                    logger.debug(f"Duplicate timestamps: {sorted(list(duplicate_timestamps))[:5]}...")
                    
                    # Filter out rows with duplicate timestamps
                    df_to_store = df_to_store[~df_to_store['timestamp'].isin(duplicate_timestamps)]
                    
                    if df_to_store.empty:
                        logger.info(f"All data for {symbol} already exists in the database")
                        conn.close()
                        return 0
            
            # Store in database
            records_before = self._count_records(conn, symbol)
            
            try:
                df_to_store.to_sql('market_data', conn, if_exists='append', index=False)
            except sqlite3.IntegrityError as e:
                logger.error(f"IntegrityError while inserting data for {symbol}: {e}")
                # If we still get an integrity error, try inserting one by one to identify problematic records
                successful_inserts = 0
                for _, row in df_to_store.iterrows():
                    try:
                        row_df = pd.DataFrame([row])
                        row_df.to_sql('market_data', conn, if_exists='append', index=False)
                        successful_inserts += 1
                    except sqlite3.IntegrityError as e:
                        logger.error(f"Failed to insert record for {symbol} at timestamp {row['timestamp']}: {e}")
                
                logger.info(f"Inserted {successful_inserts} out of {len(df_to_store)} records individually")
            
            records_after = self._count_records(conn, symbol)
            
            conn.close()
            
            records_added = records_after - records_before
            logger.info(f"Stored {records_added} new records for {symbol} in database")
            return records_added
            
        except Exception as e:
            logger.error(f"Error storing data for {symbol}: {e}")
            return 0
            
    def _count_records(self, conn, symbol=None):
        """Count the number of records in the database for a symbol."""
        cursor = conn.cursor()
        if symbol:
            cursor.execute("SELECT COUNT(*) FROM market_data WHERE symbol = ?", (symbol,))
        else:
            cursor.execute("SELECT COUNT(*) FROM market_data")
        count = cursor.fetchone()[0]
        return count
        
    def _update_metadata(self, records_added):
        """Update the metadata table with the latest fetch information."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current database size
            db_size = os.path.getsize(self.db_path)
            
            # Get total records
            cursor.execute("SELECT COUNT(*) FROM market_data")
            total_records = cursor.fetchone()[0]
            
            # Update metadata
            cursor.execute("""
            INSERT OR REPLACE INTO data_metadata (id, last_fetch_time, total_records, db_size_bytes)
            VALUES (1, ?, ?, ?)
            """, (datetime.now().isoformat(), total_records, db_size))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated metadata: {records_added} records added, total: {total_records}, size: {db_size/1024/1024:.2f} MB")
            
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            
    def _check_and_prune_database(self):
        """Check database size and prune old data if necessary."""
        try:
            # Get current database size in GB
            db_size_gb = os.path.getsize(self.db_path) / (1024**3)
            
            if db_size_gb > MAX_DB_SIZE_GB:
                logger.warning(f"Database size ({db_size_gb:.2f} GB) exceeds limit ({MAX_DB_SIZE_GB} GB). Pruning old data...")
                
                # Calculate how much data to remove (aim to get to 80% of max size)
                target_size = MAX_DB_SIZE_GB * 0.8
                remove_percentage = 1 - (target_size / db_size_gb)
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get total number of records
                cursor.execute("SELECT COUNT(*) FROM market_data")
                total_records = cursor.fetchone()[0]
                
                # Calculate number of records to remove
                records_to_remove = int(total_records * remove_percentage)
                
                # Get the timestamp cutoff for removal
                cursor.execute("""
                SELECT timestamp FROM market_data
                ORDER BY timestamp ASC
                LIMIT 1 OFFSET ?
                """, (records_to_remove,))
                
                result = cursor.fetchone()
                if result:
                    cutoff_timestamp = result[0]
                    
                    # Remove records older than the cutoff
                    cursor.execute("DELETE FROM market_data WHERE timestamp < ?", (cutoff_timestamp,))
                    removed_count = cursor.rowcount
                    
                    conn.commit()
                    logger.info(f"Pruned {removed_count} records older than {cutoff_timestamp}")
                    
                    # Vacuum the database to reclaim space
                    cursor.execute("VACUUM")
                    conn.commit()
                    
                    # Update metadata
                    self._update_metadata(0)
                    
                conn.close()
                
                # Check new size
                new_size_gb = os.path.getsize(self.db_path) / (1024**3)
                logger.info(f"Database size after pruning: {new_size_gb:.2f} GB")
                
        except Exception as e:
            logger.error(f"Error pruning database: {e}")
            
    def get_market_data(self, symbol: str, start_date: Optional[str] = None,
                       end_date: Optional[str] = None, limit: int = 100):
        """
        Get market data for a symbol from the database.
        
        Args:
            symbol: Stock symbol
            start_date: Start date in ISO format
            end_date: End date in ISO format
            limit: Maximum number of records to return
            
        Returns:
            DataFrame: DataFrame with market data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM market_data WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
                
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                # Convert timestamp strings back to datetime
                logger.info(f"Converting timestamp strings to datetime for {symbol}")
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Check for NaN or Infinity values after timestamp conversion
                has_nan = df.isna().any().any()
                if has_nan:
                    logger.warning(f"NaN values detected after timestamp conversion for {symbol}")
                    for col in df.columns:
                        nan_count = df[col].isna().sum()
                        if nan_count > 0:
                            logger.warning(f"Column '{col}' has {nan_count} NaN values")
                
                # Sort by timestamp
                df = df.sort_values('timestamp')
                
                # Check for special float values in numeric columns
                for col in df.select_dtypes(include=['float64']).columns:
                    inf_count = np.isinf(df[col]).sum()
                    if inf_count > 0:
                        logger.warning(f"Column '{col}' has {inf_count} Infinity values for {symbol}")
                
                logger.info(f"Retrieved {len(df)} rows of data for {symbol}")
                return df
            else:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {e}")
            return pd.DataFrame()
            
    def get_database_stats(self):
        """Get statistics about the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total records
            cursor.execute("SELECT COUNT(*) FROM market_data")
            total_records = cursor.fetchone()[0]
            
            # Get records per symbol
            cursor.execute("SELECT symbol, COUNT(*) FROM market_data GROUP BY symbol")
            symbol_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get database size
            db_size = os.path.getsize(self.db_path)
            
            # Get metadata
            cursor.execute("SELECT * FROM data_metadata WHERE id = 1")
            metadata = cursor.fetchone()
            
            conn.close()
            
            stats = {
                "total_records": total_records,
                "symbol_counts": symbol_counts,
                "db_size_bytes": db_size,
                "db_size_mb": db_size / (1024 * 1024),
                "db_size_gb": db_size / (1024 * 1024 * 1024),
                "last_fetch_time": metadata[1] if metadata else None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

# Create service instance
service = YFinanceDataService()

# API endpoints
@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "YFinance Data Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/data/{symbol}",
            "/stats",
            "/fetch"
        ]
    }

@app.get("/data/{symbol}")
async def get_data(symbol: str,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   limit: int = Query(100, ge=1, le=10000)):  # Increased max limit to allow more data points
    """Get market data for a symbol."""
    df = service.get_market_data(symbol, start_date, end_date, limit)
    
    if df.empty:
        return JSONResponse(
            status_code=404,
            content={"message": f"No data found for {symbol}"}
        )
    
    # Add diagnostic logging to check for problematic values
    logger.info(f"Checking for problematic values in DataFrame for {symbol}")
    has_nan = df.isna().any().any()
    has_inf = np.isinf(df.select_dtypes(include=['float64'])).any().any()
    
    if has_nan:
        logger.warning(f"DataFrame for {symbol} contains NaN values")
        # Log columns with NaN values
        for col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"Column '{col}' has {nan_count} NaN values")
    
    if has_inf:
        logger.warning(f"DataFrame for {symbol} contains Infinity values")
        # Log columns with Infinity values
        for col in df.select_dtypes(include=['float64']).columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                logger.warning(f"Column '{col}' has {inf_count} Infinity values")
    
    # Convert DataFrame to JSON
    try:
        # Proactively clean the data before attempting serialization
        df_clean = df.copy()
        
        # Replace NaN with None (null in JSON)
        df_clean = df_clean.where(pd.notnull(df_clean), None)
        
        # Replace infinity values with None
        for col in df_clean.select_dtypes(include=['float64']).columns:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], None)
        
        # Use custom JSON encoder for better handling of any remaining problematic values
        records = df_clean.to_dict(orient="records")
        
        # Manually serialize the JSON with the custom encoder
        json_str = json.dumps(records, cls=CustomJSONEncoder)
        
        return JSONResponse(
            content=json.loads(json_str),
            media_type="application/json",
            status_code=200
        )
    except ValueError as e:
        logger.error(f"JSON serialization error for {symbol}: {e}")
        
        # If we still have an error, try a more aggressive approach
        logger.info("Attempting more aggressive cleaning of problematic values")
        df_clean = df.copy()
        
        # Convert all float columns to strings to avoid JSON serialization issues
        for col in df_clean.select_dtypes(include=['float64']).columns:
            df_clean[col] = df_clean[col].apply(lambda x: str(x) if pd.notnull(x) and not np.isinf(x) else None)
        
        logger.info("Returning aggressively cleaned DataFrame")
        return df_clean.to_dict(orient="records")

@app.get("/data_safe/{symbol}")
async def get_data_safe(symbol: str,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       limit: int = Query(100, ge=1, le=1000)):
    """Get market data for a symbol with safe JSON serialization."""
    df = service.get_market_data(symbol, start_date, end_date, limit)
    
    if df.empty:
        return JSONResponse(
            status_code=404,
            content={"message": f"No data found for {symbol}"}
        )
    
    # Convert DataFrame to a safe format for JSON serialization
    safe_records = []
    for _, row in df.iterrows():
        safe_record = {}
        for col, val in row.items():
            # Handle different data types
            if pd.isna(val):
                safe_record[col] = None
            elif isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                safe_record[col] = None
            elif isinstance(val, (pd.Timestamp, datetime)):
                safe_record[col] = val.isoformat()
            else:
                try:
                    # Test if value is JSON serializable
                    json.dumps(val)
                    safe_record[col] = val
                except (TypeError, OverflowError):
                    # If not serializable, convert to string
                    safe_record[col] = str(val)
        safe_records.append(safe_record)
    
    return JSONResponse(content=safe_records)

@app.get("/stats")
async def get_stats():
    """Get database statistics."""
    return service.get_database_stats()

@app.post("/fetch")
async def fetch_data(symbols: Optional[List[str]] = None,
                    period: str = "1d",
                    interval: str = "1m"):
    """Fetch data from Yahoo Finance and store it in the database."""
    if not symbols:
        symbols = SYMBOLS
        
    records_added = service.fetch_and_store_data(symbols, period, interval)
    
    return {
        "message": f"Fetched and stored data for {len(symbols)} symbols",
        "records_added": records_added,
        "symbols": symbols
    }

# Scheduler for periodic data fetching
scheduler = BackgroundScheduler()

def fetch_newest_data():
    """
    Process 1: Fetch the newest data for all symbols.
    This runs every minute to get the most recent market data.
    """
    logger.info(f"Fetching newest data for {len(SYMBOLS)} symbols")
    
    for symbol in SYMBOLS:
        try:
            # Fetch only the most recent minute of data
            ticker = yf.Ticker(symbol)
            # Fetch more data to ensure we have enough for strategies
            logger.info(f"Fetching 1 day of minute data for {symbol}")
            df = ticker.history(period="1d", interval="1m")
            
            if df.empty:
                logger.warning(f"No recent data found for {symbol}")
                continue
                
            # Get more recent data points, not just the last one
            # This ensures we have enough data for technical indicators
            from config import config
            required_data_points = max(config.LONG_WINDOW, config.SHORT_WINDOW) + 10  # Add buffer
            logger.info(f"Required data points for strategy: {required_data_points}")
            
            if len(df) > required_data_points:
                logger.info(f"Using the most recent {required_data_points} data points for {symbol}")
                df = df.tail(required_data_points)
            else:
                logger.info(f"Using all available {len(df)} data points for {symbol}")
            
            # Process the data
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'timestamp', 'Datetime': 'timestamp',
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })
            
            # Add technical indicators
            df = service.add_indicators(df)
            
            # Store in database
            records_added = service._store_market_data(symbol, df)
            
            logger.info(f"Fetched and stored newest data for {symbol}: {records_added} records added")
            
        except Exception as e:
            logger.error(f"Error fetching newest data for {symbol}: {e}")
    
    logger.info("Newest data fetch completed")

def fetch_historical_data():
    """
    Process 2: Fetch historical data that's older than what's in the database.
    This runs less frequently to build up historical data.
    """
    logger.info(f"Fetching historical data for {len(SYMBOLS)} symbols")
    
    for symbol in SYMBOLS:
        try:
            # Get the oldest timestamp in the database for this symbol
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MIN(timestamp) FROM market_data WHERE symbol = ?",
                (symbol,)
            )
            result = cursor.fetchone()
            conn.close()
            
            oldest_timestamp = None
            if result and result[0]:
                oldest_timestamp = pd.to_datetime(result[0])
                logger.info(f"Oldest data for {symbol} is from {oldest_timestamp}")
                
                # Calculate the start time for fetching older data
                # Fetch more data to ensure we have enough for strategies
                # (60 minutes before the oldest data we have)
                start_time = oldest_timestamp - timedelta(minutes=60)
                end_time = oldest_timestamp - timedelta(seconds=1)
                
                # Format times for yfinance (using ISO format which is compatible)
                start_str = start_time.strftime('%Y-%m-%d')
                end_str = end_time.strftime('%Y-%m-%d')
                
                logger.info(f"Fetching historical data for {symbol} from {start_str} to {end_str}")
                
                # Fetch data for this time range
                ticker = yf.Ticker(symbol)
                # Use a valid period format and interval
                # Fetch more data to ensure we have enough for strategies
                logger.info(f"Fetching 7 days of minute data for {symbol} to ensure sufficient data points")
                df = ticker.history(period="7d", interval="1m")
                
                # Filter to get only the data we need (8 minutes before the oldest timestamp)
                if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                    # Convert oldest_timestamp to pandas Timestamp for comparison
                    pd_oldest_timestamp = pd.to_datetime(oldest_timestamp)
                    # Filter data to get only records older than our oldest record
                    df = df[df.index < pd_oldest_timestamp]
                    # Take more data to ensure we have enough for strategies
                    # We need at least LONG_WINDOW data points (from config)
                    from config import config
                    required_data_points = max(config.LONG_WINDOW, config.SHORT_WINDOW) + 10  # Add buffer
                    logger.info(f"Required data points for strategy: {required_data_points}")
                    df = df.tail(required_data_points)
                    
                    logger.info(f"Filtered historical data for {symbol}: got {len(df)} records")
                
                if df.empty:
                    logger.warning(f"No historical data found for {symbol} in the specified range")
                    continue
                    
                # Process the data
                df = df.reset_index()
                df = df.rename(columns={
                    'Date': 'timestamp', 'Datetime': 'timestamp',
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Volume': 'volume'
                })
                
                # Add technical indicators
                df = service.add_indicators(df)
                
                # Store in database
                records_added = service._store_market_data(symbol, df)
                
                logger.info(f"Fetched and stored historical data for {symbol}: {records_added} records added")
            else:
                logger.info(f"No existing data for {symbol}, fetching initial data")
                # If no data exists, fetch more data to ensure we have enough for strategies
                logger.info(f"Fetching 7 days of data for {symbol} to ensure sufficient data points")
                service.fetch_and_store_data([symbol], period="7d", interval="1m")
                
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
    
    logger.info("Historical data fetch completed")

# Main function
def main():
    """Main function to start the service."""
    # Schedule Process 1: Fetch newest data every minute
    scheduler.add_job(
        fetch_newest_data,
        'interval',
        minutes=1,
        id='newest_data_fetch',
        replace_existing=True
    )
    
    # Schedule Process 2: Fetch historical data every 10 minutes
    scheduler.add_job(
        fetch_historical_data,
        'interval',
        minutes=10,
        id='historical_data_fetch',
        replace_existing=True
    )
    
    # Start the scheduler
    scheduler.start()
    logger.info("Scheduler started with two processes:")
    logger.info("1. Fetching newest data every 1 minute")
    logger.info("2. Fetching historical data every 10 minutes")
    
    # Run initial data fetches
    fetch_newest_data()
    fetch_historical_data()
    
    # Start the API server
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    main()
