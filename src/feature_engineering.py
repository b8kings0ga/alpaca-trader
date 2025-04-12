"""
Feature engineering for the Alpaca Trading Bot.

This module contains functions for creating features and target variables for ML models.
"""
import numpy as np
import pandas as pd
from config import config
from src.logger import get_logger

logger = get_logger()

def add_technical_indicators(df):
    """
    Add technical indicators to a DataFrame.
    
    Args:
        df (DataFrame): DataFrame with OHLCV data
        
    Returns:
        DataFrame: DataFrame with technical indicators
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided for adding technical indicators")
        return df
        
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    try:
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns for technical indicators: {missing_columns}")
            return df
            
        # Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # Relative Strength Index
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        
        # Replace zeros with a small value to avoid division by zero
        loss = loss.replace(0, 1e-10)
        
        rs = gain / loss
        
        # Replace infinity with a large value
        rs = rs.replace([np.inf, -np.inf], 100)
            
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Average True Range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Commodity Channel Index
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma_tp = tp.rolling(window=14).mean()
        md_tp = (tp - ma_tp).abs().rolling(window=14).mean()
        df['cci'] = (tp - ma_tp) / (0.015 * md_tp)
        
        # Stochastic Oscillator
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['stoch_k'] = k.rolling(window=3).mean()
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # On-Balance Volume
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        
        # Price Rate of Change
        df['roc'] = df['close'].pct_change(periods=10) * 100
        
        # Williams %R
        highest_high = df['high'].rolling(window=14).max()
        lowest_low = df['low'].rolling(window=14).min()
        df['willr'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        
        # Momentum
        df['mom'] = df['close'].diff(periods=10)
        
        # Percentage Price Oscillator
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['ppo'] = 100 * (ema_12 - ema_26) / ema_26
        
        # Directional Movement Index
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0)
        minus_dm = minus_dm.abs().where((minus_dm < 0) & (minus_dm.abs() > plus_dm), 0)
        
        tr = pd.concat([
            (df['high'] - df['low']).abs(),
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        
        atr_14 = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr_14)
        
        df['dx'] = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        df['adx'] = df['dx'].rolling(window=14).mean()
        
        # Add price changes
        df['price_change_1d'] = df['close'].pct_change(1)
        df['price_change_5d'] = df['close'].pct_change(5)
        df['price_change_10d'] = df['close'].pct_change(10)
        df['price_change_20d'] = df['close'].pct_change(20)
        
        # Add volume changes
        df['volume_change_1d'] = df['volume'].pct_change(1)
        df['volume_change_5d'] = df['volume'].pct_change(5)
        
        # Add volatility
        df['volatility_5d'] = df['price_change_1d'].rolling(5).std()
        df['volatility_10d'] = df['price_change_1d'].rolling(10).std()
        df['volatility_20d'] = df['price_change_1d'].rolling(20).std()
        
        # Add moving average crossovers
        df['sma_5_10_cross'] = np.where(df['sma_5'] > df['sma_10'], 1, -1)
        df['sma_10_20_cross'] = np.where(df['sma_10'] > df['sma_20'], 1, -1)
        df['sma_20_50_cross'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        df['sma_50_200_cross'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)
        
        # Add price relative to moving averages
        df['price_sma_5_ratio'] = df['close'] / df['sma_5']
        df['price_sma_10_ratio'] = df['close'] / df['sma_10']
        df['price_sma_20_ratio'] = df['close'] / df['sma_20']
        df['price_sma_50_ratio'] = df['close'] / df['sma_50']
        df['price_sma_200_ratio'] = df['close'] / df['sma_200']
        
        # Add Bollinger Band position
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Add RSI changes
        df['rsi_change_1d'] = df['rsi'].diff(1)
        df['rsi_change_5d'] = df['rsi'].diff(5)
        
        # Add MACD changes
        df['macd_change_1d'] = df['macd'].diff(1)
        df['macd_hist_change_1d'] = df['macd_hist'].diff(1)
        
        # Add custom indicators
        df['custom_1'] = df['rsi'] * df['bb_position']
        df['custom_2'] = df['macd'] * df['rsi'] / 100
        
        # Replace NaN values with 0
        df = df.fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}")
        return df


def create_target_variable(df, lookahead_period=5, threshold=0.01):
    """
    Create target variable for ML models.
    
    Args:
        df (DataFrame): DataFrame with OHLCV data
        lookahead_period (int): Number of days to look ahead for price change
        threshold (float): Threshold for price change to be considered significant
        
    Returns:
        Series: Target variable (1 for buy, 0 for hold, -1 for sell)
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided for creating target variable")
        return pd.Series(index=df.index)
        
    try:
        # Calculate future price change
        future_price_change = df['close'].shift(-lookahead_period) / df['close'] - 1
        
        # Create target variable
        target = pd.Series(0, index=df.index)
        target[future_price_change > threshold] = 1  # Buy
        target[future_price_change < -threshold] = -1  # Sell
        
        return target
        
    except Exception as e:
        logger.error(f"Error creating target variable: {e}")
        return pd.Series(index=df.index)


def prepare_features_targets(historical_data, lookback_period=None):
    """
    Prepare features and targets for ML models.
    
    Args:
        historical_data (dict): Dictionary of DataFrames with historical market data
        lookback_period (int): Number of days to look back for features
        
    Returns:
        dict: Dictionary of DataFrames with features and targets
    """
    if not historical_data:
        logger.warning("No historical data provided for preparing features and targets")
        return {}
        
    lookback_period = lookback_period or config.ML_LOOKBACK_PERIOD
    
    prepared_data = {}
    
    for symbol, df in historical_data.items():
        if df is None or df.empty:
            continue
            
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Create target variable
        df['target'] = create_target_variable(df)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if df.empty:
            continue
            
        # Add lookback features
        if lookback_period > 0:
            # Log the DataFrame size before adding lag features
            logger.info(f"DataFrame size for {symbol} before adding lag features: {df.shape}")
            logger.info(f"Memory usage before adding lag features: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Create a dictionary to store all lag features
            lag_features = {}
            
            # Generate all lag features at once
            for i in range(1, lookback_period + 1):
                for feature in config.ML_FEATURES:
                    if feature in df.columns:
                        # Store the lag feature in the dictionary instead of adding to DataFrame directly
                        lag_features[f"{feature}_lag_{i}"] = df[feature].shift(i)
            
            # Add all lag features to the DataFrame at once
            df = pd.concat([df, pd.DataFrame(lag_features)], axis=1)
            
            # Log the DataFrame size after adding lag features
            logger.info(f"DataFrame size for {symbol} after adding lag features: {df.shape}")
            logger.info(f"Memory usage after adding lag features: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            # Drop rows with NaN values again
            before_dropna_shape = df.shape
            df = df.dropna()
            after_dropna_shape = df.shape
            
            # Log the effect of dropping NaN values
            logger.info(f"DataFrame shape for {symbol} before dropping NaN: {before_dropna_shape}")
            logger.info(f"DataFrame shape for {symbol} after dropping NaN: {after_dropna_shape}")
            logger.info(f"Dropped {before_dropna_shape[0] - after_dropna_shape[0]} rows with NaN values")
            
        if not df.empty:
            prepared_data[symbol] = df
            logger.info(f"Added {symbol} to prepared_data with shape {df.shape}")
        else:
            logger.warning(f"DataFrame for {symbol} is empty after processing, not adding to prepared_data")
            
    # Log the final prepared_data
    if prepared_data:
        logger.info(f"prepare_features_targets returning data for {len(prepared_data)} symbols: {list(prepared_data.keys())}")
    else:
        logger.warning("prepare_features_targets returning empty dictionary")
        
    return prepared_data
    return prepared_data
