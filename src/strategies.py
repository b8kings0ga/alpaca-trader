"""
Trading strategies for the Alpaca Trading Bot.
"""
import pandas as pd
import numpy as np
import os.path
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from config import config
from src.logger import get_logger
from src.ml_models import get_ml_model, generate_ml_signals

logger = get_logger()

class Strategy:
    """
    Base strategy class that all strategies should inherit from.
    """
    def __init__(self, name):
        """
        Initialize the strategy.
        
        Args:
            name (str): Strategy name
        """
        self.name = name
        logger.info(f"Strategy '{name}' initialized")
    
    def generate_signals(self, data):
        """
        Generate trading signals based on the strategy.
        
        Args:
            data (dict): Dictionary of DataFrames with market data
            
        Returns:
            dict: Dictionary of signals for each symbol
        """
        raise NotImplementedError("Subclasses must implement generate_signals()")


class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover strategy.
    
    Buy when short MA crosses above long MA.
    Sell when short MA crosses below long MA.
    """
    def __init__(self):
        """
        Initialize the Moving Average Crossover strategy.
        """
        super().__init__("Moving Average Crossover")
        self.short_window = config.SHORT_WINDOW
        self.long_window = config.LONG_WINDOW
        
    def generate_signals(self, data):
        """
        Generate trading signals based on Moving Average Crossover.
        
        Args:
            data (dict): Dictionary of DataFrames with market data
            
        Returns:
            dict: Dictionary of signals for each symbol
        """
        signals = {}
        
        for symbol, df in data.items():
            logger.info(f"Analyzing data for {symbol}: {len(df)} data points available")
            logger.info(f"Strategy requires at least {self.long_window} data points")
            
            if df.empty:
                logger.warning(f"DataFrame is empty for {symbol}")
                continue
                
            if len(df) < self.long_window:
                logger.warning(f"Not enough data for {symbol} to generate signals. Need {self.long_window}, got {len(df)}")
                continue
                
            # Create a copy of the DataFrame
            df = df.copy()
            
            # Create signal column (1 = buy, -1 = sell, 0 = hold)
            df['signal'] = 0
            
            # Generate signals
            # Buy signal: short MA crosses above long MA
            df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1
            
            # Sell signal: short MA crosses below long MA
            df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1
            
            # Get the latest signal
            latest_signal = df['signal'].iloc[-1]
            
            # Check for crossover (signal change)
            signal_changed = False
            if len(df) >= 2:
                prev_signal = df['signal'].iloc[-2]
                signal_changed = prev_signal != latest_signal and latest_signal != 0
            
            signals[symbol] = {
                'action': self._get_action(latest_signal),
                'signal': latest_signal,
                'signal_changed': signal_changed,
                'price': df['close'].iloc[-1],
                'timestamp': df['timestamp'].iloc[-1]
            }
            
        return signals
    
    def _get_action(self, signal):
        """
        Convert signal to action string.
        
        Args:
            signal (int): Signal value (1, -1, or 0)
            
        Returns:
            str: Action string ('buy', 'sell', or 'hold')
        """
        if signal == 1:
            return 'buy'
        elif signal == -1:
            return 'sell'
        else:
            return 'hold'


class RSIStrategy(Strategy):
    """
    Relative Strength Index (RSI) strategy.
    
    Buy when RSI crosses below oversold threshold.
    Sell when RSI crosses above overbought threshold.
    """
    def __init__(self):
        """
        Initialize the RSI strategy.
        """
        super().__init__("RSI Strategy")
        self.period = config.RSI_PERIOD
        self.oversold = config.RSI_OVERSOLD
        self.overbought = config.RSI_OVERBOUGHT
        
    def generate_signals(self, data):
        """
        Generate trading signals based on RSI.
        
        Args:
            data (dict): Dictionary of DataFrames with market data
            
        Returns:
            dict: Dictionary of signals for each symbol
        """
        signals = {}
        
        for symbol, df in data.items():
            if df.empty or len(df) < self.period:
                logger.warning(f"Not enough data for {symbol} to generate signals")
                continue
                
            # Create a copy of the DataFrame
            df = df.copy()
            
            # Create signal column (1 = buy, -1 = sell, 0 = hold)
            df['signal'] = 0
            
            # Generate signals
            # Buy signal: RSI crosses below oversold threshold
            df.loc[df['rsi'] < self.oversold, 'signal'] = 1
            
            # Sell signal: RSI crosses above overbought threshold
            df.loc[df['rsi'] > self.overbought, 'signal'] = -1
            
            # Get the latest signal
            latest_signal = df['signal'].iloc[-1]
            
            # Check for crossover (signal change)
            signal_changed = False
            if len(df) >= 2:
                prev_signal = df['signal'].iloc[-2]
                signal_changed = prev_signal != latest_signal and latest_signal != 0
            
            signals[symbol] = {
                'action': self._get_action(latest_signal),
                'signal': latest_signal,
                'signal_changed': signal_changed,
                'price': df['close'].iloc[-1],
                'rsi': df['rsi'].iloc[-1],
                'timestamp': df['timestamp'].iloc[-1]
            }
            
        return signals
    
    def _get_action(self, signal):
        """
        Convert signal to action string.
        
        Args:
            signal (int): Signal value (1, -1, or 0)
            
        Returns:
            str: Action string ('buy', 'sell', or 'hold')
        """
        if signal == 1:
            return 'buy'
        elif signal == -1:
            return 'sell'
        else:
            return 'hold'
class MLStrategy(Strategy):
    """
    Machine Learning based strategy.
    
    This is a placeholder for future ML-based strategy implementations.
    
    TODO: Implement ML-based trading strategies here. Potential approaches include:
    - Supervised learning models (Random Forest, SVM, Neural Networks)
    - Reinforcement learning for dynamic strategy optimization
    - Deep learning for pattern recognition in price data
    - Natural Language Processing for sentiment analysis of news/social media
    - Ensemble methods combining multiple ML models
    
    For implementation, consider:
    1. Feature engineering from price data and technical indicators
    2. Model training and validation pipeline
    3. Hyperparameter optimization
    4. Backtesting framework for ML models
    5. Online learning capabilities for model updates
    """
    def __init__(self, model_type=None):
        """
        Initialize the ML strategy.
        
        Args:
            model_type (str): Type of ML model to use
        """
        super().__init__("ML Strategy")
        self.model_type = model_type or config.ML_STRATEGY_TYPE
        self.model = None
        self.model_path = os.path.join("models", f"{self.model_type}_model.pkl")
        
        # Try to load a pre-trained model if it exists
        self._load_model()
        
        logger.info(f"ML Strategy initialized with model type: {self.model_type}")
        
    def _load_model(self):
        """
        Load a pre-trained ML model if it exists.
        """
        try:
            if os.path.exists(self.model_path):
                self.model = get_ml_model(self.model_type)
                if self.model and self.model.load(self.model_path):
                    logger.info(f"Loaded pre-trained {self.model_type} model from {self.model_path}")
                else:
                    logger.warning(f"Failed to load model from {self.model_path}")
            else:
                logger.info(f"No pre-trained model found at {self.model_path}")
                self.model = get_ml_model(self.model_type)
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            self.model = get_ml_model(self.model_type)
        
    def generate_signals(self, data):
        """
        Generate trading signals based on ML models.
        
        This is a placeholder implementation that uses the ml_models module.
        
        Args:
            data (dict): Dictionary of DataFrames with market data
            
        Returns:
            dict: Dictionary of signals for each symbol
        """
        if not self.model:
            logger.warning("No ML model available - returning 'hold' signals")
            signals = {}
            
            for symbol, df in data.items():
                if df.empty:
                    continue
                    
                signals[symbol] = {
                    'action': 'hold',
                    'signal': 0,
                    'signal_changed': False,
                    'price': df['close'].iloc[-1] if not df.empty else 0,
                    'timestamp': df['timestamp'].iloc[-1] if not df.empty else None,
                    'ml_confidence': 0.5  # Placeholder for ML confidence score
                }
                
            return signals
        
        # Use the ml_models module to generate signals
        try:
            return generate_ml_signals(self.model, data)
        except Exception as e:
            logger.error(f"Error generating ML signals: {e}")
            return {}


class DualMovingAverageYF(Strategy):
    """
    Dual Moving Average strategy using yfinance data.
    
    Buy when short MA crosses above long MA.
    Sell when short MA crosses below long MA.
    
    This strategy uses yfinance to fetch data instead of Alpaca API.
    """
    def __init__(self):
        """
        Initialize the Dual Moving Average strategy.
        """
        super().__init__("Dual Moving Average YF")
        self.short_window = config.SHORT_WINDOW
        self.long_window = config.LONG_WINDOW
        
    def fetch_data(self, symbols, period="1mo", interval="1d"):
        """
        Fetch data from yfinance.
        
        Args:
            symbols (list): List of stock symbols
            period (str): Period to fetch data for (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Interval between data points (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            dict: Dictionary of DataFrames with market data
        """
        data = {}
        
        for symbol in symbols:
            try:
                # Fetch data from yfinance
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Reset index to make Date a column
                df = df.reset_index()
                
                # Rename columns to match Alpaca API format
                df = df.rename(columns={
                    'Date': 'timestamp',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Add technical indicators
                df['sma_short'] = df['close'].rolling(window=self.short_window).mean()
                df['sma_long'] = df['close'].rolling(window=self.long_window).mean()
                
                # Calculate RSI
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=config.RSI_PERIOD).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=config.RSI_PERIOD).mean()
                
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                data[symbol] = df
                
                # Generate and save a plot of the dual moving averages
                self._generate_plot(df, symbol)
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                
        return data
    
    def _generate_plot(self, df, symbol):
        """
        Generate and save a plot of the dual moving averages.
        
        Args:
            df (DataFrame): DataFrame with market data
            symbol (str): Stock symbol
        """
        try:
            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the closing price
            ax.plot(df['timestamp'], df['close'], label='Close Price', alpha=0.5)
            
            # Plot the short and long moving averages
            ax.plot(df['timestamp'], df['sma_short'], label=f'SMA {self.short_window}', linewidth=1.5)
            ax.plot(df['timestamp'], df['sma_long'], label=f'SMA {self.long_window}', linewidth=1.5)
            
            # Add buy/sell signals
            buy_signals = df[(df['sma_short'] > df['sma_long']) & (df['sma_short'].shift(1) <= df['sma_long'].shift(1))]
            sell_signals = df[(df['sma_short'] < df['sma_long']) & (df['sma_short'].shift(1) >= df['sma_long'].shift(1))]
            
            ax.scatter(buy_signals['timestamp'], buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
            ax.scatter(sell_signals['timestamp'], sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
            
            # Set title and labels
            ax.set_title(f'{symbol} - Dual Moving Average Strategy')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save the figure
            os.makedirs('logs/plots', exist_ok=True)
            plt.savefig(f'logs/plots/{symbol}_dual_ma.png')
            plt.close(fig)
            
            logger.info(f"Generated plot for {symbol}")
            
        except Exception as e:
            logger.error(f"Error generating plot for {symbol}: {e}")
    
    def generate_signals(self, data=None):
        """
        Generate trading signals based on Dual Moving Average.
        
        Args:
            data (dict, optional): Dictionary of DataFrames with market data.
                                  If None, data will be fetched from yfinance.
            
        Returns:
            dict: Dictionary of signals for each symbol
        """
        if data is None:
            # Fetch data from yfinance
            data = self.fetch_data(config.SYMBOLS)
        
        signals = {}
        
        for symbol, df in data.items():
            if df.empty or len(df) < self.long_window:
                logger.warning(f"Not enough data for {symbol} to generate signals")
                continue
                
            # Create a copy of the DataFrame
            df = df.copy()
            
            # Create signal column (1 = buy, -1 = sell, 0 = hold)
            df['signal'] = 0
            
            # Generate signals
            # Buy signal: short MA crosses above long MA
            df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1
            
            # Sell signal: short MA crosses below long MA
            df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1
            
            # Get the latest signal
            latest_signal = df['signal'].iloc[-1]
            
            # Check for crossover (signal change)
            signal_changed = False
            if len(df) >= 2:
                prev_signal = df['signal'].iloc[-2]
                signal_changed = prev_signal != latest_signal and latest_signal != 0
            
            signals[symbol] = {
                'action': self._get_action(latest_signal),
                'signal': latest_signal,
                'signal_changed': signal_changed,
                'price': df['close'].iloc[-1],
                'timestamp': df['timestamp'].iloc[-1],
                'short_ma': df['sma_short'].iloc[-1],
                'long_ma': df['sma_long'].iloc[-1]
            }
            
        return signals
    
    def _get_action(self, signal):
        """
        Convert signal to action string.
        
        Args:
            signal (int): Signal value (1, -1, or 0)
            
        Returns:
            str: Action string ('buy', 'sell', or 'hold')
        """
        if signal == 1:
            return 'buy'
        elif signal == -1:
            return 'sell'
        else:
            return 'hold'


# Factory function to get strategy by name
def get_strategy(strategy_name):
    """
    Get strategy instance by name.
    
    Args:
        strategy_name (str): Strategy name
        
    Returns:
        Strategy: Strategy instance
    """
    strategies = {
        'moving_average_crossover': MovingAverageCrossover,
        'rsi': RSIStrategy,
        'ml': MLStrategy,  # Added ML strategy placeholder
        'dual_ma_yf': DualMovingAverageYF  # Added Dual Moving Average with yfinance
    }
    
    strategy_class = strategies.get(strategy_name.lower())
    if not strategy_class:
        logger.error(f"Strategy '{strategy_name}' not found")
        return None
        
    return strategy_class()