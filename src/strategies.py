"""
Trading strategies for the Alpaca Trading Bot.
"""
import pandas as pd
import numpy as np
import os.path
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
        'ml': MLStrategy  # Added ML strategy placeholder
    }
    
    strategy_class = strategies.get(strategy_name.lower())
    if not strategy_class:
        logger.error(f"Strategy '{strategy_name}' not found")
        return None
        
    return strategy_class()
        
    return strategy_class()