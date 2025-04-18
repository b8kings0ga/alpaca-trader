"""
Base strategy class for the Alpaca Trading Bot.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.logger import get_logger

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
    
    def optimize(self, historical_data, target_metric='profit', test_period='1mo'):
        """
        Optimize strategy parameters based on historical performance.
        
        This method should be implemented by subclasses to fine-tune their parameters
        based on historical data and performance metrics.
        
        Args:
            historical_data (dict): Dictionary of DataFrames with historical market data
            target_metric (str): Metric to optimize for ('profit', 'sharpe', 'win_rate', etc.)
            test_period (str): Period to use for testing optimization results
            
        Returns:
            dict: Dictionary containing optimization results and new parameters
        """
        raise NotImplementedError("Subclasses must implement optimize()")
    
    def backtest(self, data, initial_capital=10000.0):
        """
        Run a backtest of the strategy on historical data.
        
        Args:
            data (dict): Dictionary of DataFrames with historical market data
            initial_capital (float): Initial capital for the backtest
            
        Returns:
            dict: Dictionary containing backtest results
        """
        results = {}
        portfolio_value = initial_capital
        positions = {}
        trades = []
        
        for symbol, df in data.items():
            if df.empty:
                continue
                
            # Create a copy of the DataFrame
            df = df.copy()
            
            # Generate signals for the entire dataset
            signals = self.generate_signals({symbol: df})
            
            if symbol not in signals:
                continue
                
            # Add signals to the DataFrame
            df['signal'] = 0
            
            for i, row in df.iterrows():
                timestamp = row['timestamp']
                if timestamp in signals[symbol]:
                    df.at[i, 'signal'] = signals[symbol][timestamp]['signal']
            
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            
            # Calculate strategy returns
            df['strategy_returns'] = df['signal'].shift(1) * df['returns']
            
            # Calculate cumulative returns
            df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
            df['strategy_cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1
            
            # Store results
            results[symbol] = {
                'returns': df['returns'].mean(),
                'strategy_returns': df['strategy_returns'].mean(),
                'cumulative_returns': df['cumulative_returns'].iloc[-1],
                'strategy_cumulative_returns': df['strategy_cumulative_returns'].iloc[-1],
                'sharpe_ratio': df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252),
                'max_drawdown': (df['strategy_cumulative_returns'].cummax() - df['strategy_cumulative_returns']).max(),
                'win_rate': (df['strategy_returns'] > 0).sum() / (df['strategy_returns'] != 0).sum() if (df['strategy_returns'] != 0).sum() > 0 else 0,
                'profit': portfolio_value * df['strategy_cumulative_returns'].iloc[-1],
                'data': df
            }
        
        return results