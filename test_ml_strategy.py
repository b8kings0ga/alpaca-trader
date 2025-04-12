"""
Test the ML strategy for the Alpaca Trading Bot.

This script tests the ML strategy by generating signals for a set of symbols
and displaying the results.
"""
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from config import config
from src.logger import get_logger
from src.yfinance_data import YFinanceData
from src.strategies import MLStrategy
from src.ml_models import get_ml_model

logger = get_logger()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test ML strategy for stock trading')
    parser.add_argument('--model-type', type=str, default=config.ML_STRATEGY_TYPE,
                        choices=['random_forest', 'gradient_boosting', 'ensemble'],
                        help='Type of ML model to use')
    parser.add_argument('--symbols', type=str, nargs='+', default=config.SYMBOLS,
                        help='Stock symbols to test on')
    parser.add_argument('--period', type=str, default='1mo',
                        help='Period of historical data to use (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)')
    parser.add_argument('--plot', action='store_true',
                        help='Plot signals on price chart')
    
    return parser.parse_args()

def test_ml_strategy(args):
    """Test ML strategy based on command line arguments."""
    logger.info(f"Testing {args.model_type} model on {args.symbols} with {args.period} of historical data")
    
    # Check if model exists
    model_path = f"models/{args.model_type}.joblib"
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please train the model first.")
        return
    
    # Initialize ML strategy
    strategy = MLStrategy(model_type=args.model_type)
    
    # Get historical data
    logger.info(f"Fetching historical data for {args.symbols} over {args.period}")
    yf_data = YFinanceData()
    historical_data = yf_data.get_historical_data(args.symbols, period=args.period)
    
    if not historical_data:
        logger.error("Failed to fetch historical data")
        return
    
    # Generate signals
    logger.info("Generating signals")
    signals = strategy.generate_signals(historical_data)
    
    if not signals:
        logger.error("Failed to generate signals")
        return
    
    # Display signals
    logger.info("ML Strategy Signals:")
    for symbol, signal in signals.items():
        logger.info(f"  {symbol}: {signal['action']} (confidence: {signal.get('ml_confidence', 0.5):.2f})")
    
    # Plot signals
    if args.plot:
        plot_signals(historical_data, signals)

def plot_signals(data, signals):
    """Plot signals on price chart."""
    try:
        # Create plots directory if it doesn't exist
        os.makedirs('logs/plots', exist_ok=True)
        
        # Plot each symbol
        for symbol, df in data.items():
            if symbol not in signals:
                continue
                
            signal = signals[symbol]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot price
            ax.plot(df['timestamp'], df['close'], label='Close Price')
            
            # Plot moving averages if available
            if 'sma_short' in df.columns:
                ax.plot(df['timestamp'], df['sma_short'], label=f'SMA {config.SHORT_WINDOW}', alpha=0.7)
            if 'sma_long' in df.columns:
                ax.plot(df['timestamp'], df['sma_long'], label=f'SMA {config.LONG_WINDOW}', alpha=0.7)
            
            # Add signal
            if signal['action'] == 'buy':
                ax.scatter(df['timestamp'].iloc[-1], df['close'].iloc[-1], 
                          marker='^', color='green', s=200, label='Buy Signal')
            elif signal['action'] == 'sell':
                ax.scatter(df['timestamp'].iloc[-1], df['close'].iloc[-1], 
                          marker='v', color='red', s=200, label='Sell Signal')
            
            # Add title and labels
            ax.set_title(f"{symbol} - ML Strategy Signal: {signal['action'].upper()} (Confidence: {signal.get('ml_confidence', 0.5):.2f})")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f'logs/plots/{symbol}_ml_signal.png')
            plt.close(fig)
            
            logger.info(f"Signal plot saved to logs/plots/{symbol}_ml_signal.png")
            
    except Exception as e:
        logger.error(f"Error plotting signals: {e}")

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Test ML strategy
    test_ml_strategy(args)

if __name__ == '__main__':
    main()