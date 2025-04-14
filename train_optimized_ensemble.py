#!/usr/bin/env python
"""
Script to train an optimized ensemble ML strategy for trading.
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.yfinance_data import YFinanceData
from src.feature_engineering import add_technical_indicators
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('logs/plots', exist_ok=True)

def load_and_prepare_data(symbols, period='1y'):
    """Load and prepare data for model training."""
    logger.info(f"Loading data for {symbols} over {period} period")
    
    # Initialize YFinanceData
    data_source = YFinanceData()
    
    # Get historical data
    historical_data = data_source.get_historical_data(symbols, period=period)
    
    # Prepare data for ML
    X_all = []
    y_all = []
    
    for symbol, data in historical_data.items():
        logger.info(f"Preparing data for {symbol} with {len(data)} data points")
        
        # Add features
        df = add_technical_indicators(data.copy())
        df.dropna(inplace=True)
        
        # Create target variable (1 for price increase, 0 for decrease or no change)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df[:-1]  # Drop the last row since we don't have a target for it
        
        # Select features
        features = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_std',
            'atr', 'adx', 'cci', 'stoch_k', 'stoch_d',
            'obv', 'roc', 'willr', 'mom', 'ppo', 'dx'
        ]
        
        # Add to combined dataset
        X_all.append(df[features])
        y_all.append(df['target'])
    
    # Combine data from all symbols
    X = pd.concat(X_all)
    y = pd.concat(y_all)
    
    # Split data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    logger.info(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    return X_train, X_test, y_train, y_test, historical_data

def load_optimized_models():
    """Load optimized ML models and ensemble weights."""
    logger.info("Loading optimized models and weights")
    
    # Load models
    try:
        rf_model = joblib.load('models/random_forest_optimized.joblib')
        gb_model = joblib.load('models/gradient_boosting_optimized.joblib')
        logger.info("Loaded optimized models successfully")
    except FileNotFoundError:
        logger.warning("Optimized models not found, loading default models")
        rf_model = joblib.load('models/random_forest.joblib')
        gb_model = joblib.load('models/gradient_boosting.joblib')
    
    # Load ensemble weights
    try:
        with open('models/ensemble_weights_optimized.txt', 'r') as f:
            lines = f.readlines()
            weights = {}
            for line in lines:
                key, value = line.strip().split(': ')
                weights[key] = float(value)
        logger.info(f"Loaded optimized weights: {weights}")
    except FileNotFoundError:
        logger.warning("Optimized weights not found, using default weights")
        weights = {'gradient_boosting': 0.7, 'random_forest': 0.3}
    
    return rf_model, gb_model, weights

def backtest_ensemble_strategy(historical_data, rf_model, gb_model, weights):
    """Backtest the ensemble strategy on historical data."""
    logger.info("Backtesting ensemble strategy")
    
    results = {}
    
    for symbol, data in historical_data.items():
        logger.info(f"Backtesting {symbol} with {len(data)} data points")
        
        # Add features
        df = add_technical_indicators(data.copy())
        df.dropna(inplace=True)
        
        # Select features
        features = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_std',
            'atr', 'adx', 'cci', 'stoch_k', 'stoch_d',
            'obv', 'roc', 'willr', 'mom', 'ppo', 'dx'
        ]
        
        # Initialize portfolio
        initial_value = 10000.0
        cash = initial_value * 0.5
        shares = int((initial_value * 0.5) / df.iloc[0]['close'])
        trades = []
        portfolio_values = []
        
        # Add initial buy trade
        trades.append({
            'date': df.iloc[0]['timestamp'],
            'action': 'buy',
            'price': df.iloc[0]['close'],
            'shares': shares,
            'value': shares * df.iloc[0]['close']
        })
        
        # Generate predictions for each day
        X = df[features]
        rf_pred_proba = rf_model.predict_proba(X)[:, 1]
        gb_pred_proba = gb_model.predict_proba(X)[:, 1]
        
        # Combine predictions
        ensemble_pred_proba = (
            weights['gradient_boosting'] * gb_pred_proba + 
            weights['random_forest'] * rf_pred_proba
        )
        
        # Define thresholds for buy/sell signals
        buy_threshold = 0.6  # Higher threshold for buy signals
        sell_threshold = 0.4  # Lower threshold for sell signals
        
        # Simulate trading
        for i in range(len(df)):
            date = df.iloc[i]['timestamp']
            price = df.iloc[i]['close']
            signal_prob = ensemble_pred_proba[i]
            
            # Determine action based on signal probability
            if signal_prob > buy_threshold and cash > 0:
                # Buy signal
                buy_shares = int(cash / price)
                if buy_shares > 0:
                    cash -= buy_shares * price
                    shares += buy_shares
                    trades.append({
                        'date': date,
                        'action': 'buy',
                        'price': price,
                        'shares': buy_shares,
                        'value': buy_shares * price,
                        'signal': signal_prob
                    })
            
            elif signal_prob < sell_threshold and shares > 0:
                # Sell signal
                cash += shares * price
                sell_shares = shares
                shares = 0
                trades.append({
                    'date': date,
                    'action': 'sell',
                    'price': price,
                    'shares': sell_shares,
                    'value': sell_shares * price,
                    'signal': signal_prob
                })
            
            # Record portfolio value
            portfolio_values.append({
                'date': date,
                'value': cash + shares * price
            })
        
        # Calculate final portfolio value
        final_value = cash + shares * price
        
        # Calculate return
        returns = ((final_value - initial_value) / initial_value) * 100
        
        # Calculate win rate
        if len(trades) > 1:
            profitable_trades = 0
            buy_price = 0
            for trade in trades:
                if trade['action'] == 'buy':
                    buy_price = trade['price']
                elif trade['action'] == 'sell' and buy_price > 0:
                    if trade['price'] > buy_price:
                        profitable_trades += 1
                    buy_price = 0
            
            win_rate = (profitable_trades / (len(trades) // 2)) * 100 if len(trades) > 1 else 0
        else:
            win_rate = 0
        
        # Store results
        results[symbol] = {
            'initial_value': initial_value,
            'final_value': final_value,
            'return': returns,
            'trades': len(trades),
            'win_rate': win_rate,
            'trade_history': trades,
            'portfolio_history': portfolio_values
        }
        
        logger.info(f"Backtest results for {symbol}:")
        logger.info(f"  Initial Value: ${initial_value:.2f}")
        logger.info(f"  Final Value: ${final_value:.2f}")
        logger.info(f"  Return: {returns:.2f}%")
        logger.info(f"  Trades: {len(trades)}")
        logger.info(f"  Win Rate: {win_rate:.2f}%")
    
    return results

def main():
    """Main function to train and evaluate optimized ensemble strategy."""
    # Define symbols
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA']
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, historical_data = load_and_prepare_data(symbols, period='1y')
    
    # Load optimized models
    rf_model, gb_model, weights = load_optimized_models()
    
    # Backtest ensemble strategy
    results = backtest_ensemble_strategy(historical_data, rf_model, gb_model, weights)
    
    # Print summary
    logger.info("Backtest Results Summary:")
    for symbol, result in results.items():
        logger.info(f"  {symbol}: Return: {result['return']:.2f}%, Win Rate: {result['win_rate']:.2f}%")
    
    # Save results to file
    with open('logs/optimized_ensemble_results.txt', 'w') as f:
        f.write("Optimized Ensemble Strategy Backtest Results\n")
        f.write("===========================================\n\n")
        for symbol, result in results.items():
            f.write(f"{symbol}:\n")
            f.write(f"  Initial Value: ${result['initial_value']:.2f}\n")
            f.write(f"  Final Value: ${result['final_value']:.2f}\n")
            f.write(f"  Return: {result['return']:.2f}%\n")
            f.write(f"  Trades: {result['trades']}\n")
            f.write(f"  Win Rate: {result['win_rate']:.2f}%\n\n")
    
    logger.info("Results saved to logs/optimized_ensemble_results.txt")

if __name__ == "__main__":
    main()
